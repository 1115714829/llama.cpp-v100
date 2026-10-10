// bench-q8-skinny-cold: sm_70 Q8_0 skinny GEMV timing with cold weights (DRAM, not L2).
//
// test-backend-ops perf repeats one MUL_MAT on one weight, so a weight below the 6 MiB L2 of a
// V100 (the Flash-Next HC down/up projections are 3.4 MiB, the shared expert 1.7 MiB) is timed
// from L2. In a real decode step every weight is read from DRAM once. This program builds one
// graph with R independent copies of the pattern (R distinct weights and inputs, R chosen so
// that the weights of one graph exceed --min-mib), runs it through the CUDA backend (the same
// fusion matcher, skinny prepass and CUDA graph as the server) and reports the time per copy:
// the input conversion, the kernel and the gap to the next node, as in the decode graph.
//
// Patterns:
//   single  MUL_MAT(W[K,N], x[K,M])                                q8_skinny_kernel
//   gated   SWIGLU(MUL_MAT(G, x), MUL_MAT(U, x))                    q8_skinny_gated_kernel
//   ffn     MUL_MAT(D, SWIGLU(MUL_MAT(G, x), MUL_MAT(U, x)))         gated pair + down projection
//   multi   MUL_MAT(W_i, x) for 2..4 weights sharing one input     q8_skinny_multi_kernel
//   hcpost  gated DSV4_HC_POST -> RMS_NORM -> MUL (the hyper-connection combine and next mixer norm)
//   hcmix   hcpost followed by the Q8_0 down projection, SCALE -> SILU and the up projection
// Not part of the CMake build. Build against a llama.cpp build tree:
//   g++ -O2 -std=c++17 -I<src>/ggml/include tests/bench-q8-skinny-cold.cpp -o bench-q8-skinny-cold \
//       -L<build>/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,<build>/bin
// Usage:
//   bench-q8-skinny-cold [--dev 0] [--iters 30] [--min-mib 256] [--set moe|27b|mmvf|hc|all] [--what <case>] [--m 1,4,8] [--dump <prefix>]
// Output: one line per case: pattern, K, N list, M, R, us per copy, GB/s of weight bytes.
// --dump writes the F32 results of the first copy per case to <prefix>-<case>-m<M>.bin, to
// compare two kernel variants bit for bit (the weights and inputs are the same in every run).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct bench_case {
    const char *         set;
    const char *         pattern; // "single", "gated", "multi"
    const char *         what;
    int64_t              k;
    std::vector<int64_t> n;       // one entry for single, the gate/up width for gated, 2..4 for multi
    ggml_type            type = GGML_TYPE_Q8_0;
};

static std::vector<bench_case> make_cases() {
    return {
        // Qwen3.8-Flash-Next under a 6-card tensor split, per card
        { "moe", "single", "hc_down",       10240, { 320 } },
        { "moe", "single", "hc_up",           320, { 10240 } },
        { "moe", "gated",  "shexp_gate_up",  2560, { 640 } },
        { "moe", "ffn",    "shexp_ffn",      2560, { 640 } },
        { "moe", "single", "shexp_down",      640, { 2560 } },
        { "moe", "single", "ssm_out_3h",     1152, { 2560 } },
        { "moe", "single", "ssm_out_2h",      768, { 2560 } },
        { "moe", "single", "attn_output",    1024, { 2560 } },
        { "moe", "single", "eh_proj",        5120, { 2560 } },
        { "moe", "multi",  "gdn_qkv_z_3h",   2560, { 1920, 1152 } },
        { "moe", "multi",  "gdn_qkv_z_2h",   2560, { 1280, 768 } },
        { "moe", "multi",  "qsa_q_k_v",      2560, { 2048, 256, 256 } },
        // the hyper-connection combine (gated DSV4_HC_POST) and the next mixer's RMS_NORM -> MUL
        { "hc", "hcpost", "hc_post_norm",  2560, { 4 }, GGML_TYPE_F32 },
        // the whole mixer opening: the combine, the norm, the Q8_0 down/up projections (n = { hc, lr })
        { "hc", "hcmix",  "hc_mix",        2560, { 4, 320 }, GGML_TYPE_Q8_0 },
        // the F32/BF16 projections of the decode graph (mul_mat_vec_f)
        { "mmvf", "single", "hc_inject",     10240, { 4 },   GGML_TYPE_F32 },
        { "mmvf", "single", "gate_inp",       2560, { 512 }, GGML_TYPE_F32 },
        { "mmvf", "single", "gate_inp_shexp", 2560, { 1 },   GGML_TYPE_F32 },
        { "mmvf", "single", "ssm_beta",       2560, { 8 },   GGML_TYPE_F32 },
        { "mmvf", "single", "idx_q_proj",     2560, { 512 }, GGML_TYPE_BF16 },
        { "mmvf", "single", "idx_k_proj",     2560, { 128 }, GGML_TYPE_BF16 },
        // Qwen3.8-27B under a 4-card tensor split, per card (regression check)
        { "27b", "gated",  "ffn_gate_up",    5120, { 4352 } },
        { "27b", "single", "ffn_down",       4352, { 5120 } },
        { "27b", "single", "ssm_out",        1536, { 5120 } },
        { "27b", "multi",  "gdn_qkv_z",      5120, { 2560, 1536 } },
        { "27b", "multi",  "attn_q_k_v",     5120, { 3072, 256, 256 } },
    };
}

static void fill_weight(std::vector<uint8_t> & bytes, ggml_type type, int64_t k, int64_t n, std::mt19937 & rng) {
    std::vector<float> f((size_t) (k * n));
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (float & v : f) {
        v = nd(rng);
    }
    bytes.resize(ggml_row_size(type, k) * n);
    if (type == GGML_TYPE_F32) {
        memcpy(bytes.data(), f.data(), bytes.size());
    } else if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) bytes.data(), k * n);
    } else {
        ggml_quantize_chunk(type, f.data(), bytes.data(), 0, n, k, nullptr);
    }
}

// DSV4_HC_POST with the qwen4exp gate (post = 2*sigmoid(inject/hc)) followed by RMS_NORM -> MUL, as
// the combine of one block and the start of the next mixer; k = n_embd, n = { hc }. The bytes are
// those read and written by the unfused pair.
static double run_hcpost(ggml_backend_t backend, const bench_case & c, int64_t m, int iters, double min_mib,
                         int * r_out, double * gb_out, const std::string & dump) {
    const bool    mix    = strcmp(c.pattern, "hcmix") == 0;
    const int64_t n_embd = c.k;
    const int64_t hc     = c.n[0];
    const int64_t n_lr   = mix ? c.n[1] : 0;
    const size_t w_bytes = mix ? 2 * ggml_row_size(GGML_TYPE_Q8_0, n_embd*hc) * n_lr : 0;
    const size_t copy_bytes = sizeof(float) * (size_t) (n_embd*m + 3*n_embd*hc*m + hc*m) + w_bytes;
    int r = (int) std::ceil(min_mib * 1024.0 * 1024.0 / (double) copy_bytes);
    r = std::max(4, std::min(r, mix ? 512 : 1024));

    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) r * 20 + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx   = ggml_init(ip);
    ggml_context * ctx_w = ggml_init(ip);
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);
    std::vector<ggml_tensor *> xs(r), res(r), posts(r), w_down(r), w_up(r), first_outs;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    for (int i = 0; i < r; ++i) {
        xs[i]    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, m);
        res[i]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, m);
        posts[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, m);
        ggml_tensor * post = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, posts[i], 1.0f / (float) hc)), 2.0f);
        ggml_tensor * hp   = ggml_dsv4_hc_post(ctx, xs[i], res[i], post, nullptr);
        ggml_tensor * xn   = ggml_mul(ctx, ggml_rms_norm(ctx, hp, 1e-6f), gamma);
        if (mix) {
            // build_hc_mix up to the gate
            w_down[i] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q8_0, n_embd*hc, n_lr);
            w_up[i]   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q8_0, n_lr, n_embd*hc);
            ggml_tensor * x2   = ggml_reshape_2d(ctx, xn, n_embd*hc, m);
            ggml_tensor * lo   = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w_down[i], x2), 1.0f / (float) hc));
            ggml_tensor * gate = ggml_mul_mat(ctx, w_up[i], lo);
            ggml_build_forward_expand(gf, gate);
            ggml_build_forward_expand(gf, xn);
            if (i == 0) {
                first_outs = { hp, xn, gate };
            }
        } else {
            ggml_build_forward_expand(gf, xn);
            if (i == 0) {
                first_outs = { hp, xn };
            }
        }
    }
    ggml_backend_buffer_t buf   = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_t buf_w = mix ? ggml_backend_alloc_ctx_tensors(ctx_w, backend) : nullptr;
    if (buf == nullptr || (mix && buf_w == nullptr)) {
        fprintf(stderr, "allocation failed for %s m=%lld r=%d\n", c.what, (long long) m, r);
        exit(1);
    }
    if (buf_w) {
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> v(ggml_nelements(t));
        for (float & f : v) {
            f = ud(rng);
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    };
    fill(gamma);
    for (int i = 0; i < r; ++i) {
        fill(xs[i]);
        fill(res[i]);
        fill(posts[i]);
    }
    if (mix) {
        std::vector<uint8_t> wd, wu;
        fill_weight(wd, GGML_TYPE_Q8_0, n_embd*hc, n_lr, rng);
        fill_weight(wu, GGML_TYPE_Q8_0, n_lr, n_embd*hc, rng);
        for (int i = 0; i < r; ++i) {
            ggml_backend_tensor_set(w_down[i], wd.data(), 0, wd.size());
            ggml_backend_tensor_set(w_up[i],   wu.data(), 0, wu.size());
        }
    }
    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ((double) iters * r);
    *r_out  = r;
    *gb_out = (double) copy_bytes / (us * 1e3);
    if (!dump.empty()) {
        const std::string path = dump + "-" + c.what + "-m" + std::to_string(m) + ".bin";
        FILE * f = fopen(path.c_str(), "wb");
        for (ggml_tensor * o : first_outs) {
            std::vector<float> v(ggml_nelements(o));
            ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
            fwrite(v.data(), sizeof(float), v.size(), f);
        }
        fclose(f);
    }
    ggml_backend_buffer_free(buf);
    if (buf_w) {
        ggml_backend_buffer_free(buf_w);
    }
    ggml_free(ctx);
    ggml_free(ctx_w);
    return us;
}

static double run_case(ggml_backend_t backend, const bench_case & c, int64_t m, int iters, double min_mib,
                       int * r_out, double * gb_out, const std::string & dump) {
    const bool ffn    = strcmp(c.pattern, "ffn") == 0;
    const bool gated  = strcmp(c.pattern, "gated") == 0 || ffn;
    const bool multi  = strcmp(c.pattern, "multi") == 0;
    const bool hcpost = strcmp(c.pattern, "hcpost") == 0;
    if (hcpost || strcmp(c.pattern, "hcmix") == 0) {
        return run_hcpost(backend, c, m, iters, min_mib, r_out, gb_out, dump);
    }

    // weights of one copy of the pattern
    std::vector<int64_t> ws;
    if (gated) {
        ws = { c.n[0], c.n[0] };
        if (ffn) {
            ws.push_back(c.n[0]); // the down projection [n, k]: the bytes of a [k, n] weight
        }
    } else {
        ws = c.n;
    }
    size_t copy_bytes = 0;
    for (int64_t n : ws) {
        copy_bytes += ggml_row_size(c.type, c.k) * n;
    }
    int r = (int) std::ceil(min_mib * 1024.0 * 1024.0 / (double) copy_bytes);
    r = std::max(4, std::min(r, 1024));

    const size_t n_tensors = (size_t) r * (ws.size() + 4);
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_context * ctx   = ggml_init(ip);

    std::vector<std::vector<ggml_tensor *>> w(r);
    std::vector<ggml_tensor *> x(r);
    for (int i = 0; i < r; ++i) {
        for (size_t j = 0; j < ws.size(); ++j) {
            // the down projection of the ffn pattern maps the pair width back to k
            w[i].push_back(ffn && j == 2 ? ggml_new_tensor_2d(ctx_w, c.type, c.n[0], c.k)
                                         : ggml_new_tensor_2d(ctx_w, c.type, c.k, ws[j]));
        }
        // Q8_0 copies get their own input, so that the multi-weight matcher does not merge
        // neighbouring copies; the F32/BF16 GEMV is never merged and shares one hot input,
        // as in the decode graph where the input was just written
        x[i] = (c.type == GGML_TYPE_Q8_0 || i == 0) ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, m) : x[0];
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    std::vector<ggml_tensor *> first_outs; // the results of copy 0, for --dump
    for (int i = 0; i < r; ++i) {
        if (gated) {
            ggml_tensor * up   = ggml_mul_mat(ctx, w[i][1], x[i]);
            ggml_tensor * gate = ggml_mul_mat(ctx, w[i][0], x[i]);
            ggml_tensor * out  = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            if (ffn) {
                out = ggml_mul_mat(ctx, w[i][2], out);
            }
            ggml_build_forward_expand(gf, out);
            if (i == 0) {
                first_outs.push_back(out);
            }
        } else if (multi) {
            std::vector<ggml_tensor *> outs;
            for (ggml_tensor * wi : w[i]) {
                outs.push_back(ggml_mul_mat(ctx, wi, x[i]));
            }
            for (ggml_tensor * o : outs) {
                ggml_build_forward_expand(gf, o);
            }
            if (i == 0) {
                first_outs = outs;
            }
        } else {
            ggml_tensor * out = ggml_mul_mat(ctx, w[i][0], x[i]);
            ggml_build_forward_expand(gf, out);
            if (i == 0) {
                first_outs.push_back(out);
            }
        }
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_t buf   = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf_w == nullptr || buf == nullptr) {
        fprintf(stderr, "allocation failed for %s m=%lld r=%d\n", c.what, (long long) m, r);
        exit(1);
    }
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(1234);
    std::vector<std::vector<uint8_t>> wdata(ws.size());
    for (size_t j = 0; j < ws.size(); ++j) {
        if (ffn && j == 2) {
            fill_weight(wdata[j], c.type, c.n[0], c.k, rng);
        } else {
            fill_weight(wdata[j], c.type, c.k, ws[j], rng);
        }
    }
    std::vector<float> xdata((size_t) (c.k * m));
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (float & v : xdata) {
        v = ud(rng);
    }
    for (int i = 0; i < r; ++i) {
        for (size_t j = 0; j < ws.size(); ++j) {
            ggml_backend_tensor_set(w[i][j], wdata[j].data(), 0, wdata[j].size());
        }
        if (i == 0 || x[i] != x[0]) {
            ggml_backend_tensor_set(x[i], xdata.data(), 0, xdata.size() * sizeof(float));
        }
    }

    // the first compute repacks the weights, the next two warm up the graph capture
    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ((double) iters * r);

    *r_out  = r;
    *gb_out = (double) copy_bytes / (us * 1e3);

    if (!dump.empty()) {
        // raw F32 results of copy 0, to compare two kernel variants bit for bit
        const std::string path = dump + "-" + c.what + "-m" + std::to_string(m) + ".bin";
        FILE * f = fopen(path.c_str(), "wb");
        for (ggml_tensor * o : first_outs) {
            std::vector<float> v(ggml_nelements(o));
            ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
            fwrite(v.data(), sizeof(float), v.size(), f);
        }
        fclose(f);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return us;
}

int main(int argc, char ** argv) {
    int dev = 0;
    int iters = 30;
    double min_mib = 256.0;
    std::string set = "all";
    std::string what;
    std::string dump;
    std::vector<int64_t> ms = { 1, 4, 8 };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dev" && i + 1 < argc) {
            dev = atoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (a == "--min-mib" && i + 1 < argc) {
            min_mib = atof(argv[++i]);
        } else if (a == "--set" && i + 1 < argc) {
            set = argv[++i];
        } else if (a == "--what" && i + 1 < argc) {
            what = argv[++i];
        } else if (a == "--dump" && i + 1 < argc) {
            dump = argv[++i];
        } else if (a == "--m" && i + 1 < argc) {
            ms.clear();
            std::string s = argv[++i];
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) {
                    q = s.size();
                }
                ms.push_back(atoll(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else {
            fprintf(stderr, "usage: %s [--dev 0] [--iters 30] [--min-mib 256] [--set moe|27b|mmvf|hc|all] [--what <case>] [--m 1,4,8] [--dump <prefix>]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_t backend = ggml_backend_cuda_init(dev);
    if (backend == nullptr) {
        fprintf(stderr, "no CUDA device %d\n", dev);
        return 1;
    }

    printf("%-6s %-7s %-14s %6s %-18s %3s %5s %10s %8s\n", "set", "pattern", "what", "K", "N", "M", "R", "us/copy", "GB/s");
    for (const bench_case & c : make_cases()) {
        if ((set != "all" && set != c.set) || (!what.empty() && what != c.what)) {
            continue;
        }
        for (int64_t m : ms) {
            int r = 0;
            double gb = 0.0;
            const double us = run_case(backend, c, m, iters, min_mib, &r, &gb, dump);
            std::string ns;
            for (size_t j = 0; j < c.n.size(); ++j) {
                ns += (j ? "+" : "") + std::to_string(c.n[j]);
            }
            printf("%-6s %-7s %-14s %6lld %-18s %3lld %5d %10.2f %8.1f\n", c.set, c.pattern, c.what,
                   (long long) c.k, ns.c_str(), (long long) m, r, us, gb);
            fflush(stdout);
        }
    }

    ggml_backend_free(backend);
    return 0;
}
