// bench-moe-decode: the MoE FFN half layer of a Flash-Next decode/verify step on one device of an
// expert parallel split, with cold weights, through the CUDA backend.
//
// A real verify step runs this pattern 48 times per round on every device: the F32 router GEMV,
// the fused softmax/top-k/weight normalization, the gate_up MUL_MAT_ID with the SwiGLU epilogue
// (row interleaved Q4_K weight), the down MUL_MAT_ID (Q5_1/Q8_0) and the weighted sum over the
// slots. The device holds a contiguous slice of the experts (expert_offset = shard * slice), and
// the MUL_MAT_ID nodes carry the expert parallel flags, so the kernels see the routing of a real
// device: about 1/D of the (token, slot) pairs are local.
//
// The graph holds L independent copies of the pattern (distinct inputs and router weights, the
// expert tensors shared: every copy routes to different experts, so the expert reads stay cold),
// is run through the CUDA backend (the same fusion matcher and CUDA graph as the server) and the
// time per copy is reported. --dump writes the outputs of the first copy (moe_out, ids, weights)
// to compare two builds bit for bit; the inputs are seeded, so every run sees the same values.
//
// Not part of the CMake build. Build against a llama.cpp build tree:
//   g++ -O2 -std=c++17 -I<src>/ggml/include tests/bench-moe-decode.cpp -o bench-moe-decode \
//       -L<build>/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,<build>/bin
// Usage:
//   bench-moe-decode [--dev 0] [--iters 50] [--n 1,4] [--devices 4] [--shard 1] [--layers 24]
//                    [--what full|experts] [--down q5_1|q8_0] [--gu q4_k|q5_k|q8_0] [--dump <prefix>]
// Output: one line per n: what, n, devices, layers, us per copy, local pairs per copy.

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

static const int64_t N_EMBD    = 2560;
static const int64_t N_FF      = 640;
static const int64_t N_EXPERT  = 512;
static const int64_t N_USED    = 10;

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t e = str.find(',', pos);
        if (e == std::string::npos) {
            e = str.size();
        }
        v.push_back(atoi(str.substr(pos, e - pos).c_str()));
        pos = e + 1;
    }
    return v;
}

static ggml_type parse_type(const char * s) {
    if (!strcmp(s, "q4_k")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "q5_k")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "q5_1")) return GGML_TYPE_Q5_1;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q6_k")) return GGML_TYPE_Q6_K;
    fprintf(stderr, "unknown type %s\n", s);
    exit(1);
}

static void fill_weight(std::vector<uint8_t> & bytes, ggml_type type, int64_t k, int64_t rows, std::mt19937 & rng) {
    std::vector<float> f((size_t) k);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    const size_t row_size = ggml_row_size(type, k);
    bytes.resize(row_size * rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (float & v : f) {
            v = nd(rng);
        }
        ggml_quantize_chunk(type, f.data(), bytes.data() + r*row_size, 0, 1, k, nullptr);
    }
}

static void set_ep(ggml_tensor * mm, int64_t offset) {
    ggml_mul_mat_id_set_expert_parallel(mm, true);
    ggml_mul_mat_id_set_expert_offset(mm, (int32_t) offset);
    ggml_mul_mat_id_set_n_expert_total(mm, (int32_t) N_EXPERT);
}

struct run_result {
    double us_per_copy;
    double local_pairs;
};

static run_result run(ggml_backend_t backend, const std::string & what, int64_t n, int devices, int shard, int layers,
                      int iters, ggml_type type_gu, ggml_type type_down, const std::string & dump) {
    const int64_t slice  = N_EXPERT / devices;
    const int64_t offset = slice * shard;
    const bool    full   = what == "full";

    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) layers * 64 + ggml_graph_overhead_custom(16384, false), nullptr, true };
    ggml_context * ctx   = ggml_init(ip);
    ggml_context * ctx_w = ggml_init(ip);

    ggml_tensor * gate_up = ggml_new_tensor_3d(ctx_w, type_gu,   N_EMBD, 2*N_FF, slice);
    ggml_tensor * down    = ggml_new_tensor_3d(ctx_w, type_down, N_FF,   N_EMBD, slice);
    std::vector<ggml_tensor *> gate_inp(layers, nullptr), xs(layers), ids_in(layers, nullptr), w_in(layers, nullptr);
    ggml_tensor * out0 = nullptr;
    ggml_tensor * ids0 = nullptr;
    ggml_tensor * wts0 = nullptr;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    for (int l = 0; l < layers; ++l) {
        xs[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n);
        ggml_tensor * selected = nullptr;
        ggml_tensor * weights  = nullptr;
        if (full) {
            // build_moe_ffn: softmax gating, top-k, normalized weights (fused into topk_moe)
            gate_inp[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, N_EXPERT);
            ggml_tensor * logits = ggml_mul_mat(ctx, gate_inp[l], xs[l]);
            ggml_tensor * probs  = ggml_soft_max(ctx, logits);
            selected = ggml_argsort_top_k(ctx, probs, N_USED);
            weights  = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n), selected);
            weights  = ggml_reshape_2d(ctx, weights, N_USED, n);
            ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
            wsum     = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
            weights  = ggml_div(ctx, weights, wsum);
            weights  = ggml_reshape_3d(ctx, weights, 1, N_USED, n);
            ggml_build_forward_expand(gf, weights);
        } else {
            ids_in[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n);
            w_in[l]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, n);
            selected  = ids_in[l];
            weights   = w_in[l];
        }
        ggml_tensor * cur = ggml_reshape_3d(ctx, xs[l], N_EMBD, 1, n);
        ggml_tensor * h   = ggml_mul_mat_id_swiglu(ctx, gate_up, cur, selected);
        set_ep(h, offset);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, down, h, selected);
        set_ep(experts, offset);
        experts = ggml_mul(ctx, experts, weights);
        ggml_build_forward_expand(gf, experts);
        ggml_tensor * views[N_USED];
        for (int i = 0; i < N_USED; ++i) {
            views[i] = ggml_view_2d(ctx, experts, N_EMBD, n, experts->nb[2], i*experts->nb[1]);
            ggml_build_forward_expand(gf, views[i]);
        }
        ggml_tensor * moe_out = views[0];
        for (int i = 1; i < N_USED; ++i) {
            moe_out = ggml_add(ctx, moe_out, views[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        if (l == 0) {
            out0 = moe_out;
            ids0 = selected;
            wts0 = weights;
            ggml_set_output(out0);
            ggml_set_output(ids0);
            ggml_set_output(wts0);
        }
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (buf_w == nullptr) {
        fprintf(stderr, "weight allocation failed\n");
        exit(1);
    }
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "allocation failed for n=%lld layers=%d\n", (long long) n, layers);
        exit(1);
    }

    std::mt19937 rng(1234);
    {
        std::vector<uint8_t> bytes;
        fill_weight(bytes, type_gu, N_EMBD, 2*N_FF*slice, rng);
        ggml_backend_tensor_set(gate_up, bytes.data(), 0, bytes.size());
        fill_weight(bytes, type_down, N_FF, N_EMBD*slice, rng);
        ggml_backend_tensor_set(down, bytes.data(), 0, bytes.size());
    }
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<int> perm(N_EXPERT);
    for (int i = 0; i < N_EXPERT; ++i) {
        perm[i] = i;
    }
    for (int l = 0; l < layers; ++l) {
        std::vector<float> v(N_EMBD * n);
        for (float & f : v) {
            f = ud(rng);
        }
        ggml_backend_tensor_set(xs[l], v.data(), 0, ggml_nbytes(xs[l]));
        if (full) {
            std::vector<float> g(N_EMBD * N_EXPERT);
            for (float & f : g) {
                f = 0.05f * ud(rng);
            }
            ggml_backend_tensor_set(gate_inp[l], g.data(), 0, ggml_nbytes(gate_inp[l]));
        } else {
            std::vector<int32_t> ids(N_USED * n);
            std::vector<float>   w(N_USED * n);
            for (int64_t t = 0; t < n; ++t) {
                std::shuffle(perm.begin(), perm.end(), rng);
                float sum = 0.0f;
                for (int i = 0; i < N_USED; ++i) {
                    ids[t*N_USED + i] = perm[i];
                    w[t*N_USED + i]   = 0.05f + 0.5f*(ud(rng) + 1.0f);
                    sum += w[t*N_USED + i];
                }
                for (int i = 0; i < N_USED; ++i) {
                    w[t*N_USED + i] /= sum;
                }
            }
            ggml_backend_tensor_set(ids_in[l], ids.data(), 0, ggml_nbytes(ids_in[l]));
            ggml_backend_tensor_set(w_in[l], w.data(), 0, ggml_nbytes(w_in[l]));
        }
    }

    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters / layers;

    // local pairs of the first copy (the routing is the same in every build); the routed ids are a
    // strided view of the argsort result in the full pattern, so read them row by row
    std::vector<int32_t> ids_h(N_USED * n);
    for (int64_t t = 0; t < n; ++t) {
        ggml_backend_tensor_get(ids0, ids_h.data() + t*N_USED, t*ids0->nb[1], N_USED * sizeof(int32_t));
    }
    int n_local = 0;
    for (int32_t e : ids_h) {
        n_local += e >= offset && e < offset + slice;
    }

    if (!dump.empty()) {
        char path[512];
        snprintf(path, sizeof(path), "%s-%s-n%lld-d%d-s%d.bin", dump.c_str(), what.c_str(), (long long) n, devices, shard);
        FILE * f = fopen(path, "wb");
        if (f) {
            std::vector<float> o(ggml_nelements(out0));
            ggml_backend_tensor_get(out0, o.data(), 0, ggml_nbytes(out0));
            fwrite(o.data(), sizeof(float), o.size(), f);
            fwrite(ids_h.data(), sizeof(int32_t), ids_h.size(), f);
            std::vector<float> w(ggml_nelements(wts0));
            ggml_backend_tensor_get(wts0, w.data(), 0, ggml_nbytes(wts0));
            fwrite(w.data(), sizeof(float), w.size(), f);
            fclose(f);
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return { us, (double) n_local };
}

int main(int argc, char ** argv) {
    int dev = 0;
    int iters = 50;
    int devices = 4;
    int shard = 1;
    int layers = 24;
    std::string what = "full";
    std::string dump;
    std::vector<int> ns = { 1, 4 };
    ggml_type type_gu   = GGML_TYPE_Q4_K;
    ggml_type type_down = GGML_TYPE_Q5_1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "--dev") dev = atoi(next());
        else if (a == "--iters") iters = atoi(next());
        else if (a == "--devices") devices = atoi(next());
        else if (a == "--shard") shard = atoi(next());
        else if (a == "--layers") layers = atoi(next());
        else if (a == "--what") what = next();
        else if (a == "--dump") dump = next();
        else if (a == "--n") ns = parse_list(next());
        else if (a == "--gu") type_gu = parse_type(next());
        else if (a == "--down") type_down = parse_type(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (what != "full" && what != "experts") {
        fprintf(stderr, "--what must be full or experts\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(dev);
    if (!backend) {
        fprintf(stderr, "no CUDA device %d\n", dev);
        return 1;
    }
    for (int n : ns) {
        const run_result r = run(backend, what, n, devices, shard, layers, iters, type_gu, type_down, dump);
        printf("moe %s n=%d devices=%d shard=%d layers=%d gu=%s down=%s us_per_copy %.2f local_pairs %.0f\n",
               what.c_str(), n, devices, shard, layers, ggml_type_name(type_gu), ggml_type_name(type_down),
               r.us_per_copy, r.local_pairs);
    }
    ggml_backend_free(backend);
    return 0;
}
