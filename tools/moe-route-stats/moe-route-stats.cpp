// moe-route-stats : record per-layer MoE expert routing counts and a decode trace from a live request
//
// usage: llama-moe-route-stats -m model.gguf -f prompt.txt -o routes.bin [-ngl 99 -sm layer -c 32768 -n 512]
//
// the output file layout (native endian):
//   char     magic[4] = "MRS1"
//   uint32   n_layer, n_expert, n_used
//   uint32   n_prefill_tokens, n_decode_tokens
//   uint32   count_prefill[n_layer][n_expert]
//   uint32   count_decode [n_layer][n_expert]
//   uint16   ids_decode[n_decode_tokens][n_layer][n_used]
//
// prompt prefill and greedy decoding are counted separately. For both phases the tool also
// prints the hit rate of two expert cache policies (global top-N pairs, per-layer top-N/n_layer).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#define MOE_TOPK_PREFIX "ffn_moe_topk-"

struct moe_stats {
    int n_layer = 0;
    int n_expert = 0;
    int n_used = 0;

    bool prefill = true;
    int64_t decode_base = 0; // trace row of the next decode batch

    std::vector<uint32_t> count_prefill; // [n_layer][n_expert]
    std::vector<uint32_t> count_decode;

    std::vector<uint16_t> trace; // [decode tokens][n_layer][n_used]

    std::vector<uint32_t> ids; // host scratch for the current tensor
};

static moe_stats g_stats;

static bool moe_route_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    if (ask) {
        return strncmp(t->name, MOE_TOPK_PREFIX, strlen(MOE_TOPK_PREFIX)) == 0;
    }

    if (strncmp(t->name, MOE_TOPK_PREFIX, strlen(MOE_TOPK_PREFIX)) != 0) {
        return true; // not our tensor, continue the graph
    }

    moe_stats & st = g_stats;

    const int il = atoi(t->name + strlen(MOE_TOPK_PREFIX));
    if (il < 0 || il >= st.n_layer) {
        return true;
    }

    const int64_t n_ids     = t->ne[0]; // n_expert_used
    const int64_t n_tokens  = t->ne[1];

    if (t->type != GGML_TYPE_I32) {
        static bool warned = false;
        if (!warned) {
            LOG_ERR("%s : unexpected type %s for %s\n", __func__, ggml_type_name(t->type), t->name);
            warned = true;
        }
        return true;
    }

    // ffn_moe_topk is usually a view of the argsort result: rows are n_expert apart, read the strided span
    if (t->nb[0] != ggml_element_size(t)) {
        static bool warned = false;
        if (!warned) {
            LOG_ERR("%s : unexpected element stride in %s, skipping\n", __func__, t->name);
            warned = true;
        }
        return true;
    }

    if (st.n_used == 0) {
        st.n_used = (int) n_ids;
    } else if (st.n_used != n_ids) {
        static bool warned = false;
        if (!warned) {
            LOG_ERR("%s : inconsistent n_expert_used (%d vs %d) for %s\n", __func__, st.n_used, (int) n_ids, t->name);
            warned = true;
        }
        return true;
    }

    // the tensor can live on the GPU, always copy it back
    const size_t nbytes = ggml_nbytes(t);
    st.ids.resize((nbytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    ggml_backend_tensor_get(t, st.ids.data(), 0, nbytes);

    const char * data = (const char *) st.ids.data();

    if (!st.prefill) {
        const size_t row_size = (size_t) st.n_layer * st.n_used;
        const size_t needed   = ((size_t) st.decode_base + n_tokens) * row_size;
        if (st.trace.size() < needed) {
            st.trace.resize(needed);
        }
    }

    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t k = 0; k < n_ids; ++k) {
            const int32_t ex = *(const int32_t *) (data + i * t->nb[1] + k * t->nb[0]);
            if (ex < 0 || ex >= st.n_expert) {
                continue; // layers without MoE (or corrupted ids) count as zero
            }

            if (st.prefill) {
                st.count_prefill[(size_t) il * st.n_expert + ex]++;
            } else {
                st.count_decode[(size_t) il * st.n_expert + ex]++;

                const size_t off = ((size_t) st.decode_base + i) * st.n_layer * st.n_used + (size_t) il * st.n_used;
                st.trace[off + k] = (uint16_t) ex;
            }
        }
    }

    return true;
}

static void print_cache_hit_rates(const char * phase, const std::vector<uint32_t> & count, int n_layer, int n_expert) {
    static const int cache_sizes[] = { 512, 1024, 2048, 4096, 8192, 12288, 16384 };
    static const int n_cache_sizes = sizeof(cache_sizes) / sizeof(cache_sizes[0]);

    int64_t total = 0;
    for (uint32_t c : count) {
        total += c;
    }

    if (total == 0) {
        LOG_INF("%s : no expert selections recorded\n", phase);
        return;
    }

    // global top-N (layer, expert) pairs
    std::vector<int64_t> sorted(count.begin(), count.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<int64_t>());

    LOG_INF("%s : %" PRId64 " expert selections over %d layers x %d experts\n", phase, total, n_layer, n_expert);
    LOG_INF("%s : cache hit rate, global top-N (layer, expert):\n", phase);
    for (int i = 0; i < n_cache_sizes; ++i) {
        const int    n    = cache_sizes[i];
        const size_t take = std::min((size_t) n, sorted.size());

        int64_t hit = 0;
        for (size_t j = 0; j < take; ++j) {
            hit += sorted[j];
        }
        LOG_INF("  N = %5d : %6.2f%%\n", n, 100.0 * (double) hit / (double) total);
    }

    // per-layer top-(N/n_layer) experts
    LOG_INF("%s : cache hit rate, per-layer top-(N/n_layer):\n", phase);
    for (int i = 0; i < n_cache_sizes; ++i) {
        const int n         = cache_sizes[i];
        const int per_layer = n / n_layer;

        int64_t hit = 0;
        for (int il = 0; il < n_layer; ++il) {
            std::vector<int64_t> layer(count.begin() + (size_t) il * n_expert, count.begin() + (size_t) (il + 1) * n_expert);
            std::sort(layer.begin(), layer.end(), std::greater<int64_t>());

            const int take = std::min(per_layer, n_expert);
            for (int j = 0; j < take; ++j) {
                hit += layer[j];
            }
        }
        LOG_INF("  N = %5d (%3d/layer) : %6.2f%%\n", n, per_layer, 100.0 * (double) hit / (double) total);
    }
}

static bool save_stats(const std::string & fname, const moe_stats & st, uint32_t n_prefill_tokens, uint32_t n_decode_tokens) {
    std::ofstream f(fname, std::ios::binary);
    if (!f) {
        LOG_ERR("%s : failed to open '%s' for writing\n", __func__, fname.c_str());
        return false;
    }

    const char     magic[4] = { 'M', 'R', 'S', '1' };
    const uint32_t header[5] = {
        (uint32_t) st.n_layer, (uint32_t) st.n_expert, (uint32_t) st.n_used,
        n_prefill_tokens, n_decode_tokens,
    };

    f.write(magic, sizeof(magic));
    f.write((const char *) header, sizeof(header));
    f.write((const char *) st.count_prefill.data(), st.count_prefill.size() * sizeof(uint32_t));
    f.write((const char *) st.count_decode.data(),  st.count_decode.size()  * sizeof(uint32_t));

    // pad with 0xFFFF if some trace rows are missing (no decode callback ever fired)
    std::vector<uint16_t> trace = st.trace;
    const size_t trace_size = (size_t) n_decode_tokens * st.n_layer * st.n_used;
    if (trace.size() < trace_size) {
        trace.resize(trace_size, 0xFFFF);
    }
    f.write((const char *) trace.data(), trace_size * sizeof(uint16_t));

    if (!f.good()) {
        LOG_ERR("%s : failed to write '%s'\n", __func__, fname.c_str());
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s : no prompt provided, use -f or -p\n", __func__);
        return 1;
    }
    if (params.out_file.empty()) {
        LOG_ERR("%s : no output file provided, use -o\n", __func__);
        return 1;
    }
    if (params.n_predict < 0) {
        LOG_ERR("%s : no decode length provided, use -n\n", __func__);
        return 1;
    }

    params.cb_eval           = moe_route_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    {
        char arch[128] = { 0 };
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
            LOG_ERR("%s : failed to read general.architecture\n", __func__);
            return 1;
        }

        char key[256];
        snprintf(key, sizeof(key), "%s.expert_count", arch);

        char val[64] = { 0 };
        if (llama_model_meta_val_str(model, key, val, sizeof(val)) < 0) {
            LOG_ERR("%s : failed to read %s\n", __func__, key);
            return 1;
        }

        g_stats.n_layer  = llama_model_n_layer(model);
        g_stats.n_expert = atoi(val);

        if (g_stats.n_layer <= 0 || g_stats.n_expert <= 0) {
            LOG_ERR("%s : invalid model metadata: n_layer = %d, n_expert = %d\n", __func__, g_stats.n_layer, g_stats.n_expert);
            return 1;
        }

        g_stats.count_prefill.assign((size_t) g_stats.n_layer * g_stats.n_expert, 0);
        g_stats.count_decode .assign((size_t) g_stats.n_layer * g_stats.n_expert, 0);

        LOG_INF("%s : arch = %s, n_layer = %d, n_expert = %d\n", __func__, arch, g_stats.n_layer, g_stats.n_expert);
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, params.parse_special);

    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens to process\n", __func__);
        return 1;
    }

    // prefill in chunks of the logical batch size
    const int n_batch = params.n_batch > 0 ? std::min(params.n_batch, (int) llama_n_batch(ctx)) : (int) llama_n_batch(ctx);

    LOG_INF("%s : %zu prompt tokens, prefill in chunks of %d\n", __func__, tokens.size(), n_batch);

    g_stats.prefill = true;

    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int n = (int) std::min((size_t) n_batch, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("%s : llama_decode failed during prefill\n", __func__);
            return 1;
        }
    }

    // greedy decoding, one token at a time so that each trace row is a single step
    llama_sampler * smpl = llama_sampler_init_greedy();

    llama_token id = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_accept(smpl, id, true);

    g_stats.prefill = false;

    int64_t n_decode = 0;

    for (int i = 0; i < params.n_predict; ++i) {
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        g_stats.decode_base = n_decode;

        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            LOG_ERR("%s : llama_decode failed during decode\n", __func__);
            llama_sampler_free(smpl);
            return 1;
        }
        ++n_decode;

        id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id, true);
    }

    llama_sampler_free(smpl);

    LOG_INF("%s : prefill tokens = %zu, decode tokens = %" PRId64 "\n", __func__, tokens.size(), n_decode);

    print_cache_hit_rates("prefill", g_stats.count_prefill, g_stats.n_layer, g_stats.n_expert);
    print_cache_hit_rates("decode",  g_stats.count_decode,  g_stats.n_layer, g_stats.n_expert);

    if (!save_stats(params.out_file, g_stats, (uint32_t) tokens.size(), (uint32_t) n_decode)) {
        return 1;
    }

    LOG_INF("%s : stats written to '%s'\n", __func__, params.out_file.c_str());

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
