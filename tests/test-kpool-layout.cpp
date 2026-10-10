// Checks the k-pool layout that llama_memory_hybrid_idx keeps across decodes against one rebuilt from scratch.
//
// kpool_layout_update keeps each sequence's cells before the position an edit staled, with the pools made of them,
// and appends the rest; a rebuild is the same code keeping nothing. These are the edits that reach it:
//  - speculative decoding: decode a draft of k tokens, then seq_rm its rejected tail, every round - also when none is
//    rejected, which removes nothing but still stales from there, as the server does
//  - seq_rm of a range in the middle, which leaves a gap
//  - seq_add of the tail down (overlapping positions) and up (a gap), on models that can shift
//  - clearing a sequence and filling it again
//  - two sequences in their own streams, rolled back in turn and in one batch
//  - two sequences sharing cells after seq_cp, the copy rolled back and then dropped
// The layout is compared after every decode. After the speculative rounds the next token's logits must also match a
// context that decoded the accepted tokens straight.
//
// Models without a k-pool indexer are skipped. The dummy models of test-llama-archs (-o DIR) cover both pool kinds:
// qwen4exp groups cells in sequence order, glm5-next groups consecutive positions.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[m";
    }
    return "";
}

static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

// one run of tokens [p0, p1) of sequence s: its own tokens before p_bad, rejected draft tokens from there
struct span {
    llama_seq_id s;
    llama_pos    p0;
    llama_pos    p1;
    llama_pos    p_bad = std::numeric_limits<llama_pos>::max();
};

struct kpool_case {
    llama_context_ptr         ctx;
    llama_memory_hybrid_idx * mem = nullptr;

    int n_vocab  = 0;
    int n_checks = 0;

    std::mt19937 rng{42};

    const char * name = "";

    llama_memory_t mem_i() const { return llama_get_memory(ctx.get()); }

    llama_token tok(llama_seq_id s, llama_pos pos, bool bad = false) const {
        // a rejected draft token is the next id, so it always differs from the sequence's own
        return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) s + 1 + (bad ? 1 : 0)) % (uint32_t) n_vocab);
    }

    bool decode(const std::vector<span> & spans) {
        common_batch batch(ctx.get());
        for (const auto & sp : spans) {
            for (llama_pos pos = sp.p0; pos < sp.p1; ++pos) {
                batch.add(tok(sp.s, pos, pos >= sp.p_bad), pos, sp.s, pos + 1 == sp.p1);
            }
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: %s: decode failed\n", __func__, name);
            return false;
        }
        return true;
    }

    // the layout as the decode left it against a rebuild over the cache
    bool check(const char * what, int step) {
        ++n_checks;
        const std::string diff = mem->kpool_layout_diff();
        if (!diff.empty()) {
            LOG_ERR("%s: %s: %s step %d: kept layout differs from a rebuild: %s\n", __func__, name, what, step, diff.c_str());
            return false;
        }
        return true;
    }

    bool decode_check(const std::vector<span> & spans, const char * what, int step) {
        return decode(spans) && check(what, step);
    }

    llama_pos pos_next(llama_seq_id s) const {
        return llama_memory_seq_pos_max(mem_i(), s) + 1;
    }

    // verify rounds on seq s: a draft of 1..4 tokens, the first 1..k of them right, then the server's seq_rm of the rest
    bool spec_rounds(llama_seq_id s, int n_rounds, const char * what) {
        for (int r = 0; r < n_rounds; ++r) {
            const llama_pos pos   = pos_next(s);
            const int       k     = std::uniform_int_distribution<int>(1, 4)(rng);
            const int       n_acc = std::uniform_int_distribution<int>(1, k)(rng);
            if (!decode_check({ { s, pos, pos + k, pos + n_acc } }, what, r)) {
                return false;
            }
            if (!llama_memory_seq_rm(mem_i(), s, pos + n_acc, -1)) {
                LOG_ERR("%s: %s: %s step %d: rolling back %d of %d tokens was refused\n", __func__, name, what, r, k - n_acc, k);
                return false;
            }
        }
        return true;
    }
};

// false with res set when the case cannot run: FAIL if the context fails, SKIP without a k-pool indexer
static bool make_case(kpool_case & tc, const common_params & params, llama_model * model, const char * name,
        uint32_t n_seq_max, bool unified, test_status & res) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = 8;
    cparams.n_ctx      = 1024*n_seq_max;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 32;
    cparams.kv_unified = unified;

    tc.name    = name;
    tc.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    tc.ctx.reset(llama_init_from_model(model, cparams));
    if (!tc.ctx) {
        LOG_ERR("%s: %s: failed to init context\n", __func__, name);
        res = test_status::FAIL;
        return false;
    }

    tc.mem = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(tc.ctx.get()));
    if (tc.mem == nullptr || tc.mem->get_mem_idx() == nullptr || tc.mem->get_kpool() == 0) {
        LOG_INF("%s: %s: no k-pool indexer, skipping\n", __func__, name);
        res = test_status::SKIP;
        return false;
    }

    if (llama_n_rs_seq(tc.ctx.get()) < 3) {
        LOG_INF("%s: %s: n_rs_seq too small for the rollbacks, skipping\n", __func__, name);
        res = test_status::SKIP;
        return false;
    }

    return true;
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double d = (double) a[i] - b[i];
        mse_ab += d*d;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// One sequence through every edit kind.
static test_status test_single(const common_params & params, llama_model * model) {
    kpool_case tc;
    test_status res = test_status::SKIP;
    if (!make_case(tc, params, model, "single", 1, true, res)) {
        return res;
    }

    // a prefill over three ubatches, ending inside a pool
    if (!tc.decode_check({ { 0, 0, 71 } }, "prefill", 0) ||
        !tc.spec_rounds(0, 48, "spec")) {
        return test_status::FAIL;
    }

    // the next token against a context that decoded the accepted tokens straight
    {
        kpool_case ref;
        test_status res_ref = test_status::SKIP;
        if (!make_case(ref, params, model, "single-ref", 1, true, res_ref)) {
            return test_status::FAIL;
        }

        const llama_pos pos = tc.pos_next(0);
        for (llama_pos p0 = 0; p0 < pos; p0 += 256) {
            if (!ref.decode_check({ { 0, p0, std::min<llama_pos>(p0 + 256, pos) } }, "ref", p0)) {
                return test_status::FAIL;
            }
        }
        if (!tc.decode_check({ { 0, pos, pos + 1 } }, "next", 0) || !ref.decode_check({ { 0, pos, pos + 1 } }, "next", 0)) {
            return test_status::FAIL;
        }

        const float * a = llama_get_logits_ith(tc.ctx.get(),  0);
        const float * b = llama_get_logits_ith(ref.ctx.get(), 0);
        if (a == nullptr || b == nullptr) {
            LOG_ERR("%s: missing logits\n", __func__);
            return test_status::FAIL;
        }
        const double e = nmse(a, b, tc.n_vocab);
        if (e > 1e-4) {
            LOG_ERR("%s: logits after %d positions of verify rounds differ from a straight decode, nmse %g\n", __func__, (int) pos, e);
            return test_status::FAIL;
        }
        LOG_INF("%s: logits after %d positions of verify rounds match a straight decode, nmse %g\n", __func__, (int) pos, e);
        tc.n_checks += ref.n_checks;
    }

    // a range in the middle, off the pool grid, leaves a gap that later pools must step over
    {
        const llama_pos pos = tc.pos_next(0);
        const llama_pos a   = pos/3 + 1;
        if (!llama_memory_seq_rm(tc.mem_i(), 0, a, a + 6)) {
            LOG_ERR("%s: removing [%d, %d) was refused\n", __func__, (int) a, (int) a + 6);
            return test_status::FAIL;
        }
        if (!tc.decode_check({ { 0, pos, pos + 3 } }, "mid-rm", 0) || !tc.spec_rounds(0, 8, "mid-rm spec")) {
            return test_status::FAIL;
        }
    }

    // the tail shifted down onto earlier positions, then up past a gap; only where the cache can shift
    if (llama_memory_can_shift(tc.mem_i())) {
        // the hybrid's pos_min is the recurrent tail's, the cells start where the indexer cache says
        const llama_pos c = (tc.mem->get_mem_idx()->seq_pos_min(0) + tc.pos_next(0))/2;
        llama_memory_seq_add(tc.mem_i(), 0, c, -1, -3);
        llama_pos pos = tc.pos_next(0);
        if (!tc.decode_check({ { 0, pos, pos + 2 } }, "shift-down", 0)) {
            return test_status::FAIL;
        }

        const llama_pos d = tc.pos_next(0) - 5;
        llama_memory_seq_add(tc.mem_i(), 0, d, -1, 5);
        pos = tc.pos_next(0);
        if (!tc.decode_check({ { 0, pos, pos + 2 } }, "shift-up", 0) || !tc.spec_rounds(0, 8, "shift spec")) {
            return test_status::FAIL;
        }

        // a gap inside a pool's span that a rollback then removes: the pool across it forms only once the gap is
        // gone, so the scan must go back before the first pool it dropped
        const llama_pos kpool = (llama_pos) tc.mem->get_kpool();
        const llama_pos pmin  = tc.mem->get_mem_idx()->seq_pos_min(0);
        pos = tc.pos_next(0);
        const llama_pos q = pos + (kpool - (pos - pmin) % kpool) % kpool;
        if ((q > pos && !tc.decode_check({ { 0, pos, q } }, "gap align", 0)) || !tc.decode_check({ { 0, q, q + kpool } }, "gap pool", 0)) {
            return test_status::FAIL;
        }
        llama_memory_seq_add(tc.mem_i(), 0, q + 2, -1, 1);
        if (!tc.decode_check({ { 0, q + kpool + 1, q + kpool + 2 } }, "gap open", 0)) {
            return test_status::FAIL;
        }
        if (!llama_memory_seq_rm(tc.mem_i(), 0, q + 2, -1)) {
            LOG_ERR("%s: removing the gap was refused\n", __func__);
            return test_status::FAIL;
        }
        if (!tc.decode_check({ { 0, q + 2, q + 2 + kpool } }, "gap closed", 0) || !tc.spec_rounds(0, 8, "gap spec")) {
            return test_status::FAIL;
        }
    } else {
        LOG_INF("%s: the cache cannot shift, no seq_add\n", __func__);
    }

    // cleared and filled again
    if (!llama_memory_seq_rm(tc.mem_i(), 0, -1, -1)) {
        LOG_ERR("%s: clearing the sequence was refused\n", __func__);
        return test_status::FAIL;
    }
    if (!tc.decode_check({ { 0, 0, 45 } }, "refill", 0) || !tc.spec_rounds(0, 8, "refill spec")) {
        return test_status::FAIL;
    }

    LOG_INF("%s: %d layout checks\n", __func__, tc.n_checks);
    return test_status::PASS;
}

// Two sequences in their own streams: rolled back in turn, and together in one batch.
static test_status test_streams(const common_params & params, llama_model * model) {
    kpool_case tc;
    test_status res = test_status::SKIP;
    if (!make_case(tc, params, model, "streams", 2, false, res)) {
        return res;
    }

    if (!tc.decode_check({ { 0, 0, 37 } }, "prefill 0", 0) || !tc.decode_check({ { 1, 0, 53 } }, "prefill 1", 0)) {
        return test_status::FAIL;
    }

    for (int r = 0; r < 16; ++r) {
        if (!tc.spec_rounds(r % 2, 1, "turns")) {
            return test_status::FAIL;
        }
    }

    for (int r = 0; r < 12; ++r) {
        const llama_pos p0 = tc.pos_next(0);
        const llama_pos p1 = tc.pos_next(1);
        const int       k  = std::uniform_int_distribution<int>(1, 4)(tc.rng);
        const int       a0 = std::uniform_int_distribution<int>(1, k)(tc.rng);
        const int       a1 = std::uniform_int_distribution<int>(1, k)(tc.rng);
        if (!tc.decode_check({ { 0, p0, p0 + k, p0 + a0 }, { 1, p1, p1 + k, p1 + a1 } }, "together", r)) {
            return test_status::FAIL;
        }
        if (!llama_memory_seq_rm(tc.mem_i(), 0, p0 + a0, -1) || !llama_memory_seq_rm(tc.mem_i(), 1, p1 + a1, -1)) {
            LOG_ERR("%s: together step %d: a rollback was refused\n", __func__, r);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: %d layout checks\n", __func__, tc.n_checks);
    return test_status::PASS;
}

// Two sequences sharing cells after seq_cp: the copy grows and is rolled back, then dropped.
static test_status test_shared(const common_params & params, llama_model * model) {
    kpool_case tc;
    test_status res = test_status::SKIP;
    if (!make_case(tc, params, model, "shared", 2, true, res)) {
        return res;
    }

    if (!tc.decode_check({ { 0, 0, 41 } }, "prefill", 0)) {
        return test_status::FAIL;
    }

    llama_memory_seq_cp(tc.mem_i(), 0, 1, -1, -1);

    if (!tc.decode_check({ { 1, 41, 44 } }, "copy grows", 0) || !tc.decode_check({ { 0, 41, 43 } }, "source grows", 0)) {
        return test_status::FAIL;
    }

    if (!llama_memory_seq_rm(tc.mem_i(), 1, 42, -1)) {
        LOG_ERR("%s: rolling back the copy was refused\n", __func__);
        return test_status::FAIL;
    }
    if (!tc.decode_check({ { 1, 42, 43 } }, "copy rolled back", 0) ||
        !tc.spec_rounds(0, 4, "shared spec 0") || !tc.spec_rounds(1, 4, "shared spec 1")) {
        return test_status::FAIL;
    }

    if (!llama_memory_seq_rm(tc.mem_i(), 1, -1, -1)) {
        LOG_ERR("%s: dropping the copy was refused\n", __func__);
        return test_status::FAIL;
    }
    const llama_pos pos = tc.pos_next(0);
    if (!tc.decode_check({ { 0, pos, pos + 1 } }, "copy dropped", 0) || !tc.spec_rounds(0, 8, "unshared spec")) {
        return test_status::FAIL;
    }

    LOG_INF("%s: %d layout checks\n", __func__, tc.n_checks);
    return test_status::PASS;
}

struct test_results {
    test_status single  = test_status::SKIP;
    test_status streams = test_status::SKIP;
    test_status shared  = test_status::SKIP;
};

static test_results run_tests_for_model(const std::string & model_path, const common_params & base_params) {
    common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }

    test_results res;
    res.single = test_single(params, model);
    if (res.single == test_status::SKIP) {
        return res;
    }
    res.streams = test_streams(params, model);
    res.shared  = test_shared(params, model);
    return res;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // our own --models DIR option, the rest goes to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    // a placeholder so the common parser's "--model is required" check passes, each model is set in the loop
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    std::vector<std::string> models;
    if (!models_dir.empty()) {
        if (!std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());
        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }
    } else {
        models.push_back(params.model.path);
    }

    size_t name_width = 5; // "Model"
    for (const auto & model_path : models) {
        name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
    }

    LOG("%-*s  %-6s  %-7s  %s\n", (int) name_width, "Model", "single", "streams", "shared");
    common_log_flush(common_log_main());

    size_t n_pass = 0;
    size_t n_fail = 0;
    for (const auto & model_path : models) {
        const test_results res = run_tests_for_model(model_path, params);

        LOG("%-*s  %s    %s     %s\n", (int) name_width, std::filesystem::path(model_path).filename().string().c_str(),
                test_status_str(res.single), test_status_str(res.streams), test_status_str(res.shared));
        common_log_flush(common_log_main());

        const test_status all = merge_status(merge_status(res.single, res.streams), res.shared);
        if (all == test_status::PASS) {
            n_pass++;
        } else if (all == test_status::FAIL) {
            n_fail++;
        }
    }

    LOG_INF("%s: %zu models with a k-pool indexer passed, %zu failed, %zu skipped\n", __func__,
            n_pass, n_fail, models.size() - n_pass - n_fail);

    return n_fail == 0 ? 0 : 1;
}
