#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// Pipeline scheduling test: consecutive ubatches through the pipeline API (n_stages > 1) must
// produce the same results as a serial computation, even though the host inputs are overwritten
// right after every submission.
//
// The CPU backend has no events and no submission workers, so the pipeline submission path that
// uploads the staged host inputs (ggml_backend_sched_compute_splits_pipeline) is not covered here.
// This test covers the pipeline ring bookkeeping (begin/submit/wait/advance, stage reuse) and the
// correct association of inputs and outputs across consecutive ubatches, on one and two CPU
// backend instances.

static constexpr int n_tokens = 32;
static constexpr int n_ubatch = 8;
static constexpr int n_stages = 3;

static void fail(const char * msg, int ubatch, int i) {
    fprintf(stderr, "FAILED: %s (ubatch %d, element %d)\n", msg, ubatch, i);
    exit(1);
}

static struct ggml_cgraph * build_graph(
        struct ggml_context * ctx, struct ggml_tensor ** a, struct ggml_tensor ** b, struct ggml_tensor ** c) {
    *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens);
    *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens);
    ggml_set_input(*a);
    ggml_set_input(*b);

    *c = ggml_add(ctx, *a, *b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, *c);
    return gf;
}

static void run_pipeline(int n_backends) {
    std::vector<ggml_backend_t> backends(n_backends);
    std::vector<ggml_backend_buffer_type_t> bufts(n_backends);
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev == NULL) {
            fprintf(stderr, "FAILED: no CPU backend\n");
            exit(1);
        }
        backends[b] = ggml_backend_dev_init(dev, NULL);
        bufts[b]    = ggml_backend_get_default_buffer_type(backends[b]);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), bufts.data(), n_backends, 2048, true, false);
    ggml_backend_sched_set_n_stages(sched, n_stages);

    std::vector<struct ggml_context *> ctxs(n_ubatch, nullptr);

    for (int ub = 0; ub < n_ubatch; ub++) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 8,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        ctxs[ub] = ggml_init(params);
        if (ctxs[ub] == NULL) {
            fprintf(stderr, "FAILED: ggml_init\n");
            exit(1);
        }

        struct ggml_tensor * a = nullptr;
        struct ggml_tensor * b = nullptr;
        struct ggml_tensor * c = nullptr;
        struct ggml_cgraph * gf = build_graph(ctxs[ub], &a, &b, &c);

        if (!ggml_backend_sched_pipeline_begin(sched, gf, ub)) {
            fail("pipeline_begin", ub, -1);
        }

        // distinct input values per ubatch
        for (int i = 0; i < n_tokens; i++) {
            ((float *) a->data)[i] = (float) (10*ub + i);
            ((float *) b->data)[i] = (float) (1000 + ub);
        }

        if (ggml_backend_sched_pipeline_submit(sched, ub) != GGML_STATUS_SUCCESS) {
            fail("pipeline_submit", ub, -1);
        }

        // overwrite the inputs immediately: the ubatch must have taken its own copy
        memset(a->data, 0, ggml_nbytes(a));
        memset(b->data, 0, ggml_nbytes(b));

        ggml_backend_sched_pipeline_wait(sched, ub);

        const float * out = (const float *) c->data;
        for (int i = 0; i < n_tokens; i++) {
            const float expect = (float) (10*ub + i) + (float) (1000 + ub);
            if (memcmp(&out[i], &expect, sizeof(float)) != 0) {
                fail("wrong output", ub, i);
            }
        }
    }

    ggml_backend_sched_pipeline_advance(sched);

    // all ubatches done: the results above were read before the buffers were reused
    for (int ub = 0; ub < n_ubatch; ub++) {
        ggml_free(ctxs[ub]);
    }
    ggml_backend_sched_free(sched);
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_free(backends[b]);
    }
}

// GPU variant: exercises the worker submission path, where a split is posted to a submission
// thread instead of running synchronously on the main thread. The weights are pre-allocated on the
// GPUs, so the ops are pinned there and the host inputs (and views of them) must be staged by the
// pipeline. The views make sure that a view of an input is treated as a user input too.
static struct ggml_cgraph * build_graph_gpu(
        struct ggml_context * ctx, struct ggml_tensor * w0, struct ggml_tensor * w1,
        int a_len, bool w1_view, struct ggml_tensor ** a, struct ggml_tensor ** b, struct ggml_tensor ** out) {
    *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a_len);
    *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens/4, 4);
    ggml_set_input(*a);
    ggml_set_input(*b);

    // a view with an offset: its data lives in the host input buffer of a, like the views built on
    // the real model inputs
    struct ggml_tensor * x0 = ggml_add(ctx, w0, ggml_view_1d(ctx, *a, n_tokens, 8*sizeof(float)));
    struct ggml_tensor * x1;
    if (w1 != NULL) {
        // w1_view wraps the weight in a view of the same shape, so it is a cross-backend constant
        // that is a graph node instead of a leaf (the weight itself is a leaf)
        struct ggml_tensor * w1_arg = w1_view ? ggml_view_1d(ctx, w1, n_tokens, 0) : w1;
        x1 = ggml_add(ctx, w1_arg, ggml_add(ctx, x0, ggml_reshape_1d(ctx, *b, n_tokens)));
    } else {
        x1 = ggml_add(ctx, x0, ggml_reshape_1d(ctx, *b, n_tokens));
    }
    ggml_set_output(x1);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x1);
    *out = x1;
    return gf;
}

// Growing variant: the input and the weight slice span the whole a_len, so the staging entry of the
// backend must hold a different size for every ubatch. The weight is allocated at the largest size
// and sliced, so the ops stay pinned on the GPUs.
static struct ggml_cgraph * build_graph_gpu_grow(
        struct ggml_context * ctx, struct ggml_tensor * w0, struct ggml_tensor * w1,
        int a_len, bool w1_view, struct ggml_tensor ** a, struct ggml_tensor ** out) {
    *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a_len);
    ggml_set_input(*a);

    struct ggml_tensor * x0 = ggml_add(ctx, ggml_view_1d(ctx, w0, a_len, 0), *a);
    struct ggml_tensor * x1 = x0;
    if (w1 != NULL) {
        // w1_view wraps the weight in a view of the same shape, so it is a cross-backend constant
        // that is a graph node instead of a leaf
        struct ggml_tensor * w1_arg = w1_view ? ggml_view_1d(ctx, w1, a_len, 0) : w1;
        x1 = ggml_add(ctx, w1_arg, x0);
    }
    ggml_set_output(x1);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x1);
    *out = x1;
    return gf;
}

static int run_pipeline_gpu(bool growing, bool reserve, bool w1_view) {
    std::vector<ggml_backend_dev_t> gpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_devs.push_back(dev);
        }
    }
    if (gpu_devs.empty()) {
        // capability branch: the worker submission path only exists for devices with workers
        printf("pipeline gpu: skipped (no GPU)\n");
        return 0;
    }
    if (gpu_devs.size() > 2) {
        gpu_devs.resize(2);
    }
    const int n_gpu = (int) gpu_devs.size();

    printf("pipeline gpu%s%s: %d GPU ... ",
            growing ? (reserve ? " grow" : " grow-noreserve") : "",
            w1_view ? "-w1view" : "-w1direct",
            n_gpu);
    fflush(stdout);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == NULL) {
        fprintf(stderr, "FAILED: no CPU backend\n");
        exit(1);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (int g = 0; g < n_gpu; g++) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_devs[g], NULL);
        if (backend == NULL) {
            fprintf(stderr, "FAILED: gpu backend init\n");
            exit(1);
        }
        backends.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    backends.push_back(cpu_backend);
    bufts.push_back(ggml_backend_get_default_buffer_type(cpu_backend));

    const int n_backends = (int) backends.size();

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), bufts.data(), n_backends, 2048, true, false);
    ggml_backend_sched_set_n_stages(sched, n_stages);

    // constant weights, one per GPU, pre-allocated on that GPU's default buffer so the ops that
    // consume them are pinned there. the growing variant slices the weight to the current length
    const int w_len = growing ? n_tokens + 8 + 4*(n_ubatch - 1) : n_tokens;
    std::vector<struct ggml_context *>      w_ctxs;
    std::vector<ggml_backend_buffer_t>      w_bufs;
    std::vector<struct ggml_tensor *>       weights;
    for (int g = 0; g < n_gpu; g++) {
        struct ggml_init_params wparams = {
            /* .mem_size   = */ 16 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * w_ctx = ggml_init(wparams);
        struct ggml_tensor * w = ggml_new_tensor_1d(w_ctx, GGML_TYPE_F32, w_len);
        ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(w_ctx, backends[g]);
        if (w_buf == NULL) {
            fprintf(stderr, "FAILED: weight alloc\n");
            exit(1);
        }
        std::vector<float> host(w_len);
        for (int i = 0; i < w_len; i++) {
            host[i] = (float) (100*g + i);
        }
        ggml_backend_tensor_set(w, host.data(), 0, ggml_nbytes(w));

        w_ctxs.push_back(w_ctx);
        w_bufs.push_back(w_buf);
        weights.push_back(w);
    }

    std::vector<struct ggml_context *> ctxs(n_ubatch, nullptr);
    std::vector<float> out(n_tokens + 8 + 4*(n_ubatch - 1));

    // the input grows with the ubatch index: every ubatch needs a larger staging entry than the
    // previous one. reserving the largest graph up front must fix the staging size, so the growing
    // ubatches below never reallocate the ring (the reserve-before-submit path)
    const auto a_len_of = [](int ub) { return n_tokens + 8 + 4*ub; };
    if (growing && reserve) {
        struct ggml_init_params rparams = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 16,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * r_ctx = ggml_init(rparams);
        struct ggml_tensor * ra = nullptr;
        struct ggml_tensor * rx = nullptr;
        struct ggml_cgraph * rgf = build_graph_gpu_grow(r_ctx, weights[0], n_gpu > 1 ? weights[1] : NULL,
                a_len_of(n_ubatch - 1), w1_view, &ra, &rx);
        if (!ggml_backend_sched_reserve(sched, rgf)) {
            fprintf(stderr, "FAILED: reserve gpu\n");
            exit(1);
        }
        ggml_free(r_ctx);
    }

    for (int ub = 0; ub < n_ubatch; ub++) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 16,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        ctxs[ub] = ggml_init(params);
        if (ctxs[ub] == NULL) {
            fprintf(stderr, "FAILED: ggml_init\n");
            exit(1);
        }

        const int a_len = growing ? a_len_of(ub) : n_tokens + 8;
        struct ggml_tensor * a = nullptr;
        struct ggml_tensor * b = nullptr;
        struct ggml_tensor * x1 = nullptr;
        struct ggml_cgraph * gf = growing
            ? build_graph_gpu_grow(ctxs[ub], weights[0], n_gpu > 1 ? weights[1] : NULL, a_len, w1_view, &a, &x1)
            : build_graph_gpu(ctxs[ub], weights[0], n_gpu > 1 ? weights[1] : NULL, a_len, w1_view, &a, &b, &x1);

        if (!ggml_backend_sched_pipeline_begin(sched, gf, ub)) {
            fail("pipeline_begin gpu", ub, -1);
        }

        // distinct input values per ubatch
        for (int i = 0; i < a_len; i++) {
            ((float *) a->data)[i] = (float) (ub*1000 + i);
        }
        for (int i = 0; !growing && i < n_tokens; i++) {
            ((float *) b->data)[i] = (float) (ub*7 + i);
        }

        if (ggml_backend_sched_pipeline_submit(sched, ub) != GGML_STATUS_SUCCESS) {
            fail("pipeline_submit gpu", ub, -1);
        }

        // overwrite the inputs immediately: the ubatch must have taken its own copy, including the
        // data read through the views of a and b
        memset(a->data, 0, ggml_nbytes(a));
        if (!growing) {
            memset(b->data, 0, ggml_nbytes(b));
        }

        ggml_backend_sched_pipeline_wait(sched, ub);

        const int out_len = growing ? a_len : n_tokens;
        ggml_backend_tensor_get(x1, out.data(), 0, ggml_nbytes(x1));
        for (int i = 0; i < out_len; i++) {
            const float av = (float) (ub*1000 + i); // full a
            // w0 slice: (float) i, w1 slice: (float) (100 + i)
            float expect = (float) i + av;
            if (!growing) {
                const float bv = (float) (ub*7 + i); // reshape of b
                expect = (float) i + (float) (ub*1000 + i + 8) + bv; // w0 + view of a + reshape(b)
            }
            if (n_gpu > 1) {
                expect += (float) (100 + i); // + w1
            }
            if (memcmp(&out[i], &expect, sizeof(float)) != 0) {
                fail("wrong output gpu", ub, i);
            }
        }
    }

    ggml_backend_sched_pipeline_advance(sched);

    for (int ub = 0; ub < n_ubatch; ub++) {
        ggml_free(ctxs[ub]);
    }
    ggml_backend_sched_free(sched);
    for (size_t g = 0; g < w_bufs.size(); g++) {
        ggml_backend_buffer_free(w_bufs[g]);
        ggml_free(w_ctxs[g]);
    }
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_free(backends[b]);
    }

    printf("OK\n");
    return n_gpu;
}

// Deep pipeline variant: submit 2*n_stages ubatches back to back without waiting, rewriting the
// caller graph inputs right after every submit. This exercises the per-stage graph result and the
// per-stage scheduler context: a submission that is still being enqueued must not see the metadata
// of a later ubatch. Only the outputs are checked, first the last ubatch and then all of them.
static int run_pipeline_gpu_deep(void) {
    std::vector<ggml_backend_dev_t> gpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_devs.push_back(dev);
        }
    }
    if (gpu_devs.size() < 2) {
        // capability branch: the deep pipeline needs at least two devices with workers
        printf("pipeline gpu deep: skipped (need >= 2 GPU)\n");
        return 0;
    }
    if (gpu_devs.size() > 2) {
        gpu_devs.resize(2);
    }
    const int n_gpu   = (int) gpu_devs.size();
    const int stages  = n_gpu;
    const int n_ub    = 2 * stages;

    printf("pipeline gpu deep: %d GPU, %d stages ... ", n_gpu, stages);
    fflush(stdout);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == NULL) {
        fprintf(stderr, "FAILED: no CPU backend\n");
        exit(1);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (int g = 0; g < n_gpu; g++) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_devs[g], NULL);
        if (backend == NULL) {
            fprintf(stderr, "FAILED: gpu backend init\n");
            exit(1);
        }
        backends.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    backends.push_back(cpu_backend);
    bufts.push_back(ggml_backend_get_default_buffer_type(cpu_backend));

    const int n_backends = (int) backends.size();

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), bufts.data(), n_backends, 2048, true, false);
    ggml_backend_sched_set_n_stages(sched, stages);

    std::vector<struct ggml_context *>      w_ctxs;
    std::vector<ggml_backend_buffer_t>      w_bufs;
    std::vector<struct ggml_tensor *>       weights;
    for (int g = 0; g < n_gpu; g++) {
        struct ggml_init_params wparams = {
            /* .mem_size   = */ 16 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * w_ctx = ggml_init(wparams);
        struct ggml_tensor * w = ggml_new_tensor_1d(w_ctx, GGML_TYPE_F32, n_tokens);
        ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(w_ctx, backends[g]);
        if (w_buf == NULL) {
            fprintf(stderr, "FAILED: weight alloc\n");
            exit(1);
        }
        std::vector<float> host(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            host[i] = (float) (100*g + i);
        }
        ggml_backend_tensor_set(w, host.data(), 0, ggml_nbytes(w));

        w_ctxs.push_back(w_ctx);
        w_bufs.push_back(w_buf);
        weights.push_back(w);
    }

    std::vector<struct ggml_context *> ctxs(n_ub, nullptr);
    std::vector<float>                 host_out(n_ub * n_tokens, -1.0f);

    // submit everything without waiting; the input is overwritten right after each submit. the
    // output of each ubatch is copied to its own host region by a task posted to the submission
    // worker of its backend, ordered after that ubatch's compute (the compute buffer is reused by
    // the later ubatches, so the output cannot be read back in place)
    for (int ub = 0; ub < n_ub; ub++) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 16,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        ctxs[ub] = ggml_init(params);
        if (ctxs[ub] == NULL) {
            fprintf(stderr, "FAILED: ggml_init\n");
            exit(1);
        }

        struct ggml_tensor * a = nullptr;
        struct ggml_tensor * b = nullptr;
        struct ggml_tensor * x1 = nullptr;
        struct ggml_cgraph * gf = build_graph_gpu(ctxs[ub], weights[0], n_gpu > 1 ? weights[1] : NULL,
                n_tokens + 8, false, &a, &b, &x1);

        if (!ggml_backend_sched_pipeline_begin(sched, gf, ub)) {
            fail("pipeline_begin gpu deep", ub, -1);
        }

        for (int i = 0; i < n_tokens + 8; i++) {
            ((float *) a->data)[i] = (float) (ub*1000 + i);
        }
        for (int i = 0; i < n_tokens; i++) {
            ((float *) b->data)[i] = (float) (ub*7 + i);
        }

        if (ggml_backend_sched_pipeline_submit(sched, ub) != GGML_STATUS_SUCCESS) {
            fail("pipeline_submit gpu deep", ub, -1);
        }

        // rewrite the caller graph inputs immediately, then build the next ubatch
        memset(a->data, 0, ggml_nbytes(a));
        memset(b->data, 0, ggml_nbytes(b));

        // copy this ubatch's output to its own host region, ordered after its compute
        ggml_backend_t out_backend = ggml_backend_sched_get_tensor_backend(sched, x1);
        const int out_backend_id = ggml_backend_sched_get_tensor_backend_id(sched, x1);
        float * dst = host_out.data() + (size_t) ub * n_tokens;
        ggml_backend_sched_post_task(sched, out_backend_id, [=]() {
            ggml_backend_tensor_get_async(out_backend, x1, dst, 0, n_tokens*sizeof(float));
        });
    }

    const auto check = [&](int ub, const char * what) {
        for (int i = 0; i < n_tokens; i++) {
            const float av = (float) (ub*1000 + i + 8);
            const float bv = (float) (ub*7 + i);
            float expect = (float) i + av + bv;
            if (n_gpu > 1) {
                expect += (float) (100 + i);
            }
            if (memcmp(&host_out[(size_t) ub * n_tokens + i], &expect, sizeof(float)) != 0) {
                fail(what, ub, i);
            }
        }
    };

    // only the last ubatch is waited for first: the earlier submissions must still be intact. the
    // stage wait covers the output copies posted to the same workers (the copies extend the stage
    // completion event), so the host region is valid without synchronizing the backend
    {
        const int ub = n_ub - 1;
        ggml_backend_sched_pipeline_wait(sched, ub);
        check(ub, "wrong output gpu deep (last)");
    }

    // then every ubatch, in submission order
    ggml_backend_sched_pipeline_advance(sched);
    for (int ub = 0; ub < n_ub; ub++) {
        check(ub, "wrong output gpu deep");
    }

    for (int ub = 0; ub < n_ub; ub++) {
        ggml_free(ctxs[ub]);
    }
    ggml_backend_sched_free(sched);
    for (size_t g = 0; g < w_bufs.size(); g++) {
        ggml_backend_buffer_free(w_bufs[g]);
        ggml_free(w_ctxs[g]);
    }
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_free(backends[b]);
    }

    printf("OK\n");
    return n_gpu;
}

// Deep pipeline variant with a changing graph topology: consecutive ubatches alternate between
// two graphs that differ by one node (a scale by 1, so the value is unchanged). Alternating the
// node count makes ggml_gallocr_needs_realloc() return true on every ubatch, so every
// ggml_backend_sched_pipeline_begin below enters the reallocation branch of
// ggml_backend_sched_alloc_splits and frees/re-reserves the graph buffers while earlier ubatches
// are still in flight. With submission workers (GPU) the already posted but not yet enqueued tasks
// reference the old allocations, so the branch must drain the workers before it re-reserves.
static int run_pipeline_gpu_deep_realloc(void) {
    std::vector<ggml_backend_dev_t> gpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_devs.push_back(dev);
        }
    }
    if (gpu_devs.size() < 2) {
        // capability branch: the deep pipeline needs at least two devices with workers
        printf("pipeline gpu deep realloc: skipped (need >= 2 GPU)\n");
        return 0;
    }
    if (gpu_devs.size() > 2) {
        gpu_devs.resize(2);
    }
    const int n_gpu   = (int) gpu_devs.size();
    const int stages  = n_gpu;
    const int n_ub    = 4 * stages;

    printf("pipeline gpu deep realloc: %d GPU, %d stages ... ", n_gpu, stages);
    fflush(stdout);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == NULL) {
        fprintf(stderr, "FAILED: no CPU backend\n");
        exit(1);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (int g = 0; g < n_gpu; g++) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_devs[g], NULL);
        if (backend == NULL) {
            fprintf(stderr, "FAILED: gpu backend init\n");
            exit(1);
        }
        backends.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    backends.push_back(cpu_backend);
    bufts.push_back(ggml_backend_get_default_buffer_type(cpu_backend));

    const int n_backends = (int) backends.size();

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), bufts.data(), n_backends, 2048, true, false);
    ggml_backend_sched_set_n_stages(sched, stages);

    std::vector<struct ggml_context *>      w_ctxs;
    std::vector<ggml_backend_buffer_t>      w_bufs;
    std::vector<struct ggml_tensor *>       weights;
    for (int g = 0; g < n_gpu; g++) {
        struct ggml_init_params wparams = {
            /* .mem_size   = */ 16 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * w_ctx = ggml_init(wparams);
        struct ggml_tensor * w = ggml_new_tensor_1d(w_ctx, GGML_TYPE_F32, n_tokens);
        ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(w_ctx, backends[g]);
        if (w_buf == NULL) {
            fprintf(stderr, "FAILED: weight alloc\n");
            exit(1);
        }
        std::vector<float> host(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            host[i] = (float) (100*g + i);
        }
        ggml_backend_tensor_set(w, host.data(), 0, ggml_nbytes(w));

        w_ctxs.push_back(w_ctx);
        w_bufs.push_back(w_buf);
        weights.push_back(w);
    }

    std::vector<struct ggml_context *> ctxs(n_ub, nullptr);
    std::vector<float>                 host_out(n_ub * n_tokens, -1.0f);

    for (int ub = 0; ub < n_ub; ub++) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 16,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        ctxs[ub] = ggml_init(params);
        if (ctxs[ub] == NULL) {
            fprintf(stderr, "FAILED: ggml_init\n");
            exit(1);
        }

        struct ggml_tensor * a = nullptr;
        struct ggml_tensor * b = nullptr;
        struct ggml_tensor * x1 = nullptr;
        struct ggml_cgraph * gf = build_graph_gpu(ctxs[ub], weights[0], n_gpu > 1 ? weights[1] : NULL,
                n_tokens + 8, false, &a, &b, &x1);

        // the odd ubatches get one extra node: a scale by 1 keeps the value but changes the node
        // count, which forces the reallocation branch of the scheduler on the next begin
        struct ggml_tensor * out = x1;
        if (ub % 2 == 1) {
            out = ggml_scale(ctxs[ub], x1, 1.0f);
            ggml_build_forward_expand(gf, out);
        }
        ggml_set_output(out);

        if (!ggml_backend_sched_pipeline_begin(sched, gf, ub)) {
            fail("pipeline_begin gpu deep realloc", ub, -1);
        }

        for (int i = 0; i < n_tokens + 8; i++) {
            ((float *) a->data)[i] = (float) (ub*1000 + i);
        }
        for (int i = 0; i < n_tokens; i++) {
            ((float *) b->data)[i] = (float) (ub*7 + i);
        }

        if (ggml_backend_sched_pipeline_submit(sched, ub) != GGML_STATUS_SUCCESS) {
            fail("pipeline_submit gpu deep realloc", ub, -1);
        }

        // rewrite the caller graph inputs immediately, then build the next ubatch
        memset(a->data, 0, ggml_nbytes(a));
        memset(b->data, 0, ggml_nbytes(b));

        // copy this ubatch's output to its own host region, ordered after its compute
        ggml_backend_t out_backend = ggml_backend_sched_get_tensor_backend(sched, out);
        const int out_backend_id = ggml_backend_sched_get_tensor_backend_id(sched, out);
        float * dst = host_out.data() + (size_t) ub * n_tokens;
        ggml_backend_sched_post_task(sched, out_backend_id, [=]() {
            ggml_backend_tensor_get_async(out_backend, out, dst, 0, n_tokens*sizeof(float));
        });
    }

    const auto check = [&](int ub, const char * what) {
        for (int i = 0; i < n_tokens; i++) {
            const float av = (float) (ub*1000 + i + 8);
            const float bv = (float) (ub*7 + i);
            float expect = (float) i + av + bv;
            if (n_gpu > 1) {
                expect += (float) (100 + i);
            }
            if (memcmp(&host_out[(size_t) ub * n_tokens + i], &expect, sizeof(float)) != 0) {
                fail(what, ub, i);
            }
        }
    };

    // only the last ubatch is waited for first: the earlier submissions must still be intact. the
    // stage wait covers the output copies posted to the same workers
    {
        const int ub = n_ub - 1;
        ggml_backend_sched_pipeline_wait(sched, ub);
        check(ub, "wrong output gpu deep realloc (last)");
    }

    // then every ubatch, in submission order
    ggml_backend_sched_pipeline_advance(sched);
    for (int ub = 0; ub < n_ub; ub++) {
        check(ub, "wrong output gpu deep realloc");
    }

    for (int ub = 0; ub < n_ub; ub++) {
        ggml_free(ctxs[ub]);
    }
    ggml_backend_sched_free(sched);
    for (size_t g = 0; g < w_bufs.size(); g++) {
        ggml_backend_buffer_free(w_bufs[g]);
        ggml_free(w_ctxs[g]);
    }
    for (int b = 0; b < n_backends; b++) {
        ggml_backend_free(backends[b]);
    }

    printf("OK\n");
    return n_gpu;
}

// Synchronous submission (n_stages == 1, the decode path): the host inputs a GPU consumes are staged in
// pinned memory and uploaded with one copy per run of input copies that ggml-alloc placed back to back.
// The graph has
//  - an F32 input a read through a view with an offset, an I32 row index feeding get_rows and, after a
//    node forced onto the CPU, an F32 input b: GPU 0 runs two splits, the copies of the second one are
//    allocated where ggml-alloc finds room when that split starts;
//  - with two GPUs a third part on GPU 1 that reads a again through another view.
// Every mode first reserves a larger graph (a longer a), so the graphs below reuse the reserved layout
// and their copy of a leaves part of its slot unused (the upload carries that slack). The iterations
// alternate between rebuilding the graph (which rotates the copy slot of the double buffered scheduler)
// and reusing it.
//   mode 0: one input copy, asynchronous inputs: the caller synchronizes before it writes the inputs
//   mode 1: one input copy, synchronous inputs: the inputs may be overwritten once the submission
//           returns, and submissions may follow each other without a synchronization
//   mode 2: two input copies (double buffered): same as mode 1
// the compute nodes of the graph and the backend each one is placed on (0 = GPU 0, 1 = GPU 1, -1 = the
// CPU): the placement is set explicitly, otherwise the node forced onto the CPU pulls its neighbours
// along (the GPU weights are plain buffers, so they do not pin the ops by themselves)
struct sync_graph_placement {
    std::vector<std::pair<struct ggml_tensor *, int>> nodes;
};

static struct ggml_cgraph * build_graph_sync(
        struct ggml_context * ctx, struct ggml_tensor * w0, struct ggml_tensor * w0b, struct ggml_tensor * wr,
        struct ggml_tensor * w1, int a_len,
        struct ggml_tensor ** a, struct ggml_tensor ** idx, struct ggml_tensor ** b,
        struct ggml_tensor ** out, sync_graph_placement & place) {
    *a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a_len);
    *idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    *b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens);
    ggml_set_input(*a);
    ggml_set_input(*idx);
    ggml_set_input(*b);

    place.nodes.clear();
    struct ggml_tensor * rows = ggml_get_rows(ctx, wr, *idx);
    struct ggml_tensor * s0   = ggml_add(ctx, w0, ggml_view_1d(ctx, *a, n_tokens, 4*sizeof(float)));
    struct ggml_tensor * x0   = ggml_add(ctx, s0, ggml_reshape_1d(ctx, rows, n_tokens));
    struct ggml_tensor * c    = ggml_scale(ctx, x0, 2.0f);
    struct ggml_tensor * s1   = ggml_add(ctx, w0b, c);
    struct ggml_tensor * x1   = ggml_add(ctx, s1, *b);
    place.nodes.push_back({ rows, 0 });
    place.nodes.push_back({ s0,   0 });
    place.nodes.push_back({ x0,   0 });
    place.nodes.push_back({ c,   -1 });
    place.nodes.push_back({ s1,   0 });
    place.nodes.push_back({ x1,   0 });
    struct ggml_tensor * x = x1;
    if (w1 != NULL) {
        struct ggml_tensor * s2 = ggml_add(ctx, w1, x1);
        x = ggml_add(ctx, s2, ggml_view_1d(ctx, *a, n_tokens, 0));
        place.nodes.push_back({ s2, 1 });
        place.nodes.push_back({ x,  1 });
    }
    ggml_set_output(x);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);
    *out = x;
    return gf;
}

static int run_sync_inputs_gpu(int mode) {
    std::vector<ggml_backend_dev_t> gpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_devs.push_back(dev);
        }
    }
    if (gpu_devs.empty()) {
        printf("sync inputs mode %d: skipped (no GPU)\n", mode);
        return 0;
    }
    if (gpu_devs.size() > 2) {
        gpu_devs.resize(2);
    }
    const int n_gpu = (int) gpu_devs.size();

    printf("sync inputs mode %d: %d GPU ... ", mode, n_gpu);
    fflush(stdout);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == NULL) {
        fprintf(stderr, "FAILED: no CPU backend\n");
        exit(1);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (int g = 0; g < n_gpu; g++) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_devs[g], NULL);
        if (backend == NULL) {
            fprintf(stderr, "FAILED: gpu backend init\n");
            exit(1);
        }
        backends.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, NULL);
    backends.push_back(cpu_backend);
    // the host inputs live in pinned memory, like the llama.cpp inputs
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_devs[0]);
    bufts.push_back(host_buft != NULL ? host_buft : ggml_backend_get_default_buffer_type(cpu_backend));

    const int n_backends = (int) backends.size();

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), bufts.data(), n_backends, 2048, mode == 2, false);
    ggml_backend_sched_set_async_inputs(sched, mode == 0);

    // weights pinned on the GPUs: w0, w0b and the row table wr on GPU 0, w1 on GPU 1
    constexpr int n_rows = 16;
    struct ggml_init_params wparams = {
        /* .mem_size   = */ 16 * 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * w_ctx0 = ggml_init(wparams);
    struct ggml_tensor * w0  = ggml_new_tensor_1d(w_ctx0, GGML_TYPE_F32, n_tokens);
    struct ggml_tensor * w0b = ggml_new_tensor_1d(w_ctx0, GGML_TYPE_F32, n_tokens);
    struct ggml_tensor * wr  = ggml_new_tensor_2d(w_ctx0, GGML_TYPE_F32, n_tokens, n_rows);
    ggml_backend_buffer_t w_buf0 = ggml_backend_alloc_ctx_tensors(w_ctx0, backends[0]);
    struct ggml_context * w_ctx1 = NULL;
    struct ggml_tensor * w1 = NULL;
    ggml_backend_buffer_t w_buf1 = NULL;
    if (n_gpu > 1) {
        w_ctx1 = ggml_init(wparams);
        w1 = ggml_new_tensor_1d(w_ctx1, GGML_TYPE_F32, n_tokens);
        w_buf1 = ggml_backend_alloc_ctx_tensors(w_ctx1, backends[1]);
    }
    if (w_buf0 == NULL || (n_gpu > 1 && w_buf1 == NULL)) {
        fprintf(stderr, "FAILED: weight alloc\n");
        exit(1);
    }
    {
        std::vector<float> host(n_tokens*n_rows);
        for (int i = 0; i < n_tokens; i++) {
            host[i] = (float) i;
        }
        ggml_backend_tensor_set(w0, host.data(), 0, ggml_nbytes(w0));
        for (int i = 0; i < n_tokens; i++) {
            host[i] = (float) (1000 + i);
        }
        ggml_backend_tensor_set(w0b, host.data(), 0, ggml_nbytes(w0b));
        for (int r = 0; r < n_rows; r++) {
            for (int i = 0; i < n_tokens; i++) {
                host[r*n_tokens + i] = (float) (10*r + i);
            }
        }
        ggml_backend_tensor_set(wr, host.data(), 0, ggml_nbytes(wr));
        if (w1 != NULL) {
            for (int i = 0; i < n_tokens; i++) {
                host[i] = (float) (100 + i);
            }
            ggml_backend_tensor_set(w1, host.data(), 0, ggml_nbytes(w1));
        }
    }

    constexpr int n_iter = 8;
    const auto a_len_of = [](int it) { return n_tokens + 8 + 2*(it/2); };

    const auto graph_ctx = []() {
        struct ggml_init_params params = {
            /* .mem_size   = */ (ggml_tensor_overhead() + ggml_graph_overhead()) * 32,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (ctx == NULL) {
            fprintf(stderr, "FAILED: ggml_init\n");
            exit(1);
        }
        return ctx;
    };

    const auto place_nodes = [&](const sync_graph_placement & p) {
        for (const auto & np : p.nodes) {
            ggml_backend_sched_set_tensor_backend(sched, np.first, np.second < 0 ? cpu_backend : backends[np.second]);
        }
    };

    // reserve a larger graph first (a longer a): the graphs below fit its allocation
    {
        struct ggml_context * r_ctx = graph_ctx();
        struct ggml_tensor * ra, * ridx, * rb, * rx;
        sync_graph_placement rplace;
        struct ggml_cgraph * rgf = build_graph_sync(r_ctx, w0, w0b, wr, w1, a_len_of(n_iter) + 16, &ra, &ridx, &rb, &rx, rplace);
        ggml_backend_sched_reset(sched);
        place_nodes(rplace);
        if (!ggml_backend_sched_reserve(sched, rgf)) {
            fprintf(stderr, "FAILED: reserve sync inputs\n");
            exit(1);
        }
        ggml_free(r_ctx);
    }

    struct ggml_context * ctx = NULL;
    struct ggml_tensor * a = nullptr, * idx = nullptr, * b = nullptr, * out = nullptr;
    sync_graph_placement place;
    struct ggml_cgraph * gf = nullptr;
    int a_len = 0;

    const auto set_inputs = [&](int it) {
        for (int i = 0; i < a_len; i++) {
            ((float *) a->data)[i] = (float) (it*1000 + i);
        }
        ((int32_t *) idx->data)[0] = it % n_rows;
        for (int i = 0; i < n_tokens; i++) {
            ((float *) b->data)[i] = (float) (it*7 + i);
        }
    };
    const auto check_values = [&](int it, const std::vector<float> & res) {
        for (int i = 0; i < n_tokens; i++) {
            const float x0 = (float) i + (float) (it*1000 + i + 4) + (float) (10*(it % n_rows) + i);
            float expect = (float) (1000 + i) + 2.0f*x0 + (float) (it*7 + i);
            if (n_gpu > 1) {
                expect += (float) (100 + i) + (float) (it*1000 + i);
            }
            if (memcmp(&res[i], &expect, sizeof(float)) != 0) {
                fprintf(stderr, "got %f expected %f\n", res[i], expect);
                fail("wrong output sync inputs", it, i);
            }
        }
    };
    const auto check = [&](int it) {
        std::vector<float> res(n_tokens);
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        check_values(it, res);
    };

    for (int it = 0; it < n_iter; it++) {
        // even iterations rebuild the graph (a new layout, the next copy slot), odd ones reuse it
        if (it % 2 == 0) {
            if (ctx != NULL) {
                ggml_free(ctx);
            }
            ctx = graph_ctx();
            a_len = a_len_of(it);
            gf = build_graph_sync(ctx, w0, w0b, wr, w1, a_len, &a, &idx, &b, &out, place);
            ggml_backend_sched_reset(sched);
            place_nodes(place);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                fail("alloc_graph sync inputs", it, -1);
            }
            if (ggml_backend_sched_get_n_splits(sched) < 3) {
                fail("expected a CPU split between the GPU 0 splits", it, ggml_backend_sched_get_n_splits(sched));
            }
        }

        const auto clobber_inputs = [&]() {
            memset(a->data, 0, ggml_nbytes(a));
            memset(b->data, 0, ggml_nbytes(b));
            ((int32_t *) idx->data)[0] = n_rows - 1;
        };

        set_inputs(it);
        if (ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS) {
            fail("compute sync inputs", it, -1);
        }
        if (mode != 0) {
            // the submission has taken the inputs: overwriting them must not change its result
            clobber_inputs();
        }
        ggml_backend_sched_synchronize(sched);
        check(it);

        if (mode != 0) {
            // two submissions of the same graph back to back, the second one with other inputs: the
            // second upload must neither overwrite the staged inputs of the first one before they
            // reached the device nor the device inputs before the first compute read them
            set_inputs(it + 50);
            if (ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS) {
                fail("compute sync inputs (back to back, first)", it, -1);
            }
            // the result of the first submission, read on the output stream before the second one
            std::vector<float> first(n_tokens);
            ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, out), out, first.data(), 0, ggml_nbytes(out));
            set_inputs(it);
            if (ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS) {
                fail("compute sync inputs (back to back, second)", it, -1);
            }
            clobber_inputs();
            ggml_backend_sched_synchronize(sched);
            check_values(it + 50, first);
            check(it);
        }
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(w_buf0);
    ggml_free(w_ctx0);
    if (w_buf1 != NULL) {
        ggml_backend_buffer_free(w_buf1);
        ggml_free(w_ctx1);
    }
    for (int bi = 0; bi < n_backends; bi++) {
        ggml_backend_free(backends[bi]);
    }

    printf("OK\n");
    return n_gpu;
}

int main() {
    printf("pipeline: 1 backend ... ");
    fflush(stdout);
    run_pipeline(1);
    printf("OK\n");

    printf("pipeline: 2 backends ... ");
    fflush(stdout);
    run_pipeline(2);
    printf("OK\n");

    // w1direct is the control: the cross-backend constant is the weight leaf itself. w1view makes
    // the constant a view node instead, which the producer detection must not treat as a producer.
    // grow-noreserve is the case that exposed the bug; grow reserves the largest graph up front.
    run_pipeline_gpu(false, false, false);
    run_pipeline_gpu(false, false, true);
    run_pipeline_gpu(true,  false, true);
    run_pipeline_gpu(true,  true,  true);
    run_pipeline_gpu_deep();
    run_pipeline_gpu_deep_realloc();

    run_sync_inputs_gpu(0);
    run_sync_inputs_gpu(1);
    run_sync_inputs_gpu(2);

    return 0;
}
