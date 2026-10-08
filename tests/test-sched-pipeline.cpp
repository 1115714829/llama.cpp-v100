#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    return 0;
}
