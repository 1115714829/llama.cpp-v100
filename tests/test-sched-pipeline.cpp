#include "ggml.h"
#include "ggml-backend.h"

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

int main() {
    printf("pipeline: 1 backend ... ");
    fflush(stdout);
    run_pipeline(1);
    printf("OK\n");

    printf("pipeline: 2 backends ... ");
    fflush(stdout);
    run_pipeline(2);
    printf("OK\n");

    return 0;
}
