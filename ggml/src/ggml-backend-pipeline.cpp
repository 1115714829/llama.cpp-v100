#include "ggml-backend-sched.h"

// Pipeline parallelism primitives for multi-ubatch prefill.
//
// n_stages is the pipeline depth: the number of ubatches that may be in flight at once.
// This first version supports n_stages == 1, where submit/wait/advance are equivalent to
// ggml_backend_sched_graph_compute_async / ggml_backend_sched_synchronize. The multi-stage
// overlap path is added on top of these primitives in a later step.

void ggml_backend_sched_set_n_stages(ggml_backend_sched_t sched, int n_stages) {
    GGML_ASSERT(sched);
    GGML_ASSERT(n_stages >= 1 && n_stages <= GGML_SCHED_MAX_STAGES);
    GGML_ASSERT(!sched->is_alloc); // cannot change the pipeline depth while a graph is allocated

    sched->n_stages = n_stages;
    sched->next_stage = 0;
    sched->oldest_stage = 0;
}

enum ggml_status ggml_backend_sched_pipeline_submit(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int64_t ubatch_id) {
    GGML_ASSERT(sched);
    GGML_ASSERT(graph);
    GGML_ASSERT(sched->n_stages == 1); // multi-stage submission is added in a later step

    const int s = sched->next_stage;
    struct ggml_backend_sched_stage * stage = &sched->stages[s];

    // a stage can only be reused after the previous batch on it has completed
    if (stage->in_flight) {
        ggml_backend_sched_pipeline_wait(sched, stage->ubatch_id);
    }

    const enum ggml_status status = ggml_backend_sched_graph_compute_async(sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    stage->graph     = &sched->graph;
    stage->splits    = sched->splits;
    stage->n_splits  = sched->n_splits;
    stage->ubatch_id = ubatch_id;
    stage->in_flight = true;

    sched->next_stage = (s + 1) % sched->n_stages;

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_pipeline_wait(ggml_backend_sched_t sched, int64_t ubatch_id) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->n_stages == 1);

    struct ggml_backend_sched_stage * stage = &sched->stages[0];
    if (!stage->in_flight || stage->ubatch_id != ubatch_id) {
        return;
    }

    // with n_stages == 1 no other batch can be in flight, so a full synchronize is exact
    ggml_backend_sched_synchronize(sched);
    stage->in_flight = false;
}

void ggml_backend_sched_pipeline_advance(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->n_stages == 1);

    struct ggml_backend_sched_stage * stage = &sched->stages[0];
    if (stage->in_flight) {
        ggml_backend_sched_synchronize(sched);
        stage->in_flight = false;
    }

    sched->next_stage = 0;
    sched->oldest_stage = 0;
}
