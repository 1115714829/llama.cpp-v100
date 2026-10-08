#include "ggml-backend-sched.h"

#include <string.h>

// Pipeline parallelism primitives for multi-ubatch prefill.
//
// n_stages is the pipeline depth: the number of ubatches that may be in flight at once. The caller
// builds a distinct graph per ubatch, so the scheduler can split and allocate a ubatch independently
// while the previous ones keep running on the devices:
//
//   begin():  wait until the stage is reusable, split+allocate the graph on the stage's input slot
//   submit(): enqueue all splits of the stage without waiting
//   wait():   wait for one ubatch (before reading its outputs)
//   advance(): wait for all in-flight ubatches and reset the ring
//
// The input copies are double buffered independently of the depth: ubatch i uses slot i % n_copies
// on every backend. A slot may be written again only after the previous user finished with it; this
// is enforced by the per-(backend, slot) events, recorded after each split.
//
// Cross-device activation copies are chained by events inside the CUDA backend, and consecutive
// ubatches on the same device are ordered by the device stream FIFO. The only scheduler-level waits
// are the stage reuse (ring wrap-around) and the output reads.
//
// The host inputs of a pipeline ubatch are copied synchronously in ggml_backend_sched_compute_splits,
// so once submit() returns the caller can overwrite the host input tensors for the next ubatch
// without waiting for the device.

void ggml_backend_sched_set_n_stages(ggml_backend_sched_t sched, int n_stages) {
    GGML_ASSERT(sched);
    GGML_ASSERT(n_stages >= 1 && n_stages <= GGML_SCHED_MAX_STAGES);
    GGML_ASSERT(!sched->is_alloc); // cannot change the pipeline depth while a graph is allocated

    sched->n_stages = n_stages;
    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
}

static void ggml_backend_sched_pipeline_wait_stage(ggml_backend_sched_t sched, int stage_id) {
    struct ggml_backend_sched_stage * stage = &sched->stages[stage_id];

    // the stage events are recorded by the submitted tasks: wait until those tasks have been
    // enqueued, otherwise the synchronize below could observe a stale recording
    if (ggml_backend_sched_has_workers(sched)) {
        for (int b = 0; b < sched->n_backends; b++) {
            if (stage->backend_seq[b] > 0) {
                ggml_backend_sched_worker_wait_done(sched, b, stage->backend_seq[b]);
            }
        }
    }

    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_event_t ev = sched->events[b][stage->slot];
        if (ev != NULL) {
            ggml_backend_event_synchronize(ev);
        }
    }

    stage->in_flight = false;
}

// split and allocate the graph on the next pipeline stage, without computing it
bool ggml_backend_sched_pipeline_begin(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int64_t ubatch_id) {
    GGML_ASSERT(sched);
    GGML_ASSERT(graph);
    GGML_ASSERT(sched->n_stages > 1);

    const int s = sched->next_stage;
    struct ggml_backend_sched_stage * stage = &sched->stages[s];

    // the stage can only be reused after its previous ubatch completed
    if (stage->in_flight) {
        ggml_backend_sched_pipeline_wait_stage(sched, s);
    }
    memset(stage->backend_seq, 0, sizeof(stage->backend_seq));

    // input copies are double buffered: ubatch i uses slot i % n_copies on every backend, so
    // in-flight ubatches never overwrite each other's device inputs. the previous user of a slot is
    // waited for by the writers (host input copies and pushed cross-device copies), see
    // ggml_backend_sched_compute_splits_pipeline
    const int slot = (int) (ubatch_id % sched->n_copies);
    ggml_backend_sched_reset(sched);
    sched->next_copy = slot;

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        return false;
    }

    stage->slot = sched->cur_copy; // == slot
    sched->pipeline_stage = s;

    return true;
}

enum ggml_status ggml_backend_sched_pipeline_submit(ggml_backend_sched_t sched, int64_t ubatch_id) {
    GGML_ASSERT(sched);

    if (sched->n_stages == 1) {
        // single stage: the caller computed the graph through ggml_backend_sched_graph_compute_async
        return GGML_STATUS_SUCCESS;
    }

    GGML_ASSERT(sched->pipeline_stage >= 0);
    GGML_ASSERT(ubatch_id > sched->pipeline_last_ubatch_id); // submission order follows the batch order
    sched->pipeline_last_ubatch_id = ubatch_id;

    const int s = sched->pipeline_stage;
    struct ggml_backend_sched_stage * stage = &sched->stages[s];

    const enum ggml_status status = ggml_backend_sched_compute_splits(sched);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // compute_splits records sched->events[b][cur_copy] at the end of each split; cur_copy is the
    // input slot of this ubatch
    stage->ubatch_id = ubatch_id;
    stage->in_flight = true;
    sched->pipeline_stage = -1;

    sched->next_stage = (s + 1) % sched->n_stages;

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_pipeline_wait(ggml_backend_sched_t sched, int64_t ubatch_id) {
    GGML_ASSERT(sched);

    for (int s = 0; s < sched->n_stages; s++) {
        struct ggml_backend_sched_stage * stage = &sched->stages[s];
        if (!stage->in_flight || stage->ubatch_id != ubatch_id) {
            continue;
        }

        ggml_backend_sched_pipeline_wait_stage(sched, s);
        return;
    }
}

void ggml_backend_sched_pipeline_advance(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    // nothing submitted through the pipeline since the last advance: no outputs to wait for. decode
    // batches run on the other schedulers and must keep their host/device overlap
    bool in_flight = false;
    for (int s = 0; s < sched->n_stages; s++) {
        in_flight = in_flight || sched->stages[s].in_flight;
    }
    if (!in_flight) {
        return;
    }

    // the per-stage events only cover the compute, not the asynchronous copies of the outputs to
    // the host. synchronize the backends so those reads complete before the host output buffers are
    // reused by the next batch. this runs once per prefill, not per ubatch
    ggml_backend_sched_workers_drain(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_synchronize(sched->backends[b]);
    }

    for (int s = 0; s < sched->n_stages; s++) {
        sched->stages[s].in_flight = false;
    }

    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
}
