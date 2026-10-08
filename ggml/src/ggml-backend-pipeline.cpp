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
// are the stage reuse (ring wrap-around, distance n_stages) and the output reads.
//
// The host inputs of a pipeline ubatch are copied by the main thread into a pinned staging ring and
// uploaded by the submission workers before their graph computes, so once submit() returns the
// caller can overwrite the host input tensors for the next ubatch without waiting for the device.

void ggml_backend_sched_set_n_stages(ggml_backend_sched_t sched, int n_stages) {
    GGML_ASSERT(sched);
    GGML_ASSERT(n_stages >= 1 && n_stages <= GGML_SCHED_MAX_STAGES);
    GGML_ASSERT(!sched->is_alloc); // cannot change the pipeline depth while a graph is allocated

    sched->n_stages = n_stages;
    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_post_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
    sched->pipeline_error = GGML_STATUS_SUCCESS;

    // the submission workers only exist when the scheduler overlaps pipeline stages. a scheduler
    // with double buffered inputs but no overlap (n_stages == 1) submits synchronously
    if (n_stages > 1) {
        // every stage gets its own input copy context: the submission workers of a stage keep
        // reading its copy tensor structs until the stage is reused, n_stages ubatches later. the
        // default context built by ggml_backend_sched_new becomes stage 0's
        for (int s = 0; s < n_stages; s++) {
            if (s == 0) {
                sched->stages[s].context_buffer      = sched->context_buffer;
                sched->stages[s].context_buffer_size = sched->context_buffer_size;
                sched->stages[s].ctx                 = sched->ctx;
            } else {
                sched->stages[s].context_buffer      = (char *) malloc(sched->context_buffer_size);
                sched->stages[s].context_buffer_size = sched->context_buffer_size;
                sched->stages[s].ctx                 = NULL;
            }
        }
        sched->ctx                 = sched->stages[0].ctx;
        sched->context_buffer      = sched->stages[0].context_buffer;
        sched->context_buffer_size = sched->stages[0].context_buffer_size;

        // per-stage completion events and the staging events of the workers. devices without event
        // support (CPU) yield NULL and never take the pipeline submission path
        for (int s = 0; s < n_stages; s++) {
            for (int b = 0; b < sched->n_backends; b++) {
                if (sched->stages[s].done[b] == NULL) {
                    sched->stages[s].done[b] = ggml_backend_event_new(ggml_backend_get_device(sched->backends[b]));
                }
                if (sched->staging[b].done[s] == NULL) {
                    sched->staging[b].done[s] = ggml_backend_event_new(ggml_backend_get_device(sched->backends[b]));
                }
            }
        }
        ggml_backend_sched_workers_start(sched);
    }
}

static void ggml_backend_sched_pipeline_wait_stage(ggml_backend_sched_t sched, int stage_id) {
    struct ggml_backend_sched_stage * stage = &sched->stages[stage_id];

    // wait for the completion of this stage on every backend that ran a part of it. the distance to
    // the previous user of the stage is n_stages, so the pipeline keeps n_stages ubatches in flight
    for (int b = 0; b < sched->n_backends; b++) {
        if (stage->backend_seq[b] == 0) {
            continue;
        }

        // the completion events are recorded by the submitted tasks: wait until those tasks have
        // been enqueued, otherwise the synchronize below could observe a stale recording
        if (ggml_backend_sched_has_workers(sched)) {
            ggml_backend_sched_worker_wait_done(sched, b, stage->backend_seq[b]);
        }
        if (stage->done[b] != NULL) {
            ggml_backend_event_synchronize(stage->done[b]);
        }
    }

    // the tasks of the stage have run by now: their error flag is final. the first failure is
    // sticky, the caller checks it through pipeline_advance before reading the stage outputs
    if (stage->error != NULL) {
        const int ec = stage->error->load();
        if (ec != (int) GGML_STATUS_SUCCESS) {
            sched->pipeline_error = (enum ggml_status) ec;
        }
        stage->error.reset();
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
    if (sched->pipeline_error != GGML_STATUS_SUCCESS) {
        // a previous ubatch of this pipeline failed; abort the prefill. the flag is sticky until
        // the caller drains the pipeline through pipeline_advance
        return false;
    }
    // this stage is being prepared: no submission is outstanding for the post-task target anymore,
    // so a task posted before the submit below must run inline instead of extending a stale stage
    sched->pipeline_post_stage = -1;
    memset(stage->backend_seq, 0, sizeof(stage->backend_seq));

    // switch the scheduler to this stage's input copy context. the context of the previous user of
    // the stage was freed by the wait above (or has not been built yet), so it is safe to rebuild
    sched->ctx                 = stage->ctx;
    sched->context_buffer      = stage->context_buffer;
    sched->context_buffer_size = stage->context_buffer_size;

    // input copies are double buffered: ubatch i uses slot i % n_copies on every backend, so
    // in-flight ubatches never overwrite each other's device inputs. the previous user of a slot is
    // waited for by the pushed cross-device copies, and the host input uploads are ordered after it
    // by the backend stream FIFO, see ggml_backend_sched_compute_splits_pipeline
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

    // compute_splits records sched->events[b][cur_copy] and the stage completion events at the end
    // of each split; cur_copy is the input slot of this ubatch
    stage->ubatch_id = ubatch_id;
    stage->in_flight = true;
    // worker tasks of this ubatch (including the output copies posted later) extend this stage, so
    // the stage reuse wait must cover tasks posted after the submission returns
    sched->pipeline_post_stage = s;
    sched->pipeline_stage = -1;
    sched->next_stage = (s + 1) % sched->n_stages;

    return GGML_STATUS_SUCCESS;
}

int ggml_backend_sched_pipeline_next_stage(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->next_stage;
}

void ggml_backend_sched_pipeline_wait_next_stage(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    const int s = sched->next_stage;

    // the stage seqs cover the output copies posted after the compute (see
    // ggml_backend_sched_post_task), so this also waits for them
    if (sched->stages[s].in_flight) {
        ggml_backend_sched_pipeline_wait_stage(sched, s);
    }
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

    if (in_flight) {
        // the per-stage events only cover the compute, not the asynchronous copies of the outputs
        // to the host (those are posted to the same workers, after the compute). drain the workers
        // and synchronize the backends so those reads complete before the host output buffers are
        // reused by the next batch. this runs once per prefill, not per ubatch
        ggml_backend_sched_workers_drain(sched);
        for (int b = 0; b < sched->n_backends; b++) {
            ggml_backend_synchronize(sched->backends[b]);
        }
    }

    // the tasks of every stage have run: collect their errors before the stages are reused
    for (int s = 0; s < sched->n_stages; s++) {
        if (sched->stages[s].error != NULL) {
            const int ec = sched->stages[s].error->load();
            if (ec != (int) GGML_STATUS_SUCCESS) {
                sched->pipeline_error = (enum ggml_status) ec;
            }
            sched->stages[s].error.reset();
        }
        sched->stages[s].in_flight = false;
    }

    // the caller has observed any failure through pipeline_begin/pipeline_wait: the next prefill
    // starts clean
    sched->pipeline_error = GGML_STATUS_SUCCESS;

    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_post_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
}
