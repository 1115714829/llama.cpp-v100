#include "ggml-backend-sched.h"

#include <string.h>
























void ggml_backend_sched_set_n_stages(ggml_backend_sched_t sched, int n_stages) {
    GGML_ASSERT(sched);
    GGML_ASSERT(n_stages >= 1 && n_stages <= GGML_SCHED_MAX_STAGES);
    GGML_ASSERT(!sched->is_alloc);

    sched->n_stages = n_stages;
    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_post_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
    sched->pipeline_error = GGML_STATUS_SUCCESS;



    if (n_stages > 1) {



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



    for (int b = 0; b < sched->n_backends; b++) {
        if (stage->backend_seq[b] == 0) {
            continue;
        }



        if (ggml_backend_sched_has_workers(sched)) {
            ggml_backend_sched_worker_wait_done(sched, b, stage->backend_seq[b]);
        }
        if (stage->done[b] != NULL) {
            ggml_backend_event_synchronize(stage->done[b]);
        }
    }



    if (stage->error != NULL) {
        const int ec = stage->error->load();
        if (ec != (int) GGML_STATUS_SUCCESS) {
            sched->pipeline_error = (enum ggml_status) ec;
        }
        stage->error.reset();
    }

    stage->in_flight = false;
}


bool ggml_backend_sched_pipeline_begin(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int64_t ubatch_id) {
    GGML_ASSERT(sched);
    GGML_ASSERT(graph);
    GGML_ASSERT(sched->n_stages > 1);

    const int s = sched->next_stage;
    struct ggml_backend_sched_stage * stage = &sched->stages[s];


    if (stage->in_flight) {
        ggml_backend_sched_pipeline_wait_stage(sched, s);
    }
    if (sched->pipeline_error != GGML_STATUS_SUCCESS) {


        return false;
    }


    sched->pipeline_post_stage = -1;
    memset(stage->backend_seq, 0, sizeof(stage->backend_seq));



    sched->ctx                 = stage->ctx;
    sched->context_buffer      = stage->context_buffer;
    sched->context_buffer_size = stage->context_buffer_size;





    const int slot = (int) (ubatch_id % sched->n_copies);
    ggml_backend_sched_reset(sched);
    sched->next_copy = slot;

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        return false;
    }

    stage->slot = sched->cur_copy;
    sched->pipeline_stage = s;

    return true;
}

enum ggml_status ggml_backend_sched_pipeline_submit(ggml_backend_sched_t sched, int64_t ubatch_id) {
    GGML_ASSERT(sched);

    if (sched->n_stages == 1) {

        return GGML_STATUS_SUCCESS;
    }

    GGML_ASSERT(sched->pipeline_stage >= 0);
    GGML_ASSERT(ubatch_id > sched->pipeline_last_ubatch_id);
    sched->pipeline_last_ubatch_id = ubatch_id;

    const int s = sched->pipeline_stage;
    struct ggml_backend_sched_stage * stage = &sched->stages[s];

    const enum ggml_status status = ggml_backend_sched_compute_splits(sched);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }



    stage->ubatch_id = ubatch_id;
    stage->in_flight = true;


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



    bool in_flight = false;
    for (int s = 0; s < sched->n_stages; s++) {
        in_flight = in_flight || sched->stages[s].in_flight;
    }

    if (in_flight) {




        ggml_backend_sched_workers_drain(sched);
        for (int b = 0; b < sched->n_backends; b++) {
            ggml_backend_synchronize(sched->backends[b]);
        }
    }


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



    sched->pipeline_error = GGML_STATUS_SUCCESS;

    sched->next_stage = 0;
    sched->oldest_stage = 0;
    sched->pipeline_stage = -1;
    sched->pipeline_post_stage = -1;
    sched->pipeline_last_ubatch_id = -1;
}
