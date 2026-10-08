#pragma once

// ggml-backend internal header: scheduler state shared between ggml-backend.cpp
// and ggml-backend-pipeline.cpp

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#ifdef  __cplusplus
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#endif

#ifdef  __cplusplus
// one submission thread per non-CPU backend; created only in pipeline mode (n_copies > 1).
// the thread only runs tasks posted with ggml_backend_sched_worker_post()
struct ggml_backend_sched_worker {
    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;       // wakes the worker
    std::condition_variable cv_done;  // signals progress of n_done
    std::deque<std::function<void()>> tasks;
    uint64_t                n_posted = 0; // tasks posted so far
    uint64_t                n_done   = 0; // tasks executed (enqueued on the device stream) so far
    bool                    stop     = false;
};
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

// input copies of the pipeline. the copies are double buffered and independent of the pipeline
// depth: ubatch i uses copy i % n_copies on every backend, so two copies suffice for any depth
#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 2
#endif

// pipeline parallelism: max number of ubatches of one prefill in flight at once
#ifndef GGML_SCHED_MAX_STAGES
#define GGML_SCHED_MAX_STAGES 8
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

// one in-flight ubatch of a pipeline-parallel prefill
//
// every ubatch has its own graph (the caller rebuilds it) and uses the input copy of its
// submission index, so a later ubatch can be split and allocated while the earlier ones are still
// running. the input copies are small: only the graph inputs and the split inputs are duplicated
struct ggml_backend_sched_stage {
    int slot;          // input copy slot used by this stage (ubatch_id % n_copies)
    int64_t ubatch_id; // logical batch index, for ordering checks
    bool in_flight;
    // last submission-worker task posted for this stage, per backend (0 = none); the stage events
    // can only be synchronized once these tasks have been enqueued
    uint64_t backend_seq[GGML_SCHED_MAX_BACKENDS];
};

// shared with ggml-backend-pipeline.cpp: enqueue the splits of the currently allocated graph
enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched);

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    // pipeline depth; 1 = no pipeline (single ubatch at a time)
    int n_stages;
    struct ggml_backend_sched_stage stages[GGML_SCHED_MAX_STAGES];
    int next_stage;   // ring index for submission
    int oldest_stage; // ring index of the oldest in-flight batch

    // pipeline state; each in-flight ubatch has its own freshly built graph and input copy, so the
    // scheduler only tracks which stage is being prepared and the submission order
    int pipeline_stage;              // stage prepared by ggml_backend_sched_pipeline_begin, or -1
    int64_t pipeline_last_ubatch_id; // last submitted ubatch id, for ordering checks

#ifdef __cplusplus
    // per-backend submission threads, pipeline mode only (n_copies > 1), indexed by backend id;
    // NULL for the CPU backend. pointer because the scheduler is calloc'ed
    std::vector<std::unique_ptr<struct ggml_backend_sched_worker>> * workers;

    // handoff events of the pipeline submission, one per split, recorded by the producer split after
    // its output copies; created on first use and kept for the lifetime of the scheduler
    std::vector<ggml_backend_event_t> * handoff_events;
#endif

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // if true, user inputs are copied to the split backends asynchronously
    // in that case, the caller must synchronize the scheduler before overwriting the inputs
    bool async_inputs;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

#ifdef  __cplusplus
}

// per-backend submission workers, implemented in ggml-backend.cpp; pipeline mode only
uint64_t ggml_backend_sched_worker_post(ggml_backend_sched_t sched, int backend_id, std::function<void()> fn);
void ggml_backend_sched_worker_wait_done(ggml_backend_sched_t sched, int backend_id, uint64_t seq);
void ggml_backend_sched_workers_drain(ggml_backend_sched_t sched);
bool ggml_backend_sched_has_workers(ggml_backend_sched_t sched);
#endif
