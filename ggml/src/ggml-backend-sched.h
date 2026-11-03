#pragma once




#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#ifdef  __cplusplus
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#endif

#ifdef  __cplusplus


struct ggml_backend_sched_worker {
    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::deque<std::function<void()>> tasks;
    uint64_t                n_posted = 0;
    uint64_t                n_done   = 0;
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



#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 2
#endif


#ifndef GGML_SCHED_MAX_STAGES
#define GGML_SCHED_MAX_STAGES 8
#endif




struct ggml_backend_sched_upload_run {
    struct ggml_tensor * window;
    size_t               stage_offs;
    size_t               size;
};

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;



    int    * inputs_cpy_node;
    size_t * inputs_stage_offs;
    int n_inputs;
    int inputs_capacity;

    struct ggml_backend_sched_upload_run * runs;
    int n_runs;
    // graph view of this split
    struct ggml_cgraph graph;
};






struct ggml_backend_sched_stage {
    int slot;
    int64_t ubatch_id;
    bool in_flight;



    ggml_backend_event_t done[GGML_SCHED_MAX_BACKENDS];


    uint64_t backend_seq[GGML_SCHED_MAX_BACKENDS];


    char *                context_buffer;
    size_t                context_buffer_size;
    struct ggml_context * ctx;



    std::vector<ggml_backend_event_t> * handoff_events;


    std::shared_ptr<std::atomic<int>> error;
};





struct ggml_backend_sched_staging {
    ggml_backend_buffer_t buf;
    size_t size;
    ggml_backend_event_t done[GGML_SCHED_MAX_STAGES];
};


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


    struct ggml_backend_sched_staging staging[GGML_SCHED_MAX_BACKENDS];


    size_t staging_reserve[GGML_SCHED_MAX_BACKENDS];




    ggml_backend_buffer_t inputs_staging[GGML_SCHED_MAX_BACKENDS];
    size_t                inputs_staging_size[GGML_SCHED_MAX_BACKENDS];
    size_t                inputs_stage_need[GGML_SCHED_MAX_BACKENDS];
    size_t                inputs_stage_reserve[GGML_SCHED_MAX_BACKENDS];


    int n_stages;
    struct ggml_backend_sched_stage stages[GGML_SCHED_MAX_STAGES];
    int next_stage;
    int oldest_stage;



    int pipeline_stage;
    int pipeline_post_stage;
    int64_t pipeline_last_ubatch_id;
    enum ggml_status pipeline_error;

#ifdef __cplusplus


    std::vector<std::unique_ptr<struct ggml_backend_sched_worker>> * workers;



    std::vector<ggml_backend_event_t> * handoff_events;
#endif

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;



    bool async_inputs;




    ggml_backend_event_t ev_inputs[GGML_SCHED_MAX_BACKENDS];

    int debug;



    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;


    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;



    int graph_size;
    int prev_graph_size;

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



void ggml_backend_sched_workers_start(ggml_backend_sched_t sched);
uint64_t ggml_backend_sched_worker_post(ggml_backend_sched_t sched, int backend_id, std::function<void()> fn);
void ggml_backend_sched_worker_wait_done(ggml_backend_sched_t sched, int backend_id, uint64_t seq);
void ggml_backend_sched_workers_drain(ggml_backend_sched_t sched);
bool ggml_backend_sched_has_workers(ggml_backend_sched_t sched);
#endif
