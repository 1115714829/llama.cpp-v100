#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_BACKEND_API extern
#endif

#ifdef  __cplusplus
#include <functional>
#endif

#ifdef  __cplusplus
extern "C" {
#endif
    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;


    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n  (ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
    GGML_API size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size    (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API size_t                ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
    GGML_API bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device      (ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
    GGML_API void                           ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
    GGML_API bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
    GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // Backend (stream)
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

    // "offset" refers to the offset in tensor->data for setting/getting data
    GGML_API void ggml_backend_tensor_set   (      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get   (const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_memset(      struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);

    //
    // Events
    //

    GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
    GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
    GGML_API void                 ggml_backend_event_synchronize(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);

    //
    // Backend device
    //

    enum ggml_backend_dev_type {
        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
        // "meta" device wrapping multiple other devices for tensor parallelism
        GGML_BACKEND_DEVICE_TYPE_META,
    };

    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
        // mmap is supported for loading
        bool mmap_support;
    };

    // all the device properties
    struct ggml_backend_dev_props {
        // device name
        const char * name;
        // device description
        const char * description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum ggml_backend_dev_type type;
        // device id
        //   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
        //   if the id is unknown, this should be NULL
        const char * device_id;
        // device capabilities
        struct ggml_backend_dev_caps caps;
    };

    GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
    GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
    GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
    GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
    GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
    GGML_API bool                          ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
    GGML_API bool                          ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);



    GGML_API bool                          ggml_backend_dev_supports_bf16(ggml_backend_dev_t device);

    //
    // Backend (reg)
    //

    GGML_API const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
    GGML_API size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
    GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
    GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

    // Common functions that may be obtained using ggml_backend_reg_get_proc_address

    // Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)
    typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t * backends, size_t n_backends);
    typedef void   (*ggml_backend_comm_free_t)(void * comm_ctx);
    typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);






    typedef bool   (*ggml_backend_comm_allreduce_rank_can_t)(void * comm_ctx, struct ggml_tensor ** tensors, bool exact);
    typedef bool   (*ggml_backend_comm_allreduce_rank_t)    (void * comm_ctx, size_t rank, struct ggml_tensor * tensor, bool exact);






    typedef bool   (*ggml_backend_capture_begin_t) (ggml_backend_t backend);
    typedef void * (*ggml_backend_capture_end_t)   (ggml_backend_t backend);




    typedef void * (*ggml_backend_capture_update_t)(ggml_backend_t backend, void * exec);
    typedef bool   (*ggml_backend_capture_launch_t)(ggml_backend_t backend, void * exec);
    typedef void   (*ggml_backend_capture_free_t)  (void * exec);




    typedef bool   (*ggml_backend_graph_prefill_sized_t)(const struct ggml_cgraph * cgraph);
    typedef void   (*ggml_backend_set_graph_uncached_t)(ggml_backend_t backend, bool uncached);
    typedef bool   (*ggml_backend_comm_allreduce_rank_capturable_t)(void * comm_ctx, struct ggml_tensor ** tensors, bool exact);









    typedef bool   (*ggml_backend_comm_reduce_scatter_rank_can_t)(void * comm_ctx, struct ggml_tensor ** srcs, struct ggml_tensor ** dsts, int axis);
    typedef bool   (*ggml_backend_comm_reduce_scatter_rank_t)    (void * comm_ctx, size_t rank, struct ggml_tensor * src_full, struct ggml_tensor * dst_slice, int axis, int64_t col_off);




    typedef bool   (*ggml_backend_comm_allgather_rank_can_t)(void * comm_ctx, struct ggml_tensor ** srcs, struct ggml_tensor ** dsts, int axis);
    typedef bool   (*ggml_backend_comm_allgather_rank_t)    (void * comm_ctx, size_t rank, struct ggml_tensor * src_slice, struct ggml_tensor * dst_full, int axis, int64_t col_off);




    typedef bool   (*ggml_backend_comm_allgather_rank_available_t)(void * comm_ctx);







    typedef void   (*ggml_backend_comm_connect_t)(void * comm_ctx, int64_t ne_allreduce, int64_t ne_reduce_scatter);

    // Split buffer type for tensor parallelism (old)
    typedef ggml_backend_buffer_type_t   (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
    // Set the number of threads for the backend
    typedef void                         (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);
    // Get additional buffer types provided by the device (returns a NULL-terminated array)
    typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);
    // Set the abort callback for the backend
    typedef void                         (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);

    //
    // Backend registry
    //

    GGML_API void ggml_backend_register(ggml_backend_reg_t reg);

    GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);

    // Backend (reg) enumeration
    GGML_API size_t             ggml_backend_reg_count(void);
    GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
    GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    GGML_API size_t             ggml_backend_dev_count(void);
    GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
    GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
    GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
    GGML_API ggml_backend_t ggml_backend_init_best(void);

    // Load a backend from a dynamic library and register it
    GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
    // Unload a backend if loaded dynamically and unregister it
    GGML_API void               ggml_backend_unload(ggml_backend_reg_t reg);
    // Load all known backends from dynamic libraries
    GGML_API void               ggml_backend_load_all(void);
    GGML_API void               ggml_backend_load_all_from_path(const char * dir_path);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
        ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via ggml_backend_alloc_ctx_tensors
    }
    */

    typedef struct ggml_backend_sched * ggml_backend_sched_t;

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Callback while copying input weights of a split
    // if the user returns false the scheduler simply copies the entire weight
    // the callback is called only for input weights in host buffers
    // the callback is called after all non-weight inputs of the split have been copied
    // `src` is the tensor in the previous split
    // `dst` is the copy of `src` in the split
    // `graph` is the compute graph of the split
    typedef bool (*ggml_backend_sched_copy_callback)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst, struct ggml_cgraph * graph, void * user_data);

    // Callback when the buffers reserved for a graph are not sufficient and have to be reserved again
    // it happens on the graph allocation path and costs a synchronization of the backends
    // `sched` is the scheduler
    // `graph` is the compute graph being allocated
    // `backend_ids_changed` is true if the backend assignment of the graph nodes changed since the last allocation
    // `graph_size_changed` is true if the graph size changed since the last allocation
    typedef void (*ggml_backend_sched_realloc_callback)(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_stages(ggml_backend_sched_t sched);

    GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);


    GGML_API int                  ggml_backend_sched_get_tensor_backend_id(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);




    GGML_API void                 ggml_backend_sched_set_n_stages(ggml_backend_sched_t sched, int n_stages);


    GGML_API bool                 ggml_backend_sched_pipeline_begin(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int64_t ubatch_id);


    GGML_API int                  ggml_backend_sched_pipeline_next_stage(ggml_backend_sched_t sched);



    GGML_API void                 ggml_backend_sched_pipeline_wait_next_stage(ggml_backend_sched_t sched);


    GGML_API enum ggml_status     ggml_backend_sched_pipeline_submit(ggml_backend_sched_t sched, int64_t ubatch_id);

    GGML_API void                 ggml_backend_sched_pipeline_wait(ggml_backend_sched_t sched, int64_t ubatch_id);

    GGML_API void                 ggml_backend_sched_pipeline_advance(ggml_backend_sched_t sched);



    GGML_API void                 ggml_backend_sched_wait_submitted(ggml_backend_sched_t sched, ggml_backend_t backend);

#ifdef  __cplusplus




    GGML_API void                 ggml_backend_sched_post_task(ggml_backend_sched_t sched, int backend_id, std::function<void()> fn);
#endif







    GGML_API void                 ggml_backend_sched_set_async_inputs(ggml_backend_sched_t sched, bool async_inputs);






    GGML_API void                 ggml_backend_sched_set_inputs_event(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_event_t event);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
    // The correct way to use this API is to discard the deallocated tensors and create new ones.
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

    // Set a callback to be called when the inputs weights of a split are being copied
    GGML_API void                 ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);

    // Set a callback to be called when the compute buffers of a graph are being reallocated
    GGML_API void                 ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data);

    //
    // Meta backend
    //

#define GGML_BACKEND_META_MAX_DEVICES 16


#define GGML_BACKEND_META_MAX_SEGMENTS 4

    enum ggml_backend_meta_split_axis {
        // tensor split by tensor dimensions:
        GGML_BACKEND_SPLIT_AXIS_0 = 0,
        GGML_BACKEND_SPLIT_AXIS_1 = 1,
        GGML_BACKEND_SPLIT_AXIS_2 = 2,
        GGML_BACKEND_SPLIT_AXIS_3 = 3,

        GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
        GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum


        GGML_BACKEND_SPLIT_AXIS_LOCAL_PARTIAL = 12,

        // for internal bookkeeping only:
        GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
        GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
    };
    GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

    struct ggml_backend_meta_split_state {
        enum ggml_backend_meta_split_axis axis;

        // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
        //   - each device has a slice of the tensor along the split axis
        //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
        //   - some tensors have an inhomogenenous data layout along the split axis,
        //     those tensors are divided into segments which are each individually split across devices
        //     (this usually happens when multiple tensors are fused into a single one)
        //   - ne has one entry per segment and device and that segment repeats nr times;
        //     when has_off is false, in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
        //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
        //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
        //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
        //     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
        //     where the segment for K/V repeats twice
        //   - if has_off is true the source offsets in off are used instead of the accumulated lengths of ne,
        //     which allows the slices of different devices to overlap; off has the same shape/units as ne,
        //     nr[s] must be 1 in this case as repeats have no explicit offsets
        int64_t  ne[GGML_BACKEND_META_MAX_SEGMENTS*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[GGML_BACKEND_META_MAX_SEGMENTS];
        uint32_t n_segments;

        int64_t  off[GGML_BACKEND_META_MAX_SEGMENTS*GGML_BACKEND_META_MAX_DEVICES];
        bool     has_off;
    };

    // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
    typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

    // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
    // TODO: this looks a bit strange - a backend API creates a device. I think we should try
    //       express this as a backend registry functionality instead
    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);


    GGML_API bool               ggml_backend_dev_is_meta        (ggml_backend_dev_t dev);
    GGML_API size_t             ggml_backend_meta_dev_n_devs    (ggml_backend_dev_t meta_dev);
    GGML_API ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index);


    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);




    GGML_API void ggml_backend_meta_comm_connect(ggml_backend_t backend, int64_t ne_allreduce, int64_t ne_reduce_scatter);





    GGML_API ggml_backend_buffer_type_t ggml_backend_meta_mapped_weights_buffer_type(ggml_backend_dev_t meta_dev);





    GGML_API bool ggml_backend_meta_buft_is_device_mapped(ggml_backend_buffer_type_t buft);



    GGML_API bool ggml_backend_buft_is_device_mapped(ggml_backend_buffer_type_t buft);




    GGML_API void ggml_backend_meta_copy_mirrored_async(ggml_backend_t backend,
            const struct ggml_tensor * src, size_t src_offs, struct ggml_tensor * dst, size_t dst_offs, size_t nbytes);



    #define GGML_BACKEND_META_ASSIST_MIN_Q  256
    #define GGML_BACKEND_META_ASSIST_MIN_KV 32768



    #define GGML_BACKEND_META_ASSIST_KV_WINDOW 131072





    struct ggml_backend_meta_assist_split {
        int32_t kv_end;
        int32_t s;
    };















    struct ggml_backend_meta_assist_rank {
        int32_t role;
        int32_t n_owners;
        int32_t owners[2];
        int32_t assist_rank;
        int32_t idle_load;
        int32_t owner_dev[2];




        int32_t kv_view;
        int32_t kv_window;

        int32_t q_pad;
        int32_t q_len;
        int32_t heads_q;
        int32_t heads_kv;
        int32_t batch;
        int32_t kv_type;
        float   scale;


        const void * mask;
        int64_t mask_row_stride;
        int64_t mask_batch_stride;
        bool    mask_is_range;


        const void * k_src[2];
        const void * v_src[2];
        const void * qs_src[2];
        float * p_out_dst[2];
        float * p_max_dst[2];
        float * p_sum_dst[2];
        uint32_t * epoch_kv[2];
        uint32_t * epoch_partial[2];
        uint32_t * epoch_free[2];

        uint32_t   free_lag[2];

        uint32_t * gen[2];
        const struct ggml_backend_meta_assist_split * split_src[2];
        struct ggml_backend_meta_assist_split * split_local;


        float * p_out_local;
        float * p_max_local;
        float * p_sum_local;
        uint32_t * epoch_kv_local;
        uint32_t * epoch_partial_local;
        uint32_t * gen_local;
        uint32_t * epoch_free_local;


        void * work;
        size_t work_bytes;
    };

    typedef bool   (*ggml_backend_set_assist_t)(ggml_backend_t backend, const struct ggml_backend_meta_assist_rank * desc);
    typedef bool   (*ggml_backend_assist_run_t)(ggml_backend_t backend, const struct ggml_backend_meta_assist_rank * desc);
    typedef size_t (*ggml_backend_assist_workspace_size_t)(int q_pad, int heads_q);




    struct ggml_backend_meta_assist_scratch {
        size_t  qs_offset;
        size_t  p_out_offset;
        size_t  p_max_offset;
        size_t  p_sum_offset;
        int32_t q_pad;
    };

    typedef bool (*ggml_backend_assist_scratch_layout_t)(const struct ggml_tensor * dst, struct ggml_backend_meta_assist_scratch * layout);


    typedef int (*ggml_backend_device_ordinal_t)(ggml_backend_t backend);

    //
    // Utils
    //

    struct ggml_backend_graph_copy {
        ggml_backend_buffer_t buffer;
        struct ggml_context * ctx_allocated;
        struct ggml_context * ctx_unallocated;
        struct ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);

    typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);

    // Tensor initialization
    GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);

    // CPU buffer types are always available
    GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

#ifdef  __cplusplus
}
#endif
