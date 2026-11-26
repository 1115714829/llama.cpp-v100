#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    #define GGML_BACKEND_API_VERSION 3

    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)        (ggml_backend_buffer_type_t buft);

        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);

        ggml_backend_buffer_t (*alloc_buffer_n)  (ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);

        size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft);

        size_t                (*get_max_size)    (ggml_backend_buffer_type_t buft);

        size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);

        size_t                (*get_alloc_size_n)(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);

        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        void * context;
    };

    GGML_API bool ggml_op_alloc_size_may_expand(enum ggml_op op);

    struct ggml_backend_buffer_i {

        void         (*free_buffer)  (ggml_backend_buffer_t buffer);

        void *       (*get_base)     (ggml_backend_buffer_t buffer);

        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

        void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);

        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);

        void         (*reset)        (ggml_backend_buffer_t buffer);
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            struct ggml_backend_buffer_i      iface,
                   void *                     context,
                   size_t                     size);

    GGML_API bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API void                  ggml_backend_meta_buffer_set_usage (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);

    struct ggml_backend_graph_optimize_params {

        void (*add_alloc_dep)(void * user_data, struct ggml_tensor * tensor, struct ggml_tensor * until);
        void * user_data;
    };

    typedef void (*ggml_backend_graph_optimize_fusion_deps_t)(struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);

    struct ggml_backend_pipeline_peer_item {
        ggml_backend_t backend_dst;
        const struct ggml_tensor * src;
        struct ggml_tensor * dst;
        ggml_backend_event_t dst_slot;
    };

    typedef bool (*ggml_backend_pipeline_peer_copy_t)(ggml_backend_t,
            const struct ggml_backend_pipeline_peer_item *, size_t, size_t, size_t, ggml_backend_event_t);

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        void (*synchronize)(ggml_backend_t backend);

        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);

        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);

        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);
    };

    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_dev_t device;
        void * context;
    };

    struct ggml_backend_event {
        struct ggml_backend_device * device;
        void * context;
    };

    struct ggml_backend_device_i {

        const char * (*get_name)(ggml_backend_dev_t dev);

        const char * (*get_description)(ggml_backend_dev_t dev);

        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);

        enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);

        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    struct ggml_backend_device {
        struct ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        void * context;
    };

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);

        size_t             (*get_device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
    };

    struct ggml_backend_reg {
        int api_version;
        struct ggml_backend_reg_i iface;
        void * context;
    };

    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            extern "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(void) {                 \
                return reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            extern "C" {                                   \
            GGML_BACKEND_API int ggml_backend_score(void); \
            }                                              \
            int ggml_backend_score(void) {                 \
                return score_fn();                         \
            }
#    else
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);  \
            ggml_backend_reg_t                  ggml_backend_init(void) { \
                return reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API int ggml_backend_score(void);  \
            int                  ggml_backend_score(void) { \
                return score_fn();                          \
            }
#    endif
#else
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
