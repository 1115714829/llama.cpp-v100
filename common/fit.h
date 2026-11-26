#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstddef>
#include <vector>

inline constexpr size_t COMMON_FIT_TARGET_DEFAULT = 1024ull*1024*1024;

inline constexpr size_t COMMON_FIT_EP_MARGIN_DEFAULT = 800ull*1024*1024;

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0,
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1,
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2,
};

struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;

    bool     dflash_sink = false;
    uint32_t h_sink_rows = 0;

    bool     mtp_host    = false;
};

common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,
   llama_model_tensor_buft_override * tensor_buft_overrides,
                             size_t * margins,
                               bool   margins_set,

                           uint32_t   n_ctx_min,
                           int32_t    cache_ram_mib,
      const common_fit_extra_model * extra,
                     ggml_log_level   log_level);

void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_fit_log_memory_stage(
        const struct llama_model * model,
                                  const char * stage);

void common_fit_log_memory_stage_devs(
        struct ggml_backend_device ** devs,
                                       int   nd,
                                const char * stage);

void common_fit_log_host_resident_pools(const struct llama_model * model);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
