#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstddef>
#include <vector>

// default margin per device of the parameter fit (--fit-target), in bytes
inline constexpr size_t COMMON_FIT_TARGET_DEFAULT = 1024ull*1024*1024;

// Margin per device of the expert parallel fit when the user sets none (--fit-target). The fit
// budgets the communication contexts and the page tables itself; the margin holds what it cannot
// measure before the load: the CUDA graph executables of the split graphs (0-2 MiB each, about
// 130-230 MiB per card after a 16K and a 200K request), the operator pools of the CUDA backend
// (they keep their peak), the cuBLAS handle, lazily loaded module code, and the connection
// buffers NCCL >= 2.22 allocates at the first large collective (78-116 MiB per card; NCCL 2.21
// allocates them at the init, where the fit measures them). Measured on 6 x V100 (AC922, short
// prompt + 16K + 200K requests, with and without MTP): at most 476 MiB per card, plus about
// 125 MiB of headroom (mimo-runs/margin-report.md).
inline constexpr size_t COMMON_FIT_EP_MARGIN_DEFAULT = 600ull*1024*1024;

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;

    // device feature sinks are raw backend buffers outside any memory breakdown (see
    // llama_context::enable_layer_inp_sink/link_embd and enable_h_nextn_sink): the fit
    // reserves their bytes on the devices that will hold them
    bool     dflash_sink = false; // DFlash2: reserve the layer-input sink (target + draft staging)
    uint32_t h_sink_rows = 0;     // r-mtpdev: MTP nextn sink extra columns on top of n_batch (0 = off)
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes, see margins_set
                               bool   margins_set,           // true: the margins are the user's and every fit mode leaves exactly them;
                                                             // false: they are only what other loads reserve on the device (e.g. a
                                                             // multimodal projector) and each fit mode adds its default margin
                                                             // (COMMON_FIT_TARGET_DEFAULT; the expert parallel fit budgets the
                                                             // runtime growth itself and adds a smaller one)
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

// record the free VRAM of every device of a loaded model at one startup stage. Prints one
// line per device at info level, plus the growth since the previous call at debug level.
// Used to measure what the CUDA runtime adds after the fit measurement (module code,
// cuBLAS workspaces, CUDA graphs, ...), which the expert slot budget must leave room for.
// Pass nullptr to reset the recorded reference between runs.
void common_fit_log_memory_stage(
        const struct llama_model * model,
                                  const char * stage);

// same sampling on a raw device list; used when no model is available yet (the fit itself)
void common_fit_log_memory_stage_devs(
        struct ggml_backend_device ** devs,
                                       int   nd,
                                const char * stage);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
