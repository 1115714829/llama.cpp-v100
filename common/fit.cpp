#include "fit.h"

#include "json.h"
#include "log.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// a device-readable host buffer (e.g. the CUDA mapped host buffer) keeps the weights in host
// memory while the device reads them in place; such buffers are not device memory
static bool common_buft_is_device_mapped(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    typedef bool (*ggml_backend_buft_is_mapped_t)(ggml_backend_buffer_type_t);
    auto is_mapped = (ggml_backend_buft_is_mapped_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_buft_is_device_mapped");
    return is_mapped != nullptr && is_mapped(buft);
}

// the device-readable host buffer type of a device, or nullptr if the device has none
static ggml_backend_buffer_type_t common_dev_mapped_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t * extra = get_extra(dev);
    while (extra != nullptr && *extra != nullptr) {
        if (common_buft_is_device_mapped(*extra)) {
            return *extra;
        }
        ++extra;
    }
    return nullptr;
}

// an expert buffer type keeps part of the experts in VRAM slots and the complement in
// device-readable host memory; its device share is part of the device memory accounting
static bool common_buft_is_device_expert(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    typedef bool (*ggml_backend_buft_is_expert_t)(ggml_backend_buffer_type_t);
    auto is_expert = (ggml_backend_buft_is_expert_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_buft_is_device_expert");
    return is_expert != nullptr && is_expert(buft);
}

// set the expert VRAM budget (slots + staging) of one buffer type before the model loads
static void common_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    typedef void (*ggml_backend_expert_set_budget_t)(ggml_backend_buffer_type_t, size_t, size_t);
    auto set_budget = (ggml_backend_expert_set_budget_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_set_budget");
    if (set_budget != nullptr) {
        set_budget(buft, slot_bytes, stage_bytes);
    }
}

// the memory that the last expert buffer allocation of this device actually used:
// slot_bytes and stage_bytes are VRAM, host_bytes is host memory
static bool common_expert_get_allocation(ggml_backend_buffer_type_t buft, size_t * slot_bytes, size_t * host_bytes, size_t * stage_bytes) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    typedef bool (*ggml_backend_expert_get_allocation_t)(ggml_backend_buffer_type_t, size_t *, size_t *, size_t *);
    auto get_alloc = (ggml_backend_expert_get_allocation_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_get_allocation");
    return get_alloc != nullptr && get_alloc(buft, slot_bytes, host_bytes, stage_bytes);
}

// one MoE expert tensor as seen by the fit code: collected from the tensor metadata of the
// loaded model, then handed to the pure plan function of the device that computes the layer
struct common_expert_tensor {
    int     layer      = -1; // block index from the tensor name
    int64_t n_expert   = 0;
    size_t  blob_bytes = 0;  // bytes of one expert
};

// parse "blk.<n>.ffn_..._exps.weight" names, the tensors that the LAYER_FRACTION_MOE
// overrides move to the expert buffer type. Returns -1 for anything else.
static int common_expert_tensor_layer(const std::string & name) {
    if (name.compare(0, 4, "blk.") != 0) {
        return -1;
    }
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4) {
        return -1;
    }
    int layer = 0;
    for (size_t i = 4; i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return -1;
        }
        layer = layer * 10 + (name[i] - '0');
    }
    if (name.find("ffn_", dot) == std::string::npos || name.find("_exps.weight") == std::string::npos) {
        return -1;
    }
    return layer;
}

// pure placement plan of the expert tensors of one device (backend proc): the same store
// function that the expert buffer allocation uses, so the fit budgets match the allocation
static bool common_expert_plan(ggml_backend_buffer_type_t buft,
        const std::vector<common_expert_tensor> & tensors, size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    typedef void (*ggml_backend_expert_plan_t)(ggml_backend_buffer_type_t,
        const int *, const int64_t *, const size_t *, size_t, size_t, size_t *, size_t *, size_t *);
    auto plan = (ggml_backend_expert_plan_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_plan");
    if (plan == nullptr) {
        return false;
    }

    std::vector<int>     layers;
    std::vector<int64_t> n_experts;
    std::vector<size_t>  blobs;
    layers.reserve(tensors.size());
    n_experts.reserve(tensors.size());
    blobs.reserve(tensors.size());
    for (const common_expert_tensor & t : tensors) {
        layers.push_back(t.layer);
        n_experts.push_back(t.n_expert);
        blobs.push_back(t.blob_bytes);
    }

    plan(buft, layers.data(), n_experts.data(), blobs.data(), tensors.size(),
        slot_budget, slot_bytes, host_bytes, host_peak);
    return true;
}

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        std::vector<common_expert_tensor> * expert_tensors = nullptr) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (common_buft_is_device_expert(buft)) {
            // expert store: the VRAM slots and the staging area are device memory, the
            // complement is host memory. The split comes from the allocation that the load
            // of the model just made, so the fit accounting sees the same numbers as the
            // kernels will use.
            size_t slot_bytes = 0;
            size_t host_bytes = 0;
            size_t stage_bytes = 0;
            const bool allocated = common_expert_get_allocation(buft, &slot_bytes, &host_bytes, &stage_bytes);
            if (allocated) {
                ret.back().mb.model += host_bytes;
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                for (size_t i = 0; i < nd; i++) {
                    if (dev == llama_model_get_device(model, i)) {
                        ret[i].mb.model += slot_bytes + stage_bytes;
                        break;
                    }
                }
            } else {
                // no expert buffer was allocated on this device yet: all experts are in host memory
                ret.back().mb.model   += mb.model;
                ret.back().mb.context += mb.context;
                ret.back().mb.compute += mb.compute;
            }
            continue;
        }
        if (ggml_backend_buft_is_host(buft) || common_buft_is_device_mapped(buft)) {
            // a device-readable host buffer is host memory, not device memory
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    if (expert_tensors != nullptr) {
        // collect the MoE expert tensor metadata of the model: fit plans the expert store of
        // each device from it, without another trial load. The names and the layout follow
        // the LAYER_FRACTION_MOE overrides, so the set matches the tensors that the real
        // load places in the expert buffer type.
        expert_tensors->clear();
        for (const auto & [name, t] : llama_internal_get_tensor_map(model)) {
            if (t == nullptr || t->view_src != nullptr || t->ne[2] <= 1) {
                continue;
            }
            if (!ggml_is_quantized(t->type) && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
                continue;
            }
            const int layer = common_expert_tensor_layer(name);
            if (layer < 0) {
                continue;
            }
            expert_tensors->push_back({ layer, t->ne[2], (size_t) t->nb[2] });
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, const common_fit_extra_model * extra, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        // in tensor mode every layer runs on every device, so there is no layer-to-device
        // mapping that the MoE overrides could follow; keep the upstream abort
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // with non-unified kv, we need to take into account n_streams
    // for example, if memory can hold more than model's trained context size, we must extend the n_ctx to hold enough n_streams
    const uint32_t n_streams  = cparams->kv_unified ? 1 : std::max<uint32_t>(1, cparams->n_seq_max);
    const bool     n_ctx_auto = cparams->n_ctx == 0;

    dmds_t   dmds_extra;       // memory of the extra model, laid out on the devices of the main model
    uint32_t n_ctx_extra = 0;  // context that memory was measured at

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes
    auto add_extra_memory = [&](dmds_t & dmds) {
        if (extra == nullptr) {
            return;
        }

        if (dmds_extra.empty() || n_ctx_extra != cparams->n_ctx) {
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;

            extra->cparams->n_ctx = cparams->n_ctx;

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n",
                __func__, cparams->n_ctx);

            dmds_t measured;
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
            } catch (const std::runtime_error & e) {
                // the extra model is optional, fit the main model alone rather than giving up
                LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n", __func__, e.what());
                dmds_extra = dmds_t(devs.size() + 1);
                n_ctx_extra = cparams->n_ctx;
                return;
            }

            dmds_extra = dmds_t(devs.size() + 1);
            dmds_extra.back().mb = measured.back().mb;
            for (size_t je = 0; je < devs_extra.size(); je++) {
                for (size_t id = 0; id < devs.size(); id++) {
                    if (devs_extra[je] == devs[id]) {
                        dmds_extra[id].mb.model   += measured[je].mb.model;
                        dmds_extra[id].mb.context += measured[je].mb.context;
                        dmds_extra[id].mb.compute += measured[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra->shares_model) {
                for (llama_device_memory_data & dmd : dmds_extra) {
                    dmd.mb.model = 0;
                }
            }

            n_ctx_extra = cparams->n_ctx;
        }

        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += dmds_extra[id].mb.compute;
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_streams, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_streams > 1) {
            LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_streams);
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        }
    }
    add_extra_memory(dmds_full);

    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min_total;
                    dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    add_extra_memory(dmds_min_ctx);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        // round down context for CUDA backend, keep it divisible by the number of streams:
                        const uint32_t align = 256 * n_streams;
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % align, n_ctx_min_total);

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (n_ctx_max - n_ctx_min_total);
                        const int64_t memory_reduction = (n_ctx_max - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, n_ctx_max);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, n_ctx_max, n_ctx_min_total);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    // only the expert weight tensor itself, not its per-expert scale tensors:
                    // the expert store places whole experts and does not take the scales
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps\\.weight";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // per-device buffer for the MoE weights that do not fit in device memory:
    // use the device-readable host buffer when the device has one, otherwise the CPU
    std::vector<ggml_backend_buffer_type_t> moe_bufts;
    moe_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        ggml_backend_buffer_type_t mapped = common_dev_mapped_buft(devs[id]);
        if (mapped != nullptr && ggml_backend_buft_get_device(mapped) != devs[id]) {
            LOG_ERR("%s: mapped buffer type %s does not belong to device %s\n",
                __func__, ggml_backend_buft_name(mapped), ggml_backend_dev_name(devs[id]));
            throw common_params_fit_exception("mapped buffer type does not belong to the device, abort");
        }
        moe_bufts.push_back(mapped != nullptr ? mapped : ggml_backend_cpu_buffer_type());
    }

    // work buffers for the trial configurations: the real tensor_split and tensor_buft_overrides
    // arrays are only written once the fit succeeded, so a failed fit cannot leave a partial
    // configuration behind for the real model load
    std::vector<float> ts_work(nd > 0 ? nd : 1, 0.0f);
    std::vector<llama_model_tensor_buft_override> tbo_work(ntbo);

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams,
            float * ts_out,
            llama_model_tensor_buft_override * tbo_out) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                ts_out[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);

        // the layer-to-device mapping must match llama_model::load_tensors: the GPU layers are
        // split by the normalized cumulative tensor_split. any MoE weights that overflow to host
        // memory must use the device-readable host buffer of the device that computes the layer
        const uint32_t n_gpu       = uint32_t(mparams.n_gpu_layers);
        const uint32_t i_gpu_start = hp_ngl + 1 > n_gpu ? hp_ngl + 1 - n_gpu : 0;
        const uint32_t act_gpu     = n_gpu < hp_ngl + 1 ? n_gpu : hp_ngl + 1;

        std::vector<float> split_points(nd);
        float split_sum = 0.0f;
        for (size_t id = 0; id < nd; id++) {
            split_sum += ts_out[id];
            split_points[id] = split_sum;
        }
        if (split_sum > 0.0f) {
            for (size_t id = 0; id < nd; id++) {
                split_points[id] /= split_sum;
            }
        }

        auto get_layer_dev = [&](uint32_t il) -> size_t {
            if (nd == 1 || act_gpu == 0) {
                return 0;
            }
            GGML_ASSERT(il >= i_gpu_start && il - i_gpu_start < act_gpu);
            return std::upper_bound(split_points.begin(), split_points.end(), float(il - i_gpu_start)/act_gpu) - split_points.begin();
        };

        uint32_t il0 = i_gpu_start; // start index for tensor buft overrides

        mparams.tensor_split = ts_out;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 2 >= ntbo) {
                    tbo_out[itbo].pattern = nullptr;
                    tbo_out[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tbo_out;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }

                const size_t id_dev = get_layer_dev(il);
                if (id_dev != id) {
                    LOG_ERR("%s: layer %" PRIu32 " is computed on device %s, but its overflowing weights would be placed on device %s\n",
                        __func__, il, ggml_backend_dev_name(devs[id]), id_dev < nd ? ggml_backend_dev_name(devs[id_dev]) : "out of range");
                    tbo_out[itbo].pattern = nullptr;
                    tbo_out[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tbo_out;
                    throw common_params_fit_exception("layer-to-device mapping does not match llama_model::load_tensors, abort");
                }

                const common_layer_fraction_t lf = il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE;

                // the MoE weights that do not fit in device memory go to the device-readable host
                // buffer of the device; this override comes first so it wins over the fraction below
                tbo_out[itbo].pattern = get_overflow_pattern(il, LAYER_FRACTION_MOE);
                tbo_out[itbo].buft    = moe_bufts[id_dev];
                itbo++;

                if (common_buft_is_device_mapped(moe_bufts[id_dev])) {
                    LOG_TRC("%s: layer %" PRIu32 " (compute device %s): MoE weights on %s (device %s)\n",
                        __func__, il, ggml_backend_dev_name(devs[id]),
                        ggml_backend_buft_name(moe_bufts[id_dev]),
                        ggml_backend_dev_name(ggml_backend_buft_get_device(moe_bufts[id_dev])));
                }

                // the other fractions of the first partial layer keep overflowing to the configured buffer
                if (lf != LAYER_FRACTION_MOE) {
                    tbo_out[itbo].pattern = get_overflow_pattern(il, lf);
                    tbo_out[itbo].buft    = overflow_bufts[id];
                    itbo++;
                }
            }
            il0 += ngl_per_device[id].n_part;
        }
        tbo_out[itbo].pattern = nullptr;
        tbo_out[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tbo_out;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    int64_t host_used = 0; // host memory of the last measurement, updated by get_memory_for_layers
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy, ts_work.data(), tbo_work.data());

        dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmd_nl);

        host_used = dmd_nl.back().mb.total();

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    // the device-readable host buffers hold the weights in host memory, which fit does not account
    // for; check the final placement and fail with a clear error instead of silently falling back
    auto check_host_memory = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) {
        // only relevant for MoE models whose expert weights actually overflow to host memory
        if (hp_nex == 0) {
            return;
        }
        bool uses_mapped = false;
        for (size_t id = 0; id < nd; id++) {
            // the MoE tensors of every partial layer overflow to the device-readable host buffer
            if (ngl_per_device[id].n_part > 0 && common_buft_is_device_mapped(moe_bufts[id])) {
                uses_mapped = true;
                break;
            }
        }
        if (!uses_mapped) {
            return;
        }
        get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

        const int64_t host_free   = dmds_full.back().free;
        const int64_t host_margin = nd < llama_max_devices() ? (int64_t) margins_s[nd] : 0;
        LOG_TRC("%s: mapped MoE weights use %" PRId64 " MiB of host memory, %" PRId64 " MiB free, margin %" PRId64 " MiB\n",
            __func__, host_used/MiB, host_free/MiB, host_margin/MiB);
        if (host_used + host_margin > host_free) {
            throw common_params_fit_exception(
                "the expert weights that do not fit in device memory need " + std::to_string((host_used + host_margin)/MiB) +
                " MiB of host memory, but only " + std::to_string(host_free/MiB) +
                " MiB is available; reduce the context size, use more devices or a lower quantization");
        }
    };

    int64_t global_surplus_cpu_moe = 0;
    dmds_t dmds_cpu_moe; // memory data with all MoE tensors in system memory
    std::vector<common_expert_tensor> moe_expert_tensors; // MoE expert tensors of the model
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tbo_work[0] = {pattern_moe_all.c_str(), cpu_buft};
        tbo_work[1] = {nullptr, nullptr};

        llama_model_params mparams_copy = *mparams;
        mparams_copy.tensor_buft_overrides = tbo_work.data();

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level, &moe_expert_tensors);
        add_extra_memory(dmds_cpu_moe);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tbo_work[0] = {nullptr, nullptr};
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // expert VRAM budget (slots + staging) of the final layer split. The placement is
    // computed with the pure backend plan function, the same one that alloc_buffer_n uses,
    // so the fit numbers match the real allocation byte for byte. Staging is sized for one
    // prefill batch, which routes to almost every expert of a layer: the largest host
    // complement of a single expert tensor of the device. More slots mean a smaller
    // complement and vice versa, so the two are solved by iterating the plan function to a
    // fixed point (a few microseconds per round). Only the partial layers of a device place
    // their MoE weights on the expert buffer; the full layers keep them in device memory.
    auto apply_expert_budgets = [&]() {
        if (hp_nex == 0) {
            return;
        }

        // layer -> device of the MoE overflow, the same mapping that set_ngl_tensor_split_tbo
        // wrote into the tensor buft overrides
        std::vector<int> partial_dev((size_t) hp_ngl + 2, -1);
        {
            const uint32_t n_gpu       = (uint32_t) mparams->n_gpu_layers;
            const uint32_t i_gpu_start = hp_ngl + 1 > n_gpu ? hp_ngl + 1 - n_gpu : 0;
            const uint32_t act_gpu     = n_gpu < hp_ngl + 1 ? n_gpu : hp_ngl + 1;

            std::vector<float> split_points(nd);
            float split_sum = 0.0f;
            for (size_t id = 0; id < nd; id++) {
                split_sum += tensor_split[id];
                split_points[id] = split_sum;
            }
            if (split_sum > 0.0f) {
                for (size_t id = 0; id < nd; id++) {
                    split_points[id] /= split_sum;
                }
            }
            auto get_layer_dev = [&](uint32_t il) -> size_t {
                if (nd == 1 || act_gpu == 0) {
                    return 0;
                }
                if (il < i_gpu_start || il - i_gpu_start >= act_gpu) {
                    return (size_t) -1;
                }
                return std::upper_bound(split_points.begin(), split_points.end(), float(il - i_gpu_start)/act_gpu) - split_points.begin();
            };

            uint32_t il0 = i_gpu_start;
            for (size_t id = 0; id < nd; id++) {
                il0 += ngl_per_device[id].n_full();
                for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                    if (get_layer_dev(il) == id && (size_t) il < partial_dev.size()) {
                        partial_dev[il] = (int) id;
                    }
                }
                il0 += ngl_per_device[id].n_part;
            }
        }

        std::vector<std::vector<common_expert_tensor>> dev_tensors(nd);
        for (const common_expert_tensor & t : moe_expert_tensors) {
            if (t.layer >= 0 && (size_t) t.layer < partial_dev.size() && partial_dev[t.layer] >= 0) {
                dev_tensors[(size_t) partial_dev[t.layer]].push_back(t);
            }
        }

        for (size_t id = 0; id < nd; id++) {
            if (!common_buft_is_device_expert(moe_bufts[id])) {
                continue;
            }
            const int64_t surplus = dmds_cpu_moe[id].free - int64_t(dmds_cpu_moe[id].mb.total()) - margins[id];
            if (surplus <= 0) {
                common_expert_set_budget(moe_bufts[id], 0, 0);
                LOG_TRC("%s: device %zu: no expert VRAM surplus, all experts stay in host memory\n", __func__, id);
                continue;
            }
            if (dev_tensors[id].empty()) {
                // no partial layer is computed on this device, no tensor uses its expert store
                common_expert_set_budget(moe_bufts[id], 0, 0);
                continue;
            }

            size_t stage_bytes = 0; // staging area: one prefill batch
            size_t stage_peak  = 0;
            size_t slot_budget = (size_t) surplus;
            size_t slot_used   = 0;
            size_t host_bytes  = 0;
            for (int round = 0; round < 8; ++round) {
                if (!common_expert_plan(moe_bufts[id], dev_tensors[id], slot_budget, &slot_used, &host_bytes, &stage_peak)) {
                    return; // no plan function: keep the (0, 0) budget
                }
                const size_t next = std::min(stage_peak, (size_t) surplus);
                if (next == stage_bytes) {
                    break;
                }
                stage_bytes = next;
                slot_budget = (size_t) surplus - stage_bytes;
            }
            if (!common_expert_plan(moe_bufts[id], dev_tensors[id], slot_budget, &slot_used, &host_bytes, &stage_peak)) {
                return;
            }
            common_expert_set_budget(moe_bufts[id], slot_budget, stage_bytes);

            LOG_INF("%s: device %zu: expert store: slots %.2f MiB (plan %.2f MiB), host complement %.2f MiB, staging %.2f MiB, prefill direct-read %.0f%%\n",
                __func__, id, slot_budget/1024.0/1024.0, slot_used/1024.0/1024.0, host_bytes/1024.0/1024.0,
                stage_bytes/1024.0/1024.0,
                stage_peak > stage_bytes ? 100.0 * (double) (stage_peak - stage_bytes) / (double) stage_peak : 0.0);
            if (stage_peak > stage_bytes) {
                LOG_WRN("%s: device %zu: one prefill batch needs %.2f MiB of staging for the host experts, only %.2f MiB fit: about %.0f%% of the routed host experts are read directly from host memory\n",
                    __func__, id, stage_peak/1024.0/1024.0, stage_bytes/1024.0/1024.0,
                    100.0 * (double) (stage_peak - stage_bytes) / (double) stage_peak);
            }
        }
    };

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        check_host_memory(ngl_per_device, overflow_bufts);
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams, tensor_split, tensor_buft_overrides);
        apply_expert_budgets();
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    check_host_memory(ngl_per_device, overflow_bufts);
    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams, tensor_split, tensor_buft_overrides);
    apply_expert_budgets();
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        const common_fit_extra_model * extra,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, extra, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());

    // same data as the table below, for --log-jsonl consumers
    common_json rows = common_json::array();
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (common_buft_is_device_expert(buft)) {
            // expert store: VRAM slots and staging are device memory, the complement is host
            size_t slot_bytes = 0;
            size_t host_bytes = 0;
            size_t stage_bytes = 0;
            if (common_expert_get_allocation(buft, &slot_bytes, &host_bytes, &stage_bytes)) {
                mb_host.model += host_bytes;
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                for (size_t i = 0; i < devices.size(); i++) {
                    if (devices[i] == dev) {
                        mb_dev[i].model += slot_bytes + stage_bytes;
                        break;
                    }
                }
            } else {
                mb_host.model   += mb.model;
                mb_host.context += mb.context;
                mb_host.compute += mb.compute;
            }
            seen_buffer_types.insert(buft);
            continue;
        }
        if (ggml_backend_buft_is_host(buft) || common_buft_is_device_mapped(buft)) {
            // a device-readable host buffer is host memory, not device memory
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});

        rows.push_back({
            {"kind",        "device"},
            {"name",        name},
            {"description", desc},
            {"total",       total / MiB},
            {"free",        free / MiB},
            {"self",        self / MiB},
            {"model",       mb.model / MiB},
            {"context",     mb.context / MiB},
            {"compute",     mb.compute / MiB},
            {"unaccounted", unaccounted / static_cast<int64_t>(MiB)},
        });
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "host"},
            {"name",    "Host"},
            {"self",    self / MiB},
            {"model",   mb_host.model / MiB},
            {"context", mb_host.context / MiB},
            {"compute", mb_host.compute / MiB},
        });
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "buffer_type"},
            {"name",    name},
            {"self",    self / MiB},
            {"model",   mb.model / MiB},
            {"context", mb.context / MiB},
            {"compute", mb.compute / MiB},
        });

        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }

    LOG_JSON("fit_memory_breakdown", common_json({
        {"unit", "MiB"},
        {"rows", rows},
    }));
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
