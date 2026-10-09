#include "fit.h"

#include "json.h"
#include "log.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
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

// the expert buffer type of one device: the same extra buffer type the meta backend uses for
// the expert tensors of a device in tensor-parallel mode. Returns nullptr when the device has
// no expert buffer type (non-CUDA backends).
static ggml_backend_buffer_type_t common_dev_expert_buft(ggml_backend_dev_t dev) {
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
        if (common_buft_is_device_expert(*extra)) {
            return *extra;
        }
        ++extra;
    }
    return nullptr;
}

// drop the recorded expert allocation of the backend (a model loaded earlier in this process
// would otherwise leak its slot/host bytes into the no-alloc trials of this fit)
static void common_expert_reset_allocation(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    typedef void (*ggml_backend_expert_reset_allocation_t)(ggml_backend_buffer_type_t);
    auto reset = (ggml_backend_expert_reset_allocation_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_reset_allocation");
    if (reset != nullptr) {
        reset(buft);
    }
}

// build the CUDA context of the device behind this buffer type (through the backend proc),
// so the driver bookkeeping the real load pays for is already present when fit samples the
// free memory. This makes the sampled free memory the same state the expert slots are
// allocated in; without it the first sample of an untouched device is too large.
static void common_expert_device_ready(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    typedef void (*ggml_backend_expert_device_ready_t)(ggml_backend_buffer_type_t);
    auto ready = (ggml_backend_expert_device_ready_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_device_ready");
    if (ready != nullptr) {
        ready(buft);
    }
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
    int        layer      = -1; // block index from the tensor name
    int64_t    expert_base = 0; // global index of local expert 0 (expert parallelism)
    int64_t    n_expert   = 0;
    size_t     blob_bytes = 0;  // data bytes of one expert
    ggml_type  type       = GGML_TYPE_COUNT; // for the per-expert row padding of the backend
    int64_t    ne0        = 0;  // contraction size, same padding rule as the backend
};

// layer index of a tensor name "blk.<n>...", -1 for anything else. Used on tensors that
// already carry the expert mark, so the name is metadata and not recognition. The exact
// parse mirrors ggml_cuda_expert_tensor_layer(): the fit plan and the real allocation must
// see the same layer number for every tensor, otherwise the two plans differ.
static int common_expert_tensor_layer(const std::string & name) {
    const size_t pos = name.find("blk.");
    if (pos == std::string::npos) {
        return -1;
    }
    size_t p = pos + 4;
    if (p >= name.size() || name[p] < '0' || name[p] > '9') {
        return -1;
    }
    int layer = 0;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        layer = layer * 10 + (name[p] - '0');
        p++;
    }
    return layer;
}

// pure placement plan of the expert tensors of one device (backend proc): the same store
// function that the expert buffer allocation uses, so the fit budgets match the allocation.
// host_each, when not null, receives the host complement bytes of every tensor in allocation
// order (used by the NUMA check of the host complement).
static bool common_expert_plan(ggml_backend_buffer_type_t buft,
        const std::vector<common_expert_tensor> & tensors, size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak,
        std::vector<size_t> * host_each = nullptr) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    typedef void (*ggml_backend_expert_plan_t)(ggml_backend_buffer_type_t,
        const int *, const int64_t *, const int64_t *, const size_t *, const int *, const int64_t *, size_t,
        size_t, size_t *, size_t *, size_t *, size_t *);
    auto plan = (ggml_backend_expert_plan_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_plan");
    if (plan == nullptr) {
        return false;
    }

    std::vector<int>     layers;
    std::vector<int64_t> expert_bases;
    std::vector<int64_t> n_experts;
    std::vector<size_t>  blobs;
    std::vector<int>     types;
    std::vector<int64_t> ne0s;
    layers.reserve(tensors.size());
    expert_bases.reserve(tensors.size());
    n_experts.reserve(tensors.size());
    blobs.reserve(tensors.size());
    types.reserve(tensors.size());
    ne0s.reserve(tensors.size());
    for (const common_expert_tensor & t : tensors) {
        layers.push_back(t.layer);
        expert_bases.push_back(t.expert_base);
        n_experts.push_back(t.n_expert);
        blobs.push_back(t.blob_bytes);
        types.push_back((int) t.type);
        ne0s.push_back(t.ne0);
    }

    if (host_each != nullptr) {
        host_each->assign(tensors.size(), 0);
    }
    plan(buft, layers.data(), expert_bases.data(), n_experts.data(), blobs.data(), types.data(), ne0s.data(), tensors.size(),
        slot_budget, slot_bytes, host_bytes, host_peak, host_each != nullptr ? host_each->data() : nullptr);
    return true;
}

// record the placement plan fit computed for one device (slot and host bytes). The real
// allocation replans from the same tensors and the same budget and must come out at exactly
// these bytes; it aborts when it does not.
static void common_expert_record_plan(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t host_bytes) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    typedef void (*ggml_backend_expert_record_plan_t)(ggml_backend_buffer_type_t, size_t, size_t);
    auto record = (ggml_backend_expert_record_plan_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_record_plan");
    if (record != nullptr) {
        record(buft, slot_bytes, host_bytes);
    }
}

// NUMA check of the expert host arenas: hand the host complement allocations of all devices
// to the backend in the order the real load performs them (the model tensor order), the
// backend simulates the node selection of the real allocations against the per-node
// available memory (MemFree plus reclaimable file pages minus the margin, the same figures
// the real selection uses). On success the bytes placed on every node are written to
// node_bytes/node_ids (up to the capacity in *n_nodes, which receives the actual number of
// nodes). Returns 1 when the placement fits, -1 when some node cannot hold its planned
// bytes (the failing node, allocation size and available memory are returned), and 0 when
// there is no NUMA information at all; the caller then falls back to a plain total check.
static int common_expert_check_host_nodes(
        const std::vector<ggml_backend_dev_t> & devs,
        const std::vector<int> & seq_dev,
        const std::vector<size_t> & seq_size,
        int64_t * node_bytes, int * node_ids, size_t * n_nodes,
        int * fail_node, size_t * fail_bytes, int64_t * fail_avail) {
    if (seq_dev.empty()) {
        if (n_nodes != nullptr) {
            *n_nodes = 0;
        }
        return 1; // no expert host arena at all
    }
    if (seq_dev[0] < 0 || (size_t) seq_dev[0] >= devs.size() || devs[(size_t) seq_dev[0]] == nullptr) {
        return 0;
    }
    ggml_backend_dev_t dev = devs[(size_t) seq_dev[0]];
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return 0;
    }
    typedef int (*ggml_backend_expert_check_host_nodes_t)(
        const int *, const size_t *, size_t, int64_t *, int *, size_t *, int *, size_t *, int64_t *);
    auto check = (ggml_backend_expert_check_host_nodes_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_check_host_nodes");
    if (check == nullptr) {
        return 0;
    }
    return check(seq_dev.data(), seq_size.data(), seq_dev.size(),
        node_bytes, node_ids, n_nodes, fail_node, fail_bytes, fail_avail);
}

// collect the per-device memory breakdown of one loaded model: one entry per device of
// the model plus the host entry at the back
static void collect_device_memory_breakdown(
        const llama_model * model, const llama_context * ctx, std::vector<llama_device_memory_data> & ret) {
    const size_t nd = llama_model_n_devices(model);
    GGML_ASSERT(ret.size() == nd + 1);

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
        std::vector<common_expert_tensor> * expert_tensors = nullptr,
        const common_fit_extra_model * extra = nullptr,
        std::vector<llama_device_memory_data> * extra_mem = nullptr,
        std::vector<ggml_backend_dev_t> * extra_devs = nullptr) {
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

    collect_device_memory_breakdown(model, ctx, ret);

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
        // each device from it, without another trial load. Recognition is the loader mark
        // (GGML_TENSOR_FLAG_EXPERT), the same one the expert store uses, so the set matches
        // the tensors that the real load places in the expert buffer type. The layer index is
        // read from the name only as metadata for the per-tensor plan.
        expert_tensors->clear();
        for (const auto & [name, t] : llama_internal_get_tensor_map(model)) {
            if (t == nullptr || (t->flags & GGML_TENSOR_FLAG_EXPERT) == 0) {
                continue;
            }
            if (t->view_src != nullptr || t->ne[2] <= 1) {
                continue;
            }
            if (!ggml_is_quantized(t->type) && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
                continue;
            }
            const int layer = common_expert_tensor_layer(name);
            if (layer < 0) {
                continue;
            }
            expert_tensors->push_back({ layer, 0, t->ne[2], (size_t) t->nb[2], t->type, t->ne[0] });
        }
    }

    // measure the extra model while the target model is still alive: a draft head borrows
    // its shared tensors from the target (borrow_shared_tensor), so the measurement graph
    // matches the real load - real tensor types, real devices, and the borrowed tensors
    // add no bytes to the model breakdown of the extra model (they are counted in the
    // target measurement above). The draft context also links itself to the target while
    // it is constructed (shared embeddings/output, device sinks, the MTP h sink), so the
    // measured target context is handed to the draft as ctx_other below - same as the real
    // load in common_speculative_init_result - and outlives the extra measurement.
    if (extra != nullptr && extra_mem != nullptr) {
        llama_model_params mparams_extra = *extra->mparams;
        mparams_extra.no_alloc      = true;
        mparams_extra.load_mode     = LLAMA_LOAD_MODE_NONE;
        mparams_extra.model_shared  = model;

        llama_model * model_extra = llama_model_load_from_file(extra->path_model, mparams_extra);
        if (model_extra == nullptr) {
            llama_free(ctx);
            llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
            throw std::runtime_error(std::string("failed to load the extra model '") + extra->path_model + "'");
        }

        // the draft context is created against the measured target context, exactly like the
        // real draft context: every draft links itself to the target during construction
        // (shared embeddings/output, device sinks), so ctx_other is handed over here
        extra->cparams->ctx_other = ctx;

        // the MTP draft additionally checks the h sink layout at construction; the sink bytes
        // are counted in below, so the linked sink only has to exist while the measurement
        // graph is built
        if (extra->h_sink_rows > 0) {
            extra->cparams->n_h_sink_rows = extra->h_sink_rows;
        }

        llama_context * ctx_extra = llama_init_from_model(model_extra, *extra->cparams);
        if (ctx_extra == nullptr) {
            llama_model_free(model_extra);
            llama_free(ctx);
            llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
            throw std::runtime_error(std::string("failed to create llama_context from the extra model '") + extra->path_model + "'");
        }

        const size_t nd_extra = llama_model_n_devices(model_extra);
        std::vector<llama_device_memory_data> ret_extra(nd_extra + 1);

        collect_device_memory_breakdown(model_extra, ctx_extra, ret_extra);

        // the device feature sinks are raw backend buffers, so no memory breakdown covers
        // them. Reserve their bytes on the devices the real load allocates them on
        {
            const size_t sink_row = (size_t) llama_model_n_embd(model) * sizeof(float);

            if (extra->dflash_sink) {
                const uint32_t n_lids = llama_model_target_layer_ids_n(model_extra);
                if (n_lids > 0) {
                    // DFlash2 layer-input sink [n_embd * n_lids, n_batch] on the first target
                    // device and the draft staging sink [n_embd_inp_enc, n_ubatch] on the first
                    // draft device: both are allocated from the main device buffer type of
                    // backends[0] (llama_context::enable_layer_inp_sink / link_embd)
                    ret      [0].mb.context += sink_row * n_lids * cparams->n_batch; // the host entry when there is no device
                    ret_extra[0].mb.context += sink_row * n_lids * extra->cparams->n_ubatch;
                }
            }

            if (extra->h_sink_rows > 0) {
                // MTP nextn sink [n_embd_out, n_batch + extra rows] where the extra rows hold a
                // second batch-column group (the prefill ring) and the per-sequence bookkeeping
                // columns: allocated once by the target on its output device
                // (llama_context::enable_h_nextn_sink); the draft shares those rows via
                // link_h_nextn and allocates nothing. With a layer split the output device is
                // the last target device
                const size_t row = (size_t) llama_model_n_embd_out(model) * sizeof(float);
                ret[nd > 0 ? nd - 1 : 0].mb.context +=
                    row * ((size_t) cparams->n_batch + extra->h_sink_rows);
            }
        }

        llama_free(ctx_extra);
        if (extra_devs != nullptr) {
            extra_devs->clear();
            for (size_t i = 0; i < nd_extra; i++) {
                extra_devs->push_back(llama_model_get_device(model_extra, i));
            }
        }
        llama_model_free(model_extra);

        *extra_mem = std::move(ret_extra);
    }

    if (extra != nullptr) {
        // the measured target context dies here, do not leave a dangling ctx_other behind
        extra->cparams->ctx_other = nullptr;
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

// free VRAM of every device of a loaded model, sampled at the startup stages. The CUDA
// runtime adds device memory lazily (module code and constants, cuBLAS workspaces, CUDA
// graphs, communication buffers), so the free memory of a device shrinks after the fit
// measured it. The expert slot budget must leave room for that growth; these samples are
// what it is measured from. One line per device shows the stage free memory; the growth per
// stage and the total are printed alongside, all at trace level.
void common_fit_log_memory_stage(const struct llama_model * model, const char * stage) {
    if (model == nullptr || stage == nullptr) {
        common_fit_log_memory_stage_devs(nullptr, 0, nullptr);
        return;
    }

    const int nd = llama_model_n_devices(model);
    if (nd <= 0) {
        return;
    }

    std::vector<ggml_backend_dev_t> devs((size_t) nd);
    for (int i = 0; i < nd; i++) {
        devs[(size_t) i] = llama_model_get_device(model, i);
    }
    common_fit_log_memory_stage_devs(devs.data(), nd, stage);
}

void common_fit_log_memory_stage_devs(struct ggml_backend_device ** devs, int nd, const char * stage) {
    static std::vector<size_t> prev_free;
    static int64_t             sum_growth = 0;

    if (devs == nullptr || stage == nullptr) {
        prev_free.clear();
        sum_growth = 0;
        return;
    }
    if (nd <= 0) {
        return;
    }

    std::vector<size_t> cur_free((size_t) nd, 0);
    for (int i = 0; i < nd; i++) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(devs[i], &free, &total);
        cur_free[(size_t) i] = free;
    }

    int64_t stage_growth = 0;
    for (int i = 0; i < nd; i++) {
        const int64_t growth = prev_free.size() == (size_t) nd ? (int64_t) prev_free[(size_t) i] - (int64_t) cur_free[(size_t) i] : 0;
        stage_growth += growth;
        LOG_TRC("%s: stage '%s' device %d (%s): free %.0f MiB, %+.0f MiB since the previous stage\n",
            __func__, stage, i, ggml_backend_dev_name(devs[i]),
            (double) cur_free[(size_t) i] / 1024.0 / 1024.0, (double) growth / 1024.0 / 1024.0);
    }
    sum_growth += stage_growth;
    LOG_TRC("%s: stage '%s': device free memory changed by %+.0f MiB (%.0f MiB since the reference sample)\n",
        __func__, stage, (double) stage_growth / 1024.0 / 1024.0, (double) sum_growth / 1024.0 / 1024.0);

    prev_free = std::move(cur_free);
}

// Tensor-parallel split with routed experts (expert parallelism): one meta device wraps all
// simple devices, every layer runs on every device and each device holds a contiguous slice of
// the expert dim. There is no layer-to-device map, so the fit budgets the expert store of each
// device directly: it measures the model no-alloc (the TP slice each device holds), reads the
// free memory of every simple device, and gives what remains after the dense weights, the KV
// cache, the compute buffers and the margin to the expert slots of that device.
static void common_fit_ep(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        size_t * margins_s, enum ggml_log_level log_level) {
    constexpr int64_t MiB = 1024*1024;

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        throw common_params_fit_exception("failed to load model");
    }
    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        throw common_params_fit_exception("failed to create llama_context from model");
    }

    // the model device is the meta device; its simple devices are the cards
    std::vector<ggml_backend_dev_t> simple_devs;
    if (llama_model_n_devices(model) == 1 && model->devices[0].is_meta) {
        ggml_backend_dev_t meta_dev = llama_model_get_device(model, 0);
        const size_t ns = ggml_backend_meta_dev_n_devs(meta_dev);
        for (size_t j = 0; j < ns; j++) {
            simple_devs.push_back(ggml_backend_meta_dev_simple_dev(meta_dev, j));
        }
    }
    const size_t nd = simple_devs.size();
    if (nd < 1) {
        llama_free(ctx);
        llama_model_free(model);
        throw common_params_fit_exception("expert parallel fit needs a meta device with at least one simple device, abort");
    }

    // every simple device must offer an expert buffer type: the store is the only expert
    // placement in tensor mode
    std::vector<ggml_backend_buffer_type_t> expert_bufts(nd, nullptr);
    for (size_t j = 0; j < nd; j++) {
        expert_bufts[j] = common_dev_expert_buft(simple_devs[j]);
        if (expert_bufts[j] == nullptr) {
            llama_free(ctx);
            llama_model_free(model);
            throw common_params_fit_exception("device " + std::to_string(j) +
                " has no expert buffer type, expert parallelism needs the expert store on every device, abort");
        }
    }

    // per-simple-device dense model bytes: the TP slice each device holds. Sliced tensors
    // contribute their per-device slice, replicated ones the full tensor. Routed experts are
    // skipped: they are placed by the expert store, not counted as dense weights. ggml_nbytes
    // is exact for the contiguous weight layout, so no per-tensor stride math is needed.
    std::vector<int64_t> dev_model(nd, 0);
    for (const auto & kv : llama_internal_get_tensor_map(model)) {
        const ggml_tensor * t = kv.second;
        if (t == nullptr || t->view_src != nullptr || ggml_nelements(t) == 0) {
            continue;
        }
        if ((t->flags & GGML_TENSOR_FLAG_EXPERT) != 0) {
            continue;
        }
        const ggml_backend_meta_split_state ss = llama_meta_device_get_split_state(
            t, const_cast<llama_meta_device_get_split_state_userdata *>(&model->get_split_state_ud));
        if (!(ss.axis >= 0 && ss.axis < GGML_MAX_DIMS)) {
            // MIRRORED / PARTIAL / LOCAL_PARTIAL: every device holds the whole tensor
            for (size_t j = 0; j < nd; j++) {
                dev_model[j] += (int64_t) ggml_nbytes(t);
            }
            continue;
        }
        const int64_t ne_axis = t->ne[ss.axis];
        if (ne_axis <= 0) {
            continue;
        }
        for (size_t j = 0; j < nd; j++) {
            int64_t ne_j = 0;
            for (size_t s = 0; s < ss.n_segments; s++) {
                ne_j += ss.ne[s*nd + j] * ss.nr[s];
            }
            dev_model[j] += (int64_t) ggml_nbytes(t) * ne_j / ne_axis;
        }
    }

    // expert tensors of the model (loader mark), with the per-device expert counts and bases
    // of the same split the meta backend applies (llama_meta_device_get_split_state).
    struct ep_tensor { int layer; std::vector<int64_t> expert_base; std::vector<int64_t> n_expert; size_t blob_bytes; ggml_type type; int64_t ne0; };
    std::vector<ep_tensor> expert_tensors;
    for (const auto & kv : llama_internal_get_tensor_map(model)) {
        const ggml_tensor * t = kv.second;
        if (t == nullptr || (t->flags & GGML_TENSOR_FLAG_EXPERT) == 0) {
            continue;
        }
        if (t->view_src != nullptr || t->ne[2] <= 1) {
            continue;
        }
        if (!ggml_is_quantized(t->type) && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
            continue;
        }
        const int layer = common_expert_tensor_layer(kv.first);
        if (layer < 0) {
            continue;
        }
        const ggml_backend_meta_split_state ss = llama_meta_device_get_split_state(
            t, const_cast<llama_meta_device_get_split_state_userdata *>(&model->get_split_state_ud));
        if (ss.axis != GGML_BACKEND_SPLIT_AXIS_2) {
            throw common_params_fit_exception("tensor split with experts: expert tensor '" + kv.first +
                "' is not split along the expert dim, abort");
        }
        std::vector<int64_t> n_expert(nd, 0);
        std::vector<int64_t> expert_base(nd, 0);
        int64_t prefix = 0;
        for (size_t j = 0; j < nd; j++) {
            int64_t ne_j = 0;
            for (size_t s = 0; s < ss.n_segments; s++) {
                ne_j += ss.ne[s*nd + j] * ss.nr[s];
            }
            n_expert[j]   = ne_j;
            expert_base[j] = prefix;
            prefix += ne_j;
        }
        expert_tensors.push_back({ layer, std::move(expert_base), std::move(n_expert), (size_t) t->nb[2], t->type, t->ne[0] });
    }

    // context (KV cache) and compute bytes: the meta device reports one number per buffer
    // type for the whole model
    int64_t context_total = 0;
    int64_t compute_total = 0;
    {
        const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
        for (const auto & [buft, data] : mb) {
            context_total += (int64_t) data.context;
            compute_total += (int64_t) data.compute;
        }
    }

    llama_free(ctx);
    llama_model_free(model);

    if (expert_tensors.empty()) {
        throw common_params_fit_exception("tensor split with experts: the model carries no expert tensor, abort");
    }

    // drop the recorded allocation of an earlier model in this process and build the CUDA
    // context of every device before sampling its free memory: a device without its context
    // reports more free memory than the real load has, which would over-commit the slots
    for (size_t j = 0; j < nd; j++) {
        common_expert_reset_allocation(expert_bufts[j]);
        common_expert_device_ready(expert_bufts[j]);
    }

    std::vector<int64_t> free_mem(nd, 0);
    for (size_t j = 0; j < nd; j++) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(simple_devs[j], &free, &total);
        free_mem[j] = (int64_t) free;
    }

    // per-device expert tensors: the contiguous prefix split of every expert tensor
    std::vector<std::vector<common_expert_tensor>> dev_tensors(nd);
    int64_t host_complement = 0;
    std::vector<std::vector<size_t>> host_each(nd);
    std::vector<size_t> slot_budgets(nd, 0);
    bool has_experts = false;

    // The meta no-alloc measurement reports one context (KV) and one compute figure per
    // buffer type. get_alloc_size is the maximum over the simple devices of the full tensor,
    // so the context sum is the whole-cache size as if one device held it; the per-device
    // slice is that divided by the device count. The compute figure is the scheduler buffer
    // size of the single meta backend, which is the per-rank compute buffer already. Both are
    // estimates; the exact per-device KV follows the attention-head split and is validated by
    // the real load.
    LOG_TRC("%s: expert parallel: context %.0f MiB total, compute %.0f MiB per rank, %zu devices\n",
        __func__, context_total/(1024.0*1024.0), compute_total/(1024.0*1024.0), nd);

    for (size_t j = 0; j < nd; j++) {
        const int64_t context_j = context_total / (int64_t) nd;
        const int64_t compute_j = compute_total;
        const int64_t margin_j  = margins_s[j];
        const int64_t surplus   = free_mem[j] - dev_model[j] - context_j - compute_j - margin_j;

        for (const ep_tensor & t : expert_tensors) {
            dev_tensors[j].push_back({ t.layer, t.expert_base[j], t.n_expert[j], t.blob_bytes, t.type, t.ne0 });
        }

        if (surplus <= 0) {
            // no VRAM for slots: the prefill staging of the host complement could not exist
            size_t slot_used_none = 0;
            size_t host_bytes_none = 0;
            size_t host_peak_none = 0;
            if (!common_expert_plan(expert_bufts[j], dev_tensors[j], 0, &slot_used_none, &host_bytes_none,
                    &host_peak_none, &host_each[j])) {
                throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(j) +
                    " has no placement plan, abort");
            }
            throw common_params_fit_exception(
                "expert store: device " + std::to_string(j) + " has no VRAM surplus for expert slots,"
                " so the " + std::to_string((host_peak_none + MiB - 1)/MiB) +
                " MiB staging of one prefill batch cannot be allocated; reduce the context size,"
                " use more devices or a lower quantization");
        }

        size_t stage_bytes = 0;
        size_t stage_peak  = 0;
        size_t slot_budget = (size_t) surplus;
        size_t slot_used   = 0;
        size_t host_bytes  = 0;
        for (int round = 0; round < 64; ++round) {
            if (!common_expert_plan(expert_bufts[j], dev_tensors[j], slot_budget, &slot_used, &host_bytes, &stage_peak)) {
                throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(j) +
                    " has no placement plan, abort");
            }
            const size_t next = std::min(stage_peak, (size_t) surplus);
            if (next == stage_bytes) {
                break;
            }
            stage_bytes = next;
            slot_budget = (size_t) surplus - stage_bytes;
        }
        if (!common_expert_plan(expert_bufts[j], dev_tensors[j], slot_budget, &slot_used, &host_bytes, &stage_peak, &host_each[j])) {
            throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(j) +
                " has no placement plan, abort");
        }
        if (stage_peak > stage_bytes) {
            throw common_params_fit_exception(
                "expert store: device " + std::to_string(j) + " needs " +
                std::to_string((stage_peak + MiB - 1)/MiB) + " MiB of staging for the host experts of one prefill batch, but only " +
                std::to_string(stage_bytes/MiB) + " MiB fit (short by " +
                std::to_string((stage_peak - stage_bytes + MiB - 1)/MiB) +
                " MiB); reduce the context size, use more devices or a lower quantization");
        }
        common_expert_set_budget(expert_bufts[j], slot_budget, stage_bytes);
        common_expert_record_plan(expert_bufts[j], slot_used, host_bytes);

        const double hit = slot_used + host_bytes > 0
            ? 100.0 * (double) slot_used / (double) (slot_used + host_bytes) : 0.0;
        LOG_INF("%s: device %zu: dense %.0f MiB, context %.0f MiB, compute %.0f MiB, expert slots %.0f MiB (budget %.0f MiB), staging %.0f MiB, host complement %.0f MiB, decode hit %.0f%% (uniform)\n",
            __func__, j,
            (double) dev_model[j]/MiB, (double) context_j/MiB, (double) compute_j/MiB,
            slot_used/1024.0/1024.0, slot_budget/1024.0/1024.0, stage_bytes/1024.0/1024.0,
            host_bytes/1024.0/1024.0, hit);
        slot_budgets[j] = slot_budget;
        host_complement += (int64_t) host_bytes;
        has_experts = true;
    }

    if (has_experts) {
        size_t min_slot = SIZE_MAX;
        size_t max_slot = 0;
        size_t min_id   = 0;
        size_t max_id   = 0;
        for (size_t j = 0; j < nd; j++) {
            if (slot_budgets[j] < min_slot) {
                min_slot = slot_budgets[j];
                min_id   = j;
            }
            if (slot_budgets[j] > max_slot) {
                max_slot = slot_budgets[j];
                max_id   = j;
            }
        }
        LOG_INF("%s: expert slot budgets: min %.0f MiB (device %zu), max %.0f MiB (device %zu), difference %.0f MiB\n",
            __func__, min_slot/1024.0/1024.0, min_id, max_slot/1024.0/1024.0, max_id,
            (max_slot - min_slot)/1024.0/1024.0);
    }

    // host complement NUMA check, same backend proc the layer-split path uses
    {
        std::vector<int>    seq_dev;
        std::vector<size_t> seq_size;
        for (size_t j = 0; j < nd; j++) {
            for (const size_t bytes : host_each[j]) {
                seq_dev.push_back((int) j);
                seq_size.push_back(bytes);
            }
        }
        constexpr size_t max_nodes = 64;
        int64_t node_bytes[max_nodes] = {};
        int     node_ids  [max_nodes] = {};
        size_t  n_nodes   = max_nodes;
        int     fail_node  = -1;
        size_t  fail_bytes = 0;
        int64_t fail_avail = 0;
        const int check = common_expert_check_host_nodes(simple_devs, seq_dev, seq_size,
            node_bytes, node_ids, &n_nodes, &fail_node, &fail_bytes, &fail_avail);
        if (check < 0) {
            const int64_t deficit = (int64_t) fail_bytes - fail_avail;
            throw std::runtime_error(
                "expert store host complement: a slice of " + std::to_string(fail_bytes/MiB) + " MiB does not fit on NUMA node " +
                std::to_string(fail_node) + ", which has only " + std::to_string(fail_avail/MiB) +
                " MiB available (short by " + std::to_string((deficit > 0 ? deficit : 0)/MiB) +
                " MiB); reduce the context size, use more devices or a lower quantization");
        }
        if (check > 0) {
            std::string per_node;
            int64_t total = 0;
            for (size_t n = 0; n < n_nodes; n++) {
                if (node_bytes[n] == 0) {
                    continue;
                }
                total += node_bytes[n];
                per_node += " node " + std::to_string(node_ids[n]) + ": " + std::to_string(node_bytes[n]/MiB) + " MiB,";
            }
            if (per_node.empty()) {
                per_node = " none";
            }
            LOG_INF("%s: expert store host complement %" PRId64 " MiB fits, per NUMA node:%s\n",
                __func__, total/MiB, per_node.c_str());
        }
    }
    GGML_UNUSED(host_complement);
    GGML_UNUSED(log_level);
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, const common_fit_extra_model * extra, enum ggml_log_level log_level) {
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

    // the extra model competes for the same memory as the main model, add it to every
    // measurement. The fresh measurement comes from the same common_get_device_memory_data_impl
    // call that measured the main model (with the main model still alive, so the extra model
    // can borrow its shared tensors); it is empty when the cached values are still valid.
    auto add_extra_memory = [&](dmds_t & dmds, const dmds_t & measured_extra, const std::vector<ggml_backend_dev_t> & devs_extra) {
        if (extra == nullptr) {
            return;
        }

        if (!measured_extra.empty()) {
            dmds_extra = dmds_t(devs.size() + 1);
            dmds_extra.back().mb = measured_extra.back().mb;
            for (size_t je = 0; je < devs_extra.size(); je++) {
                for (size_t id = 0; id < devs.size(); id++) {
                    if (devs_extra[je] == devs[id]) {
                        dmds_extra[id].mb.model   += measured_extra[je].mb.model;
                        dmds_extra[id].mb.context += measured_extra[je].mb.context;
                        dmds_extra[id].mb.compute += measured_extra[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra->shares_model) {
                for (llama_device_memory_data & dmd : dmds_extra) {
                    dmd.mb.model = 0;
                }
            }

            // one INFO line per (re)measurement: what the extra model adds on every device
            // and on the host. This is the budget the expert slots and the layer split
            // compete with.
            std::string per_dev;
            for (size_t id = 0; id < devs.size(); id++) {
                const llama_memory_breakdown_data & mb = dmds_extra[id].mb;
                if (mb.total() == 0) {
                    continue;
                }
                per_dev += std::string(" ") + ggml_backend_dev_name(devs[id]) + " "
                    + std::to_string(mb.total()/MiB) + " MiB (model " + std::to_string(mb.model/MiB)
                    + ", context " + std::to_string(mb.context/MiB) + ", compute " + std::to_string(mb.compute/MiB) + "),";
            }
            LOG_INF("%s: extra model '%s' memory: host %.0f MiB,%s\n",
                __func__, extra->path_model, (double) dmds_extra.back().mb.total()/MiB, per_dev.c_str());

            n_ctx_extra = cparams->n_ctx;
        }

        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += dmds_extra[id].mb.compute;
        }
    };

    // measure the main model and account the extra model on top of it. The extra model is
    // never measured on its own: borrowing its shared tensors needs the main model alive,
    // so it is measured by the same impl call whenever its cached memory is stale (the
    // context it follows changed).
    auto measure_main = [&](dmds_t & dmds, std::vector<common_expert_tensor> * expert_tensors = nullptr,
            const llama_model_params * mparams_over = nullptr) {
        const bool need_extra = extra != nullptr && (dmds_extra.empty() || n_ctx_extra != cparams->n_ctx);
        dmds_t measured_extra;
        std::vector<ggml_backend_dev_t> devs_extra;
        if (need_extra) {
            extra->cparams->n_ctx = cparams->n_ctx;

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n",
                __func__, cparams->n_ctx);
        }

        dmds = common_get_device_memory_data_impl(
            path_model, mparams_over != nullptr ? mparams_over : mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level,
            expert_tensors, extra, need_extra ? &measured_extra : nullptr, need_extra ? &devs_extra : nullptr);

        add_extra_memory(dmds, measured_extra, devs_extra);
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    dmds_t dmds_full;
    measure_main(dmds_full);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_streams, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_streams > 1) {
            LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_streams);
            measure_main(dmds_full);
        }
    }

    const size_t nd = devs.size(); // number of devices

    // Tensor split mode: every layer runs on every device. Without routed experts there is no
    // per-device placement fit can choose, so dense models keep the abort and the caller treats
    // the fit as "not possible" (the 27B tensor-split configuration has always logged it). With
    // routed experts the EP branch budgets the per-device expert store; the expert count comes
    // from the model metadata, not from an architecture name. This runs before the early
    // "no changes needed" returns below, because in EP mode the expert budgets must be set even
    // when the dense weights already fit.
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        if (hp_nex == 0) {
            throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
        }
        common_fit_ep(path_model, mparams, cparams, margins_s, log_level);
        return;
    }

    // baseline of the startup memory sampling series: the free memory the fit measures with.
    // model_loaded / sched_reserve / warmup / first_prefill / first_decode follow in the same
    // process and are logged relative to it, so the runtime growth is visible per stage.
    common_fit_log_memory_stage_devs(devs.data(), (int) nd, "fit");

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
                    dmds_t dmds_min_ctx;
                    measure_main(dmds_min_ctx);
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

    // a user-provided layer assignment stays untouched: fit then only budgets the expert
    // store from the resulting layer-to-device map. Without the expert store there is
    // nothing fit could add, keep the abort of the previous rounds.
    const bool user_ngl = mparams->n_gpu_layers != default_mparams.n_gpu_layers;
    bool user_ts = false;
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    user_ts = true;
                    break;
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

    // a model loaded earlier in this process recorded its expert allocation; the no-alloc
    // trials of this fit must not count that allocation as their own
    for (size_t id = 0; id < nd; id++) {
        common_expert_reset_allocation(moe_bufts[id]);
    }

    // build the CUDA context of every device before any measurement below. The free memory
    // a device reports without its context is larger than at allocation time, so budgeting
    // the expert slots from such a sample over-commits the device. The context lives for the
    // whole process, so every later trial and the real load measure the same state.
    for (size_t id = 0; id < nd; id++) {
        common_expert_device_ready(moe_bufts[id]);
    }
    // the first sample (dmds_full) was taken before the contexts existed; re-measure so the
    // per-device targets of the layer split use the same free-memory state as everything else.
    // The stage series restarts here: the runtime growth of the later stages is measured from
    // the state the expert slots are allocated in, not from the pre-context sample.
    measure_main(dmds_full);
    common_fit_log_memory_stage_devs(nullptr, 0, nullptr); // restart the stage series at the new baseline
    common_fit_log_memory_stage_devs(devs.data(), (int) nd, "fit_ready");

    // expert store (the MoE tiered cache): when every device supports it, all MoE expert
    // tensors are routed to the expert buffer type of the device that computes the layer, not
    // only the overflowing ones. The store then owns the whole expert set of the device and
    // decides per expert between VRAM slots and host memory, and the fit budget is what
    // remains after the dense weights, the KV cache and the compute buffers. Devices without
    // the expert store keep the per-layer overflow placement of the previous rounds.
    bool all_expert_store = hp_nex > 0;
    for (size_t id = 0; id < nd; id++) {
        if (!common_buft_is_device_expert(moe_bufts[id])) {
            all_expert_store = false;
            break;
        }
    }
    if (hp_nex > 0 && !all_expert_store) {
        LOG_WRN("%s: not every device supports the expert store, keeping the per-layer MoE overflow placement\n", __func__);
    }
    if ((user_ngl || user_ts) && !all_expert_store) {
        if (user_ngl) {
            throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
        }
        throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
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
            llama_model_tensor_buft_override * tbo_out,
            bool keep_split = false) {
        // every trial rebuilds the overrides from scratch: clear the whole array, so no
        // pattern of an earlier trial can survive behind the terminator
        std::fill(tbo_out, tbo_out + ntbo, llama_model_tensor_buft_override{nullptr, nullptr});

        // a user-provided layer split stays untouched: only the expert overrides are
        // rebuilt from the resulting layer-to-device map
        if (!keep_split) {
            mparams.n_gpu_layers = 0;
            for (size_t id = 0; id < nd; id++) {
                mparams.n_gpu_layers += ngl_per_device[id].n_layer;
                if (nd > 1) {
                    ts_out[id] = ngl_per_device[id].n_layer;
                }
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

        // a layer that no device computes: its weights are on the host and its experts stay
        // on the CPU path. Trial splits can contain such layers (all layers on one device),
        // they must not get an expert buft override.
        auto is_gpu_layer = [&](uint32_t il) {
            return act_gpu > 0 && il >= i_gpu_start && il - i_gpu_start < act_gpu;
        };

        auto get_layer_dev = [&](uint32_t il) -> size_t {
            GGML_ASSERT(is_gpu_layer(il));
            return nd == 1 ? 0 :
                std::upper_bound(split_points.begin(), split_points.end(), float(il - i_gpu_start)/act_gpu) - split_points.begin();
        };

        uint32_t il0 = i_gpu_start; // start index for tensor buft overrides

        mparams.tensor_split = ts_out;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            // expert store: the expert tensors of the full layers are placed on the expert
            // buffer type of the computing device as well, so the store owns the whole expert
            // set of each device (the layer-to-device mapping is the same as the overflow one)
            if (all_expert_store) {
                for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_full(); il++) {
                    if (!is_gpu_layer(il)) {
                        continue; // CPU layer: its experts keep the CPU path
                    }
                    if (itbo + 1 >= ntbo) {
                        tbo_out[itbo].pattern = nullptr;
                        tbo_out[itbo].buft    = nullptr;
                        itbo++;
                        mparams.tensor_buft_overrides = tbo_out;
                        throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                            + std::to_string(ntbo) + " is insufficient for model");
                    }

                    const size_t id_dev = get_layer_dev(il);
                    if (id_dev != id) {
                        LOG_ERR("%s: layer %" PRIu32 " is computed on device %s, but its expert weights would be placed on device %s\n",
                            __func__, il, ggml_backend_dev_name(devs[id]), id_dev < nd ? ggml_backend_dev_name(devs[id_dev]) : "out of range");
                        tbo_out[itbo].pattern = nullptr;
                        tbo_out[itbo].buft    = nullptr;
                        itbo++;
                        mparams.tensor_buft_overrides = tbo_out;
                        throw common_params_fit_exception("layer-to-device mapping does not match llama_model::load_tensors, abort");
                    }

                    tbo_out[itbo].pattern = get_overflow_pattern(il, LAYER_FRACTION_MOE);
                    tbo_out[itbo].buft    = moe_bufts[id_dev];
                    itbo++;
                }
            }
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (!is_gpu_layer(il)) {
                    continue; // CPU layer: its experts keep the CPU path
                }
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
    dmds_t  dmds_last;     // device memory data of the last get_memory_for_layers call, for the logs
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy, ts_work.data(), tbo_work.data());

        dmds_t dmd_nl;
        measure_main(dmd_nl, nullptr, &mparams_copy);

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

        dmds_last = std::move(dmd_nl);
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
            // the MoE tensors of every partial layer overflow to the device-readable host buffer;
            // with the expert store this check is skipped, its host complement is checked after
            // the placement plan is known (apply_expert_budgets), which is the smaller number
            if (ngl_per_device[id].n_part > 0 && common_buft_is_device_mapped(moe_bufts[id]) &&
                    !common_buft_is_device_expert(moe_bufts[id])) {
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
        measure_main(dmds_cpu_moe, &moe_expert_tensors, &mparams_copy);

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

        // reset: the trial overrides below are rebuilt from scratch
        std::fill(tbo_work.begin(), tbo_work.end(), llama_model_tensor_buft_override{nullptr, nullptr});
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
    // so the fit numbers match the real allocation byte for byte. With the expert store the
    // dense weights, the KV cache and the compute buffers are placed first and everything
    // that remains is the expert budget; the store then decides per expert between slots
    // and host memory. Staging is sized for one prefill batch, which routes to almost every
    // expert of a layer: the largest host complement of a single expert tensor of the
    // device. More slots mean a smaller complement and vice versa, so the two are solved by
    // iterating the plan function to a fixed point (a few microseconds per round).
    auto apply_expert_budgets = [&]() {
        if (hp_nex == 0) {
            return;
        }

        // layer -> device of the MoE placement, the same mapping that set_ngl_tensor_split_tbo
        // wrote into the tensor buft overrides
        std::vector<int> layer_dev((size_t) hp_ngl + 2, -1);
        std::vector<int> dev_first(nd, -1);
        std::vector<int> dev_last(nd, -1);
        uint32_t i_gpu_start = 0;
        uint32_t act_gpu     = 0;
        {
            const uint32_t n_gpu = (uint32_t) mparams->n_gpu_layers;
            i_gpu_start = hp_ngl + 1 > n_gpu ? hp_ngl + 1 - n_gpu : 0;
            act_gpu     = n_gpu < hp_ngl + 1 ? n_gpu : hp_ngl + 1;

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
            // the same mapping llama_model::load_tensors uses: i_gpu_start and act_gpu from
            // the layer count, the device from the normalized cumulative tensor_split
            auto get_layer_dev = [&](uint32_t il) -> size_t {
                if (act_gpu == 0 || il < i_gpu_start || il - i_gpu_start >= act_gpu) {
                    return (size_t) -1; // CPU layer: not computed on any device
                }
                return nd == 1 ? 0 :
                    std::upper_bound(split_points.begin(), split_points.end(), float(il - i_gpu_start)/act_gpu) - split_points.begin();
            };

            for (uint32_t il = i_gpu_start; il < i_gpu_start + act_gpu && (size_t) il < layer_dev.size(); il++) {
                const size_t id_dev = get_layer_dev(il);
                if (id_dev < nd) {
                    layer_dev[il] = (int) id_dev;
                    if (dev_first[id_dev] < 0) {
                        dev_first[id_dev] = (int) il;
                    }
                    dev_last[id_dev] = (int) il;
                }
            }

            // one INFO line with the resulting layer ranges: the real load reproduces this
            // mapping from n_gpu_layers and tensor_split, check it against load_tensors
            std::string ranges = i_gpu_start > 0 ? "CPU 0.." + std::to_string(i_gpu_start - 1) : "CPU none";
            for (size_t id = 0; id < nd; id++) {
                ranges += ", " + dev_names[id] + " ";
                if (dev_first[id] < 0) {
                    ranges += "none";
                } else {
                    ranges += std::to_string(dev_first[id]) + ".." + std::to_string(dev_last[id]);
                    if (dev_last[id] == (int) hp_ngl) {
                        ranges += " (out)";
                    }
                }
            }
            LOG_INF("%s: layer ranges: %s\n", __func__, ranges.c_str());
        }

        // the MoE layers whose expert tensors use the expert buffer type: with the expert
        // store all of them, otherwise only the partial layers (like the tensor buft
        // overrides of set_ngl_tensor_split_tbo)
        std::vector<int> tensor_dev = layer_dev;
        if (!all_expert_store) {
            std::fill(tensor_dev.begin(), tensor_dev.end(), -1);
            uint32_t il0 = i_gpu_start;
            for (size_t id = 0; id < nd; id++) {
                il0 += ngl_per_device[id].n_full();
                for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part && (size_t) il < tensor_dev.size(); il++) {
                    if (layer_dev[il] == (int) id) {
                        tensor_dev[il] = (int) id;
                    }
                }
                il0 += ngl_per_device[id].n_part;
            }
        }

        std::vector<std::vector<common_expert_tensor>> dev_tensors(nd);
        for (const common_expert_tensor & t : moe_expert_tensors) {
            if (t.layer >= 0 && (size_t) t.layer < tensor_dev.size() && tensor_dev[t.layer] >= 0) {
                dev_tensors[(size_t) tensor_dev[t.layer]].push_back(t);
            }
        }

        // with the expert store the whole expert set of the device is in the store, so the
        // surplus comes from a final measurement of the split: the no-alloc trial counts the
        // experts as host memory, what remains on the device is the dense weights, the KV
        // cache and the compute buffers, and the surplus is the expert budget
        std::vector<int64_t> mem_final;
        if (all_expert_store) {
            mem_final = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);
        }

        int64_t host_complement = 0;
        // expert slot budget of every device that has experts, for the balance report below
        std::vector<size_t> slot_budgets(nd, 0);
        bool has_experts = false;
        // host complement bytes of every tensor of each device, in allocation order: the
        // NUMA check needs the same sequence of allocations the real store will perform
        std::vector<std::vector<size_t>> host_each(nd);
        for (size_t id = 0; id < nd; id++) {
            if (!common_buft_is_device_expert(moe_bufts[id])) {
                continue;
            }

            int64_t surplus;
            if (all_expert_store) {
                // basis: the free memory of the same measurement that produced mem_final (the
                // last trial, right before the real load), not the first measurement of the fit.
                // The trials themselves create contexts and load device code, so the first free
                // reading is stale and larger; budgeting from it would over-commit the device.
                surplus = dmds_last[id].free - mem_final[id] - margins[id];
            } else {
                surplus = dmds_cpu_moe[id].free - int64_t(dmds_cpu_moe[id].mb.total()) - margins[id];
            }
            if (surplus <= 0) {
                // no slot fits: the staging area of one prefill batch cannot exist either,
                // and reading the routed host experts directly would be a silent slowdown
                size_t slot_used_none = 0;
                size_t host_bytes_none = 0;
                size_t host_peak_none = 0;
                if (!common_expert_plan(moe_bufts[id], dev_tensors[id], 0, &slot_used_none, &host_bytes_none,
                        &host_peak_none, &host_each[id])) {
                    throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(id) +
                        " has no placement plan, abort");
                }
                if (!dev_tensors[id].empty()) {
                    throw common_params_fit_exception(
                        "expert store: device " + std::to_string(id) + " has no VRAM surplus for expert slots,"
                        " so the " + std::to_string((host_peak_none + MiB - 1)/MiB) +
                        " MiB staging of one prefill batch cannot be allocated; reduce the context size,"
                        " use more devices or a lower quantization");
                }
                // no tensor of this device uses the expert store
                common_expert_set_budget(moe_bufts[id], 0, 0);
                continue;
            }
            if (dev_tensors[id].empty()) {
                // no MoE layer is computed on this device, no tensor uses its expert store
                common_expert_set_budget(moe_bufts[id], 0, 0);
                continue;
            }

            size_t stage_bytes = 0; // staging area: one prefill batch
            size_t stage_peak  = 0;
            size_t slot_budget = (size_t) surplus;
            size_t slot_used   = 0;
            size_t host_bytes  = 0;
            for (int round = 0; round < 64; ++round) {
                if (!common_expert_plan(moe_bufts[id], dev_tensors[id], slot_budget, &slot_used, &host_bytes, &stage_peak)) {
                    throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(id) +
                        " has no placement plan, abort");
                }
                const size_t next = std::min(stage_peak, (size_t) surplus);
                if (next == stage_bytes) {
                    break;
                }
                stage_bytes = next;
                slot_budget = (size_t) surplus - stage_bytes;
            }
            if (!common_expert_plan(moe_bufts[id], dev_tensors[id], slot_budget, &slot_used, &host_bytes, &stage_peak, &host_each[id])) {
                throw common_params_fit_exception("expert store: the buffer type of device " + std::to_string(id) +
                    " has no placement plan, abort");
            }
            if (stage_peak > stage_bytes) {
                // a physical limit: the routed host experts of one prefill batch cannot all be
                // staged, and a direct host read of the rest would be a silent slowdown
                throw common_params_fit_exception(
                    "expert store: device " + std::to_string(id) + " needs " +
                    std::to_string((stage_peak + MiB - 1)/MiB) + " MiB of staging for the host experts of one prefill batch, but only " +
                    std::to_string(stage_bytes/MiB) + " MiB fit (short by " +
                    std::to_string((stage_peak - stage_bytes + MiB - 1)/MiB) +
                    " MiB); reduce the context size, use more devices or a lower quantization");
            }
            common_expert_set_budget(moe_bufts[id], slot_budget, stage_bytes);
            // hand the same plan to the backend: the real allocation replans from the same
            // tensors and the same budget and aborts when it comes out at different bytes
            common_expert_record_plan(moe_bufts[id], slot_used, host_bytes);

            // one line per device: the layer range, what the dense weights, the context
            // (KV cache) and the compute buffers take, how the surplus is split into slots
            // and staging, and how much of the expert set stays in host memory (a uniform
            // routing estimate). The slot figure is the plan the allocation must reproduce
            // byte for byte; it is always <= the budget.
            const llama_memory_breakdown_data & mb = all_expert_store ? dmds_last[id].mb : dmds_cpu_moe[id].mb;
            const double hit = slot_used + host_bytes > 0
                ? 100.0 * (double) slot_used / (double) (slot_used + host_bytes) : 0.0;
            const std::string range = dev_first[id] < 0 ? "none" :
                std::to_string(dev_first[id]) + ".." + std::to_string(dev_last[id]) +
                (dev_last[id] == (int) hp_ngl ? " (out)" : "");
            LOG_INF("%s: device %zu (layers %s): dense %.0f MiB, context %.0f MiB, compute %.0f MiB, expert slots %.0f MiB (budget %.0f MiB), staging %.0f MiB, host complement %.0f MiB, decode hit %.0f%% (uniform)\n",
                __func__, id, range.c_str(),
                (double) mb.model/MiB, (double) mb.context/MiB, (double) mb.compute/MiB,
                slot_used/1024.0/1024.0, slot_budget/1024.0/1024.0, stage_bytes/1024.0/1024.0,
                host_bytes/1024.0/1024.0, hit);
            // budget accounting of this device: the surplus is real free device memory at the
            // measurement that produced mem_final, minus the dense weights, the KV cache and
            // the compute buffers, minus the user margin. The slots are planned with whole
            // experts only, so the plan uses as much of the budget as whole experts allow and
            // never more; the remainder stays free for what the runtime adds after the fit.
            const size_t slot_slack    = slot_budget > slot_used  ? slot_budget - slot_used  : 0;
            const size_t staging_slack = stage_bytes > stage_peak ? stage_bytes - stage_peak : 0;
            LOG_TRC("%s: device %zu budget: margin %.0f MiB (user target), slot remainder %.0f MiB (whole experts), staging slack %.0f MiB; basis surplus %.0f MiB = free %.0f - accounted %.0f - margin %.0f\n",
                __func__, id,
                margins[id]/1024.0/1024.0, slot_slack/1024.0/1024.0, staging_slack/1024.0/1024.0,
                surplus/1024.0/1024.0, dmds_last[id].free/1024.0/1024.0, mem_final[id]/1024.0/1024.0, margins[id]/1024.0/1024.0);
            slot_budgets[id] = slot_budget;
            has_experts = true;
            host_complement += (int64_t) host_bytes;
        }

        // balance metric of the expert slots: the user wants to see the peak and the spread
        if (has_experts) {
            size_t min_slot = SIZE_MAX;
            size_t max_slot = 0;
            size_t min_id   = 0;
            size_t max_id   = 0;
            for (size_t id = 0; id < nd; id++) {
                if (slot_budgets[id] < min_slot) {
                    min_slot = slot_budgets[id];
                    min_id   = id;
                }
                if (slot_budgets[id] > max_slot) {
                    max_slot = slot_budgets[id];
                    max_id   = id;
                }
            }
            LOG_INF("%s: expert slot budgets: min %.0f MiB (device %zu), max %.0f MiB (device %zu), difference %.0f MiB\n",
                __func__, min_slot/1024.0/1024.0, min_id, max_slot/1024.0/1024.0, max_id,
                (max_slot - min_slot)/1024.0/1024.0);
        }

        // the host complement of the expert store lives in host memory: check its NUMA
        // placement node by node with the same selection and the same availability figures
        // the real allocation uses; a plain total check remains as fallback when the backend
        // has no NUMA information (single node machines, unsupported platforms)
        if (all_expert_store) {
            // the real load walks the per-buft contexts in buft name order
            // (llama_model_loader::ctx_key_comparator) and every device has its own expert
            // buffer type; within one context the tensors keep the model order. Rebuild that
            // allocation sequence so the node simulation follows the real order.
            std::vector<int> order(nd);
            for (size_t id = 0; id < nd; id++) {
                order[id] = (int) id;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return strcmp(ggml_backend_buft_name(moe_bufts[(size_t) a]),
                              ggml_backend_buft_name(moe_bufts[(size_t) b])) < 0;
            });
            std::vector<int>    seq_dev;
            std::vector<size_t> seq_size;
            for (int id : order) {
                if (!common_buft_is_device_expert(moe_bufts[(size_t) id])) {
                    continue;
                }
                for (const size_t bytes : host_each[(size_t) id]) {
                    seq_dev.push_back(id);
                    seq_size.push_back(bytes);
                }
            }

            constexpr size_t max_nodes = 64;
            int64_t node_bytes[max_nodes] = {};
            int     node_ids  [max_nodes] = {};
            size_t  n_nodes   = max_nodes;
            int     fail_node  = -1;
            size_t  fail_bytes = 0;
            int64_t fail_avail = 0;
            const int check = common_expert_check_host_nodes(devs, seq_dev, seq_size,
                node_bytes, node_ids, &n_nodes, &fail_node, &fail_bytes, &fail_avail);
            if (check < 0) {
                // a physical limit, not a fallback: report the deficit at error level
                const int64_t deficit = (int64_t) fail_bytes - fail_avail;
                throw std::runtime_error(
                    "expert store host complement: a slice of " + std::to_string(fail_bytes/MiB) + " MiB does not fit on NUMA node " +
                    std::to_string(fail_node) + ", which has only " + std::to_string(fail_avail/MiB) +
                    " MiB available (MemFree plus reclaimable page cache, minus the reserved margin; short by " +
                    std::to_string((deficit > 0 ? deficit : 0)/MiB) + " MiB); reduce the context size, use more devices or a lower quantization");
            }
            if (check == 0) {
                const int64_t host_free   = dmds_full.back().free;
                const int64_t host_margin = nd < llama_max_devices() ? (int64_t) margins_s[nd] : 0;
                const int64_t host_needed = host_complement + host_margin;
                LOG_TRC("%s: expert store host complement %" PRId64 " MiB, %" PRId64 " MiB free, margin %" PRId64 " MiB\n",
                    __func__, host_complement/MiB, host_free/MiB, host_margin/MiB);
                if (host_needed > host_free) {
                    // a physical limit, not a fallback: report the deficit at error level
                    throw std::runtime_error(
                        "expert store host complement needs " + std::to_string(host_needed/MiB) +
                        " MiB of host memory, but only " + std::to_string(host_free/MiB) +
                        " MiB is available (short by " + std::to_string((host_needed - host_free)/MiB) +
                        " MiB); reduce the context size, use more devices or a lower quantization");
                }
            } else {
                // one INFO line with the bytes the store places on every NUMA node
                std::string per_node;
                int64_t     total = 0;
                for (size_t n = 0; n < n_nodes; n++) {
                    if (node_bytes[n] == 0) {
                        continue;
                    }
                    total += node_bytes[n];
                    per_node += " node " + std::to_string(node_ids[n]) + ": " + std::to_string(node_bytes[n]/MiB) + " MiB,";
                }
                if (per_node.empty()) {
                    per_node = " none";
                }
                LOG_INF("%s: expert store host complement %" PRId64 " MiB fits, per NUMA node:%s\n",
                    __func__, total/MiB, per_node.c_str());
            }
        }
    };

    // with the expert store all layers run on the GPU and the layer split only decides how
    // much of each device memory is left for the expert slots: distribute the layers in
    // proportion to the memory that remains after the dense weights, the KV cache and the
    // compute buffers, so the devices end up with about equal slot budgets. The first guess
    // uses the target memory, the fixed cost then comes from the same measurement the real
    // load uses, and every further round re-weights the split by the dense demand of the
    // last measurement, until the split stops changing. Integer layers use the largest
    // remainder method; the output layer is computed by the last device and counts as part
    // of its fixed cost. This split is the only placement of the expert store: when no
    // integer split keeps the dense part of every device within its target, the fit reports
    // the per-device deficit and aborts.
    auto balanced_expert_split = [&](std::vector<ngl_t> & ngl_out, std::vector<int64_t> & mem_out) {
        const uint32_t n_gpu_layers = hp_ngl + 1;

        // largest remainder method: n_gpu_layers split by the given weights, the last device
        // keeps at least one layer (the output layer)
        auto split_layers = [&](const std::vector<int64_t> & weight, std::vector<uint32_t> & out) -> bool {
            // the caller may pass a fresh vector: size it before the loop below writes out[id]
            out.assign(nd, 0);

            int64_t sum = 0;
            for (size_t id = 0; id < nd; id++) {
                sum += std::max<int64_t>(weight[id], 0);
            }
            if (sum <= 0) {
                return false;
            }
            std::vector<int64_t> rem(nd, 0);
            uint32_t assigned = 0;
            for (size_t id = 0; id < nd; id++) {
                const int64_t scaled = (int64_t) n_gpu_layers * std::max<int64_t>(weight[id], 0);
                out[id] = (uint32_t) (scaled / sum);
                rem[id] = scaled % sum;
                assigned += out[id];
            }
            while (assigned < n_gpu_layers) {
                size_t best_id = nd;
                for (size_t id = 0; id < nd; id++) {
                    if (rem[id] > 0 && (best_id == nd || rem[id] > rem[best_id])) {
                        best_id = id;
                    }
                }
                if (best_id == nd) {
                    break;
                }
                out[best_id]++;
                rem[best_id] = 0;
                assigned++;
            }
            if (nd > 1 && out[nd - 1] == 0) {
                size_t best_id = 0;
                for (size_t id = 1; id + 1 < nd; id++) {
                    if (out[id] > out[best_id]) {
                        best_id = id;
                    }
                }
                if (out[best_id] == 0) {
                    return false;
                }
                out[best_id]--;
                out[nd - 1] = 1;
            }
            return true;
        };

        std::vector<int64_t> weight(targets.begin(), targets.end());
        std::vector<uint32_t> layers(nd);
        if (!split_layers(weight, layers)) {
            throw common_params_fit_exception("expert store: cannot split the " + std::to_string(n_gpu_layers) +
                " layers over " + std::to_string(nd) + " devices with a non-negative memory target, abort");
        }

        std::vector<uint32_t> best = layers;
        std::vector<int64_t> best_mem;
        int64_t best_min = INT64_MIN;
        bool measured = false;

        // closest measured configuration: the error report uses it, the last round of an
        // oscillating iteration can be a much worse split than one already seen
        std::vector<int64_t> fail_mem;
        int64_t fail_deficit = INT64_MAX;

        for (int round = 0; round < 32; round++) {
            for (size_t id = 0; id < nd; id++) {
                ngl_out[id].n_layer = layers[id];
                ngl_out[id].n_part  = 0;
            }
            mem_out = get_memory_for_layers(__func__, ngl_out, overflow_bufts);

            bool fits = true;
            int64_t deficit = 0;
            for (size_t id = 0; id < nd; id++) {
                if (mem_out[id] > targets[id]) {
                    fits = false;
                    deficit += mem_out[id] - targets[id];
                }
            }
            if (deficit < fail_deficit) {
                fail_deficit = deficit;
                fail_mem     = mem_out;
            }
            if (fits) {
                // keep the split with the largest minimum expert budget: the integer layer
                // count can make the iteration oscillate between two neighbors
                int64_t worst = INT64_MAX;
                for (size_t id = 0; id < nd; id++) {
                    worst = std::min(worst, targets[id] - mem_out[id]);
                }
                if (!measured || worst > best_min) {
                    best_min = worst;
                    best     = layers;
                    best_mem = mem_out;
                    measured = true;
                }
            }

            std::vector<int64_t> avail(nd);
            for (size_t id = 0; id < nd; id++) {
                avail[id] = std::max<int64_t>(targets[id] - mem_out[id], 0);
            }
            std::vector<uint32_t> next;
            if (!split_layers(avail, next) || next == layers) {
                break;
            }
            layers = next;
        }

        if (!measured) {
            // no integer split keeps the dense part (weights + KV cache + compute buffers)
            // of every device within its target: report the closest attempt
            const std::vector<int64_t> & need = fail_mem.empty() ? mem_out : fail_mem;
            std::string detail;
            for (size_t id = 0; id < nd; id++) {
                const int64_t short_by = need[id] > targets[id] ? need[id] - targets[id] : 0;
                detail += "\n  " + dev_names[id] + ": needs " + std::to_string(need[id]/MiB) +
                    " MiB, target " + std::to_string(targets[id]/MiB) + " MiB, short " +
                    std::to_string(short_by/MiB) + " MiB";
            }
            throw common_params_fit_exception(
                "expert store: no integer layer split keeps the dense weights, the KV cache and the compute buffers"
                " of every device within its memory target:" + detail +
                "\nreduce the context size, use more devices or a lower quantization");
        }
        for (size_t id = 0; id < nd; id++) {
            ngl_out[id].n_layer = best[id];
            ngl_out[id].n_part  = 0;
        }
        mem_out = best_mem;
    };

    // a user-provided layer assignment: fit leaves it untouched and only budgets the expert
    // store from the resulting layer-to-device map (the same math as llama_model::load_tensors)
    if (user_ngl || user_ts) {
        int64_t n_gpu = mparams->n_gpu_layers;
        if (n_gpu < 0 || n_gpu > (int64_t) hp_ngl + 1) {
            n_gpu = (int64_t) hp_ngl + 1; // auto or "all": every layer runs on the GPU
        }
        mparams->n_gpu_layers = (int) n_gpu;

        std::vector<float> split(nd, 1.0f);
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                split[id] = user_ts ? mparams->tensor_split[id] : (float) dmds_full[id].free;
            }
        }
        // the user did not set a split: write the one load_tensors uses (by free memory)
        if (!user_ts && nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                tensor_split[id] = split[id];
            }
        }

        // number of GPU layers per device: walk the layers in the order load_tensors assigns
        // them; the count per device reproduces the same mapping through tensor_split
        std::vector<ngl_t> ngl_user(nd);
        {
            const uint32_t n_all       = hp_ngl + 1;
            const uint32_t i_gpu_start = n_all > uint32_t(n_gpu) ? n_all - uint32_t(n_gpu) : 0;
            const uint32_t act_gpu     = std::min(uint32_t(n_gpu), n_all);

            float split_sum = 0.0f;
            for (size_t id = 0; id < nd; id++) {
                split_sum += split[id];
                split[id] = split_sum;
            }
            if (split_sum > 0.0f) {
                for (size_t id = 0; id < nd; id++) {
                    split[id] /= split_sum;
                }
            }
            for (uint32_t il = i_gpu_start; il < i_gpu_start + act_gpu; il++) {
                const size_t id_dev = nd == 1 ? 0 :
                    std::upper_bound(split.begin(), split.end(), float(il - i_gpu_start)/act_gpu) - split.begin();
                ngl_user[id_dev].n_layer++;
            }
        }

        check_host_memory(ngl_user, overflow_bufts);
        set_ngl_tensor_split_tbo(ngl_user, overflow_bufts, *mparams, tensor_split, tensor_buft_overrides, /*keep_split=*/true);
        apply_expert_budgets();
        return;
    }

    // expert store: all layers run on the GPU and the split is chosen for equal expert
    // slot budgets, so no device keeps most of its experts in host memory while others
    // stay empty. This is the only placement of the expert store; a dense part that no
    // integer split can fit is reported by balanced_expert_split itself.
    if (all_expert_store) {
        balanced_expert_split(ngl_per_device, mem);
        check_host_memory(ngl_per_device, overflow_bufts);
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams, tensor_split, tensor_buft_overrides);
        apply_expert_budgets();
        return;
    }

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
            // no partial layer left to move when the dense start passed the last device
            while (delta > 1 && id_dense_start < nd) {
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

        // try to fit at least part of one more layer; the device must still hold partial
        // layers, a layer of a fully placed device cannot be split (the output layer of the
        // last device is always full)
        if (id_dense_start < nd && ngl_per_device[id_dense_start].n_part > 0 &&
                ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
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

    // start a fresh startup memory sampling series: the model/context stages below are
    // logged relative to this fit measurement
    common_fit_log_memory_stage(nullptr, nullptr);

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
