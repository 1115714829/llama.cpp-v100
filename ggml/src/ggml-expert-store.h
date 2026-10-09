#pragma once

// Host-side expert store for MoE weights spread over per-device VRAM slots and a
// device-readable host arena.
//
// Layout follows Strata's expert cache (docs/analysis/strata-tiers-src.md A1/A2/A3/A10/A15):
//   - every registered tensor owns a contiguous array of VRAM slots, one expert per slot;
//   - layers are isolated by construction because a tensor spans one layer only, so a slot
//     array is exactly the per-layer quota of Strata A2;
//   - a (expert) -> slot residency table per tensor says where each expert lives;
//   - experts without a slot are packed into the host arena (the complement, A15);
//   - a static routing profile (A3) decides which experts get slots, otherwise the first
//     experts of each tensor are placed in the slots;
//   - every expert is stored exactly once, either in a slot or in the host arena.
//
// The kernels read the weights through a per-tensor device pointer table whose address is
// fixed for the lifetime of the tensor; the table content is written by this module at load
// time and may be rewritten for prefill staging (the main table stays untouched).
//
// This file is internal to ggml. The CUDA backend provides the memory operations through
// ggml_expert_store_backend, so the store itself stays platform independent and the CPU
// in-place / copy-to-VRAM choices of docs/design/r200-x86-worklist.md 4.1 can reuse it.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// an expert with no slot and no host copy (future NVMe tier)
#define GGML_EXPERT_STORE_INVALID (-1)

// upper bound of expert tensors per store (48 layers x gate/up/down with margin)
#define GGML_EXPERT_STORE_MAX_TENSORS 512

// byte alignment of one expert inside the tiers. The kernels address an expert through a
// pointer table and copy whole experts with 64-bit loads, so every expert starts aligned.
#define GGML_EXPERT_STORE_ALIGN 256

static inline size_t ggml_expert_store_align_up(size_t v) {
    return (v + GGML_EXPERT_STORE_ALIGN - 1) & ~(size_t) (GGML_EXPERT_STORE_ALIGN - 1);
}

// bytes one expert occupies in a slot / in the host arena: the data bytes, the row padding
// the backend needs past the last row (see ggml_expert_store_plan_tensor::blob_pad) and the
// alignment of the next expert. The store, the plan and the backend all use this one rule,
// so the fit estimate and the real allocation match byte for byte.
static inline size_t ggml_expert_store_stride(size_t blob_bytes, size_t blob_pad) {
    return ggml_expert_store_align_up(blob_bytes + blob_pad);
}

// where an expert is served from
enum ggml_expert_tier {
    GGML_EXPERT_TIER_VRAM = 0, // VRAM slot
    GGML_EXPERT_TIER_HOST = 1, // device-readable host arena
    GGML_EXPERT_TIER_CPU  = 2, // computed in place on the CPU (x86 worklist 4.1)
};

// memory operations of one device; all calls happen on the thread that loads the model
struct ggml_expert_store_backend {
    void * user = nullptr;

    // contiguous device memory for the slot array of one tensor
    void * (*alloc_device)(void * user, size_t size) = nullptr;
    void   (*free_device) (void * user, void * ptr)   = nullptr;

    // device-readable host memory (CUDA<n>_Mapped); returns the host address and
    // writes the device alias to dev_ptr
    void * (*alloc_host)(void * user, size_t size, void ** dev_ptr) = nullptr;
    void   (*free_host) (void * user, void * ptr)                   = nullptr;

    // synchronous copies used while loading and while verifying
    void (*copy_h2d)(void * user, void * dst_dev, const void * src_host, size_t size) = nullptr;
    void (*copy_d2h)(void * user, void * dst_host, const void * src_dev, size_t size) = nullptr;

    // copy the host-side pointer table into fixed device memory
    void (*table_upload)(void * user, void * dst_dev, const char * const * src_host, size_t n) = nullptr;
};

// per-tensor placement plan
struct ggml_expert_store_tensor {
    int      layer      = -1;
    int      kind       = -1; // llm tensor id of one of the ffn_*_exps tensors
    // global index of expert 0 of this tensor: with expert parallelism each device holds a
    // contiguous slice of the expert dim, so the profile and the routing dump need the base
    // to address global (layer, expert) entries. Always 0 for layer-split tensors.
    int64_t  expert_base = 0;
    int64_t  n_expert   = 0;
    size_t   blob_bytes = 0;  // data bytes of one expert inside this tensor
    size_t   blob_pad   = 0;  // row padding after the data of one expert
    size_t   stride     = 0;  // bytes from one expert to the next in a slot / in the host arena

    // placement
    std::vector<int32_t> residency;  // slot index (>= 0), or GGML_EXPERT_STORE_INVALID for host tier
    std::vector<int64_t> host_offset; // byte offset in the host arena, or INVALID
    int64_t n_slot       = 0; // slots assigned to this tensor
    int64_t n_host       = 0; // experts in the host arena
    int64_t host_used    = 0; // bytes actually used in the host arena
    int64_t load_bytes   = 0; // total bytes written through this tensor (load time)

    // addresses
    char * slots_dev  = nullptr; // device base of the slot array
    char * host_host  = nullptr; // host base of the arena slice used by this tensor
    char * host_dev   = nullptr; // device alias of host_host
    char ** table_host = nullptr; // host shadow of the main pointer table

    // byte-level verification (Strata A10): while loading, the bytes of two sampled experts
    // (one in the slots, one in the host arena) are kept aside; ggml_expert_store_verify()
    // compares the stored copies against them after loading.
    int64_t sample_expert[2] = { GGML_EXPERT_STORE_INVALID, GGML_EXPERT_STORE_INVALID };
    std::vector<uint8_t> sample_src[2];
    bool    sample_ready[2] = { false, false };
    bool    sample_skip[2]  = { false, false };

    // statistics (filled at runtime by the kernels' dispatch code)
    int64_t stat_hits_vram = 0;
    int64_t stat_hits_host = 0;
    std::vector<uint32_t> stat_experts; // [n_expert] routing count per local expert, for the profile dump
};

// one store per device
struct ggml_expert_store {
    const ggml_expert_store_backend * backend = nullptr;
    int device = -1;

    std::string profile_path;

    std::vector<ggml_expert_store_tensor> tensors;
    // map from a ggml tensor pointer to the index in tensors; filled by the buffer
    std::vector<std::pair<const ggml_tensor *, int>> tensor_map;

    int64_t slot_used = 0; // bytes allocated in slot arrays
    int64_t host_used = 0; // bytes allocated in host arenas
};

// profile: text file, one line per entry "layer expert count", lines starting with '#'
// are ignored. Counts are per (layer, expert) hits; higher is hotter. Missing entries
// count as zero. The format is intentionally trivially writable from shell pipelines.
struct ggml_expert_store_profile {
    int64_t n_layer  = 0;
    int64_t n_expert = 0;
    std::vector<int64_t> count; // [n_layer * n_expert], 0 if absent

    int64_t at(int64_t layer, int64_t expert) const {
        return count[layer * n_expert + expert];
    }
};

GGML_API ggml_expert_store * ggml_expert_store_init(const ggml_expert_store_backend * backend, int device, const char * profile_path);
GGML_API void ggml_expert_store_free(ggml_expert_store * store);

// routing profile (see ggml_expert_store_profile) used by stores created afterwards. The CLI
// layer sets it before the model loads; without a profile the placement is uniform within
// the per-layer quota and the store logs that.
#ifdef __cplusplus
extern "C" {
#endif
GGML_API void ggml_expert_store_set_profile(const char * path);

// runtime routing dump: when set, every store writes the accumulated per-(layer, expert)
// routing counts to this path when the model is unloaded. The file format is the same as
// the one ggml_expert_store_set_profile() reads, so one run can produce the profile of the
// next run. With more than one device the path gets a ".devN" suffix per store.
GGML_API void ggml_expert_store_set_dump_path(const char * path);

// per-token routing trace: when a path is set, every MUL_MAT_ID records the ids of the
// launch into a device ring buffer and the CUDA backend drains it to this path. This is used
// for the offline replay of mimo-runs/tools/expert-replay.py and for building a routing
// profile from real traffic; see docs/design/r200-routing-trace.md.
GGML_API void ggml_expert_store_set_trace_path(const char * path, int size_mib);
#ifdef __cplusplus
}
#endif

// trace path/size as set by the CLI layer, empty when the trace is disabled
GGML_API std::string ggml_expert_store_get_trace_path();
GGML_API int         ggml_expert_store_get_trace_size_mib();

// true when a trace path is set; a lock-free check for the per-launch hot path
GGML_API bool ggml_expert_store_trace_enabled();

// true when a routing dump path is set. The per-launch routing counts only feed that dump, so
// without it there is nothing to accumulate; this is the lock-free check for that decision.
GGML_API bool ggml_expert_store_dump_enabled();

// write the per-expert routing statistics collected in the tensor (see stat_experts) in the
// profile format. Called by the backend when the buffer is released.
GGML_API void ggml_expert_store_dump_profile(ggml_expert_store * store, int n_device);

// input of ggml_expert_store_plan: one expert tensor to place
struct ggml_expert_store_plan_tensor {
    int     layer      = -1; // profile row of the tensor
    int64_t n_expert   = 0;
    // global index of expert 0 of this tensor (expert parallelism). The profile is addressed
    // with expert_base + local expert, so the per-device slices of a tensor share one row.
    int64_t expert_base = 0;
    size_t  blob_bytes = 0;  // data bytes of one expert
    // trailing bytes one expert needs past its data for the row padding of the kernels.
    // The quantized MMQ/MMVQ loaders read the last row up to MATRIX_ROW_PADDING, which the
    // regular CUDA buffer covers with get_alloc_size(); the store keeps the same padding
    // after every expert so the kernels never read a neighbour's bytes.
    size_t  blob_pad   = 0;
};

// result of ggml_expert_store_plan; sizes are bytes
struct ggml_expert_store_placement {
    std::vector<int64_t> n_slot;       // slots assigned to each tensor
    std::vector<int64_t> host_used;    // host complement of each tensor
    std::vector<std::vector<int64_t>> slot_experts; // expert of each slot, in slot order
    int64_t slot_used  = 0; // slot array bytes across all tensors
    int64_t host_bytes = 0; // host complement across all tensors
    int64_t host_peak  = 0; // largest single-tensor complement: the staging of one prefill batch
};

// pure placement plan of one store allocation: given the expert tensors of one device and a
// slot budget in bytes, compute the per-tensor slot counts, the host complements and the
// totals. No allocation, repeatable. slot_budget < 0 gives every expert a slot. profile_path
// is the routing profile that ranks the experts; nullptr means the process-wide profile
// (ggml_expert_store_set_profile), which is the file that stores created afterwards use - so
// fit and the real allocation plan with the same profile.
GGML_API void ggml_expert_store_plan(
        const ggml_expert_store_plan_tensor * tensors, size_t n_tensors,
        int64_t slot_budget, const char * profile_path,
        ggml_expert_store_placement * placement);

// register one expert tensor at the placement computed by ggml_expert_store_plan:
// slot_experts[0..n_slot) are the experts that get slots 0..n_slot-1, the others are packed
// into the host arena. host_budget < 0 means "as many experts as needed".
// expert_base is the global index of local expert 0 (0 without expert parallelism); it only
// shifts the profile/routing rows, the placement itself stays per local expert.
// returns the tensor index, or -1 when it does not fit at all.
GGML_API int ggml_expert_store_add_tensor(
        ggml_expert_store * store,
        int layer, int kind,
        int64_t expert_base, int64_t n_expert, size_t blob_bytes, size_t blob_pad,
        const int64_t * slot_experts, int64_t n_slot,
        int64_t host_budget);

// distribute one contiguous byte range of a tensor. offset/size are in tensor bytes, the
// same convention as ggml_backend_buffer_i.set_tensor. Bytes of one expert that arrive in
// separate calls are written to the same target.
GGML_API void ggml_expert_store_write(
        ggml_expert_store * store, int tensor_id,
        const void * data, size_t offset, size_t size);

// read back a byte range of a tensor (for ggml_backend_buffer_i.get_tensor)
GGML_API void ggml_expert_store_read(
        ggml_expert_store * store, int tensor_id,
        void * data, size_t offset, size_t size);

// fill a byte range with a constant (for ggml_backend_buffer_i.memset_tensor)
GGML_API void ggml_expert_store_memset(
        ggml_expert_store * store, int tensor_id,
        uint8_t value, size_t offset, size_t size);

// byte-for-byte check of the sampled experts of one tensor: the loader kept the source
// bytes of one slot expert and one host expert aside (see ggml_expert_store_tensor), the
// check reads the placed copies back and compares them. A wrong slot/host index is caught
// at startup instead of silently computing wrong results (Strata A10).
GGML_API bool ggml_expert_store_verify(
        ggml_expert_store * store, int tensor_id);

// print per-layer slot/host counts and the profile coverage
GGML_API void ggml_expert_store_print_plan(const ggml_expert_store * store);

// print the runtime tier statistics (called periodically by the dispatch code under -v)
GGML_API void ggml_expert_store_log_stats(const ggml_expert_store * store, const char * tag);

// runtime statistics
GGML_API void ggml_expert_store_reset_stats(ggml_expert_store * store);
GGML_API const ggml_expert_store_tensor * ggml_expert_store_tensor_at(const ggml_expert_store * store, int tensor_id);

GGML_API int ggml_expert_store_tensor_id(const ggml_expert_store * store, const ggml_tensor * tensor);
GGML_API void ggml_expert_store_set_tensor_id(ggml_expert_store * store, const ggml_tensor * tensor, int tensor_id);

// store the per-expert routing counts of one tensor read back from the device; they are
// merged per layer and written by ggml_expert_store_dump_profile()
GGML_API void ggml_expert_store_set_tensor_stats(ggml_expert_store * store, int tensor_id, const uint32_t * counts, size_t n);
