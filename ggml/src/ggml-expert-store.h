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
#include "ggml-impl.h" // ggml_expert_store_node_lut, ggml_expert_store_assignment_version (C linkage)

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

// align is a multiple of GGML_EXPERT_STORE_ALIGN but in general no power of two (see
// ggml_expert_store_stride_align), so round with a division instead of a bit mask
static inline size_t ggml_expert_store_align_up(size_t v, size_t align) {
    return (v + align - 1) / align * align;
}

// floor(a * b / c) for 64-bit unsigned operands, exact for every a, b and c > 0 whose result
// fits in 64 bits, without a 128-bit type: the product is formed in two 64-bit limbs and
// divided limb-wise. Used where a byte quota must not lose a bit to rounding (the slot plan).
static inline uint64_t ggml_expert_store_mul_div_floor(uint64_t a, uint64_t b, uint64_t c) {
    GGML_ASSERT(c > 0);

    // 64 x 64 -> 128 in 32-bit limbs. The four half products are exact in 64 bits; the middle
    // column (p0 high half plus the low halves of p1 and p2) carries into both result halves.
    const uint64_t a_lo = a & 0xFFFFFFFFULL;
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = b & 0xFFFFFFFFULL;
    const uint64_t b_hi = b >> 32;
    const uint64_t p0   = a_lo * b_lo;
    const uint64_t p1   = a_lo * b_hi;
    const uint64_t p2   = a_hi * b_lo;
    const uint64_t p3   = a_hi * b_hi;
    const uint64_t mid  = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    const uint64_t lo   = (mid << 32) | (p0 & 0xFFFFFFFFULL);
    const uint64_t hi   = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);

    // the quotient needs more than 64 bits (a * b >= c * 2^64). No caller reaches this: the
    // product is a byte budget times a byte count and the quotient is smaller than the budget.
    GGML_ASSERT(hi < c);

    // long division, top half first. hi < c, so the quotient bits all come from lo. The
    // remainder stays below c, hence doubling it can wrap; the wrap is exact because the value
    // subtracted brings it back below c.
    uint64_t q = 0;
    uint64_t r = hi;
    for (int i = 63; i >= 0; --i) {
        const uint64_t bit = (lo >> i) & 1;
        if (r >> 63) {
            // 2*r + bit >= 2^64 > c: the quotient bit is set whatever the value is
            q |= (uint64_t) 1 << i;
            r = (r << 1 | bit) - c;
        } else {
            r = r << 1 | bit;
            if (r >= c) {
                q |= (uint64_t) 1 << i;
                r -= c;
            }
        }
    }
    return q;
}

// alignment of the expert stride: the least common multiple of the store alignment and the
// block size of the quantized type. The fully resident (uniform) kernels address an expert as
// base + expert * stride in units of type blocks, so the stride has to be a whole number of
// blocks; keeping the 256-byte store alignment in the lcm preserves the vectorized loads and
// the MMQ tile requirement.
static inline size_t ggml_expert_store_stride_align(size_t type_size) {
    size_t a = GGML_EXPERT_STORE_ALIGN;
    size_t b = type_size;
    while (b != 0) {
        const size_t t = a % b;
        a = b;
        b = t;
    }
    return GGML_EXPERT_STORE_ALIGN / a * type_size;
}

// bytes one expert occupies in a slot / in the host arena: the data bytes, the row padding
// the backend needs past the last row (see ggml_expert_store_plan_tensor::blob_pad) and the
// alignment of the next expert. The store, the plan and the backend all use this one rule,
// so the fit estimate and the real allocation match byte for byte.
static inline size_t ggml_expert_store_stride(size_t blob_bytes, size_t blob_pad, size_t type_size) {
    return ggml_expert_store_align_up(blob_bytes + blob_pad, ggml_expert_store_stride_align(type_size));
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
    // start of this device's segment of the expert dim (the prefix sum of the segment lengths
    // of the preceding devices): with expert parallelism each device holds n_expert experts of
    // the tensor. Which global expert a local one is comes from the assignment table
    // (ggml_expert_store_active_global_of_slice with expert_base and n_expert), not from
    // expert_base + local: the loader lays the experts out by that table. Always 0 for
    // layer-split tensors, which hold the whole expert dim.
    int64_t  expert_base = 0;
    int64_t  n_expert   = 0;
    size_t   blob_bytes = 0;  // data bytes of one expert inside this tensor
    size_t   blob_pad   = 0;  // row padding after the data of one expert
    size_t   type_size  = 1;  // block size of the type of the tensor
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

    // placement check: hash of the bytes every local expert received through
    // ggml_expert_store_write() (see ggml_expert_store_hash_expert), compared by
    // ggml_expert_store_verify_placement() against the hash the loader registered for the
    // global expert the active table maps the local one to. state: 0 nothing written, 1 the
    // expert arrived whole in one write (hash valid), 2 written in pieces (no hash)
    std::vector<uint64_t> write_hash;
    std::vector<uint8_t>  write_hash_state;

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
    // start of this device's segment of the expert dim (expert parallelism): the prefix sum of
    // the segment lengths of the preceding devices. Together with n_expert it names the device
    // (ggml_expert_store_slice_device); the profile row of a local expert is the one of the
    // global expert the assignment maps it to, not expert_base + local.
    int64_t expert_base = 0;
    size_t  blob_bytes = 0;  // data bytes of one expert
    // trailing bytes one expert needs past its data for the row padding of the kernels.
    // The quantized MMQ/MMVQ loaders read the last row up to MATRIX_ROW_PADDING, which the
    // regular CUDA buffer covers with get_alloc_size(); the store keeps the same padding
    // after every expert so the kernels never read a neighbour's bytes.
    size_t  blob_pad   = 0;
    // block size of the quantized type of the tensor (ggml_type_size). The expert stride is a
    // multiple of it, which the fully resident kernels rely on.
    size_t  type_size  = 1;
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
// fit and the real allocation plan with the same profile. n_device is the number of devices
// the experts are split over; it is needed for the per-device assignment (see
// ggml_expert_store_owner) and must be the count of the whole run, not of one store.
GGML_API void ggml_expert_store_plan(
        const ggml_expert_store_plan_tensor * tensors, size_t n_tensors,
        int n_device,
        int64_t slot_budget, const char * profile_path,
        ggml_expert_store_placement * placement);

// compute the expert assignment (the per-layer LPT split of the experts over the devices) and
// cache it for the process. profile is the routing profile of the run, nullptr (or an empty one)
// means the counts are unknown and the contiguous split is used. Called by
// ggml_expert_store_plan(); the table it produces is read through ggml_expert_store_owner().
GGML_API void ggml_expert_store_compute_assignment(
        const char * path, const ggml_expert_store_profile * profile,
        int n_device, int n_layer, int n_expert);

// result of the expert assignment (the per-layer LPT split of the experts over the devices)
struct ggml_expert_store_assignment_info {
    int    n_dev    = 0;  // devices the experts are split over
    int    n_layer  = 0;  // layers covered by the assignment
    int    n_expert = 0;  // experts per layer
    int    cap_max  = 0;  // most experts on one device
    bool   has_profile = false; // the assignment was computed from a routing profile
    // mean over the layers of max(dev load) / mean(dev load), before and after the assignment
    double imbalance_contiguous = 0.0;
    double imbalance_permuted    = 0.0;
    int    worst_layer           = -1;  // layer with the largest imbalance_permuted
    double worst_layer_imbalance = 0.0;
};

// device that holds one expert, or -1 when no assignment has been computed yet
GGML_API int ggml_expert_store_owner(int layer, int expert);

// local expert number of every expert of one layer on one device, -1 for the experts of the
// other devices; the row has ggml_expert_store_assignment_info::n_expert entries and stays
// valid until the next assignment is computed. nullptr when there is no assignment for dev.
GGML_API const int16_t * ggml_expert_store_local_of(int dev, int layer);

// global expert number of one local expert, or -1 when there is no such expert on dev
GGML_API int ggml_expert_store_global_of(int dev, int layer, int local);

// state of the cached assignment, for logging and tests
GGML_API bool ggml_expert_store_assignment_info_get(ggml_expert_store_assignment_info * info);

// The assignment the loaded expert data follows (the "active" table), as opposed to the proposal
// above. It starts as the contiguous split of the proposal's dimensions - device j holds its
// segment of the expert numbering, which is how the meta backend slices the weights - whether or
// not a profile is set. Only the loader changes it: once it has written the experts of every
// device in the order of the proposal (local k of device j holds global_of(j, layer, k)) it calls
// ggml_expert_store_activate_assignment(), and from then on the expert maps of the MUL_MAT_ID
// nodes (ggml_expert_store_node_lut, ggml-impl.h) are the rows of this table. Every change bumps
// ggml_expert_store_assignment_version(), which the device copies of the maps are keyed by.
GGML_API void ggml_expert_store_activate_assignment(void);

// back to the contiguous split of the current dimensions (a test that activated a permuted table
// restores the state for the cases after it)
GGML_API void ggml_expert_store_reset_assignment(void);

// the same three lookups as ggml_expert_store_owner / _local_of / _global_of on the active table
GGML_API int ggml_expert_store_active_owner(int layer, int expert);
GGML_API const int16_t * ggml_expert_store_active_local_of(int dev, int layer);
GGML_API int ggml_expert_store_active_global_of(int dev, int layer, int local);

// dimensions of the active table; false when no assignment was computed yet
GGML_API bool ggml_expert_store_active_dims(int * n_dev, int * n_layer, int * n_expert);

// the device whose segment of the expert dim is exactly [expert_base, expert_base + n_expert):
// >= 0 that device, -1 when the range is the whole expert dim (a tensor that is not split over
// the devices holds every expert, so its local numbers are the global ones), -2 when the range
// matches no segment of the assignment (or there is no assignment)
GGML_API int ggml_expert_store_slice_device(int64_t expert_base, int64_t n_expert);

// global expert of local expert `local` of the tensor slice [expert_base, expert_base + n_expert)
// according to the active table (-1 for the whole dim is the identity); -1 when the slice
// matches no segment or local is out of range
GGML_API int64_t ggml_expert_store_active_global_of_slice(int64_t expert_base, int64_t n_expert, int layer, int64_t local);

// hash of the bytes of one expert as the placement check uses it on both sides (loader and store):
// for a small expert all bytes, otherwise three windows (head, middle, tail) of
// GGML_EXPERT_STORE_HASH_WINDOW bytes each, so the check costs a fixed small fraction of the load.
// Two different experts of one tensor never share a hash in practice; the full-byte integrity of
// the stored copies is covered separately by ggml_expert_store_verify().
#define GGML_EXPERT_STORE_HASH_WINDOW (16*1024)
GGML_API uint64_t ggml_expert_store_hash_expert(const void * data, size_t size);

// Placement check of the loader's layout (the second half of ggml_expert_store_set_source_hash in
// ggml.h): the loader registered the hash of every global expert of a tensor by tensor name, the
// store hashed the bytes it received for every local expert. The check maps every local expert
// of this store's slice to its global expert through the active table and compares the two
// hashes. A tensor the loader did not register is counted as skipped (not a failure: the store
// can be filled by other code), a registered expert whose hash differs is a failure. n_checked and
// n_skipped (optional) receive the counts.
GGML_API bool ggml_expert_store_verify_placement(
        ggml_expert_store * store, int tensor_id, const char * tensor_name,
        int64_t * n_checked, int64_t * n_skipped);

// drop every registered source hash (after all stores verified their placement)
GGML_API void ggml_expert_store_clear_source_hashes(void);

// register one expert tensor at the placement computed by ggml_expert_store_plan:
// slot_experts[0..n_slot) are the experts that get slots 0..n_slot-1, the others are packed
// into the host arena. host_budget < 0 means "as many experts as needed".
// expert_base is the global index of local expert 0 (0 without expert parallelism); it only
// shifts the profile/routing rows, the placement itself stays per local expert.
// returns the tensor index, or -1 when it does not fit at all.
GGML_API int ggml_expert_store_add_tensor(
        ggml_expert_store * store,
        int layer, int kind,
        int64_t expert_base, int64_t n_expert, size_t blob_bytes, size_t blob_pad, size_t type_size,
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
