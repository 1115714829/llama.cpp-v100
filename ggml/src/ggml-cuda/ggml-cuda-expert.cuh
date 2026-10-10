#pragma once

// CUDA side of the expert store (ggml-expert-store.h): implements the memory operations
// with device allocations + device-readable host memory, owns the per-tensor device
// pointer tables, and stages the routed host tier experts into VRAM for prefill.
//
// Staging follows Strata's prefill ring (docs/analysis/strata-tiers-src.md 2.5): only the
// experts that the current batch routes to AND that live in the host arena are copied into
// the staging area, organized as per-expert slots. A small device kernel de-duplicates the
// routed ids into a bitmap first, then a second kernel builds the slot assignment, rewrites
// the staging pointer table and copies the routed experts in; the following MMQ launch on
// the same stream reads the staged bytes. The host does not wait for the bitmap, so the
// prefetch decision never blocks the launch queue; the bitmap is read back asynchronously
// for the logs.

#include "common.cuh"

#include "../ggml-expert-store.h"

// trailing row padding one expert of the given type needs past its data. The quantized MMQ
// and MMVQ loaders read the last row up to MATRIX_ROW_PADDING elements; the regular CUDA
// buffer covers that with get_alloc_size(). The expert store keeps the same bytes after
// every expert, so no kernel reads a neighbour's bytes or uninitialized host memory.
static inline size_t ggml_cuda_expert_blob_pad(ggml_type type, int64_t ne0) {
    if (!ggml_is_quantized(type) || ne0 % MATRIX_ROW_PADDING == 0) {
        return 0;
    }
    return ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
}

// lookup of one expert tensor by the mul_mat_id dispatch
struct ggml_cuda_expert_table {
    const char * const * main  = nullptr;   // device table, one entry per local expert
    const char * const * stage = nullptr;   // device table through the prefill staging area
    // host shadow of the main table: the ids of the fallback path are read on the host, which
    // cannot dereference the device table. Same entries, same expert_start offset. The entries
    // are device addresses held in a host array (the store owns it as char **), the host only
    // reads the addresses and never dereferences them, so the entries are plain char *.
    char * const * main_host = nullptr;
    int64_t     n_expert   = 0;
    size_t      blob_bytes = 0;
    const char * stage_dev = nullptr;       // staging area base of the stream (shared)
    size_t      stage_size = 0;             // bytes available in the staging area
    bool        has_host   = false;         // some experts live in the host arena
    bool        has_stage  = false;         // staging is possible for this tensor
    int         tensor_id  = -1;
    // fully resident tensor: every expert sits at plain_base + e*plain_pitch, so the kernels can
    // address it arithmetically like a plain tensor instead of reading the pointer table. Set
    // together with plain_base / plain_pitch; the store numbers the slots by expert number, so a
    // tensor without a host tier is exactly one strided block of device memory.
    bool        uniform     = false;
    const char * plain_base = nullptr;
    size_t       plain_pitch = 0;
    // when src0 is a view of the store tensor: the store expert number of the first expert of
    // the view. main / main_host / plain_base already point at that entry, so the ids of the
    // view stay view-local expert numbers (ggml semantics). Only the arrays that are numbered
    // by store expert number (residency, counts, the routing bitmap) use
    // view-local number + expert_start.
    int64_t     expert_start = 0;
};

bool ggml_cuda_expert_buft_is_expert(ggml_backend_buffer_type_t buft);

// Explicitly mark a tensor as an expert weight of the given buffer type: the store then
// places it in the tiered layout (VRAM slots + host arena). The llama loader sets the flag
// while the tensor is created, test-backend-ops marks its tensors through this proc. No-op
// for other buffer types or tensors that are not a 3D expert weight layout.
void ggml_cuda_expert_mark_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * t);

// true when this tensor is registered in an expert store: its weights are then read through
// the device pointer table instead of t->data. Tensors of the same buffer type that are not
// registered (test sentinels, plain weights) return false. Kernel files can call this
// without knowing the internals of ggml_backend_buffer.
bool ggml_cuda_expert_is_store(const ggml_tensor * t);

// false on backends where the expert store is not implemented (HIP/MUSA stubs); the buffer
// type is then not registered and no tensor ever uses it
bool ggml_cuda_expert_supported(void);

// look up the placement table of an expert tensor; when src0 is a view, the view is traced
// back to the store tensor and the expert range of the view is applied to the table. Returns
// false when src0 does not belong to an expert store. Aborts when src0 looks like a store
// view but cannot be traced back, because then no kernel could know where the weights are.
bool ggml_cuda_expert_table_get(const ggml_tensor * src0, ggml_cuda_expert_table * table);

// prepare the staging area for one prefill launch: the routed host experts of this batch
// are copied into VRAM slots and the staging pointer table is updated. All work is queued
// on the given stream, so the launch that follows reads the staged bytes. The pointer
// table that the kernels must use is returned. Requires the tensor to be an expert store
// tensor and the batch to be a prefill batch (see ggml_cuda_mul_mat_id).
const char * const * ggml_cuda_expert_stage_prepare(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream);

// Same for a decode batch (the MMVQ mmid path): one fused kernel stages the routed host experts
// and rewrites the staging pointer table, with no host side state, so the launch can be recorded
// into a CUDA graph. The pointer table that the kernels must use is returned.
const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream);

// record hits of one mul_mat_id launch: every (token, expert) entry is counted in the
// per-expert statistics, vram/host tier hits are accumulated separately. Runs on the same
// stream, no synchronization.
void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream);

// run the startup byte check of all tensors once. Called by the llama loader through the
// "ggml_backend_expert_store_model_loaded" backend proc after all tensor data was written,
// before any graph runs; a mismatch aborts. Also logs the stats under -v.
void ggml_backend_cuda_expert_verify_all(void);

// log the accumulated tier statistics (called periodically by the dispatch code under -v)
void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream);

// buffer interface of the expert buffer type: the type objects are registered in ggml-cuda.cu,
// which includes ggml-backend-impl.h for the interface struct. This header is included by the
// kernel files, so it must not depend on that internal header; the interface object is declared
// where it is used instead.
const char * ggml_backend_cuda_expert_buffer_type_name(ggml_backend_buffer_type_t buft);

// VRAM budget of one expert buffer type, set by fit before the model loads. slot_bytes is
// the VRAM reserved for resident expert slots, stage_bytes the VRAM reserved per stream for
// the prefill staging area; both are part of the fit device memory accounting. A buffer
// with expert tensors and no budget at all is refused: the host experts of a prefill batch
// could not be staged without a staging area.
void ggml_cuda_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes);

// record the placement plan fit computed for the budget of this buffer type. The store
// allocation replans from the same tensors and the same budget, so it must come out at these
// bytes; the allocation aborts when it does not. Called by fit right after ggml_cuda_expert_plan
// and before the model loads.
void ggml_cuda_expert_record_plan(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t host_bytes);

// register the global index of local expert 0 of one expert tensor before its buffer is
// allocated. The meta backend creates one simple tensor per device and knows the axis-2 slice
// start of every one of them; it calls this so the store profile and the routing dump address
// global (layer, expert) entries. Without expert parallelism the base is 0.
void ggml_cuda_expert_set_tensor_base(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor, int64_t expert_base);

// pure placement plan of the expert tensors of one device: the same store function that
// alloc_buffer_n uses, so fit sizes the staging area without a trial load. The tensors arrive
// as parallel arrays (layer, expert_base, n_expert, blob_bytes). expert_base may be null (all
// bases 0). slot_budget is the VRAM budget for the slots. Outputs: slot bytes, host
// complement bytes, the largest single-tensor host complement (the staging requirement of one
// prefill batch), and, when host_bytes_each is not null, the host complement bytes of every
// tensor in allocation order (the fit code checks their NUMA placement against the nodes).
void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each);

// actual allocation of the most recent expert buffer on the device of buft, for the fit
// memory accounting: slot_bytes is real VRAM, host_bytes host memory, stage_bytes VRAM.
// Returns false when no expert buffer was allocated on that device yet.
bool ggml_backend_cuda_expert_get_allocation(
        ggml_backend_buffer_type_t buft, size_t * slot_bytes, size_t * host_bytes, size_t * stage_bytes);

// drop the recorded allocation of all devices; the fit code calls this before its no-alloc
// trials so that a model loaded earlier in the process does not leak into the measurements
void ggml_cuda_expert_reset_allocation(ggml_backend_buffer_type_t buft);

// build or touch the CUDA context of the device of this buffer type, so that the driver
// bookkeeping the real load pays for (context, peer mappings) is present when fit samples
// the free memory. Without this the sample of a device that no trial touched reads too
// large and the slot budget derived from it over-commits the device.
void ggml_backend_cuda_expert_device_ready(ggml_backend_buffer_type_t buft);

// largest host complement of a single expert tensor in the most recent allocation on the
// device of buft. Returns false when no expert buffer was allocated yet.
bool ggml_backend_cuda_expert_get_host_peak(ggml_backend_buffer_type_t buft, size_t * peak_bytes);

// cap the VRAM slot budget of one expert buffer type in bytes (0 = automatic). Exposed for
// test-backend-ops through a backend proc address, and reserved for --expert-vram-gib later.
void ggml_cuda_expert_set_slot_limit(ggml_backend_buffer_type_t buft, size_t bytes);

// test hook: release the per-stream staging state of the expert buffers of the device, so the next
// launch creates it again (the first-launch path of the staging)
void ggml_cuda_expert_reset_streams(ggml_backend_buffer_type_t buft);

// test hook: the number of host experts the last prefill staging launch of the device copied into
// the staging area; false when no staging launch ran yet
bool ggml_cuda_expert_last_stage_hits(ggml_backend_buffer_type_t buft, size_t * n_hit);

// Host NUMA placement of the expert host arena, shared with the CUDA<n>_Mapped buffer type
// (defined in ggml-cuda.cu): the machine is planned once from the bytes every device reads, the
// node capacities (MemTotal of the nodes) and the read times measured while all the devices
// read at the same time, and every host arena of a device then goes to the node of that plan.
// The plan is made by ggml_cuda_numa_plan::plan() (ggml-cuda-numa-plan.h), which fit and the
// real allocation share: fit commits it and the load reads the committed nodes, so a fit that
// passes does not fail later and the load never places the machine from the bytes of one card.
// The return values of ggml_cuda_numa_reserve() that are not a node number:
#define GGML_CUDA_NUMA_NODE_ANY  (-2) // no NUMA node information: allocate with no policy
#define GGML_CUDA_NUMA_NODE_FAIL (-1) // there is no placement, or the bytes do not fit the nodes

// hand the host bytes of every device to the placement before the first arena of the machine is
// allocated: the plan needs all the devices, not only the one that allocates now. Returns 0 when
// the machine has no placement afterwards (the bytes do not fit the node capacities).
int ggml_cuda_numa_prepare_host_plan(const size_t * dev_bytes, size_t n_dev);

// reserve `size` bytes on the node of the plan for the device and return it (or
// GGML_CUDA_NUMA_NODE_ANY / GGML_CUDA_NUMA_NODE_FAIL). The bytes count as placed while the
// allocation is live; a failed allocation must call ggml_cuda_numa_release().
int ggml_cuda_numa_reserve(int device, size_t size);

// release a reservation of ggml_cuda_numa_reserve(), e.g. when cudaHostAlloc failed
void ggml_cuda_numa_release(int device, int node, size_t size);

// Reserve `size` bytes of host memory for an allocation that the machine plan does not place:
// the mapped weights tables (the n-gram table and the token embedding table) are read by every
// device, so they go on the node with the most free capacity and their bytes leave the capacity
// of that node, which the plan of the expert arenas then sees. Returns the node, or
// GGML_CUDA_NUMA_NODE_ANY when the machine has no host NUMA node and GGML_CUDA_NUMA_NODE_FAIL
// when the bytes do not fit into the node capacities.
int ggml_cuda_numa_fixed_alloc(size_t size);

// allocate pinned mapped host memory on the node, nullptr when the allocation fails or when
// the pages can not be bound to that node
void * ggml_cuda_numa_host_alloc(int device, size_t size, int node);

// startup log of the placement: one line per device with its node and bytes, then one line with
// the estimated step time and the two placements it is compared against
void ggml_cuda_numa_log_placement(const char * what);

// log the placement the arenas really got, once every arena of the model is allocated
void ggml_cuda_numa_log_final_placement(void);

// CPU affinity of a host memory fill: bind the calling thread to the CPUs of host NUMA node
// `node` before it writes device-readable host memory that lives there, and restore the
// previous affinity with ggml_cuda_numa_unbind_cpus() afterwards. On the AC922 a GPU reads the
// pages of its own node at 44 GB/s when the CPUs of that node wrote them last and at 17-30 GB/s
// when the CPUs of the other socket did (tests/bench-pinned-read.cu), so every fill of a host
// arena or a mapped table runs on the node of its pages. Returns false and binds nothing when
// the node is unknown, has no CPU list, or the thread is already bound by an outer call; the
// unbind is a no-op then.
bool ggml_cuda_numa_bind_cpus(int node);
void ggml_cuda_numa_unbind_cpus(void);
