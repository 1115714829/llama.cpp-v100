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

// lookup of one expert tensor by the mul_mat_id dispatch
struct ggml_cuda_expert_table {
    const char * const * main  = nullptr;   // device table, one entry per local expert
    const char * const * stage = nullptr;   // device table through the prefill staging area
    // host shadow of the main table: the ids of the fallback path are read on the host, which
    // cannot dereference the device table. Same entries, same expert_start offset.
    const char * const * main_host = nullptr;
    int64_t     n_expert   = 0;
    size_t      blob_bytes = 0;
    const char * stage_dev = nullptr;       // staging area base of the stream (shared)
    size_t      stage_size = 0;             // bytes available in the staging area
    bool        has_host   = false;         // some experts live in the host arena
    bool        has_stage  = false;         // staging is possible for this tensor
    int         tensor_id  = -1;
    // when src0 is a view of the store tensor: the first expert of the view. main already
    // points at that entry, the dispatch adds expert_start to the expert offset so that the
    // kernels index the view-local expert numbers.
    int64_t     expert_start = 0;
};

bool ggml_cuda_expert_buft_is_expert(ggml_backend_buffer_type_t buft);

// Explicitly mark a tensor as an expert weight of the given buffer type: the store then
// places it in the tiered layout (VRAM slots + host arena) even when its name does not
// look like a llama MoE expert tensor. Callers that place MoE weights should call this
// instead of relying on the "_exps.weight" name match. No-op for other buffer types or
// tensors that are not a 3D expert weight layout.
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
// the prefill staging area; both are part of the fit device memory accounting. The budget
// (0, 0) keeps all experts in the host arena.
void ggml_cuda_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes);

// pure placement plan of the expert tensors of one device: the same store function that
// alloc_buffer_n uses, so fit sizes the staging area without a trial load. The tensors arrive
// as parallel arrays (layer, n_expert, blob_bytes). slot_budget is the VRAM budget for the
// slots. Outputs: slot bytes, host complement bytes, the largest single-tensor host
// complement (the staging requirement of one prefill batch), and, when host_bytes_each is
// not null, the host complement bytes of every tensor in allocation order (the fit code
// checks their NUMA placement against the nodes).
void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * n_expert, const size_t * blob_bytes, size_t n_tensors,
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

// largest host complement of a single expert tensor in the most recent allocation on the
// device of buft. Returns false when no expert buffer was allocated yet.
bool ggml_backend_cuda_expert_get_host_peak(ggml_backend_buffer_type_t buft, size_t * peak_bytes);

// cap the VRAM slot budget of one expert buffer type in bytes (0 = automatic). Exposed for
// test-backend-ops through a backend proc address, and reserved for --expert-vram-gib later.
void ggml_cuda_expert_set_slot_limit(ggml_backend_buffer_type_t buft, size_t bytes);

// Host NUMA placement of the expert host arena, shared with the CUDA<n>_Mapped buffer type
// (defined in ggml-cuda.cu): the same one-time bandwidth probe and the same cudaHostAlloc
// path, so both halves of the host tier follow one placement. The node of every tensor arena
// is picked from the measured bandwidth matrix: the best node of the device first, then the
// next best node when a node cannot hold the arena (the nodes fill up in bandwidth order).
// The return values of ggml_cuda_numa_reserve() that are not a node number:
#define GGML_CUDA_NUMA_NODE_ANY  (-2) // no NUMA node information: allocate with no policy
#define GGML_CUDA_NUMA_NODE_FAIL (-1) // no candidate node has enough available memory

// reserve `size` bytes on a node for the device and return it (or GGML_CUDA_NUMA_NODE_ANY /
// GGML_CUDA_NUMA_NODE_FAIL). The bytes count as placed while the allocation is live; a
// failed allocation must call ggml_cuda_numa_release().
int ggml_cuda_numa_reserve(int device, size_t size);

// release a reservation of ggml_cuda_numa_reserve(), e.g. when cudaHostAlloc failed
void ggml_cuda_numa_release(int device, int node, size_t size);

// allocate pinned mapped host memory on the node, nullptr when the allocation fails
void * ggml_cuda_numa_host_alloc(size_t size, int node);

// startup log: the live host arena bytes of one device per NUMA node and the measured
// read bandwidth for those nodes (one line)
void ggml_cuda_numa_log_placement(int device, const char * what);
