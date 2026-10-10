#include "ggml-cuda-expert.cuh"

#include "common.cuh"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "../ggml-expert-store.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <sys/mman.h>
#endif

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)

// CUDA implementation of the expert store memory operations. The store itself keeps the
// placement plan; this file owns the device allocations, the per-tensor device pointer
// tables and the per-stream prefill staging areas.
//
// Prefill staging (Strata prefill ring, docs/analysis/strata-tiers-src.md 2.5, adapted to
// "only what this batch routes to"):
//   1. a routing kernel de-duplicates the ids of the batch into a per-tensor bitmap of
//      n_expert bits and counts the per-expert routes;
//   2. a plan kernel assigns the routed host experts to staging slots, rewrites the staging
//      pointer table (staged address for the routed host experts, main table for all
//      others) and fills the copy list;
//   3. a copy kernel moves the routed host experts into their slots as ONE copy per expert
//      (the previous version copied the whole host complement of the layer).
// Steps 2/3 are device side, so the host never waits for the bitmap: the launch queue keeps
// running. The bitmap is read back asynchronously for the logs only. Everything is queued on
// the stream of the MUL_MAT_ID, so the MMQ launch that follows reads the staged bytes.
//
// The staging buffer is sized by fit: one block, large enough for the host complement of the
// largest expert tensor of the device, which is the worst case of one prefill batch (a batch
// of a few thousand tokens routes to almost every expert of a layer). It is per
// (device, stream): a stream may run several contexts at once while sharing the weights, each
// one needs its own block. Within one stream consecutive launches are serialized by the
// stream, so one block is enough; there is no double buffering because in the single-stream
// llama.cpp graph a copy of layer L+1 can never overlap the compute of layer L (the ids of
// L+1 are computed from the output of L).

// free and total device memory of one device: the input of the allocation logs and of the
// pre-allocation invariant check. The free memory the fit measured must still be there when
// the expert slots are allocated, otherwise the cudaMalloc below would fail without saying
// how much was needed or missing.
static void ggml_cuda_expert_get_mem(int device, size_t * free_bytes, size_t * total_bytes) {
    ggml_cuda_set_device(device);
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        *free_bytes  = 0;
        *total_bytes = 0;
    }
}

// device context ready: touch the device so its CUDA context and the peer mappings of the
// device exist, exactly like the real load does when it walks the devices. fit samples the
// free memory of the devices; a device whose context was never built still lacks that
// driver bookkeeping in the sample, so its free memory reads too large and the slot budget
// derived from it over-commits the device. Building the context before the sample makes the
// measured state and the allocation-time state the same.
void ggml_backend_cuda_expert_device_ready(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    ggml_cuda_set_device(buft_ctx->device);
    // cudaFree(nullptr) is the CUDA recommended way to force the primary context to exist
    CUDA_CHECK(cudaFree(nullptr));
}

static void * ggml_cuda_expert_alloc_device(void * user, size_t size) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);

    size_t free_before = 0;
    size_t total       = 0;
    ggml_cuda_expert_get_mem(device, &free_before, &total);

    void * ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: device %d: cudaMalloc of %.2f MiB failed (%s), %.2f MiB free of %.2f MiB\n",
            __func__, device, size / 1024.0 / 1024.0, cudaGetErrorString(err),
            free_before / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
        return nullptr;
    }
    GGML_LOG_DEBUG("%s: device %d: cudaMalloc %.2f MiB, free %.2f -> %.2f MiB of %.2f MiB\n",
        __func__, device, size / 1024.0 / 1024.0,
        free_before / 1024.0 / 1024.0, (free_before - size) / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
    return ptr;
}

static void ggml_cuda_expert_free_device(void * user, void * ptr) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaFree(ptr));
}

// node and size of every live host arena allocation, for the NUMA accounting and for the
// page drop on free; keyed by the host pointer
static std::mutex g_expert_host_mutex;
static std::map<void *, std::pair<int, size_t>> g_expert_host_alloc;

static void * ggml_cuda_expert_alloc_host(void * user, size_t size, void ** dev_ptr) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);

    // the host arena of the device follows the node of the machine plan: the plan is computed
    // once for all devices from their bytes and the measured read times, so every arena of this
    // device goes to the same node
    const int node = ggml_cuda_numa_reserve(device, size);
    ggml_cuda_set_device(device); // the probe may have left another device current

    void * host_ptr = ggml_cuda_numa_host_alloc(device, size, node);
    if (host_ptr == nullptr) {
        if (node != GGML_CUDA_NUMA_NODE_FAIL) {
            ggml_cuda_numa_release(device, node, size);
        }
        return nullptr;
    }
    cudaError_t err = cudaHostGetDevicePointer(dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        ggml_cuda_numa_release(device, node, size);
        CUDA_CHECK(cudaFreeHost(host_ptr));
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_expert_host_mutex);
        g_expert_host_alloc[host_ptr] = { node, size };
    }
    return host_ptr;
}

static void ggml_cuda_expert_free_host(void * user, void * ptr) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);

    int    node = GGML_CUDA_NUMA_NODE_ANY;
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_expert_host_mutex);
        const auto it = g_expert_host_alloc.find(ptr);
        if (it != g_expert_host_alloc.end()) {
            node = it->second.first;
            size = it->second.second;
            g_expert_host_alloc.erase(it);
        }
    }
    if (size > 0) {
        ggml_cuda_numa_release(device, node, size);
    }

    // Strata A9: an arena that is no longer needed only drops its physical pages first
    // (MADV_DONTNEED), cudaFreeHost then returns the pinned mapping itself
#if defined(MADV_DONTNEED)
    if (ptr != nullptr && size > 0 && madvise(ptr, size, MADV_DONTNEED) != 0) {
        GGML_LOG_DEBUG("%s: MADV_DONTNEED of %.2f MiB failed: %s\n",
            __func__, size / 1024.0 / 1024.0, strerror(errno));
    }
#endif
    CUDA_CHECK(cudaFreeHost(ptr));
}

static void ggml_cuda_expert_copy_h2d(void * user, void * dst_dev, const void * src_host, size_t size) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);
    ggml_cuda_memcpy_h2d_async(dst_dev, src_host, size, cudaStreamPerThread);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_cuda_expert_copy_d2h(void * user, void * dst_host, const void * src_dev, size_t size) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpy(dst_host, src_dev, size, cudaMemcpyDeviceToHost));
}

static void ggml_cuda_expert_table_upload(void * user, void * dst_dev, const char * const * src_host, size_t n) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpy(dst_dev, src_host, n * sizeof(char *), cudaMemcpyHostToDevice));
}

static ggml_expert_store_backend ggml_cuda_expert_backend(int device) {
    ggml_expert_store_backend backend;
    backend.user         = (void *) (intptr_t) device;
    backend.alloc_device = ggml_cuda_expert_alloc_device;
    backend.free_device  = ggml_cuda_expert_free_device;
    backend.alloc_host   = ggml_cuda_expert_alloc_host;
    backend.free_host    = ggml_cuda_expert_free_host;
    backend.copy_h2d     = ggml_cuda_expert_copy_h2d;
    backend.copy_d2h     = ggml_cuda_expert_copy_d2h;
    backend.table_upload = ggml_cuda_expert_table_upload;
    return backend;
}

// parsing helpers: expert tensors are 3D weight tensors, one expert per ne[2] slice

// shape and type of a tensor that the store can place
static bool ggml_cuda_expert_layout_ok(const ggml_tensor * t) {
    if (t == nullptr || t->view_src != nullptr || t->op != GGML_OP_NONE) {
        return false;
    }
    if (t->ne[2] <= 1) {
        return false;
    }
    return ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
}

// expert weight of the store: the caller marked it explicitly (GGML_TENSOR_FLAG_EXPERT; the
// llama loader sets the flag from the tensor role, test-backend-ops uses the backend proc)
// and the layout matches. The tensor name plays no part.
static bool ggml_cuda_expert_tensor_is_expert(const ggml_tensor * t) {
    if (!ggml_cuda_expert_layout_ok(t)) {
        return false;
    }
    return (t->flags & GGML_TENSOR_FLAG_EXPERT) != 0;
}

void ggml_cuda_expert_mark_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * t) {
    if (!ggml_cuda_expert_buft_is_expert(buft) || !ggml_cuda_expert_layout_ok(t)) {
        return;
    }
    t->flags |= GGML_TENSOR_FLAG_EXPERT;
}

static int ggml_cuda_expert_tensor_layer(const ggml_tensor * t) {
    const char * p = strstr(t->name, "blk.");
    if (p == nullptr) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    return layer;
}

static int ggml_cuda_expert_tensor_kind(const ggml_tensor * t) {
    const char * p = strstr(t->name, "_exps");
    if (p == nullptr) {
        return -1;
    }
    // hash the name prefix, only used for logging
    int kind = 0;
    for (const char * q = t->name; q < p; ++q) {
        kind = kind * 31 + (unsigned char) *q;
    }
    return kind;
}

// device kernels

// One pass over the ids of the launch: counts the routes per expert, records the tier hits
// and (for prefill) sets the routing bits. The bitmap is a plain u32 word per 32 experts.
// The ids may be a view of a larger tensor, so the row stride is taken from the tensor
// instead of assuming a contiguous [n_expert_used, n_tokens] array.
// The ids hold view-local expert numbers, residency / counts / the bitmap are numbered by the
// store expert number (global). expert_base is the store number of the first expert of the view
// (0 when src0 is not a view), expert_offset the first expert of this device under expert
// parallelism, so e = id + expert_base - expert_offset is the store number of the routed expert.
static __global__ void expert_routing_kernel(
        const int32_t * __restrict__ ids, const int64_t n_entries, const int n_expert_used, const int64_t ids_stride,
        const int expert_offset, const int expert_base, const int n_local, const int32_t * __restrict__ residency, const int64_t host_bytes,
        unsigned long long * __restrict__ counters, uint32_t * __restrict__ counts,
        uint32_t * __restrict__ bitmap_or_null) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) {
        return;
    }
    const int64_t token = i / n_expert_used;
    const int     slot  = (int) (i - token * n_expert_used);
    const int e = ids[token*ids_stride + slot] + expert_base - expert_offset;
    if (e < 0 || e >= n_local) {
        return;
    }
    if (counts != nullptr) {
        atomicAdd(&counts[e], 1u);
    }
    if (residency[e] >= 0) {
        atomicAdd(&counters[0], 1ULL);
    } else {
        atomicAdd(&counters[1], 1ULL);
        atomicAdd(&counters[2], (unsigned long long) host_bytes);
    }
    if (bitmap_or_null != nullptr) {
        atomicOr(&bitmap_or_null[e >> 5], 1u << (e & 31));
    }
}

// Assigns the routed host experts of this batch to staging slots and rewrites the staging
// pointer table: staged address for the routed host experts, main table entry for everyone
// else. fit sizes the staging area for the largest host complement of one tensor, so every
// routed host expert of the launch always gets a slot. The ranges use the global expert
// numbers of the store; the caller shifts the table pointer for a view (expert_start).
//
// The whole table is written, not only the range of the view: the table is a reused buffer, so
// an entry outside the range would keep the pointer of an earlier launch (or, on the first
// launch, the uninitialized bytes of the allocation) and the mul_mat_id kernel would follow it.
// n_table entries outside [expert_lo, expert_hi) are not reachable through this launch, they
// take the main table entry so that no entry of the table is ever undefined.
//
// fault counts the launches whose plan does not fit into the staging area. The area cannot be
// exceeded: fit sizes it for the host complement of the largest tensor and the allocation
// refuses a smaller budget. The check stays because an overrun here would be a silent write
// past the end of the staging area, and a device-side count reports it instead.
static __global__ void expert_stage_plan_kernel(
        const uint32_t * __restrict__ bitmap, const int expert_lo, const int expert_hi, const int n_table,
        const int32_t * __restrict__ residency,
        const char * const * __restrict__ table_main, const char ** __restrict__ table_stage,
        char * __restrict__ stage_dev, const size_t blob_stride, const size_t blob_bytes, const size_t stage_bytes,
        int32_t * __restrict__ hit_list, const size_t hit_cap, int * __restrict__ n_hit,
        int * __restrict__ slot_counter, int * __restrict__ fault, unsigned long long * __restrict__ counters) {
    const int stride = gridDim.x * blockDim.x;
    for (int x = (int) (blockIdx.x * blockDim.x + threadIdx.x); x < n_table; x += stride) {
        if (x < expert_lo || x >= expert_hi) {
            table_stage[x] = table_main[x];
            continue;
        }
        const bool routed = ((bitmap[x >> 5] >> (x & 31)) & 1u) != 0;
        if (routed && residency[x] < 0) {
            const int slot = atomicAdd(slot_counter, 1);
            if ((size_t) slot >= hit_cap || (size_t) (slot + 1) * blob_stride > stage_bytes) {
                // no slot for this expert: count the launch as faulty and leave the entry at the
                // main table, so the pointer stays a weight address the kernel can read. The
                // host turns the counter into an abort; nothing over a correct result depends
                // on it.
                atomicExch(fault, 1);
                table_stage[x] = table_main[x];
                continue;
            }
            atomicAdd(n_hit, 1);
            hit_list[slot] = x;
            table_stage[x] = stage_dev + (size_t) slot * blob_stride;
            atomicAdd(&counters[3], (unsigned long long) blob_bytes);
        } else {
            table_stage[x] = table_main[x];
        }
    }
}

// bytes one block of the staging copy moves. The copy reads the host arena through UVA, so it
// is latency bound and the bandwidth comes from the number of concurrent requests: one block
// per expert moved 8 bytes per thread, which is what made a copy as slow as reading the same
// expert in place. The chunk is a multiple of 16, so the 16-byte vector addresses stay aligned.
#define EXPERT_STAGE_COPY_CHUNK 65536

// Moves n bytes of one expert blob, four 16-byte vectors per thread per round so the block
// keeps 64 bytes per thread in flight. src and dst are both 16-byte aligned (the arena offsets
// and the slot pitch are 256-byte aligned, the chunk is a multiple of 16).
static __device__ __forceinline__ void expert_stage_copy_bytes(
        const char * __restrict__ src, char * __restrict__ dst, const size_t n) {
    const uint4 * src16 = (const uint4 *) src;
    uint4 * dst16 = (uint4 *) dst;

    const size_t n16      = n / 16;
    const size_t n16_full = (n16 / (4 * blockDim.x)) * (4 * blockDim.x);
    for (size_t i = threadIdx.x; i < n16_full; i += 4 * blockDim.x) {
        const uint4 v0 = src16[i];
        const uint4 v1 = src16[i + blockDim.x];
        const uint4 v2 = src16[i + 2 * blockDim.x];
        const uint4 v3 = src16[i + 3 * blockDim.x];
        dst16[i]                  = v0;
        dst16[i + blockDim.x]     = v1;
        dst16[i + 2 * blockDim.x] = v2;
        dst16[i + 3 * blockDim.x] = v3;
    }
    for (size_t i = n16_full + threadIdx.x; i < n16; i += blockDim.x) {
        dst16[i] = src16[i];
    }
    for (size_t i = n16 * 16 + threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Copies the routed host experts from the mapped host arena into their staging slots and
// clears the row padding after each copy. grid = (chunks of one expert, hit entries): every
// block moves EXPERT_STAGE_COPY_CHUNK bytes of one expert, so a 3.0 MiB expert is 48 blocks
// instead of one. Blocks of a hit slot that the plan did not fill return at once.
static __global__ void expert_stage_copy_kernel(
        const int32_t * __restrict__ hit_list, const int * __restrict__ n_hit,
        const char * __restrict__ host_dev, const int64_t * __restrict__ host_offset,
        char * __restrict__ stage_dev, const size_t blob_stride, const size_t blob_bytes,
        const int expert_lo, const int expert_hi, const char * const * __restrict__ table_stage,
        int * __restrict__ fault) {
    const int slot = blockIdx.y;
    if (slot >= *n_hit) {
        return;
    }
    const int64_t expert = hit_list[slot];
    if (expert < expert_lo || expert >= expert_hi || table_stage[expert] != stage_dev + (size_t) slot * blob_stride) {
        atomicExch(fault, 2);
        return;
    }
    char * dst = stage_dev + (size_t) slot * blob_stride;

    const size_t begin = (size_t) blockIdx.x * EXPERT_STAGE_COPY_CHUNK;
    if (begin < blob_bytes) {
        const size_t left = blob_bytes - begin;
        expert_stage_copy_bytes(host_dev + host_offset[expert] + begin, dst + begin,
            left < EXPERT_STAGE_COPY_CHUNK ? left : EXPERT_STAGE_COPY_CHUNK);
    }
    // The staging area is shared by tensors with different strides, so the bytes past the
    // data can still hold the weights of an earlier launch. Clear them: the MMQ loaders
    // read that far and must see zeros, not a previous tensor's bytes. Only the last chunk
    // of an expert reaches them.
    if (blockIdx.x == gridDim.x - 1) {
        for (size_t i = blob_bytes + threadIdx.x; i < blob_stride; i += blockDim.x) {
            dst[i] = 0;
        }
    }
}

// Staging for a decode batch (the MMVQ mmid path, at most MMVQ_MMID_MAX_BATCH_SIZE tokens):
// the routed host experts are copied into VRAM slots and the staging pointer table is
// rewritten, all in one kernel. grid = (chunks of one expert, local experts): block (c, y)
// owns expert e = blockIdx.y and the slice [c * CHUNK, ...) of its blob. A decode batch has a
// few hundred ids at most, so one block reads them from shared memory and derives everything
// from them: whether e is routed, which slot it gets, and whether it is copied. That is why
// there is no bitmap, no memset and no atomic on the slot numbers here, and why the host side
// of a decode launch is a single launch with nothing to wait for.
//
// The slot number is the number of routed host experts with a smaller expert number, so every
// expert computes the same number without talking to the other blocks. The staging table is a
// reused buffer: every entry is written on every launch and the entries outside the view of
// this launch take the main table (same contract as expert_stage_plan_kernel).
static __global__ void expert_stage_decode_kernel(
        const int32_t * __restrict__ ids, const int n_entries, const int n_expert_used, const int64_t ids_stride,
        const int expert_offset, const int expert_base,
        const int expert_lo, const int expert_hi,
        const int32_t * __restrict__ residency, const char * __restrict__ host_dev,
        const int64_t * __restrict__ host_offset,
        const char * const * __restrict__ table_main, const char ** __restrict__ table_stage,
        char * __restrict__ stage_dev, const size_t blob_stride, const size_t blob_bytes, const size_t stage_bytes,
        const int seq, int * __restrict__ fault, unsigned long long * __restrict__ counters) {
    const int e = (int) blockIdx.y;

    if (e < expert_lo || e >= expert_hi) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            table_stage[e] = table_main[e];
        }
        return;
    }

    extern __shared__ int32_t s_ids[];
    for (int i = threadIdx.x; i < n_entries; i += blockDim.x) {
        const int token = i / n_expert_used;
        s_ids[i] = ids[(int64_t) token * ids_stride + (i - token * n_expert_used)];
    }
    __syncthreads();

    // is expert e routed by this batch?
    int found = 0;
    for (int i = threadIdx.x; i < n_entries; i += blockDim.x) {
        found |= s_ids[i] + expert_base - expert_offset == e;
    }
    unsigned int any = (unsigned int) found;
    for (int off = 16; off > 0; off >>= 1) {
        any |= __shfl_down_sync(0xffffffffu, any, off);
    }
    __shared__ int s_any;
    if (threadIdx.x == 0) {
        s_any = 0;
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0 && any) {
        atomicOr(&s_any, 1);
    }
    __syncthreads();

    const bool hit = s_any && residency[e] < 0;
    if (!hit) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            table_stage[e] = table_main[e];
        }
        return;
    }

    // slot: how many routed host experts have a smaller expert number than e. Only a hit
    // block needs it, so the second pass over the ids runs for the few experts of the batch
    // that are actually in the host arena.
    int cnt = 0;
    for (int f = expert_lo + (int) threadIdx.x; f < e; f += blockDim.x) {
        int f_found = 0;
        for (int i = 0; i < n_entries; ++i) {
            f_found |= s_ids[i] + expert_base - expert_offset == f;
        }
        cnt += f_found && residency[f] < 0;
    }
    unsigned int sum = (unsigned int) cnt;
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    __shared__ int s_slot;
    if (threadIdx.x == 0) {
        s_slot = 0;
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0 && sum) {
        atomicAdd(&s_slot, (int) sum);
    }
    __syncthreads();
    const size_t slot = (size_t) s_slot;

    // The host side sizes the area for the host complement of this tensor (its stride
    // included), so a routed host expert always has a slot. A launch number in the fault
    // counter reports the case where it does not; the entry stays at the main table, which is
    // where the bytes of this expert are, and the host turns the counter into an abort.
    if ((slot + 1) * blob_stride > stage_bytes) {
        atomicMax(fault, seq);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            table_stage[e] = table_main[e];
        }
        return;
    }

    char * dst = stage_dev + slot * blob_stride;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        table_stage[e] = dst;
        atomicAdd(&counters[3], (unsigned long long) blob_bytes);
    }
    const size_t begin = (size_t) blockIdx.x * EXPERT_STAGE_COPY_CHUNK;
    if (begin < blob_bytes) {
        const size_t left = blob_bytes - begin;
        expert_stage_copy_bytes(host_dev + host_offset[e] + begin, dst + begin,
            left < EXPERT_STAGE_COPY_CHUNK ? left : EXPERT_STAGE_COPY_CHUNK);
    }
    if (blockIdx.x == gridDim.x - 1) {
        for (size_t i = blob_bytes + threadIdx.x; i < blob_stride; i += blockDim.x) {
            dst[i] = 0;
        }
    }
}

// per-tensor runtime state
struct ggml_cuda_expert_tensor_state {
    int tensor_id = -1;
    const ggml_tensor * tensor = nullptr;

    int64_t n_expert = 0;
    size_t  blob_bytes = 0;  // data bytes of one expert
    size_t  blob_stride = 0; // data + row padding + alignment: the staging slot pitch
    bool    has_host = false;

    const char ** table_main_dev = nullptr;
    int32_t * residency_dev   = nullptr;
    int64_t * host_offset_dev = nullptr; // device copy of the host arena byte offsets
    uint32_t * counts_dev     = nullptr; // per-expert routing counts, for --dump-expert-routing
    // the prefill staging of this tensor is possible: a host tier to stage and a staging area to
    // stage it into. The bitmap the routing kernel writes is per stream (see
    // ggml_cuda_expert_stream_state), this is the property of the tensor alone that says so.
    bool     has_stage  = false;

    bool verify_done = false;
};

// per-stream staging area: contexts sharing the weights may run on different streams, each
// stream owns its block so no cross-stream synchronization is needed
struct ggml_cuda_expert_stream_state {
    char * stage_dev   = nullptr;
    size_t stage_bytes = 0; // from fit: the largest host complement of one expert tensor

    // per tensor: device pointer table through the staging area, the routing bitmap and its
    // readback, and the copy plan buffers. The bitmap is written by the routing kernel of one
    // launch and read by the plan kernel of the same launch, so two streams that prepare the same
    // tensor would clear and set each other's bits; it belongs to the stream like the rest.
    std::vector<const char **> tables;
    std::vector<uint32_t *>     bitmaps;        // n_expert bits, for the tensors that stage
    std::vector<uint32_t *>     bitmaps_pinned; // async readback of the last bitmap, for the logs
    int32_t *              hit_list   = nullptr;
    // [n_hit, slot_counter, fault_plan, fault_copy, fault_decode]. The two counters are cleared
    // before every prefill launch, the faults are not: a fault of one launch is reported by the
    // next launch whose readback has landed, so an asynchronous check cannot miss it. The decode
    // staging writes the launch number of the launch that did not fit, which makes the counter
    // monotonic and frees it from a clear before every launch.
    int *                  plan_state  = nullptr;
    int *                  plan_pinned = nullptr; // async readback of the last plan, logs and fault check
    cudaEvent_t            plan_event  = nullptr; // marks the readback into plan_pinned
    size_t                 hit_cap    = 0;        // == largest n_local of the tensors
    int                    stage_seq  = 0;        // launch counter of the decode staging, for its fault
};

struct ggml_backend_cuda_expert_buffer_context {
    int device = -1;
    ggml_expert_store_backend backend;
    ggml_expert_store * store = nullptr;

    size_t stage_budget = 0; // per stream, bytes, from fit

    // non-expert tensors of the buffer (test sentinels, mixed contexts): one block of plain
    // device memory, like the regular CUDA buffer type would allocate for them
    void * plain_dev  = nullptr;
    size_t plain_size = 0;

    std::vector<ggml_cuda_expert_tensor_state> tensors;
    std::map<cudaStream_t, ggml_cuda_expert_stream_state> streams;

    // usage counters, updated on device and read back periodically
    unsigned long long * usage_dev = nullptr;
    unsigned long long * usage_host = nullptr; // pinned
    cudaEvent_t usage_event = nullptr;          // marks the last async copy into usage_host
    int64_t launches = 0;

    ggml_cuda_expert_stream_state & get_stream(cudaStream_t stream);
    ~ggml_backend_cuda_expert_buffer_context();
};

static ggml_backend_cuda_expert_buffer_context * ggml_cuda_expert_buffer_ctx(const ggml_tensor * t) {
    return (ggml_backend_cuda_expert_buffer_context *) t->buffer->context;
}

// VRAM budgets of the expert buffer types, set by fit (or by the tests) before the model
// loads; indices are device numbers
static std::mutex g_expert_budget_mutex;
static size_t g_expert_slot_budget [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_stage_budget[GGML_CUDA_MAX_DEVICES] = {};

// actual allocation of the most recent expert buffer per device, for the fit accounting
static size_t g_expert_alloc_slot [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_alloc_host [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_alloc_stage[GGML_CUDA_MAX_DEVICES] = {};
// largest host complement of a single tensor: the staging requirement of one prefill batch,
// which fit uses to size the staging area
static size_t g_expert_alloc_host_peak[GGML_CUDA_MAX_DEVICES] = {};

// plan the fit computed for the same budget and the same tensors as the real allocation. The
// plan is a pure function of (tensors, slot budget), so the allocation must reproduce these
// values byte for byte; it aborts when it does not. valid is false until fit records a plan
// (and after a new budget), so an allocation without a fit plan is not compared against a
// stale one.
static size_t g_expert_plan_slot[GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_plan_host[GGML_CUDA_MAX_DEVICES] = {};
static bool   g_expert_plan_valid[GGML_CUDA_MAX_DEVICES] = {};

// host bytes of a device, once they are known: the plan fit recorded for the device, or the bytes
// its last allocation used. The NUMA placement of the machine is computed from the bytes of all
// the devices at once, so this has to outlive the fit plan check above.
static bool   g_expert_host_known [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_host_bytes[GGML_CUDA_MAX_DEVICES] = {};

static void ggml_cuda_expert_set_budget_impl(int device, size_t slot_bytes, size_t stage_bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_slot_budget [device] = slot_bytes;
    g_expert_stage_budget[device] = stage_bytes;
    g_expert_plan_valid[device] = false;
    // a new budget invalidates the recorded bytes; the plan of the new budget is recorded next
    g_expert_host_known [device] = false;
    g_expert_host_bytes[device] = 0;
}

// record the fit plan (slot and host bytes) of one device: the allocation of the same budget
// must come out at exactly these bytes
static void ggml_cuda_expert_set_plan_impl(int device, size_t slot_bytes, size_t host_bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_plan_slot[device] = slot_bytes;
    g_expert_plan_host[device] = host_bytes;
    g_expert_plan_valid[device] = true;
    g_expert_host_known [device] = true;
    g_expert_host_bytes[device] = host_bytes;
}

static size_t ggml_cuda_expert_get_slot_budget(int device) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    return g_expert_slot_budget[device];
}

static size_t ggml_cuda_expert_get_stage_budget(int device) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    return g_expert_stage_budget[device];
}

// global index of local expert 0 of every expert tensor registered by the meta backend. The
// base belongs to the simple tensor of one device and is consumed by the allocation of that
// tensor; no base is kept after the allocation. Layer-split tensors are never registered.
static std::mutex g_expert_base_mutex;
static std::map<const ggml_tensor *, int64_t> g_expert_base;

void ggml_cuda_expert_set_tensor_base(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor, int64_t expert_base) {
    if (tensor == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_expert_base_mutex);
    g_expert_base[tensor] = expert_base;
}

static int64_t ggml_cuda_expert_get_tensor_base(const ggml_tensor * tensor) {
    std::lock_guard<std::mutex> lock(g_expert_base_mutex);
    const auto it = g_expert_base.find(tensor);
    return it == g_expert_base.end() ? 0 : it->second;
}

static void ggml_cuda_expert_clear_tensor_base(const ggml_tensor * tensor) {
    std::lock_guard<std::mutex> lock(g_expert_base_mutex);
    g_expert_base.erase(tensor);
}

static void ggml_cuda_expert_set_allocation(int device, size_t slot_bytes, size_t host_bytes, size_t stage_bytes, size_t host_peak) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_alloc_slot [device] = slot_bytes;
    g_expert_alloc_host [device] = host_bytes;
    g_expert_alloc_stage[device] = stage_bytes;
    g_expert_alloc_host_peak[device] = host_peak;
    // the host bytes of the device are known from here on: the NUMA placement of the machine
    // needs them for every device, not only for the one that allocates first
    g_expert_host_known [device] = true;
    g_expert_host_bytes[device] = host_bytes;
}

// Register the host bytes of every device with the NUMA placement before this device allocates
// its first arena. The bytes of the other devices come from the plans fit recorded for them
// (ggml_cuda_expert_record_plan); the bytes of this device come from the plan just computed, and
// become the recorded ones, so the whole machine is known before any arena exists. A device
// that has an expert VRAM budget but no plan leaves its bytes unknown: planning the machine
// without them would place it from partial information, which is the single-card placement, so
// this is an error instead.
static bool ggml_cuda_expert_register_host_plan(int device, size_t slot_bytes, size_t host_bytes,
        size_t stage_bytes, size_t host_peak) {
    const int n_dev = ggml_backend_cuda_get_device_count();

    std::vector<size_t> bytes((size_t) n_dev, 0);
    {
        std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
        for (int d = 0; d < n_dev; d++) {
            if (d == device) {
                bytes[(size_t) d] = host_bytes;
                continue;
            }
            if (g_expert_host_known[d]) {
                bytes[(size_t) d] = g_expert_host_bytes[d];
                continue;
            }
            if (g_expert_slot_budget[d] == 0 && g_expert_stage_budget[d] == 0) {
                continue; // no expert store on that device: it reads no host arena
            }
            GGML_LOG_ERROR("%s: device %d has an expert VRAM budget but no host byte plan, so the host bytes it reads are unknown; the NUMA placement of the machine needs the bytes of every device. Run fit (it records a plan per device before the load) or load one model per process.\n",
                __func__, d);
            return false;
        }

        g_expert_alloc_slot [device] = slot_bytes;
        g_expert_alloc_host [device] = host_bytes;
        g_expert_alloc_stage[device] = stage_bytes;
        g_expert_alloc_host_peak[device] = host_peak;
        g_expert_host_known [device] = true;
        g_expert_host_bytes[device] = host_bytes;
    }

    return ggml_cuda_numa_prepare_host_plan(bytes.data(), bytes.size()) == 1;
}

// drop the allocation accounting of all devices. The fit code calls this before its no-alloc
// trials: the numbers of a model loaded earlier in the process must not leak into them.
void ggml_cuda_expert_reset_allocation(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        g_expert_alloc_slot [i] = 0;
        g_expert_alloc_host [i] = 0;
        g_expert_alloc_stage[i] = 0;
        g_expert_alloc_host_peak[i] = 0;
        g_expert_plan_valid[i] = false;
        g_expert_host_known [i] = false;
        g_expert_host_bytes[i] = 0;
    }
}

// all expert buffers of the process, for the load-finished byte check
static std::mutex g_expert_buffers_mutex;
static std::vector<ggml_backend_cuda_expert_buffer_context *> g_expert_buffers;

// test hook: cap the VRAM slot budget of one device (bytes, 0 = automatic)
static void ggml_cuda_expert_set_slot_limit_impl(int device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_slot_budget[device] = bytes;
}

// traces a view back to the registered store tensor and returns the byte offset of the
// view inside it; nullptr when the chain is broken
static const ggml_tensor * ggml_cuda_expert_trace_view(const ggml_tensor * t, int64_t * view_offset) {
    const ggml_tensor * cur = t;
    int64_t offset = 0;
    int depth = 0;
    while (cur->view_src != nullptr) {
        if (cur->data == nullptr || cur->view_src->data == nullptr) {
            return nullptr;
        }
        offset += (const char *) cur->data - (const char *) cur->view_src->data;
        cur = cur->view_src;
        if (++depth > 16) {
            return nullptr;
        }
    }
    *view_offset = offset;
    return cur;
}

bool ggml_cuda_expert_table_get(const ggml_tensor * src0, ggml_cuda_expert_table * table) {
    if (src0 == nullptr || src0->buffer == nullptr || !ggml_cuda_expert_buft_is_expert(src0->buffer->buft)) {
        return false;
    }
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

    // a view of an expert tensor carries the same buffer; trace it back and convert the
    // byte offset into an expert offset. Anything else cannot be expressed by the pointer
    // table, so abort instead of reading wrong weights.
    int64_t view_offset = 0;
    const ggml_tensor * store_tensor = ggml_cuda_expert_trace_view(src0, &view_offset);
    if (store_tensor == nullptr) {
        GGML_ABORT("%s: cannot trace view '%s' back to the expert store tensor\n", __func__, src0->name);
    }
    const int tensor_id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, store_tensor) : -1;
    if (tensor_id < 0) {
        // a plain tensor of the same buffer type (test sentinels, mixed contexts): it is not
        // a store tensor, so the generic dispatch reads it directly from device memory
        if (!ggml_cuda_expert_tensor_is_expert(store_tensor)) {
            return false;
        }
        GGML_ABORT("%s: tensor '%s' is not registered in the expert store\n", __func__, src0->name);
    }
    const ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);

    int64_t expert_start = 0;
    if (view_offset != 0) {
        const int64_t blob = (int64_t) st->blob_bytes;
        if (view_offset % blob != 0 || src0->nb[0] != store_tensor->nb[0] ||
                src0->nb[1] != store_tensor->nb[1] || src0->nb[2] != store_tensor->nb[2]) {
            GGML_ABORT("%s: view '%s' is not a contiguous expert range of '%s'\n",
                __func__, src0->name, store_tensor->name);
        }
        expert_start = view_offset / blob;
        if (expert_start + src0->ne[2] > s.n_expert) {
            GGML_ABORT("%s: view '%s' exceeds the expert range of '%s'\n",
                __func__, src0->name, store_tensor->name);
        }
    }

    table->main         = s.table_main_dev + expert_start;
    table->stage        = nullptr;
    table->main_host    = st->table_host + expert_start;
    table->n_expert     = src0->ne[2];
    table->blob_bytes   = (size_t) src0->nb[2];
    table->stage_dev    = nullptr;
    table->stage_size   = 0;
    table->has_host     = s.has_host;
    table->has_stage    = s.has_stage;
    table->tensor_id    = tensor_id;
    table->expert_start = expert_start;
    // without a host tier every expert is a slot and slot e holds expert e (see
    // ggml_expert_store_plan), so the slots are one strided block starting at expert_start
    table->uniform      = !s.has_host;
    if (table->uniform) {
        // the uniform (fully resident) kernels address an expert as base + expert * plain_pitch
        // in units of type blocks, so the pitch has to be a whole number of blocks. The store
        // aligns the pitch to lcm(GGML_EXPERT_STORE_ALIGN, type_size) to guarantee it.
        const size_t ts = ggml_type_size(src0->type);
        GGML_ASSERT(st->stride % ts == 0);
    }
    table->plain_base   = st->slots_dev + expert_start * (int64_t) st->stride;
    table->plain_pitch  = st->stride;
    return true;
}

ggml_cuda_expert_stream_state & ggml_backend_cuda_expert_buffer_context::get_stream(cudaStream_t stream) {
    const auto it = streams.find(stream);
    if (it != streams.end()) {
        return it->second;
    }

    ggml_cuda_set_device(device);

    // The state belongs to the stream, so its clearing is queued on that stream: the kernels of
    // the first launch follow it there in order. A clear through the synchronous cudaMemset would
    // run on the legacy default stream, which has no ordering with the non-blocking streams of
    // the backend (common.cuh, cudaStreamNonBlocking): it could run while the plan kernel of the
    // first launch has already counted the routed host experts and the copy kernel is filling the
    // area, and zero the count or the staged bytes behind them - the consumer then reads zeroed
    // slots and the rows of every routed host expert come out as zero. Observed in
    // test-backend-ops on x86 (every test case is a first launch there), where the result of a
    // case lost all host experts about once in a hundred first launches.
    ggml_cuda_expert_stream_state ss;
    if (stage_budget > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.stage_dev, stage_budget));
        // the copy kernel writes only the data bytes of a staged expert; the row padding
        // after them is never touched again, so clearing the area once keeps it at zero
        CUDA_CHECK(cudaMemsetAsync(ss.stage_dev, 0, stage_budget, stream));
        ss.stage_bytes = stage_budget;
    }

    for (const ggml_cuda_expert_tensor_state & s : tensors) {
        ss.hit_cap = std::max(ss.hit_cap, (size_t) s.n_expert);
    }
    if (ss.hit_cap > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.hit_list, ss.hit_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &ss.plan_state, 5 * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ss.plan_state, 0, 5 * sizeof(int), stream));
        CUDA_CHECK(cudaHostAlloc((void **) &ss.plan_pinned, 5 * sizeof(int), cudaHostAllocDefault));
        memset(ss.plan_pinned, 0, 5 * sizeof(int));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss.plan_event, cudaEventDisableTiming));
    }
    ss.tables.assign(tensors.size(), nullptr);
    ss.bitmaps.assign(tensors.size(), nullptr);
    ss.bitmaps_pinned.assign(tensors.size(), nullptr);
    // Everything a launch needs is allocated here, while the device of the buffer is current and
    // before any launch runs: the prepare of a launch must not allocate, so nothing of the host
    // side of a staging launch lands in the middle of a launch queue (or of a graph capture).
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_cuda_expert_tensor_state & s = tensors[i];
        if (!s.has_stage) {
            continue;
        }
        CUDA_CHECK(cudaMalloc((void **) &ss.tables[i], (size_t) s.n_expert * sizeof(char *)));
        // every launch rewrites the whole table (see expert_stage_plan_kernel); the clear only
        // takes the uninitialized bytes of the allocation out of the state space, so an entry
        // that a launch did not write is a null pointer, not the pointer of an earlier tensor
        CUDA_CHECK(cudaMemsetAsync(ss.tables[i], 0, (size_t) s.n_expert * sizeof(char *), stream));
        const size_t n_words = ((size_t) s.n_expert + 31) / 32 * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc((void **) &ss.bitmaps[i], n_words));
        CUDA_CHECK(cudaMemsetAsync(ss.bitmaps[i], 0, n_words, stream));
        CUDA_CHECK(cudaHostAlloc((void **) &ss.bitmaps_pinned[i], n_words, cudaHostAllocDefault));
        memset(ss.bitmaps_pinned[i], 0, n_words);
    }

    return streams.emplace(stream, std::move(ss)).first->second;
}

ggml_backend_cuda_expert_buffer_context::~ggml_backend_cuda_expert_buffer_context() {
    ggml_cuda_set_device(device);

    // read the per-expert routing counts back and write the profile file before the device
    // buffers go away; the store skips the file when nothing was routed (memory-fit loads)
    if (store != nullptr) {
        for (ggml_cuda_expert_tensor_state & s : tensors) {
            if (s.counts_dev == nullptr || s.n_expert <= 0) {
                continue;
            }
            std::vector<uint32_t> counts((size_t) s.n_expert);
            CUDA_CHECK(cudaMemcpy(counts.data(), s.counts_dev, counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            ggml_expert_store_set_tensor_stats(store, s.tensor_id, counts.data(), counts.size());
        }
        ggml_expert_store_dump_profile(store, ggml_backend_cuda_get_device_count());
    }

    for (auto & kv : streams) {
        ggml_cuda_expert_stream_state & ss = kv.second;
        for (size_t i = 0; i < ss.tables.size(); ++i) {
            if (ss.tables[i] != nullptr) {
                CUDA_CHECK(cudaFree(ss.tables[i]));
            }
            if (ss.bitmaps[i] != nullptr) {
                CUDA_CHECK(cudaFree(ss.bitmaps[i]));
            }
            if (ss.bitmaps_pinned[i] != nullptr) {
                CUDA_CHECK(cudaFreeHost(ss.bitmaps_pinned[i]));
            }
        }
        if (ss.stage_dev) {
            CUDA_CHECK(cudaFree(ss.stage_dev));
        }
        if (ss.hit_list) {
            CUDA_CHECK(cudaFree(ss.hit_list));
        }
        if (ss.plan_state) {
            CUDA_CHECK(cudaFree(ss.plan_state));
        }
        if (ss.plan_pinned) {
            CUDA_CHECK(cudaFreeHost(ss.plan_pinned));
        }
        if (ss.plan_event) {
            CUDA_CHECK(cudaEventDestroy(ss.plan_event));
        }
    }
    for (ggml_cuda_expert_tensor_state & s : tensors) {
        if (s.table_main_dev) {
            CUDA_CHECK(cudaFree(s.table_main_dev));
        }
        if (s.residency_dev) {
            CUDA_CHECK(cudaFree(s.residency_dev));
        }
        if (s.host_offset_dev) {
            CUDA_CHECK(cudaFree(s.host_offset_dev));
        }
        if (s.counts_dev) {
            CUDA_CHECK(cudaFree(s.counts_dev));
        }
    }
    if (usage_dev) {
        CUDA_CHECK(cudaFree(usage_dev));
    }
    if (usage_host) {
        CUDA_CHECK(cudaFreeHost(usage_host));
    }
    if (usage_event) {
        CUDA_CHECK(cudaEventDestroy(usage_event));
    }
    if (plain_dev) {
        CUDA_CHECK(cudaFree(plain_dev));
    }
    ggml_expert_store_free(store);
}

// records the tier hits and the per-expert routing counts of one mul_mat_id launch. The
// per-expert counts use the view-local expert numbers of the tensor, so a view of a store
// tensor counts into its own range of the full count array.
// The counts have exactly one reader: the routing dump that ggml_expert_store_dump_profile()
// writes at unload. Without a dump path there is nothing to accumulate, and the kernel is pure
// overhead - one launch per mul_mat_id plus every routing entry hammering the same counter
// address - so it only runs when a dump is configured.
void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);
    if (ctx->usage_dev == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return;
    }
    if (!ggml_expert_store_dump_enabled()) {
        return;
    }
    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table)) {
        return;
    }
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];

    const int64_t n_entries = ggml_nelements(ids);
    if (n_entries <= 0) {
        return;
    }

    const int nthreads = 256;
    const int nblocks = (int) ((n_entries + nthreads - 1) / nthreads);
    // counts / residency are arrays of the whole store tensor, so the ids of the view are
    // counted with their store expert number (id + expert_start) and the range check is against
    // the expert count of the store tensor, not of the view.
    expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
        (const int32_t *) ids->data, n_entries, (int) ids->ne[0], ids->nb[1] / sizeof(int32_t),
        expert_offset, (int) ex_table.expert_start, (int) s.n_expert,
        s.residency_dev, (int64_t) ex_table.blob_bytes,
        ctx->usage_dev, s.counts_dev, nullptr);
    // check right here: a sticky error would otherwise surface at the next unrelated API
    // call and hide which kernel actually faulted
    CUDA_CHECK(cudaGetLastError());
}

// Tier counters of one tick. The copy is queued on the stream and the log line is printed from
// the pinned buffer only once it has landed, so the host never waits for the GPU here: a
// blocking readback in the middle of a decode step stops the launch queue and lets the cards
// drift apart, which shows up as waiting in the tensor parallel all-reduce.
static void ggml_cuda_expert_stats_readback(ggml_backend_cuda_expert_buffer_context * ctx, cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusActive) {
        CUDA_CHECK(cudaMemcpyAsync(ctx->usage_host, ctx->usage_dev, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ctx->usage_event, stream));
        if (cudaEventQuery(ctx->usage_event) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        const unsigned long long total = ctx->usage_host[0] + ctx->usage_host[1];
        if (total > 0) {
            GGML_LOG_DEBUG("%s: device %d: expert hits vram %llu / host %llu (%.2f%% vram), host hit %.2f MiB, staged %.2f MiB\n",
                __func__, ctx->device, ctx->usage_host[0], ctx->usage_host[1],
                100.0 * ctx->usage_host[0] / total, ctx->usage_host[2] / 1024.0 / 1024.0,
                ctx->usage_host[3] / 1024.0 / 1024.0);
        }
    }
}

void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);
    ctx->launches++;
    if (ctx->launches % 64 != 0) {
        return;
    }
    ggml_cuda_expert_stats_readback(ctx, stream);

    // how many experts the last prefill batch routed to (the bitmap was read back
    // asynchronously, so this may describe an earlier batch of the tensor; logs only)
    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table) || !ex_table.has_stage) {
        return;
    }
    const ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];

    // staging plan of the last prefill batch of this stream: how many routed host experts
    // were staged (async readback, logs only). The plan and the bitmap are read back into
    // pinned memory that the stream may still be writing, so they are read only once the event
    // of the readback has landed; otherwise the log shows a torn read of a launch that has not
    // finished yet. Skipping the line costs nothing, the next tick reads the same buffers.
    const auto it = ctx->streams.find(stream);
    const bool plan_ready = it != ctx->streams.end() && it->second.plan_pinned != nullptr &&
        cudaEventQuery(it->second.plan_event) == cudaSuccess;
    if (!plan_ready) {
        (void) cudaGetLastError();
        return;
    }

    const uint32_t * bitmap_pinned = it->second.bitmaps_pinned[(size_t) ex_table.tensor_id];
    if (bitmap_pinned != nullptr && s.n_expert > 0) {
        const int n_words = (int) ((s.n_expert + 31) / 32);
        int n_routed = 0;
        for (int i = 0; i < n_words; ++i) {
            uint32_t v = bitmap_pinned[i];
            while (v != 0) {
                n_routed += (int) (v & 1u);
                v >>= 1;
            }
        }
        GGML_LOG_DEBUG("%s: device %d tensor %s: last batch routed %d/%" PRId64 " experts\n",
            __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?", n_routed, s.n_expert);
    }

    const int n_hit = it->second.plan_pinned[0];
    GGML_LOG_DEBUG("%s: device %d tensor %s: staging last batch: %d host experts staged (%.2f MiB)\n",
        __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?",
        n_hit, (double) (n_hit * (int64_t) s.blob_bytes) / 1024.0 / 1024.0);
}

// prepare the staging area for one prefill launch: de-duplicate the routed experts of the
// batch into the bitmap, assign the routed host experts to slots, rewrite the staging table
// and copy the routed experts in. All of it is device side and queued on the stream, the
// host never waits; the bitmap is read back asynchronously for the logs.
const char * const * ggml_cuda_expert_stage_prepare(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table)) {
        return nullptr;
    }
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id);

    ggml_cuda_set_device(ctx->device);

    // without a host tier there is nothing to stage: the main table is complete (every
    // entry is a VRAM slot) and is used as-is.
    if (!s.has_host) {
        return s.table_main_dev + ex_table.expert_start;
    }

    // fit sizes the staging area for the largest host complement of one tensor and refuses
    // a load that cannot hold it (common_params_fit_impl reports the deficit), so the area
    // of this tensor is always complete. Configuration invariants, not a fallback path.
    GGML_ASSERT(ids != nullptr && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(s.has_stage);
    GGML_ASSERT(ctx->stage_budget > 0 && ctx->stage_budget >= (size_t) st->host_used);

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
    GGML_ASSERT(ss.hit_list != nullptr && ss.hit_cap >= (size_t) s.n_expert && ss.stage_dev != nullptr);
    GGML_ASSERT(ss.stage_bytes >= s.blob_stride && ss.stage_bytes >= (size_t) st->host_used);

    const char ** table_stage = ss.tables[(size_t) ex_table.tensor_id];
    uint32_t *   bitmap_dev   = ss.bitmaps[(size_t) ex_table.tensor_id];
    GGML_ASSERT(table_stage != nullptr && bitmap_dev != nullptr);

    const int nthreads = 256;
    const int64_t n_entries = ggml_nelements(ids);
    const int n_words = (int) ((s.n_expert + 31) / 32);
    char * block_dev = ss.stage_dev;

    // [n_hit, slot_counter]; the two fault counters stay set until they are reported
    CUDA_CHECK(cudaMemsetAsync(ss.plan_state, 0, 2 * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(bitmap_dev, 0, (size_t) n_words * sizeof(uint32_t), stream));

    if (n_entries > 0) {
        const int nblocks = (int) ((n_entries + nthreads - 1) / nthreads);
        // counts == nullptr: the routing statistics of this launch were recorded separately.
        // The bitmap is read by the plan kernel with store expert numbers, so the routing kernel
        // writes it with the store number of the view (id + expert_start), not with the id itself.
        expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
            (const int32_t *) ids->data, n_entries, (int) ids->ne[0], ids->nb[1] / sizeof(int32_t),
            expert_offset, (int) ex_table.expert_start, (int) s.n_expert,
            s.residency_dev, (int64_t) s.blob_bytes, ctx->usage_dev, nullptr, bitmap_dev);
        CUDA_CHECK(cudaGetLastError());
    }

    // the whole staging table is rewritten every launch, so no entry keeps the pointer of an
    // earlier one (see the kernel)
    const int expert_lo = (int) ex_table.expert_start;
    const int expert_hi = expert_lo + (int) ex_table.n_expert;
    const int n_table    = (int) s.n_expert;
    const int plan_blocks = std::max(1, std::min((n_table + nthreads - 1) / nthreads, 1024));
    expert_stage_plan_kernel<<<plan_blocks, nthreads, 0, stream>>>(
        bitmap_dev, expert_lo, expert_hi, n_table, s.residency_dev,
        s.table_main_dev, table_stage, block_dev, s.blob_stride, s.blob_bytes, ss.stage_bytes,
        ss.hit_list, ss.hit_cap, ss.plan_state, ss.plan_state + 1, ss.plan_state + 2, ctx->usage_dev);
    CUDA_CHECK(cudaGetLastError());

    const int copy_chunks = (int) ((s.blob_bytes + EXPERT_STAGE_COPY_CHUNK - 1) / EXPERT_STAGE_COPY_CHUNK);
    expert_stage_copy_kernel<<<dim3(copy_chunks, expert_hi - expert_lo), nthreads, 0, stream>>>(
        ss.hit_list, ss.plan_state, st->host_dev, s.host_offset_dev,
        block_dev, s.blob_stride, s.blob_bytes,
        expert_lo, expert_hi, table_stage, ss.plan_state + 3);
    CUDA_CHECK(cudaGetLastError());

    // read the bitmap and the copy plan back without waiting: the log shows how many experts
    // the last batches staged, and a launch whose plan and copy disagree is reported here.
    // The decode staging fault rides in the same readback: a decode launch records nothing
    // itself, so the next prefill launch is the one that reports it.
    CUDA_CHECK(cudaMemcpyAsync(ss.bitmaps_pinned[(size_t) ex_table.tensor_id], bitmap_dev,
        (size_t) n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ss.plan_pinned, ss.plan_state, 5 * sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ss.plan_event, stream));

    // the fault counters are sticky, so a launch that could not be checked here (its
    // readback has not landed) is still reported by the next launch that can. None of them can
    // fire on a correct staging: the area holds every routed host expert by the budget
    // invariant above, and the copy reads the plan of the same launch on the same stream.
    if (cudaEventQuery(ss.plan_event) == cudaSuccess) {
        const int fault_plan  = ss.plan_pinned[2];
        const int fault_copy  = ss.plan_pinned[3];
        const int fault_decode = ss.plan_pinned[4];
        if (fault_plan != 0 || fault_copy != 0 || fault_decode != 0) {
            const size_t slot_cap = ss.stage_bytes / s.blob_stride;
            GGML_ABORT("%s: tensor '%s' on stream %p: staging plan does not fit (fault %d/%d/%d): "
                "staging area %.2f MiB holds %zu experts of %.2f MiB, the host complement of the tensor "
                "is %" PRId64 " experts; fault 1 is the plan (too many routed host experts), "
                "fault 2 is the copy (hit list and pointer table disagree), fault 3 is the decode "
                "staging (launch number %d did not fit)\n",
                __func__, s.tensor != nullptr ? s.tensor->name : "?", (void *) stream,
                fault_plan, fault_copy, fault_decode,
                (double) ss.stage_bytes / 1024.0 / 1024.0, slot_cap,
                (double) s.blob_stride / 1024.0 / 1024.0, st->n_host, fault_decode);
        }
    } else {
        (void) cudaGetLastError();
    }

    return table_stage + ex_table.expert_start;
}

// Staging for one decode launch: the experts this batch routes to that live in the host arena
// are copied into VRAM slots by one fused kernel, and the staging pointer table is rewritten
// for the MMVQ mmid kernel that follows. Nothing is read back and nothing is waited for, so a
// decode launch can be recorded into a CUDA graph; the kernel recomputes the routing of the ids
// it is given when the graph is replayed.
const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

    ggml_cuda_expert_table ex_table;
    GGML_ASSERT(ggml_cuda_expert_table_get(src0, &ex_table));
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id);

    ggml_cuda_set_device(ctx->device);

    // without a host tier there is nothing to stage: the main table is complete (every
    // entry is a VRAM slot) and is used as-is.
    if (!s.has_host) {
        return s.table_main_dev + ex_table.expert_start;
    }

    // Configuration invariants, not a fallback path: the area holds the host complement of
    // this tensor with the stride of a slot, and a decode batch routes to at most one expert
    // per id entry, so every routed host expert has a slot.
    GGML_ASSERT(ids != nullptr && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ex_table.has_stage);
    GGML_ASSERT(ctx->stage_budget > 0 && ctx->stage_budget >= (size_t) st->host_used);

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
    GGML_ASSERT(ss.stage_dev != nullptr && ss.stage_bytes >= s.blob_stride);

    const int n_entries = (int) ggml_nelements(ids);
    const int n_expert_used = (int) ids->ne[0];
    GGML_ASSERT(n_entries > 0 && n_entries <= (int) (32 * 1024 / sizeof(int32_t)));
    GGML_ASSERT((size_t) std::min<int64_t>(n_entries, st->n_host) * s.blob_stride <= ss.stage_bytes);

    // the table is allocated with the stream state, so a launch allocates nothing (see
    // ggml_backend_cuda_expert_buffer_context::get_stream)
    const char ** table_stage = ss.tables[(size_t) ex_table.tensor_id];
    GGML_ASSERT(table_stage != nullptr);

    const int nthreads = 256;
    const int expert_lo = (int) ex_table.expert_start;
    const int expert_hi = expert_lo + (int) ex_table.n_expert;
    const int n_chunks = (int) ((s.blob_bytes + EXPERT_STAGE_COPY_CHUNK - 1) / EXPERT_STAGE_COPY_CHUNK);

    expert_stage_decode_kernel<<<dim3(n_chunks, (unsigned int) s.n_expert), nthreads,
        n_entries * sizeof(int32_t), stream>>>(
        (const int32_t *) ids->data, n_entries, n_expert_used, ids->nb[1] / sizeof(int32_t),
        expert_offset, (int) ex_table.expert_start, expert_lo, expert_hi,
        s.residency_dev, st->host_dev, s.host_offset_dev, s.table_main_dev, table_stage,
        ss.stage_dev, s.blob_stride, s.blob_bytes, ss.stage_bytes,
        ++ss.stage_seq, ss.plan_state + 4, ctx->usage_dev);
    CUDA_CHECK(cudaGetLastError());

    return table_stage + ex_table.expert_start;
}

// buffer interface

static void ggml_backend_cuda_expert_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        g_expert_buffers.erase(std::remove(g_expert_buffers.begin(), g_expert_buffers.end(), ctx), g_expert_buffers.end());
    }
    delete ctx;
}

static void * ggml_backend_cuda_expert_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    return ctx->plain_dev != nullptr ? ctx->plain_dev : (void *) ctx->store;
}

static enum ggml_status ggml_backend_cuda_expert_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

// The store addresses its bytes by an offset relative to the tensor start, one expert per
// blob_bytes (see ggml_expert_store_write), so a view of an expert tensor is traced back to the
// store tensor and its own offset added. The same view check as ggml_cuda_expert_table_get:
// anything the store cannot express aborts instead of reading or writing the wrong experts.
// Returns false for a plain tensor of the same buffer type (test sentinels, mixed contexts):
// it has no store entry and is addressed through its data pointer.
static bool ggml_backend_cuda_expert_buffer_store_target(
        const ggml_tensor * tensor, ggml_backend_cuda_expert_buffer_context * ctx, const char * func,
        int * tensor_id, size_t * store_offset) {
    int64_t view_offset = 0;
    const ggml_tensor * store_tensor = ggml_cuda_expert_trace_view(tensor, &view_offset);
    if (store_tensor == nullptr) {
        GGML_ABORT("%s: cannot trace view '%s' back to the expert store tensor\n", func, tensor->name);
    }

    const int id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, store_tensor) : -1;
    if (id < 0) {
        if (!ggml_cuda_expert_tensor_is_expert(store_tensor)) {
            return false;
        }
        GGML_ABORT("%s: tensor '%s' is not registered in the expert store\n", func, tensor->name);
    }

    if (view_offset != 0) {
        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, id);
        if (view_offset % (int64_t) st->blob_bytes != 0 || tensor->nb[0] != store_tensor->nb[0] ||
                tensor->nb[1] != store_tensor->nb[1] || tensor->nb[2] != store_tensor->nb[2]) {
            GGML_ABORT("%s: view '%s' is not a contiguous expert range of '%s'\n",
                func, tensor->name, store_tensor->name);
        }
        if (view_offset / (int64_t) st->blob_bytes + tensor->ne[2] > st->n_expert) {
            GGML_ABORT("%s: view '%s' exceeds the expert range of '%s'\n",
                func, tensor->name, store_tensor->name);
        }
    }

    *tensor_id    = id;
    *store_offset = (size_t) view_offset;
    return true;
}

static void ggml_backend_cuda_expert_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    int tensor_id = -1;
    size_t store_offset = 0;
    if (!ggml_backend_cuda_expert_buffer_store_target(tensor, ctx, __func__, &tensor_id, &store_offset)) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_expert_store_memset(ctx->store, tensor_id, value, store_offset + offset, size);
}

static void ggml_backend_cuda_expert_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    int tensor_id = -1;
    size_t store_offset = 0;
    if (!ggml_backend_cuda_expert_buffer_store_target(tensor, ctx, __func__, &tensor_id, &store_offset)) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        ggml_cuda_memcpy_h2d_async((char *) tensor->data + offset, data, size, cudaStreamPerThread);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);
    ggml_expert_store_write(ctx->store, tensor_id, data, store_offset + offset, size);
}

static void ggml_backend_cuda_expert_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    int tensor_id = -1;
    size_t store_offset = 0;
    if (!ggml_backend_cuda_expert_buffer_store_target(tensor, ctx, __func__, &tensor_id, &store_offset)) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);
    ggml_expert_store_read(ctx->store, tensor_id, data, store_offset + offset, size);
}

static bool ggml_backend_cuda_expert_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_cuda_expert_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    if (ctx->store != nullptr) {
        for (size_t i = 0; i < ctx->tensors.size(); ++i) {
            ggml_expert_store_memset(ctx->store, (int) i, value, 0, (size_t) ggml_nbytes(ctx->tensors[i].tensor));
        }
    }
    if (ctx->plain_dev != nullptr) {
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemsetAsync(ctx->plain_dev, value, ctx->plain_size, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static const ggml_backend_buffer_i ggml_backend_cuda_expert_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_expert_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_expert_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_expert_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_expert_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_expert_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_expert_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cuda_expert_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_expert_buffer_clear,
    /* .reset           = */ NULL,
};

static size_t ggml_backend_cuda_expert_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_cuda_expert_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_expert_tensor_is_expert(tensor)) {
        // the store keeps its experts at the padded stride (data + row padding, aligned), so
        // the size reported here matches what alloc_buffer_n actually reserves per expert
        const size_t blob_pad = ggml_cuda_expert_blob_pad(tensor->type, tensor->ne[0]);
        return (size_t) tensor->ne[2] * ggml_expert_store_stride((size_t) tensor->nb[2], blob_pad, ggml_type_size(tensor->type));
    }
    // plain tensor: the size the regular CUDA buffer type would allocate
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    return ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(buft_ctx->device), tensor);
}

static size_t ggml_backend_cuda_expert_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        if (tensors[i]->view_src != nullptr) {
            continue; // views do not allocate, they share the memory of their source
        }
        total += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensors[i]), alignment);
    }
    return total;
}

const char * ggml_backend_cuda_expert_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_cuda_expert_buft_is_expert(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_cuda_expert_buffer_type_name;
}

bool ggml_cuda_expert_is_store(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr || !ggml_cuda_expert_buft_is_expert(t->buffer->buft)) {
        return false;
    }
    // tensor-level test: only tensors registered in the store are read through the pointer
    // table, plain tensors of the same buffer type take the generic path. A fused gate weight
    // therefore cannot hit the store path just because it shares the buffer type.
    ggml_cuda_expert_table table;
    return ggml_cuda_expert_table_get(t, &table);
}

static ggml_backend_buffer_t ggml_backend_cuda_expert_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // a plain size request carries no tensors to place: allocate ordinary device memory,
    // like the regular CUDA buffer type. Expert placement goes through the tensor list.
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    const int device = buft_ctx->device;
    ggml_cuda_set_device(device);

    void * dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_expert_buffer_context * ctx = new ggml_backend_cuda_expert_buffer_context();
    ctx->device     = device;
    ctx->plain_dev  = dev_ptr;
    ctx->plain_size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_expert_buffer_interface, ctx, size);
}

static ggml_backend_buffer_t ggml_backend_cuda_expert_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    const int device = buft_ctx->device;
    ggml_cuda_set_device(device);

    if (n_tensors == 0) {
        return nullptr;
    }

    // The buffer type accepts any tensor: expert tensors go into the tiered store (VRAM
    // slots plus a device-readable host arena), every other tensor is plain device memory
    // placed like the regular CUDA buffer type does. This keeps test sentinels and mixed
    // contexts working, and the store never has to rely on the tensor name alone.
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    size_t expert_bytes = 0; // bytes of all expert tensors: the basis of the per-tensor slot quota
    size_t plain_size   = 0;
    std::vector<size_t> plain_offset((size_t) n_tensors, 0);
    std::vector<bool>   plain_alloc ((size_t) n_tensors, false);
    std::vector<ggml_tensor *> expert_tensors;

    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = tensors[i];
        if (t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        if (ggml_cuda_expert_tensor_is_expert(t)) {
            // data bytes plus the per-expert row padding: the same span get_alloc_size and
            // the store use, so the reported buffer size matches the real allocation
            const size_t blob_pad = ggml_cuda_expert_blob_pad(t->type, t->ne[0]);
            expert_bytes += (size_t) t->ne[2] * ggml_expert_store_stride((size_t) t->nb[2], blob_pad, ggml_type_size(t->type));
            expert_tensors.push_back(t);
        } else {
            plain_offset[i] = plain_size;
            plain_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
            plain_alloc[i] = true;
        }
    }

    // VRAM budgets decided by fit (part of its device memory accounting): slot_bytes for the
    // resident expert slots, stage_bytes for the per-stream prefill staging area. fit refuses
    // a configuration whose staging cannot hold the host complement of one tensor, so a load
    // that reaches this point with host experts always has a complete staging area.
    const size_t slot_budget  = ggml_cuda_expert_get_slot_budget(device);
    const size_t stage_budget = ggml_cuda_expert_get_stage_budget(device);
    if (expert_bytes > 0 && slot_budget == 0 && stage_budget == 0) {
        GGML_LOG_ERROR("%s: device %d: no expert VRAM budget was set; the host experts of a prefill batch cannot be staged, refusing the buffer\n",
            __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_expert_buffer_context * ctx = new ggml_backend_cuda_expert_buffer_context();
    ctx->device       = device;
    ctx->backend      = ggml_cuda_expert_backend(device);
    ctx->store        = ggml_expert_store_init(&ctx->backend, device, nullptr);
    ctx->stage_budget = stage_budget;

    // the placement of every expert tensor is computed by the store's pure plan function
    // before anything is allocated: fit calls the same function through the backend proc, so
    // the fit estimate and this allocation match byte for byte. The profile path is the one
    // of the store, which is already the process-wide profile the CLI set.
    std::vector<ggml_expert_store_plan_tensor> plan_in;
    plan_in.reserve(expert_tensors.size());
    for (const ggml_tensor * t : expert_tensors) {
        plan_in.push_back({ ggml_cuda_expert_tensor_layer(t), t->ne[2], ggml_cuda_expert_get_tensor_base(t),
            (size_t) t->nb[2], ggml_cuda_expert_blob_pad(t->type, t->ne[0]), ggml_type_size(t->type) });
    }
    ggml_expert_store_placement plan;
    ggml_expert_store_plan(plan_in.data(), plan_in.size(), (int64_t) slot_budget, ctx->store->profile_path.c_str(), &plan);

    // the staging area must hold the host complement of the largest tensor: fit sizes it that
    // way and reports a deficit, so a smaller budget is a configuration error
    if ((size_t) plan.host_peak > stage_budget) {
        GGML_LOG_ERROR("%s: device %d: staging budget %.2f MiB is smaller than the host complement %.2f MiB of one tensor, refusing the buffer\n",
            __func__, device, stage_budget / 1024.0 / 1024.0, plan.host_peak / 1024.0 / 1024.0);
        delete ctx;
        return nullptr;
    }

    // The bytes of every device are registered with the NUMA placement before the first arena of
    // this buffer exists: the placement of the machine is planned from all of them, not from this
    // device alone, and the plan fit committed is the one the arenas then follow.
    if (!ggml_cuda_expert_register_host_plan(device, (size_t) plan.slot_used, (size_t) plan.host_bytes,
            stage_budget, (size_t) plan.host_peak)) {
        delete ctx;
        return nullptr;
    }

    // invariant before the first cudaMalloc of this buffer: the bytes this buffer still has to
    // allocate (slots + staging + the plain block of the non-expert tensors of the same load)
    // must be present as free device memory right now. fit planned them from a free-memory
    // sample of the same state; if the state changed in between (a context or a workspace of
    // the runtime was built after the sample), the numbers here diverge and the cudaMalloc
    // below would fail with a bare OOM. Report need, available and the shortfall instead.
    {
        size_t free_now  = 0;
        size_t total_now = 0;
        ggml_cuda_expert_get_mem(device, &free_now, &total_now);
        const size_t need_now = (size_t) plan.slot_used + stage_budget + plain_size;
        GGML_LOG_DEBUG("%s: device %d: expert buffer needs %.2f MiB (slots %.2f + staging %.2f + dense %.2f), %.2f MiB free of %.2f MiB\n",
            __func__, device, need_now / 1024.0 / 1024.0, plan.slot_used / 1024.0 / 1024.0,
            stage_budget / 1024.0 / 1024.0, plain_size / 1024.0 / 1024.0,
            free_now / 1024.0 / 1024.0, total_now / 1024.0 / 1024.0);
        if (need_now > free_now) {
            GGML_LOG_ERROR("%s: device %d: the expert buffer needs %.2f MiB (slots %.2f + staging %.2f + dense %.2f), but only %.2f MiB are free; short by %.2f MiB. The fit budget was planned from a different free-memory state.\n",
                __func__, device, need_now / 1024.0 / 1024.0, plan.slot_used / 1024.0 / 1024.0,
                stage_budget / 1024.0 / 1024.0, plain_size / 1024.0 / 1024.0,
                free_now / 1024.0 / 1024.0, (need_now - free_now) / 1024.0 / 1024.0);
            delete ctx;
            return nullptr;
        }
    }

    CUDA_CHECK(cudaMalloc((void **) &ctx->usage_dev, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx->usage_dev, 0, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **) &ctx->usage_host, 4 * sizeof(unsigned long long), cudaHostAllocDefault));
    memset(ctx->usage_host, 0, 4 * sizeof(unsigned long long));
    CUDA_CHECK(cudaEventCreateWithFlags(&ctx->usage_event, cudaEventDisableTiming));

    ctx->tensors.reserve((size_t) n_tensors);

    // expert tensors first: their data pointers must be set before any view of them attaches
    for (size_t k = 0; k < expert_tensors.size(); ++k) {
        ggml_tensor * t = expert_tensors[k];
        const int64_t n_expert = t->ne[2];
        const size_t blob_bytes = t->nb[2];
        const size_t blob_pad   = ggml_cuda_expert_blob_pad(t->type, t->ne[0]);

        const int tensor_id = ggml_expert_store_add_tensor(ctx->store,
            ggml_cuda_expert_tensor_layer(t), ggml_cuda_expert_tensor_kind(t),
            ggml_cuda_expert_get_tensor_base(t),
            n_expert, blob_bytes, blob_pad, ggml_type_size(t->type),
            plan.slot_experts[k].data(), plan.n_slot[k],
            -1); // the complement must always fit in host memory
        ggml_cuda_expert_clear_tensor_base(t);
        if (tensor_id < 0) {
            GGML_LOG_ERROR("%s: cannot place expert tensor %s\n", __func__, t->name);
            delete ctx;
            return nullptr;
        }

        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);

        // poison the freshly allocated tiers with zeros so reading an unfilled slot or host
        // entry is a deterministic error instead of stale bytes (Strata A1). This also clears
        // the row padding of every expert: only the data bytes are copied in later, so the
        // kernels read zeros past the last row instead of a neighbour or uninitialized bytes.
        if (st->slots_dev != nullptr) {
            CUDA_CHECK(cudaMemset(st->slots_dev, 0, (size_t) st->n_slot * st->stride));
        }
        if (st->host_host != nullptr) {
            memset(st->host_host, 0, (size_t) st->host_used);
        }

        ggml_cuda_expert_tensor_state s;
        s.tensor_id    = tensor_id;
        s.tensor       = t;
        s.n_expert     = n_expert;
        s.blob_bytes   = blob_bytes;
        s.blob_stride  = st->stride;
        s.has_host     = st->n_host > 0;

        CUDA_CHECK(cudaMalloc((void **) &s.table_main_dev, (size_t) n_expert * sizeof(char *)));
        CUDA_CHECK(cudaMemcpy(s.table_main_dev, st->table_host, (size_t) n_expert * sizeof(char *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **) &s.residency_dev, (size_t) n_expert * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(s.residency_dev, st->residency.data(), (size_t) n_expert * sizeof(int32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &s.host_offset_dev, (size_t) n_expert * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(s.host_offset_dev, st->host_offset.data(), (size_t) n_expert * sizeof(int64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &s.counts_dev, (size_t) n_expert * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(s.counts_dev, 0, (size_t) n_expert * sizeof(uint32_t)));

        // the staging area only helps the MMQ prefill path, only when host experts exist
        s.has_stage = s.has_host && stage_budget > 0;

        // the tensor data pointer is not used by the kernels on this path; point it at one
        // of the real allocations so generic ggml code never sees a dangling pointer
        t->data = st->host_dev != nullptr ? (void *) st->host_dev : (void *) st->slots_dev;

        ggml_expert_store_set_tensor_id(ctx->store, t, tensor_id);
        ctx->tensors.push_back(std::move(s));

        GGML_LOG_INFO("%s: device %d tensor %s: %" PRId64 " experts, slots %" PRId64 ", host %" PRId64 ", stage %s\n",
            __func__, device, t->name, n_expert, st->n_slot, st->n_host,
            s.has_stage ? "yes" : "no");
    }

    if (expert_bytes > 0) {
        // invariant: this allocation must match the plan fit computed for the same budget and
        // the same tensors byte for byte. The plan is a pure function of the two, so a mismatch
        // means the budget would be exceeded and the device runs out of memory later; fail now
        // and print both sides. No plan was recorded when the buffer is not loaded by a fit
        // (tests, direct backend use): the allocation stands on its own then.
        {
            std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
            if (g_expert_plan_valid[device] &&
                ((size_t) ctx->store->slot_used != g_expert_plan_slot[device] ||
                 (size_t) ctx->store->host_used != g_expert_plan_host[device])) {
                GGML_ABORT("%s: device %d: expert placement (slots %.2f MiB, host %.2f MiB) differs from the fit plan"
                    " (slots %.2f MiB, host %.2f MiB) for slot budget %.2f MiB\n",
                    __func__, device,
                    (double) ctx->store->slot_used / 1024.0 / 1024.0, (double) ctx->store->host_used / 1024.0 / 1024.0,
                    (double) g_expert_plan_slot[device] / 1024.0 / 1024.0, (double) g_expert_plan_host[device] / 1024.0 / 1024.0,
                    slot_budget / 1024.0 / 1024.0);
            }
            // one allocation per budget: a second buffer of the same device must record its own
            // plan before it can be checked
            g_expert_plan_valid[device] = false;
        }

        ggml_expert_store_print_plan(ctx->store);
        ggml_cuda_expert_set_allocation(device, (size_t) ctx->store->slot_used, (size_t) ctx->store->host_used,
            stage_budget, (size_t) plan.host_peak);

        // the staging area is sized for the host complement of the largest tensor; fit refuses
        // a configuration whose staging cannot hold it (see common_params_fit_impl)
        GGML_LOG_INFO("%s: device %d: expert store: slots %.2f MiB, host complement %.2f MiB, staging %.2f MiB\n",
            __func__, device, plan.slot_used / 1024.0 / 1024.0, plan.host_bytes / 1024.0 / 1024.0,
            stage_budget / 1024.0 / 1024.0);

        // host arena placement: the node of every device and the bytes it reads there, one line
        // per device plus the summary with the estimated step time
        ggml_cuda_numa_log_placement("expert host placement");
    }

    // plain tensors: one block of device memory, laid out like the regular CUDA buffer type
    if (plain_size > 0) {
        size_t free_before = 0;
        size_t total       = 0;
        ggml_cuda_expert_get_mem(device, &free_before, &total);
        CUDA_CHECK(cudaMalloc(&ctx->plain_dev, plain_size));
        GGML_LOG_DEBUG("%s: device %d: dense block cudaMalloc %.2f MiB, free %.2f -> %.2f MiB of %.2f MiB\n",
            __func__, device, plain_size / 1024.0 / 1024.0,
            free_before / 1024.0 / 1024.0, (free_before - plain_size) / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
        ctx->plain_size = plain_size;
        for (int i = 0; i < n_tensors; ++i) {
            if (!plain_alloc[i]) {
                continue;
            }
            tensors[i]->data = (char *) ctx->plain_dev + plain_offset[i];
        }
    }

    // the buffer has to exist before the views of its tensors can attach to it
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_expert_buffer_interface, ctx,
        expert_bytes + plain_size);
    if (buffer == nullptr) {
        delete ctx;
        return nullptr;
    }

    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = tensors[i];
        if (t->view_src != nullptr || t->data == nullptr || t->buffer != nullptr) {
            continue;
        }
        t->buffer = buffer;
        if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: cannot initialize tensor %s\n", __func__, t->name);
            ggml_backend_buffer_free(buffer);
            return nullptr;
        }
    }

    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = tensors[i];
        if (t->view_src == nullptr || t->buffer != nullptr) {
            continue;
        }
        if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: cannot initialize view %s\n", __func__, t->name);
            ggml_backend_buffer_free(buffer);
            return nullptr;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        g_expert_buffers.push_back(ctx);
    }

    return buffer;
}

extern const ggml_backend_buffer_type_i ggml_backend_cuda_expert_buffer_type_interface;
const ggml_backend_buffer_type_i ggml_backend_cuda_expert_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_expert_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_expert_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_cuda_expert_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_cuda_expert_buft_get_alignment,
    /* .get_max_size        = */ NULL,
    /* .get_alloc_size      = */ ggml_backend_cuda_expert_buft_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_cuda_expert_buft_get_alloc_size_n,
    /* .is_host             = */ NULL, // device buffer type: ops using it stay on the CUDA backend
};

// startup byte check: the loader wrote the tensor data through set_tensor before this call,
// so the sampled experts of every tensor are compared now, before any graph runs. A mismatch
// means the placement is wrong and the results would be silently wrong too.
void ggml_backend_cuda_expert_verify_all(void) {
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        ggml_cuda_set_device(ctx->device);
        for (ggml_cuda_expert_tensor_state & s : ctx->tensors) {
            if (s.verify_done) {
                continue;
            }
            s.verify_done = true;
            if (!ggml_expert_store_verify(ctx->store, s.tensor_id)) {
                GGML_ABORT("%s: expert store byte check failed for tensor %s\n",
                    __func__, s.tensor != nullptr ? s.tensor->name : "?");
            }
        }
    }

    // every arena of the model is allocated now: report the placement they got, the bytes and
    // the pages move_pages confirmed, and whether it is the placement fit committed
    ggml_cuda_numa_log_final_placement();
}

// pure placement plan for the fit code: the same store function that alloc_buffer_n uses, so
// the fit estimate and the real allocation match byte for byte. The tensors arrive as
// parallel arrays to keep the interface free of internal store types. host_bytes_each, when
// not null, receives the host complement bytes of every tensor in allocation order (the fit
// code checks their NUMA placement against the nodes).
void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each) {
    if (buft == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }

    std::vector<ggml_expert_store_plan_tensor> tensors;
    tensors.reserve(n_tensors);
    for (size_t i = 0; i < n_tensors; ++i) {
        // the fit plan must use the same per-expert stride the real allocation uses, so the
        // row padding is derived here from the tensor type exactly like alloc_buffer_n does
        const size_t blob_pad = ggml_cuda_expert_blob_pad((ggml_type) type[i], ne0[i]);
        tensors.push_back({ layer[i], n_expert[i], expert_base != nullptr ? expert_base[i] : 0, blob_bytes[i], blob_pad,
            ggml_type_size((ggml_type) type[i]) });
    }

    ggml_expert_store_placement plan;
    ggml_expert_store_plan(tensors.data(), tensors.size(), (int64_t) slot_budget, nullptr, &plan);

    if (slot_bytes != nullptr) {
        *slot_bytes = (size_t) plan.slot_used;
    }
    if (host_bytes != nullptr) {
        *host_bytes = (size_t) plan.host_bytes;
    }
    if (host_peak != nullptr) {
        *host_peak = (size_t) plan.host_peak;
    }
    if (host_bytes_each != nullptr) {
        for (size_t i = 0; i < tensors.size(); ++i) {
            host_bytes_each[i] = (size_t) plan.host_used[i];
        }
    }
}

void ggml_cuda_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes) {
    if (buft == nullptr || buft->context == nullptr) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    ggml_cuda_expert_set_budget_impl(buft_ctx->device, slot_bytes, stage_bytes);
}

void ggml_cuda_expert_record_plan(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t host_bytes) {
    if (buft == nullptr || buft->context == nullptr) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    ggml_cuda_expert_set_plan_impl(buft_ctx->device, slot_bytes, host_bytes);
}

void ggml_cuda_expert_set_slot_limit(ggml_backend_buffer_type_t buft, size_t bytes) {
    if (buft == nullptr || buft->context == nullptr) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    ggml_cuda_expert_set_slot_limit_impl(buft_ctx->device, bytes);
}

bool ggml_backend_cuda_expert_get_allocation(
        ggml_backend_buffer_type_t buft, size_t * slot_bytes, size_t * host_bytes, size_t * stage_bytes) {
    if (buft == nullptr || buft->context == nullptr) {
        return false;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    const int device = buft_ctx->device;
    if (g_expert_alloc_slot[device] == 0 && g_expert_alloc_host[device] == 0 && g_expert_alloc_stage[device] == 0) {
        return false;
    }
    if (slot_bytes != nullptr) {
        *slot_bytes = g_expert_alloc_slot[device];
    }
    if (host_bytes != nullptr) {
        *host_bytes = g_expert_alloc_host[device];
    }
    if (stage_bytes != nullptr) {
        *stage_bytes = g_expert_alloc_stage[device];
    }
    return true;
}

bool ggml_cuda_expert_supported(void) {
    return true;
}

bool ggml_backend_cuda_expert_get_host_peak(ggml_backend_buffer_type_t buft, size_t * peak_bytes) {
    if (buft == nullptr || buft->context == nullptr) {
        return false;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    const int device = buft_ctx->device;
    if (g_expert_alloc_host_peak[device] == 0 && g_expert_alloc_slot[device] == 0 && g_expert_alloc_host[device] == 0) {
        return false;
    }
    if (peak_bytes != nullptr) {
        *peak_bytes = g_expert_alloc_host_peak[device];
    }
    return true;
}

#else // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)

// HIP/MUSA: the expert store is not implemented on these backends. Provide stubs with the
// same signatures so the rest of the CUDA backend compiles unchanged; the buffer type
// reports unsupported and is never registered.

static const char * ggml_backend_cuda_expert_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "Experts(unsupported)";
}

extern const ggml_backend_buffer_type_i ggml_backend_cuda_expert_buffer_type_interface;
const ggml_backend_buffer_type_i ggml_backend_cuda_expert_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_expert_buffer_type_name,
    /* .alloc_buffer        = */ NULL,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ NULL,
    /* .get_max_size        = */ NULL,
    /* .get_alloc_size      = */ NULL,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

bool ggml_cuda_expert_supported(void) {
    return false;
}

bool ggml_cuda_expert_buft_is_expert(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

void ggml_cuda_expert_mark_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * t) {
    GGML_UNUSED(buft);
    GGML_UNUSED(t);
}

bool ggml_cuda_expert_is_store(const ggml_tensor * t) {
    GGML_UNUSED(t);
    return false;
}

bool ggml_cuda_expert_table_get(const ggml_tensor * src0, ggml_cuda_expert_table * table) {
    GGML_UNUSED(src0);
    GGML_UNUSED(table);
    return false;
}

const char * const * ggml_cuda_expert_stage_prepare(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(expert_offset);
    GGML_UNUSED(stream);
    return nullptr;
}

const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(expert_offset);
    GGML_UNUSED(stream);
    return nullptr;
}

void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(expert_offset);
    GGML_UNUSED(stream);
}

void ggml_backend_cuda_expert_verify_all(void) {
}

void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(stream);
}

void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each) {
    GGML_UNUSED(buft);
    GGML_UNUSED(layer);
    GGML_UNUSED(expert_base);
    GGML_UNUSED(n_expert);    GGML_UNUSED(blob_bytes);
    GGML_UNUSED(type);
    GGML_UNUSED(ne0);
    GGML_UNUSED(n_tensors);
    GGML_UNUSED(slot_budget);
    GGML_UNUSED(slot_bytes);
    GGML_UNUSED(host_bytes);
    GGML_UNUSED(host_peak);
    GGML_UNUSED(host_bytes_each);
}

void ggml_cuda_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes) {
    GGML_UNUSED(buft);
    GGML_UNUSED(slot_bytes);
    GGML_UNUSED(stage_bytes);
}

void ggml_cuda_expert_set_tensor_base(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor, int64_t expert_base) {
    GGML_UNUSED(buft);
    GGML_UNUSED(tensor);
    GGML_UNUSED(expert_base);
}

void ggml_cuda_expert_record_plan(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t host_bytes) {
    GGML_UNUSED(buft);
    GGML_UNUSED(slot_bytes);
    GGML_UNUSED(host_bytes);
}

void ggml_cuda_expert_set_slot_limit(ggml_backend_buffer_type_t buft, size_t bytes) {
    GGML_UNUSED(buft);
    GGML_UNUSED(bytes);
}

bool ggml_backend_cuda_expert_get_allocation(
        ggml_backend_buffer_type_t buft, size_t * slot_bytes, size_t * host_bytes, size_t * stage_bytes) {
    GGML_UNUSED(buft);
    GGML_UNUSED(slot_bytes);
    GGML_UNUSED(host_bytes);
    GGML_UNUSED(stage_bytes);
    return false;
}

bool ggml_backend_cuda_expert_get_host_peak(ggml_backend_buffer_type_t buft, size_t * peak_bytes) {
    GGML_UNUSED(buft);
    GGML_UNUSED(peak_bytes);
    return false;
}

void ggml_cuda_expert_reset_allocation(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
}

#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)


