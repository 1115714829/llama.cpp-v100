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

static void * ggml_cuda_expert_alloc_device(void * user, size_t size) {
    const int device = (int) (intptr_t) user;
    ggml_cuda_set_device(device);
    void * ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
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

    // the host arena follows the same NUMA placement as the CUDA<n>_Mapped buffer type:
    // reserve picks the best measured node of the device; when that node cannot hold the
    // arena, the next best node with room takes it, so the arena spills in bandwidth order
    const int node = ggml_cuda_numa_reserve(device, size);
    ggml_cuda_set_device(device); // the probe may have left another device current

    void * host_ptr = ggml_cuda_numa_host_alloc(size, node);
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
    CUDA_CHECK(cudaMemcpy(dst_dev, src_host, size, cudaMemcpyHostToDevice));
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

// shape and type of a tensor that the store can place. The name is not part of this check:
// the caller may mark a tensor explicitly (GGML_TENSOR_FLAG_EXPERT), and the llama name
// pattern is only a fallback for tensors that arrive without a mark.
static bool ggml_cuda_expert_layout_ok(const ggml_tensor * t) {
    if (t == nullptr || t->view_src != nullptr || t->op != GGML_OP_NONE) {
        return false;
    }
    if (t->ne[2] <= 1) {
        return false;
    }
    return ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
}

static bool ggml_cuda_expert_tensor_is_expert(const ggml_tensor * t) {
    if (!ggml_cuda_expert_layout_ok(t)) {
        return false;
    }
    // marked by the caller, or named like the llama MoE expert weights blk.<l>.<name>_exps.weight
    return (t->flags & GGML_TENSOR_FLAG_EXPERT) != 0 || strstr(t->name, "_exps.weight") != nullptr;
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
static __global__ void expert_routing_kernel(
        const int32_t * __restrict__ ids, const int64_t n_entries, const int expert_offset,
        const int n_local, const int32_t * __restrict__ residency, const int64_t host_bytes,
        unsigned long long * __restrict__ counters, uint32_t * __restrict__ counts,
        uint32_t * __restrict__ bitmap_or_null) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) {
        return;
    }
    const int e = ids[i] - expert_offset;
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
// else. The ranges use the global expert numbers of the store; the caller shifts the table
// pointer for a view (expert_start).
static __global__ void expert_stage_plan_kernel(
        const uint32_t * __restrict__ bitmap, const int expert_lo, const int expert_hi,
        const int32_t * __restrict__ residency, const int stage_cap,
        const char * const * __restrict__ table_main, const char ** __restrict__ table_stage,
        char * __restrict__ stage_dev, const size_t blob_aligned, const size_t blob_bytes,
        int32_t * __restrict__ hit_list, int * __restrict__ n_hit, int * __restrict__ n_overflow,
        int * __restrict__ slot_counter, unsigned long long * __restrict__ counters) {
    const int stride = gridDim.x * blockDim.x;
    for (int x = expert_lo + (int) (blockIdx.x * blockDim.x + threadIdx.x); x < expert_hi; x += stride) {
        const bool routed = ((bitmap[x >> 5] >> (x & 31)) & 1u) != 0;
        if (routed && residency[x] < 0) {
            atomicAdd(n_hit, 1);
            const int slot = atomicAdd(slot_counter, 1);
            if (slot < stage_cap) {
                hit_list[slot] = x;
                table_stage[x] = stage_dev + (size_t) slot * blob_aligned;
                atomicAdd(&counters[3], (unsigned long long) blob_bytes);
            } else {
                atomicAdd(n_overflow, 1);
                table_stage[x] = table_main[x];
            }
        } else {
            table_stage[x] = table_main[x];
        }
    }
}

// Copies the routed host experts from the mapped host arena into their staging slots.
// One block per hit entry, grid covers n_local (blocks beyond n_hit return at once).
static __global__ void expert_stage_copy_kernel(
        const int32_t * __restrict__ hit_list, const int * __restrict__ n_hit, const int stage_cap,
        const char * __restrict__ host_dev, const int64_t * __restrict__ host_offset,
        char * __restrict__ stage_dev, const size_t blob_aligned, const size_t blob_bytes) {
    const int slot = blockIdx.x;
    if (slot >= *n_hit || slot >= stage_cap) {
        return;
    }
    const int64_t expert = hit_list[slot];
    // the arena offsets and the slot pitch are 256-byte aligned
    const uint64_t * src64 = (const uint64_t *) (host_dev + host_offset[expert]);
    uint64_t * dst64 = (uint64_t *) (stage_dev + (size_t) slot * blob_aligned);

    const size_t n64 = blob_bytes / 8;
    for (size_t i = threadIdx.x; i < n64; i += blockDim.x) {
        dst64[i] = src64[i];
    }
    const char * src = (const char *) src64;
    char * dst = (char *) dst64;
    for (size_t i = n64 * 8 + threadIdx.x; i < blob_bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// per-tensor runtime state
struct ggml_cuda_expert_tensor_state {
    int tensor_id = -1;
    const ggml_tensor * tensor = nullptr;

    int64_t n_expert = 0;
    size_t  blob_bytes = 0;
    size_t  blob_aligned = 0; // staging slot pitch
    bool    has_host = false;

    const char ** table_main_dev = nullptr;
    int32_t * residency_dev   = nullptr;
    int64_t * host_offset_dev = nullptr; // device copy of the host arena byte offsets
    uint32_t * counts_dev     = nullptr; // per-expert routing counts, for --dump-expert-routing
    uint32_t * bitmap_dev     = nullptr; // n_expert bits, only allocated when staging applies
    uint32_t * bitmap_pinned  = nullptr; // async readback of the last bitmap, for the logs

    bool verify_done = false;
};

// per-stream staging area: contexts sharing the weights may run on different streams, each
// stream owns its block so no cross-stream synchronization is needed
struct ggml_cuda_expert_stream_state {
    char * stage_dev   = nullptr;
    size_t stage_bytes = 0; // from fit: the largest host complement of one expert tensor

    // per tensor: device pointer table through the staging area, and the copy plan buffers
    std::vector<const char **> tables;
    int32_t *              hit_list   = nullptr;
    int *                  plan_state = nullptr; // [n_hit, n_overflow, slot_counter]
    int *                  plan_pinned = nullptr; // async readback of the last plan, logs only
    size_t                 hit_cap    = 0;       // == largest n_local of the tensors
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

static void ggml_cuda_expert_set_budget_impl(int device, size_t slot_bytes, size_t stage_bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_slot_budget [device] = slot_bytes;
    g_expert_stage_budget[device] = stage_bytes;
}

static size_t ggml_cuda_expert_get_slot_budget(int device) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    return g_expert_slot_budget[device];
}

static size_t ggml_cuda_expert_get_stage_budget(int device) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    return g_expert_stage_budget[device];
}

static void ggml_cuda_expert_set_allocation(int device, size_t slot_bytes, size_t host_bytes, size_t stage_bytes, size_t host_peak) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_alloc_slot [device] = slot_bytes;
    g_expert_alloc_host [device] = host_bytes;
    g_expert_alloc_stage[device] = stage_bytes;
    g_expert_alloc_host_peak[device] = host_peak;
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
    table->has_stage    = s.bitmap_dev != nullptr;
    table->tensor_id    = tensor_id;
    table->expert_start = expert_start;
    return true;
}

ggml_cuda_expert_stream_state & ggml_backend_cuda_expert_buffer_context::get_stream(cudaStream_t stream) {
    const auto it = streams.find(stream);
    if (it != streams.end()) {
        return it->second;
    }

    ggml_cuda_set_device(device);

    ggml_cuda_expert_stream_state ss;
    if (stage_budget > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.stage_dev, stage_budget));
        ss.stage_bytes = stage_budget;
    }

    for (const ggml_cuda_expert_tensor_state & s : tensors) {
        ss.hit_cap = std::max(ss.hit_cap, (size_t) s.n_expert);
    }
    if (ss.hit_cap > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.hit_list, ss.hit_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &ss.plan_state, 3 * sizeof(int)));
        CUDA_CHECK(cudaMemset(ss.plan_state, 0, 3 * sizeof(int)));
        CUDA_CHECK(cudaHostAlloc((void **) &ss.plan_pinned, 3 * sizeof(int), cudaHostAllocDefault));
        memset(ss.plan_pinned, 0, 3 * sizeof(int));
    }
    ss.tables.assign(tensors.size(), nullptr);

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
        for (const char ** table : ss.tables) {
            if (table != nullptr) {
                CUDA_CHECK(cudaFree(table));
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
        if (s.bitmap_dev) {
            CUDA_CHECK(cudaFree(s.bitmap_dev));
        }
        if (s.bitmap_pinned) {
            CUDA_CHECK(cudaFreeHost(s.bitmap_pinned));
        }
    }
    if (usage_dev) {
        CUDA_CHECK(cudaFree(usage_dev));
    }
    if (usage_host) {
        CUDA_CHECK(cudaFreeHost(usage_host));
    }
    if (plain_dev) {
        CUDA_CHECK(cudaFree(plain_dev));
    }
    ggml_expert_store_free(store);
}

// records the tier hits and the per-expert routing counts of one mul_mat_id launch. The
// per-expert counts use the view-local expert numbers of the tensor, so a view of a store
// tensor counts into its own range of the full count array.
void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, int expert_offset, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);
    if (ctx->usage_dev == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
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
    expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
        (const int32_t *) ids->data, n_entries, (int) (expert_offset + ex_table.expert_start), (int) ex_table.n_expert,
        s.residency_dev + ex_table.expert_start, (int64_t) ex_table.blob_bytes,
        ctx->usage_dev, s.counts_dev + ex_table.expert_start, nullptr);
}

static void ggml_cuda_expert_stats_readback(ggml_backend_cuda_expert_buffer_context * ctx, cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusActive) {
        CUDA_CHECK(cudaMemcpy(ctx->usage_host, ctx->usage_dev, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        const unsigned long long total = ctx->usage_host[0] + ctx->usage_host[1];
        if (total > 0) {
            GGML_LOG_DEBUG("%s: device %d: expert hits vram %llu / host %llu (%.2f%% vram), host read %.2f MiB, staged %.2f MiB\n",
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
    if (s.bitmap_pinned == nullptr || s.n_expert <= 0) {
        return;
    }
    const int n_words = (int) ((s.n_expert + 31) / 32);
    int n_routed = 0;
    for (int i = 0; i < n_words; ++i) {
        uint32_t v = s.bitmap_pinned[i];
        while (v != 0) {
            n_routed += (int) (v & 1u);
            v >>= 1;
        }
    }
    GGML_LOG_DEBUG("%s: device %d tensor %s: last batch routed %d/%" PRId64 " experts\n",
        __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?", n_routed, s.n_expert);

    // staging plan of the last prefill batch of this stream: how many routed host experts
    // were staged and how many overflowed to a direct host read (async readback, logs only)
    const auto it = ctx->streams.find(stream);
    if (it == ctx->streams.end() || it->second.plan_pinned == nullptr) {
        return;
    }
    const int n_hit      = it->second.plan_pinned[0];
    const int n_overflow = it->second.plan_pinned[1];
    const int n_staged   = n_hit - n_overflow;
    GGML_LOG_DEBUG("%s: device %d tensor %s: staging last batch: %d/%d host experts staged (%.2f MiB), %d overflow to direct host read\n",
        __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?",
        n_staged, n_hit, (double) (n_staged * (int64_t) s.blob_bytes) / 1024.0 / 1024.0, n_overflow);
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

    if (s.bitmap_dev == nullptr || ctx->stage_budget == 0 || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return s.table_main_dev + ex_table.expert_start;
    }

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
    if (ss.hit_list == nullptr || ss.stage_bytes < s.blob_aligned) {
        return s.table_main_dev + ex_table.expert_start;
    }

    const char ** table_stage = ss.tables[(size_t) ex_table.tensor_id];
    if (table_stage == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &table_stage, (size_t) s.n_expert * sizeof(char *)));
        ss.tables[(size_t) ex_table.tensor_id] = table_stage;
    }

    ggml_cuda_set_device(ctx->device);

    const int nthreads = 256;
    const int64_t n_entries = ggml_nelements(ids);
    const int n_words = (int) ((s.n_expert + 31) / 32);
    // the block is sized for the largest host complement of one tensor: the worst tensor of
    // the device fits completely, smaller tensors fit with room to spare. Beyond stage_cap
    // routed host experts keep their main table entry and are read directly from host memory.
    GGML_ASSERT(ss.hit_list != nullptr && ss.hit_cap >= (size_t) s.n_expert);
    const int stage_cap = (int) std::min<size_t>(ss.stage_bytes / s.blob_aligned, ss.hit_cap);
    char * block_dev = ss.stage_dev;

    // [n_hit, n_overflow, slot_counter]
    CUDA_CHECK(cudaMemsetAsync(ss.plan_state, 0, 3 * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(s.bitmap_dev, 0, (size_t) n_words * sizeof(uint32_t), stream));

    if (n_entries > 0) {
        const int nblocks = (int) ((n_entries + nthreads - 1) / nthreads);
        // counts == nullptr: the routing statistics of this launch were recorded separately
        expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
            (const int32_t *) ids->data, n_entries, expert_offset, (int) s.n_expert,
            s.residency_dev, (int64_t) s.blob_bytes, ctx->usage_dev, nullptr, s.bitmap_dev);
    }

    const int expert_lo = (int) ex_table.expert_start;
    const int expert_hi = expert_lo + (int) ex_table.n_expert;
    const int plan_blocks = std::max(1, std::min((expert_hi - expert_lo + nthreads - 1) / nthreads, 1024));
    expert_stage_plan_kernel<<<plan_blocks, nthreads, 0, stream>>>(
        s.bitmap_dev, expert_lo, expert_hi, s.residency_dev, stage_cap,
        s.table_main_dev, table_stage, block_dev, s.blob_aligned, s.blob_bytes,
        ss.hit_list, ss.plan_state, ss.plan_state + 1, ss.plan_state + 2, ctx->usage_dev);

    expert_stage_copy_kernel<<<expert_hi - expert_lo, nthreads, 0, stream>>>(
        ss.hit_list, ss.plan_state, stage_cap, st->host_dev, s.host_offset_dev,
        block_dev, s.blob_aligned, s.blob_bytes);
    CUDA_CHECK(cudaGetLastError());

    // read the bitmap and the copy plan back without waiting: the log shows how many experts
    // the last batches staged and how many overflowed to a direct host read
    CUDA_CHECK(cudaMemcpyAsync(s.bitmap_pinned, s.bitmap_dev, (size_t) n_words * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ss.plan_pinned, ss.plan_state, 3 * sizeof(int),
        cudaMemcpyDeviceToHost, stream));

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

static void ggml_backend_cuda_expert_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    const int tensor_id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, tensor) : -1;
    if (tensor_id < 0) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_expert_store_memset(ctx->store, tensor_id, value, offset, size);
}

static void ggml_backend_cuda_expert_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    const int tensor_id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, tensor) : -1;
    if (tensor_id < 0) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);
    ggml_expert_store_write(ctx->store, tensor_id, data, offset, size);
}

static void ggml_backend_cuda_expert_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    const int tensor_id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, tensor) : -1;
    if (tensor_id < 0) {
        // plain tensor: same behavior as the regular CUDA buffer type
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);
    ggml_expert_store_read(ctx->store, tensor_id, data, offset, size);
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
        return ggml_nbytes(tensor);
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
            expert_bytes += ggml_nbytes(t);
            expert_tensors.push_back(t);
        } else {
            plain_offset[i] = plain_size;
            plain_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
            plain_alloc[i] = true;
        }
    }

    // VRAM budgets decided by fit (part of its device memory accounting): slot_bytes for the
    // resident expert slots, stage_bytes for the per-stream prefill staging area. Without a
    // budget the experts stay in the host arena, which is the safe fallback (the mapped
    // buffer type behavior) instead of guessing a share of the currently free memory.
    const size_t slot_budget  = ggml_cuda_expert_get_slot_budget(device);
    const size_t stage_budget = ggml_cuda_expert_get_stage_budget(device);
    if (expert_bytes > 0 && slot_budget == 0 && stage_budget == 0) {
        GGML_LOG_INFO("%s: device %d: no expert VRAM budget from fit, all experts stay in host memory\n",
            __func__, device);
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
        plan_in.push_back({ ggml_cuda_expert_tensor_layer(t), t->ne[2], (size_t) t->nb[2] });
    }
    ggml_expert_store_placement plan;
    ggml_expert_store_plan(plan_in.data(), plan_in.size(), (int64_t) slot_budget, ctx->store->profile_path.c_str(), &plan);

    CUDA_CHECK(cudaMalloc((void **) &ctx->usage_dev, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx->usage_dev, 0, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **) &ctx->usage_host, 4 * sizeof(unsigned long long), cudaHostAllocDefault));
    memset(ctx->usage_host, 0, 4 * sizeof(unsigned long long));

    ctx->tensors.reserve((size_t) n_tensors);

    // expert tensors first: their data pointers must be set before any view of them attaches
    for (size_t k = 0; k < expert_tensors.size(); ++k) {
        ggml_tensor * t = expert_tensors[k];
        const int64_t n_expert = t->ne[2];
        const size_t blob_bytes = t->nb[2];

        const int tensor_id = ggml_expert_store_add_tensor(ctx->store,
            ggml_cuda_expert_tensor_layer(t), ggml_cuda_expert_tensor_kind(t),
            n_expert, blob_bytes,
            plan.slot_experts[k].data(), plan.n_slot[k],
            -1); // the complement must always fit in host memory
        if (tensor_id < 0) {
            GGML_LOG_ERROR("%s: cannot place expert tensor %s\n", __func__, t->name);
            delete ctx;
            return nullptr;
        }

        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);

        // poison the freshly allocated tiers with zeros so reading an unfilled slot or host
        // entry is a deterministic error instead of stale bytes (Strata A1)
        if (st->slots_dev != nullptr) {
            const size_t blob_aligned = (blob_bytes + 255) & ~(size_t) 255;
            CUDA_CHECK(cudaMemset(st->slots_dev, 0, (size_t) st->n_slot * blob_aligned));
        }
        if (st->host_host != nullptr) {
            memset(st->host_host, 0, (size_t) st->host_used);
        }

        ggml_cuda_expert_tensor_state s;
        s.tensor_id    = tensor_id;
        s.tensor       = t;
        s.n_expert     = n_expert;
        s.blob_bytes   = blob_bytes;
        s.blob_aligned = (blob_bytes + 255) & ~(size_t) 255;
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
        if (s.has_host && stage_budget > 0) {
            const int n_words = (int) ((n_expert + 31) / 32);
            CUDA_CHECK(cudaMalloc((void **) &s.bitmap_dev, (size_t) n_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(s.bitmap_dev, 0, (size_t) n_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaHostAlloc((void **) &s.bitmap_pinned, (size_t) n_words * sizeof(uint32_t), cudaHostAllocDefault));
            memset(s.bitmap_pinned, 0, (size_t) n_words * sizeof(uint32_t));
        }

        // the tensor data pointer is not used by the kernels on this path; point it at one
        // of the real allocations so generic ggml code never sees a dangling pointer
        t->data = st->host_dev != nullptr ? (void *) st->host_dev : (void *) st->slots_dev;

        ggml_expert_store_set_tensor_id(ctx->store, t, tensor_id);
        ctx->tensors.push_back(std::move(s));

        GGML_LOG_INFO("%s: device %d tensor %s: %" PRId64 " experts, slots %" PRId64 ", host %" PRId64 ", stage %s\n",
            __func__, device, t->name, n_expert, st->n_slot, st->n_host,
            s.bitmap_dev != nullptr ? "yes" : "no");
    }

    if (expert_bytes > 0) {
        ggml_expert_store_print_plan(ctx->store);
        ggml_cuda_expert_set_allocation(device, (size_t) ctx->store->slot_used, (size_t) ctx->store->host_used,
            stage_budget, (size_t) plan.host_peak);

        // the staging area is sized for the host complement of the largest tensor; the part
        // above it is read directly from host memory during prefill
        const double direct = plan.host_peak > 0 && plan.host_peak > (int64_t) stage_budget
            ? 100.0 * (double) (plan.host_peak - (int64_t) stage_budget) / (double) plan.host_peak
            : 0.0;
        GGML_LOG_INFO("%s: device %d: expert store: slots %.2f MiB, host complement %.2f MiB, staging %.2f MiB, prefill direct-read %.0f%%\n",
            __func__, device, plan.slot_used / 1024.0 / 1024.0, plan.host_bytes / 1024.0 / 1024.0,
            stage_budget / 1024.0 / 1024.0, direct);

        // host arena placement: the NUMA nodes, the bytes on each of them and the measured
        // read bandwidth of this device for those nodes, one line per device
        ggml_cuda_numa_log_placement(device, "expert host complement");
    }

    // plain tensors: one block of device memory, laid out like the regular CUDA buffer type
    if (plain_size > 0) {
        CUDA_CHECK(cudaMalloc(&ctx->plain_dev, plain_size));
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
}

// pure placement plan for the fit code: the same store function that alloc_buffer_n uses, so
// the fit estimate and the real allocation match byte for byte. The tensors arrive as
// parallel arrays to keep the interface free of internal store types. host_bytes_each, when
// not null, receives the host complement bytes of every tensor in allocation order (the fit
// code checks their NUMA placement against the nodes).
void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * n_expert, const size_t * blob_bytes, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each) {
    if (buft == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }

    std::vector<ggml_expert_store_plan_tensor> tensors;
    tensors.reserve(n_tensors);
    for (size_t i = 0; i < n_tensors; ++i) {
        tensors.push_back({ layer[i], n_expert[i], blob_bytes[i] });
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
        const int * layer, const int64_t * n_expert, const size_t * blob_bytes, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each) {
    GGML_UNUSED(buft);
    GGML_UNUSED(layer);
    GGML_UNUSED(n_expert);
    GGML_UNUSED(blob_bytes);
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


