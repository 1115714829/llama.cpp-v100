#include "ggml-cuda-expert.cuh"

#include "common.cuh"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "../ggml-expert-store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <sys/mman.h>
#endif

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)

// CUDA implementation of the expert store memory operations. The store itself keeps the
// placement plan; this file owns the device allocations, the per-tensor device pointer
// tables and the per-stream prefill staging regions.
//
// Prefill staging (the streamed ring of Strata src/prefill/prefill.cpp, MIT,
// licenses/LICENSE-strata): the host experts of every layer of a tensor kind stream through a
// ring of expert slots, in layer and expert order, and the consumer of a layer takes the experts
// of its layer out of the ring as they arrive. The order does not depend on the routing: a batch
// of a few thousand tokens routes to almost every expert of a layer, so the experts a batch would
// not need are a rounding error, and what the independence from the routing buys is that the
// producer needs to know nothing about the routing: the slot of an expert is a static function of
// its place in the order, so the pointer table of a tensor stays static (built once per stream:
// slot experts -> slot, host experts -> their slot of the ring).
//
// Ring: one per tensor kind (ggml_expert_kind_of_name, e.g. the gate_up tensors and the down
// tensors). Its slots hold one expert each, so a ring of a kind has one slot per host expert of
// its largest layer plus one, and the experts of a layer land in distinct slots. The regions of
// the kinds are laid out back to back in the staging area fit sized for exactly that sum. The
// rings are per (device, stream): a stream may run several contexts at once while sharing the
// weights, each one needs its own set.
//
// Producer: a host thread per ring copies the arcs (the host experts of one layer) on the copy
// stream of the ring (non-blocking), one or two cudaMemcpy2DAsync per arc (the experts of the
// arc are contiguous in the host arena of the tensor, the arc can wrap around the ring). Before
// it writes an arc it waits for the release of the arc that held the slot of its last expert
// (the `arc_released` event of that arc, recorded on the compute stream after the consumers of
// that arc were queued): the copy of the experts of layer L therefore runs while the GPU is in
// the attention half of layer L, and the consumer of layer L finds its experts in place. It also
// holds every copy back until the last collective of the compute stream is done
// (ggml_cuda_expert_comm_done), so a copy never stretches a collective it would otherwise
// overlap. The producer never waits on the host for the GPU: the waits are stream events.
//
// Consumer: the launch of a layer waits for the `arc_copied` event of its arc on the compute
// stream. The copy of the arc is ordered behind the copies of the earlier arcs of the ring, so
// waiting for the last one covers the whole arc. The launch of the first layer of a block also
// records the barrier of the block, which the copy stream waits for before the copies of the
// block: a decode launch may run between two blocks as a CUDA graph replay and write the ring
// without the host knowing, and the barrier keeps the copies of the next block behind it.
//
// Decode batches (the MMVQ mmid path) keep the per-launch staging kernel: a decode batch routes
// to a few experts, copying the whole complement for it would cost the time it saves. The
// kernel copies the routed host experts into the first slots of the region of the kind and
// rewrites a second, per-launch table, with no host side state, so a decode launch stays
// recordable into a CUDA graph.











#define GGML_CUDA_EXPERT_S3PRIME_MIN_TOKENS GGML_EXPERT_S3PRIME_MIN_TOKENS



#define GGML_CUDA_EXPERT_MODE_S3P    1
#define GGML_CUDA_EXPERT_MODE_ROUTED 2





static void ggml_cuda_expert_get_mem(int device, size_t * free_bytes, size_t * total_bytes) {
    ggml_cuda_set_device(device);
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        *free_bytes  = 0;
        *total_bytes = 0;
    }
}







void ggml_backend_cuda_expert_device_ready(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    ggml_cuda_set_device(buft_ctx->device);

    CUDA_CHECK(cudaFree(nullptr));
}


struct ggml_cuda_expert_backend_user {
    int device = -1;






    char * slab      = nullptr;
    size_t slab_size = 0;
    size_t slab_used = 0;
};

static int ggml_cuda_expert_user_device(void * user) {
    return ((const ggml_cuda_expert_backend_user *) user)->device;
}




static constexpr size_t GGML_CUDA_EXPERT_SLAB_ALIGN = 4096;

static void * ggml_cuda_expert_alloc_device(void * user, size_t size) {
    ggml_cuda_expert_backend_user * u = (ggml_cuda_expert_backend_user *) user;

    ggml_cuda_set_device(u->device);
    const size_t take = GGML_PAD(size, GGML_CUDA_EXPERT_SLAB_ALIGN);
    if (u->slab == nullptr || u->slab_used + take > u->slab_size) {
        GGML_LOG_ERROR("%s: device %d: a slot array of %.2f MiB does not fit the slot block of the plan (%.2f of %.2f MiB used)\n",
            __func__, u->device, size / 1024.0 / 1024.0, u->slab_used / 1024.0 / 1024.0, u->slab_size / 1024.0 / 1024.0);
        return nullptr;
    }
    void * ptr = u->slab + u->slab_used;
    u->slab_used += take;
    GGML_LOG_DEBUG("%s: device %d: slot array %.2f MiB at %.2f MiB of the %.2f MiB slot block\n",
        __func__, u->device, size / 1024.0 / 1024.0, (u->slab_used - take) / 1024.0 / 1024.0, u->slab_size / 1024.0 / 1024.0);
    return ptr;
}

static void ggml_cuda_expert_free_device(void * user, void * ptr) {

    GGML_UNUSED(user);
    GGML_UNUSED(ptr);
}



static std::mutex g_expert_host_mutex;
static std::map<void *, std::pair<int, size_t>> g_expert_host_alloc;



static int ggml_cuda_expert_host_node(const void * host_ptr) {
    if (host_ptr == nullptr) {
        return GGML_CUDA_NUMA_NODE_ANY;
    }
    std::lock_guard<std::mutex> lock(g_expert_host_mutex);
    const auto it = g_expert_host_alloc.find(const_cast<void *>(host_ptr));
    return it != g_expert_host_alloc.end() ? it->second.first : GGML_CUDA_NUMA_NODE_ANY;
}

static void * ggml_cuda_expert_alloc_host(void * user, size_t size, void ** dev_ptr) {
    const int device = ggml_cuda_expert_user_device(user);
    ggml_cuda_set_device(device);




    const int node = ggml_cuda_numa_reserve(device, size);
    ggml_cuda_set_device(device);

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



    ggml_cuda_host_pool_add(GGML_CUDA_HOST_POOL_EXPERT, node, size);

    return host_ptr;
}

static void ggml_cuda_expert_free_host(void * user, void * ptr) {
    const int device = ggml_cuda_expert_user_device(user);
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

    ggml_cuda_host_pool_sub(GGML_CUDA_HOST_POOL_EXPERT, node, size);



#if defined(MADV_DONTNEED)
    if (ptr != nullptr && size > 0 && madvise(ptr, size, MADV_DONTNEED) != 0) {
        GGML_LOG_DEBUG("%s: MADV_DONTNEED of %.2f MiB failed: %s\n",
            __func__, size / 1024.0 / 1024.0, strerror(errno));
    }
#endif
    CUDA_CHECK(cudaFreeHost(ptr));
}

static void ggml_cuda_expert_copy_h2d(void * user, void * dst_dev, const void * src_host, size_t size) {
    const int device = ggml_cuda_expert_user_device(user);
    ggml_cuda_set_device(device);
    ggml_cuda_memcpy_h2d_async(dst_dev, src_host, size, cudaStreamPerThread);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_cuda_expert_copy_d2h(void * user, void * dst_host, const void * src_dev, size_t size) {
    const int device = ggml_cuda_expert_user_device(user);
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpy(dst_host, src_dev, size, cudaMemcpyDeviceToHost));
}

static void ggml_cuda_expert_table_upload(void * user, void * dst_dev, const char * const * src_host, size_t n) {
    const int device = ggml_cuda_expert_user_device(user);
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpy(dst_dev, src_host, n * sizeof(char *), cudaMemcpyHostToDevice));
}

static ggml_expert_store_backend ggml_cuda_expert_backend(ggml_cuda_expert_backend_user * user) {
    ggml_expert_store_backend backend;
    backend.user         = user;
    backend.alloc_device = ggml_cuda_expert_alloc_device;
    backend.free_device  = ggml_cuda_expert_free_device;
    backend.alloc_host   = ggml_cuda_expert_alloc_host;
    backend.free_host    = ggml_cuda_expert_free_host;
    backend.copy_h2d     = ggml_cuda_expert_copy_h2d;
    backend.copy_d2h     = ggml_cuda_expert_copy_d2h;
    backend.table_upload = ggml_cuda_expert_table_upload;
    return backend;
}




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
    return (t->flags & GGML_TENSOR_FLAG_EXPERT) != 0;
}

void ggml_cuda_expert_mark_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * t) {
    if (!ggml_cuda_expert_buft_is_expert(buft) || !ggml_cuda_expert_layout_ok(t)) {
        return;
    }
    t->flags |= GGML_TENSOR_FLAG_EXPERT;
}



static int ggml_cuda_expert_tensor_layer(const ggml_tensor * t) {
    return ggml_expert_layer_of_name(t->name);
}



static int ggml_cuda_expert_tensor_kind(const ggml_tensor * t) {
    return ggml_expert_kind_of_name(t->name);
}










static __global__ void expert_routing_kernel(
        const int32_t * __restrict__ ids, const int64_t n_entries, const int n_expert_used, const int64_t ids_stride,
        const ggml_cuda_expert_lut lut, const int expert_base, const int32_t * __restrict__ residency, const int64_t host_bytes,
        unsigned long long * __restrict__ counters, uint32_t * __restrict__ counts,
        uint32_t * __restrict__ bitmap_or_null) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) {
        return;
    }
    const int64_t token = i / n_expert_used;
    const int     slot  = (int) (i - token * n_expert_used);
    const int local = ggml_cuda_expert_local(lut, ids[token*ids_stride + slot]);
    if (local < 0) {
        return;
    }
    const int e = local + expert_base;
    if (counts != nullptr) {
        atomicAdd(&counts[e], 1u);
    }
    if (bitmap_or_null != nullptr) {


        atomicOr(&bitmap_or_null[e >> 5], 1u << (e & 31));
    }
    if (residency[e] >= 0) {
        atomicAdd(&counters[0], 1ULL);
    } else {
        atomicAdd(&counters[1], 1ULL);
        atomicAdd(&counters[2], (unsigned long long) host_bytes);
    }
}





static __global__ void expert_launch_count_kernel(
        const int32_t * __restrict__ ids, const int64_t n_entries, const int n_expert_used, const int64_t ids_stride,
        const ggml_cuda_expert_lut lut, const int expert_base, const int n_expert, uint32_t * __restrict__ launches) {
    extern __shared__ uint32_t s_routed[];
    const int n_words = (n_expert + 31) / 32;
    const int n_words_pad = (n_words + 3) & ~3;
    for (int w = threadIdx.x; w < n_words_pad; w += blockDim.x) {
        s_routed[w] = 0;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n_entries; i += blockDim.x) {
        const int64_t token = i / n_expert_used;
        const int     slot  = (int) (i - token * n_expert_used);
        const int local = ggml_cuda_expert_local(lut, ids[token*ids_stride + slot]);
        if (local < 0) {
            continue;
        }
        const int e = local + expert_base;
        atomicOr(&s_routed[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        if (s_routed[e >> 5] & (1u << (e & 31))) {
            launches[e] += 1u;
        }
    }
}






#define EXPERT_STAGE_COPY_CHUNK 65536




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




















static __global__ void expert_stage_decode_kernel(
        const int32_t * __restrict__ ids, const int n_entries, const int n_expert_used, const int64_t ids_stride,
        const ggml_cuda_expert_lut lut, const int expert_base,
        const int expert_lo, const int expert_hi,
        const int32_t * __restrict__ residency, const char * __restrict__ host_dev,
        const int64_t * __restrict__ host_offset,
        const char * const * __restrict__ table_main, const char ** __restrict__ table_stage,
        char * __restrict__ stage_dev, const size_t blob_stride, const size_t blob_bytes, const size_t stage_bytes,
        const int seq, int * __restrict__ fault, unsigned long long * __restrict__ counters,
        uint32_t * __restrict__ counts) {
    const int e = (int) blockIdx.y;

    if (e < expert_lo || e >= expert_hi) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            table_stage[e] = table_main[e];
        }
        return;
    }




    extern __shared__ int32_t s_ids[];








    const int n_pad = (n_entries + 3) & ~3;
    for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
        int v = -1;
        if (i < n_entries) {
            const int token = i / n_expert_used;
            const int local = ggml_cuda_expert_local(lut, ids[(int64_t) token * ids_stride + (i - token * n_expert_used)]);
            v = local < 0 ? -1 : local + expert_base;
        }
        s_ids[i] = v;
    }
    __syncthreads();


    int found = 0;
    for (int i = threadIdx.x; i < n_pad; i += blockDim.x) {
        found += s_ids[i] == e;
    }
    unsigned int n_e = (unsigned int) found;
    for (int off = 16; off > 0; off >>= 1) {
        n_e += __shfl_down_sync(0xffffffffu, n_e, off);
    }
    __shared__ int s_routes;
    if (threadIdx.x == 0) {
        s_routes = 0;
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0 && n_e) {
        atomicAdd(&s_routes, (int) n_e);
    }
    __syncthreads();



    if (counts != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && s_routes > 0) {
        atomicAdd(&counts[e], (uint32_t) s_routes);
    }

    const bool hit = s_routes > 0 && residency[e] < 0;
    if (!hit) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            table_stage[e] = table_main[e];
        }
        return;
    }




    int cnt = 0;
    for (int f = expert_lo + (int) threadIdx.x; f < e; f += blockDim.x) {
        int f_found = 0;
        for (int i = 0; i < n_pad; ++i) {
            f_found |= s_ids[i] == f;
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



                atomicExch(fault, 1);
                table_stage[x] = table_main[x];
                continue;
            }
            atomicAdd(n_hit, 1);
            hit_list[slot] = x;
            table_stage[x] = stage_dev + (size_t) slot * blob_stride;
            if (counters != nullptr) {
                atomicAdd(&counters[3], (unsigned long long) blob_bytes);
            }
        } else {
            table_stage[x] = table_main[x];
        }
    }
}





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




    if (blockIdx.x == gridDim.x - 1) {
        for (size_t i = blob_bytes + threadIdx.x; i < blob_stride; i += blockDim.x) {
            dst[i] = 0;
        }
    }
}









#define GGML_CUDA_EXPERT_XCOPY_BLOCKS 8


#define GGML_CUDA_EXPERT_XCOPY_MAX 128
#define GGML_CUDA_EXPERT_XFLIP_MAX 96


struct ggml_cuda_expert_xcopy {
    const char *       src;
    char *             dst;
    unsigned long long bytes;
};

struct ggml_cuda_expert_xcopy_arg {
    ggml_cuda_expert_xcopy e[GGML_CUDA_EXPERT_XCOPY_MAX];
};
static_assert(sizeof(ggml_cuda_expert_xcopy_arg) + 64 <= 4096, "the copies of one launch must fit the kernel parameter block");




static __global__ void expert_xcopy_kernel(const ggml_cuda_expert_xcopy_arg arg, const int n) {
    for (int i = 0; i < n; ++i) {
        const ggml_cuda_expert_xcopy c = arg.e[i];
        const size_t n_chunks = (size_t) ((c.bytes + EXPERT_STAGE_COPY_CHUNK - 1) / EXPERT_STAGE_COPY_CHUNK);
        for (size_t k = blockIdx.x; k < n_chunks; k += gridDim.x) {
            const size_t begin = k * EXPERT_STAGE_COPY_CHUNK;
            const size_t left  = (size_t) c.bytes - begin;
            expert_stage_copy_bytes(c.src + begin, c.dst + begin, left < EXPERT_STAGE_COPY_CHUNK ? left : EXPERT_STAGE_COPY_CHUNK);
        }
    }
}


struct ggml_cuda_expert_tables {
    const char ** table;
    int32_t *     residency;
    int64_t *     host_offset;
};


struct ggml_cuda_expert_xflip {
    const char * ptr;
    long long    off;
    int          tensor;
    int          expert;
    int          res;
    int          pad;
};

struct ggml_cuda_expert_xflip_arg {
    ggml_cuda_expert_xflip e[GGML_CUDA_EXPERT_XFLIP_MAX];
};
static_assert(sizeof(ggml_cuda_expert_xflip_arg) + 64 <= 4096, "the flips of one launch must fit the kernel parameter block");



static __global__ void expert_xflip_kernel(const ggml_cuda_expert_xflip_arg arg, const int n,
        const ggml_cuda_expert_tables * __restrict__ tables) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const ggml_cuda_expert_xflip f = arg.e[i];
    const ggml_cuda_expert_tables t = tables[f.tensor];
    t.table      [f.expert] = f.ptr;
    t.residency  [f.expert] = f.res;
    t.host_offset[f.expert] = f.off;
}


struct ggml_cuda_expert_tensor_state {
    int tensor_id = -1;
    const ggml_tensor * tensor = nullptr;

    int64_t n_expert = 0;
    size_t  blob_bytes = 0;
    size_t  blob_stride = 0;
    bool    has_host = false;
    int     kind   = -1;
    int     region = -1;
    int     arc    = -1;

    const char ** table_main_dev = nullptr;
    int32_t * residency_dev   = nullptr;
    int64_t * host_offset_dev = nullptr;
    uint32_t * counts_dev     = nullptr;
    uint32_t * launches_dev   = nullptr;



    bool     has_stage  = false;




    cudaStream_t reader        = nullptr;
    bool         reader_shared = false;

    bool verify_done = false;
};



struct ggml_cuda_expert_stage_region {
    char *  base  = nullptr;
    size_t  bytes = 0;
};





struct ggml_cuda_expert_stage_arc {
    int      region     = -1;
    int      layer      = -1;
    int      tensor_id  = -1;
    int64_t  n_host     = 0;
    int64_t  pos_base   = 0;
    int      prev_arc   = -1;



    bool     prev_same_block = false;
    size_t   stride     = 0;
    size_t   blob_bytes = 0;


    char *   host_host  = nullptr;
};


struct ggml_cuda_expert_stage_ring {
    char *  base  = nullptr;
    size_t pitch = 0;
    int    slots = 0;
    int64_t total = 0;
    std::vector<int> arcs;
    cudaStream_t copy_stream = nullptr;


    int last_layer = -1;
};





struct ggml_cuda_expert_ring_control {
    std::mutex             mu;
    std::condition_variable cv;
    bool                   stop = false;




    std::unique_ptr<std::atomic<uint64_t>[]> blocks;

    std::unique_ptr<std::atomic<int64_t>[]> arc_issued;
    std::unique_ptr<std::atomic<int64_t>[]> arc_consumed;



    std::unique_ptr<std::atomic<int64_t>[]> producer_phase;

    std::unique_ptr<std::atomic<int64_t>[]> producer_arc;
    std::unique_ptr<std::atomic<int64_t>[]> producer_block;



    std::atomic<uint64_t> generation{0};
};



struct ggml_backend_cuda_expert_buffer_context;

struct ggml_cuda_expert_stream_state {
    char * stage_dev   = nullptr;
    size_t stage_bytes = 0;
    std::vector<ggml_cuda_expert_stage_region> regions;





    std::vector<const char **> tables_static;
    std::vector<char **>       tables_static_host;
    std::vector<const char **> tables_decode;





    std::vector<uint32_t *> bitmaps;
    int32_t *              hit_list   = nullptr;
    int *                  plan_state = nullptr;
    size_t                 hit_cap    = 0;





    int *       fault_dev    = nullptr;
    int *       fault_pinned = nullptr;
    cudaEvent_t fault_event  = nullptr;
    int         stage_seq    = 0;



    int64_t     last_stage_n_host = -1;




    std::vector<ggml_cuda_expert_stage_ring> rings;




    std::vector<cudaEvent_t> arc_copied;
    std::vector<cudaEvent_t> arc_released;





    cudaEvent_t init_ev = nullptr;







    int launch_mode = 0;




    int64_t swap_epoch = 0;




    std::vector<char> ring_block_open;




    cudaEvent_t block_barrier = nullptr;










    cudaEvent_t comm_done = nullptr;





    std::vector<std::thread>              producers;
    std::unique_ptr<ggml_cuda_expert_ring_control> ctl;



    bool                   block_barrier_armed = false;



    static constexpr int   WAIT_TIMING_RING = 16;
    cudaEvent_t            wait_ev[WAIT_TIMING_RING][2] = {};
    int                    wait_ring_head  = 0;
    int                    wait_ring_tail  = 0;
    int                    wait_ring_count = 0;
};
























#define GGML_CUDA_EXPERT_ROW_MAX 4


enum ggml_cuda_expert_xstate : uint8_t {
    GGML_CUDA_EXPERT_XS_HOST  = 0,
    GGML_CUDA_EXPERT_XS_SLOT  = 1,
    GGML_CUDA_EXPERT_XS_BUSY  = 2,
    GGML_CUDA_EXPERT_XS_MIXED = 3,
};


struct ggml_cuda_expert_xrow {
    int     layer    = -1;
    int     n_tensor = 0;
    int     tensor[GGML_CUDA_EXPERT_ROW_MAX] = {};
    int     pool  [GGML_CUDA_EXPERT_ROW_MAX] = {};
    int64_t n_expert = 0;
};


struct ggml_cuda_expert_xpool {
    size_t              stride   = 0;
    int64_t             capacity = 0;
    std::vector<char *> free;
};


struct ggml_cuda_expert_xreq {
    int     row   = -1;
    int32_t e_in  = -1;
    int32_t e_out = -1;
};


struct ggml_cuda_expert_xchg {
    int     row   = -1;
    int32_t e_in  = -1;
    int32_t e_out = -1;
    char *  fresh[GGML_CUDA_EXPERT_ROW_MAX] = {};
};


struct ggml_cuda_expert_xbatch {
    std::vector<ggml_cuda_expert_xchg> xs;
    cudaEvent_t ev = nullptr;
};




struct ggml_cuda_expert_xjob {
    int                               mirror = -1;
    int64_t                           budget = 0;
    std::vector<std::vector<uint8_t>> state;
};

struct ggml_cuda_expert_engine {
    std::mutex mutex;





    cudaStream_t driver  = nullptr;
    int          first_layer = -1;
    cudaStream_t copy    = nullptr;
    cudaEvent_t  ev_flip = nullptr;
    cudaEvent_t  ev_done = nullptr;



    cudaEvent_t  ev_bcopy = nullptr;
    std::vector<cudaEvent_t> ev_free;

    std::vector<ggml_cuda_expert_xpool> pools;
    std::vector<ggml_cuda_expert_xrow>  rows;
    std::vector<int>                    row_of_tensor;
    std::vector<std::vector<uint8_t>>   state;

    ggml_cuda_expert_tables * tables_dev = nullptr;

    std::deque<ggml_cuda_expert_xreq>   todo;
    std::deque<ggml_cuda_expert_xbatch> copying;
    std::deque<ggml_cuda_expert_xbatch> leaving;






    uint32_t *              counts_dev = nullptr;
    size_t                  counts_n   = 0;
    std::vector<size_t>     count_off;
    std::vector<uint32_t *> count_ptr;
    uint32_t *              counts_host[2] = {};
    cudaEvent_t             counts_ev[2]   = {};
    bool                    counts_busy[2] = {};
    int                     counts_next    = 0;
    int64_t                 window_tokens  = 0;
    int64_t                 window_budget  = 0;




    std::thread             worker;
    std::mutex              job_mutex;
    std::condition_variable job_cv;
    bool                    job_stop  = false;
    bool                    job_ready = false;
    ggml_cuda_expert_xjob   job;
    std::atomic<bool>       result_ready{false};
    std::deque<ggml_cuda_expert_xreq> result;
    std::vector<std::vector<double>>  score;


    int64_t n_issued      = 0;
    int64_t n_done        = 0;
    int64_t n_dropped     = 0;
    int64_t n_flips       = 0;
    int64_t n_inflight    = 0;
    int64_t inflight_peak = 0;
};

struct ggml_backend_cuda_expert_buffer_context {
    int device = -1;
    ggml_cuda_expert_backend_user backend_user;
    ggml_expert_store_backend backend;
    ggml_expert_store * store = nullptr;

    size_t stage_budget = 0;




    std::vector<size_t> region_bytes;




    std::vector<ggml_cuda_expert_stage_arc> arcs;
    std::vector<std::vector<int>> region_arcs;
    std::vector<int>    region_slots;
    std::vector<size_t> region_pitch;
    std::vector<int64_t> region_total;



    void * plain_dev  = nullptr;
    size_t plain_size = 0;

    std::vector<ggml_cuda_expert_tensor_state> tensors;
    std::map<cudaStream_t, ggml_cuda_expert_stream_state> streams;



    std::mutex streams_mu;


    unsigned long long * usage_dev = nullptr;
    unsigned long long * usage_host = nullptr;
    cudaEvent_t usage_event = nullptr;


    bool     usage_readback_pending = false;
    int64_t launches = 0;




    int64_t stage_host_bytes   = 0;



    std::atomic<int64_t> stage_hits{0};
    std::atomic<int64_t> stage_cold{0};
    std::atomic<int64_t> prefetch_issued{0};
    std::atomic<int64_t> stage_copy_us{0};
    std::atomic<int64_t> stage_copy_arcs{0};
    double  prefetch_wait_ms   = 0.0;
    int64_t prefetch_wait_timed = 0;


    std::unique_ptr<ggml_cuda_expert_engine> engine;





    std::atomic<int64_t> swap_epoch{0};




    std::atomic<int64_t> ring_epoch{0};






    std::atomic<int64_t> ring_blocks_open{0};















    std::atomic<int64_t> swap_in_flight{0};



    std::mutex ring_mutex;






    int prefill_mode = 0;
    ggml_cuda_expert_stream_state & get_stream(cudaStream_t stream);
    ~ggml_backend_cuda_expert_buffer_context();
};

static ggml_backend_cuda_expert_buffer_context * ggml_cuda_expert_buffer_ctx(const ggml_tensor * t) {
    return (ggml_backend_cuda_expert_buffer_context *) t->buffer->context;
}




struct ggml_cuda_expert_stage_arc_spec {
    int     region       = -1;
    int     layer        = -1;
    int     tensor_index = -1;
    int64_t n_host       = 0;
    size_t  stride       = 0;
    size_t  blob_bytes   = 0;
    char *  host_host    = nullptr;
};










static void ggml_cuda_expert_ring_log_rebuild(const ggml_backend_cuda_expert_buffer_context & ctx,
        int region, int64_t epoch_old, int64_t epoch_new, const char * reason) {
    const size_t n_arcs  = (region >= 0 && (size_t) region < ctx.region_arcs.size())
        ? ctx.region_arcs[(size_t) region].size() : (size_t) 0;
    const int    n_slots = (region >= 0 && (size_t) region < ctx.region_slots.size())
        ? ctx.region_slots[(size_t) region] : 0;
    const size_t pitch   = (region >= 0 && (size_t) region < ctx.region_pitch.size())
        ? ctx.region_pitch[(size_t) region] : (size_t) 0;
    GGML_LOG_DEBUG("stage ring rebuilt: device %d ring %d epoch %lld -> %lld arcs %zu slots %d pitch %zu regions %zu reason %s\n",
        ctx.device, region, (long long) epoch_old, (long long) epoch_new, n_arcs, n_slots, pitch,
        ctx.region_bytes.size(), reason);
}








static void ggml_cuda_expert_ring_build(ggml_backend_cuda_expert_buffer_context & ctx,
        std::vector<ggml_cuda_expert_stage_arc_spec> specs) {

    std::stable_sort(specs.begin(), specs.end(),
            [](const ggml_cuda_expert_stage_arc_spec & a, const ggml_cuda_expert_stage_arc_spec & b) {
        if (a.region != b.region) return a.region < b.region;
        if (a.layer  != b.layer)  return a.layer  < b.layer;
        return a.tensor_index < b.tensor_index;
    });

    ctx.arcs.resize(specs.size());
    ctx.region_arcs.assign(ctx.region_bytes.size(), {});
    ctx.region_slots.assign(ctx.region_bytes.size(), 0);
    ctx.region_pitch.assign(ctx.region_bytes.size(), 0);
    ctx.region_total.assign(ctx.region_bytes.size(), 0);

    for (const ggml_cuda_expert_stage_arc_spec & a : specs) {
        ctx.region_slots[(size_t) a.region] = std::max(ctx.region_slots[(size_t) a.region], (int) a.n_host + 1);
        ctx.region_pitch[(size_t) a.region] = std::max(ctx.region_pitch[(size_t) a.region], a.stride);
    }

    int region = -1;
    int64_t pos_base = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        const ggml_cuda_expert_stage_arc_spec & a = specs[i];
        if (a.region != region) {
            region   = a.region;
            pos_base = 0;
        }
        ggml_cuda_expert_stage_arc arc;
        arc.region     = a.region;
        arc.layer      = a.layer;
        arc.tensor_id  = ctx.tensors[(size_t) a.tensor_index].tensor_id;
        arc.n_host     = a.n_host;
        arc.pos_base   = pos_base;
        arc.stride     = a.stride;
        arc.blob_bytes = a.blob_bytes;
        arc.host_host  = a.host_host;
        pos_base += a.n_host;
        ctx.region_total[(size_t) a.region] = pos_base;
        ctx.arcs[i] = arc;
        ctx.region_arcs[(size_t) a.region].push_back((int) i);
        ctx.tensors[(size_t) a.tensor_index].arc = (int) i;
    }





    for (size_t r = 0; r < ctx.region_arcs.size(); ++r) {
        const std::vector<int> & arcs = ctx.region_arcs[r];
        const int64_t slots = ctx.region_slots[r];
        const int64_t total = ctx.region_total[r];
        for (size_t i = 0; i < arcs.size(); ++i) {
            ggml_cuda_expert_stage_arc & arc = ctx.arcs[(size_t) arcs[i]];
            if (total <= 0 || arc.n_host <= 0) {
                arc.prev_arc = -1;
                continue;
            }
            const int64_t p = arc.pos_base + arc.n_host - 1 - slots;
            arc.prev_same_block = p >= 0;






            const int64_t q = p >= 0 ? p : p + total;
            int prev = -1;
            for (size_t j = 0; j < arcs.size(); ++j) {
                const ggml_cuda_expert_stage_arc & other = ctx.arcs[(size_t) arcs[j]];
                if (q >= other.pos_base && q < other.pos_base + other.n_host) {
                    prev = arcs[j];
                    break;
                }
            }
            arc.prev_arc = prev;
        }
    }

    for (size_t r = 0; r < ctx.region_bytes.size(); ++r) {
        ctx.region_bytes[r] = (size_t) ctx.region_slots[r] * ctx.region_pitch[r];
    }
}




static std::vector<ggml_cuda_expert_stage_arc_spec> ggml_cuda_expert_ring_specs(
        ggml_backend_cuda_expert_buffer_context & ctx) {
    std::vector<ggml_cuda_expert_stage_arc_spec> specs;
    specs.reserve(ctx.tensors.size());
    for (size_t i = 0; i < ctx.tensors.size(); ++i) {
        const ggml_cuda_expert_tensor_state & s = ctx.tensors[i];
        if (!s.has_stage) {
            continue;
        }
        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx.store, s.tensor_id);
        if (st == nullptr) {
            continue;
        }
        ggml_cuda_expert_stage_arc_spec spec;
        spec.region       = s.region;
        spec.layer        = st->layer;
        spec.tensor_index = (int) i;
        spec.n_host       = st->n_host;
        spec.stride       = st->stride;
        spec.blob_bytes   = s.blob_bytes;
        spec.host_host    = st->host_host;
        specs.push_back(spec);
    }
    return specs;
}




static void ggml_cuda_expert_ring_block_open(ggml_backend_cuda_expert_buffer_context & ctx,
        ggml_cuda_expert_stream_state & ss, int region) {
    if (region < 0 || (size_t) region >= ss.ring_block_open.size() || ss.ring_block_open[(size_t) region]) {
        return;
    }
    ss.ring_block_open[(size_t) region] = 1;
    ctx.ring_blocks_open.fetch_add(1, std::memory_order_release);
}






















static void ggml_cuda_expert_ring_block_end(ggml_backend_cuda_expert_buffer_context & ctx,
        ggml_cuda_expert_stream_state & ss, int region, cudaStream_t stream) {
    if (region < 0 || (size_t) region >= ss.ring_block_open.size() || !ss.ring_block_open[(size_t) region]) {
        return;
    }
    ss.ring_block_open[(size_t) region] = 0;
    if (ss.ctl) {
        const std::vector<int> & arcs = ss.rings[(size_t) region].arcs;
        const int64_t blk = (int64_t) ss.ctl->blocks[(size_t) region].load(std::memory_order_relaxed);



        int n_skipped = 0;
        for (int a : arcs) {
            if (ss.ctl->arc_consumed[(size_t) a].load(std::memory_order_relaxed) < blk) {
                CUDA_CHECK(cudaEventRecord(ss.arc_released[(size_t) a], stream));
                n_skipped++;
            }
        }
        {
            std::lock_guard<std::mutex> lock(ss.ctl->mu);
            for (int a : arcs) {
                if (ss.ctl->arc_consumed[(size_t) a].load(std::memory_order_relaxed) < blk) {
                    ss.ctl->arc_consumed[(size_t) a].store(blk, std::memory_order_release);
                }
            }
        }
        ss.ctl->cv.notify_all();



        GGML_LOG_DEBUG("%s: device %d: ring %d block %" PRId64 " closed, %d of %zu arcs skipped\n",
            __func__, ctx.device, region, blk, n_skipped, arcs.size());
    }
    ctx.ring_blocks_open.fetch_sub(1, std::memory_order_release);
}




static void ggml_cuda_expert_engine_drain(ggml_backend_cuda_expert_buffer_context & ctx,
        ggml_cuda_expert_engine & eng, bool quiesce_now);


































static void ggml_cuda_expert_exchange_settle(ggml_backend_cuda_expert_buffer_context & ctx,
        cudaStream_t stream) {

    if (ctx.swap_in_flight.load(std::memory_order_acquire) == 0) {
        return;
    }
    if (ctx.engine == nullptr) {


        GGML_ABORT("%s: device %d: an exchange of the placement is in flight, but the device has no engine\n",
            __func__, ctx.device);
    }
    ggml_cuda_expert_engine & eng = *ctx.engine;
    const int64_t epoch_old = ctx.swap_epoch.load(std::memory_order_acquire);
    {


        std::lock_guard<std::mutex> lock(eng.mutex);
        if (ctx.swap_in_flight.load(std::memory_order_acquire) == 0) {


            return;
        }





        ggml_cuda_expert_engine_drain(ctx, eng, true);
    }
    if (ctx.swap_in_flight.load(std::memory_order_acquire) != 0) {
        GGML_ABORT("%s: device %d: the exchange of the placement was settled, but it is still in flight\n",
            __func__, ctx.device);
    }



    CUDA_CHECK(cudaStreamWaitEvent(stream, eng.ev_done, 0));

    GGML_LOG_DEBUG("%s: device %d: settled an exchange at a prefill boundary (epoch %lld -> %lld)\n",
        __func__, ctx.device, (long long) epoch_old,
        (long long) ctx.swap_epoch.load(std::memory_order_acquire));
}






















static void ggml_cuda_expert_ring_rebuild(ggml_backend_cuda_expert_buffer_context & ctx,
        ggml_cuda_expert_stream_state & ss, cudaStream_t stream, int64_t epoch) {
    GGML_ASSERT(ss.ctl != nullptr);

    const int64_t epoch_old = ss.swap_epoch;

    {


        std::lock_guard<std::mutex> lock(ctx.ring_mutex);

        std::vector<ggml_cuda_expert_stage_arc_spec> specs = ggml_cuda_expert_ring_specs(ctx);





        GGML_ASSERT(specs.size() == ctx.arcs.size());
        ggml_cuda_expert_ring_build(ctx, specs);








        size_t region_sum = 0;
        for (size_t r = 0; r < ctx.region_bytes.size(); ++r) {
            region_sum += ctx.region_bytes[r];
            if (r < ss.regions.size() && ss.regions[r].bytes != ctx.region_bytes[r]) {
                GGML_ABORT("%s: device %d: the region %zu of the staging area grew from %zu to %zu bytes"
                    " when the placement changed; the staging area is laid out for the plan of the"
                    " allocation and the budget of fit is computed for the same plan, so this is a"
                    " configuration error\n",
                    __func__, ctx.device, r, ss.regions[r].bytes, ctx.region_bytes[r]);
            }
        }
        if (region_sum > ss.stage_bytes) {
            GGML_ABORT("%s: device %d: the regions of the staging area need %zu bytes when the"
                " placement changed, but the staging area has %zu bytes; the staging budget of fit"
                " is computed for the same plan as the allocation, so this is a configuration error\n",
                __func__, ctx.device, region_sum, ss.stage_bytes);
        }



        for (size_t r = 0; r < ss.rings.size() && r < ctx.region_arcs.size(); ++r) {
            ss.rings[r].slots = ctx.region_slots[r];
            ss.rings[r].pitch = ctx.region_pitch[r];
            ss.rings[r].arcs  = ctx.region_arcs[r];
        }
    }





    for (size_t i = 0; i < ctx.tensors.size(); ++i) {
        const ggml_cuda_expert_tensor_state & s = ctx.tensors[i];
        if (!s.has_stage) {
            continue;
        }
        GGML_ASSERT(s.region >= 0 && (size_t) s.region < ss.regions.size());
        GGML_ASSERT(s.arc >= 0 && (size_t) s.arc < ctx.arcs.size());
        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx.store, s.tensor_id);
        GGML_ASSERT(st != nullptr);
        const ggml_cuda_expert_stage_region & R    = ss.regions[(size_t) s.region];
        const ggml_cuda_expert_stage_arc    & arc  = ctx.arcs[(size_t) s.arc];
        const ggml_cuda_expert_stage_ring   & ring = ss.rings[(size_t) s.region];
        const size_t table_bytes = (size_t) s.n_expert * sizeof(char *);
        for (int64_t e = 0; e < s.n_expert; ++e) {
            if (st->residency[(size_t) e] >= 0) {
                ss.tables_static_host[i][e] = st->table_host[(size_t) e];
                continue;
            }
            const int64_t comp = st->host_offset[(size_t) e] / (int64_t) st->stride;
            const int64_t slot = (arc.pos_base + comp) % ring.slots;
            ss.tables_static_host[i][e] = R.base + (size_t) slot * ring.pitch;
        }
        CUDA_CHECK(cudaMemcpyAsync(ss.tables_static[i], ss.tables_static_host[i], table_bytes,
            cudaMemcpyHostToDevice, stream));
    }


    {
        std::lock_guard<std::mutex> lock(ss.ctl->mu);
        ss.ctl->generation.fetch_add(1, std::memory_order_release);



        for (size_t r = 0; r < ss.rings.size(); ++r) {
            ss.ctl->blocks[r].store(0, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < ctx.arcs.size(); ++i) {
            ss.ctl->arc_issued[i].store(0, std::memory_order_relaxed);
            ss.ctl->arc_consumed[i].store(0, std::memory_order_relaxed);
        }
    }
    ss.ctl->cv.notify_all();
    for (ggml_cuda_expert_stage_ring & ring : ss.rings) {
        ring.last_layer = -1;
    }
    ss.block_barrier_armed = false;
    ss.swap_epoch = epoch;
    ctx.ring_epoch.store(epoch, std::memory_order_release);


    for (size_t r = 0; r < ss.rings.size(); ++r) {
        ggml_cuda_expert_ring_log_rebuild(ctx, (int) r, epoch_old, epoch, "swap");
    }
}



static std::mutex g_expert_budget_mutex;
static size_t g_expert_slot_budget [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_stage_budget[GGML_CUDA_MAX_DEVICES] = {};


static size_t g_expert_alloc_slot [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_alloc_host [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_alloc_stage[GGML_CUDA_MAX_DEVICES] = {};



static size_t g_expert_alloc_host_peak[GGML_CUDA_MAX_DEVICES] = {};






static size_t g_expert_plan_slot[GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_plan_host[GGML_CUDA_MAX_DEVICES] = {};
static bool   g_expert_plan_valid[GGML_CUDA_MAX_DEVICES] = {};




static bool   g_expert_host_known [GGML_CUDA_MAX_DEVICES] = {};
static size_t g_expert_host_bytes[GGML_CUDA_MAX_DEVICES] = {};

static void ggml_cuda_expert_set_budget_impl(int device, size_t slot_bytes, size_t stage_bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_slot_budget [device] = slot_bytes;
    g_expert_stage_budget[device] = stage_bytes;
    g_expert_plan_valid[device] = false;

    g_expert_host_known [device] = false;
    g_expert_host_bytes[device] = 0;
}



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


    g_expert_host_known [device] = true;
    g_expert_host_bytes[device] = host_bytes;
}








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
                continue;
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





struct ggml_cuda_expert_device_bytes {
    size_t slot       = 0;
    size_t host       = 0;
    size_t stage      = 0;
    size_t host_peak  = 0;
    bool   host_known = false;
    size_t host_bytes = 0;
};

static ggml_cuda_expert_device_bytes ggml_cuda_expert_device_bytes_get(int device) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    ggml_cuda_expert_device_bytes b;
    b.slot       = g_expert_alloc_slot [device];
    b.host       = g_expert_alloc_host [device];
    b.stage      = g_expert_alloc_stage[device];
    b.host_peak  = g_expert_alloc_host_peak[device];
    b.host_known = g_expert_host_known [device];
    b.host_bytes = g_expert_host_bytes[device];
    return b;
}

static void ggml_cuda_expert_device_bytes_set(int device, const ggml_cuda_expert_device_bytes & b) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_alloc_slot [device] = b.slot;
    g_expert_alloc_host [device] = b.host;
    g_expert_alloc_stage[device] = b.stage;
    g_expert_alloc_host_peak[device] = b.host_peak;
    g_expert_host_known [device] = b.host_known;
    g_expert_host_bytes[device] = b.host_bytes;
}



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


static std::mutex g_expert_buffers_mutex;
static std::vector<ggml_backend_cuda_expert_buffer_context *> g_expert_buffers;


static void ggml_cuda_expert_set_slot_limit_impl(int device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
    g_expert_slot_budget[device] = bytes;
}



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




    int64_t view_offset = 0;
    const ggml_tensor * store_tensor = ggml_cuda_expert_trace_view(src0, &view_offset);
    if (store_tensor == nullptr) {
        GGML_ABORT("%s: cannot trace view '%s' back to the expert store tensor\n", __func__, src0->name);
    }
    const int tensor_id = ctx->store != nullptr ? ggml_expert_store_tensor_id(ctx->store, store_tensor) : -1;
    if (tensor_id < 0) {


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


    table->uniform      = !s.has_host;
    if (table->uniform) {



        const size_t ts = ggml_type_size(src0->type);
        GGML_ASSERT(st->stride % ts == 0);
    }
    table->plain_base   = st->slots_dev + expert_start * (int64_t) st->stride;
    table->plain_pitch  = st->stride;
    return true;
}





static void ggml_cuda_expert_stage_copy_arc(const ggml_cuda_expert_stage_arc & arc,
        const ggml_cuda_expert_stage_ring & ring, int64_t slot0, int64_t src0, int64_t n) {
    if (n <= 0) {
        return;
    }
    char * dst = ring.base + (size_t) slot0 * ring.pitch;
    const char * src = arc.host_host + (size_t) src0 * arc.stride;
    CUDA_CHECK(cudaMemcpy2DAsync(dst, ring.pitch, src, arc.stride, arc.blob_bytes, (size_t) n,
        cudaMemcpyHostToDevice, ring.copy_stream));
    if (ring.pitch > arc.blob_bytes) {
        CUDA_CHECK(cudaMemset2DAsync(dst + arc.blob_bytes, ring.pitch, 0, ring.pitch - arc.blob_bytes,
            (size_t) n, ring.copy_stream));
    }
}

// ---------------------------------------------------------------------------------------------
// Ring watchdog (diagnostic only).
//
// The prefill ring has two host waits: the consumer waiting for the producer to issue the arc of
// its layer (ggml_cuda_expert_stage_prepare) and the producer waiting for the arc that held the
// slot of its last expert to be consumed (ggml_cuda_expert_ring_producer). They are the only
// points of the prefill path that can stop the stream with no CUDA error and no log line, and the
// observed failure mode is exactly that: a prefill that stops advancing (roughly one 200K prefill
// in forty) and hangs until the harness timeout, which leaves nothing behind to attribute it with.
//
// Both waits therefore carry a timeout. A wait that makes no progress for the timeout below prints
// the state of its ring once - one field per line, every line prefixed "ring-watchdog:", so a dump
// can be grepped out of a long log - and aborts the process, so the next hit leaves the state
// behind instead of hanging. This changes nothing else: the waits, the gating, the events and the
// block counters are untouched. The normal path pays one counter increment and one compare per
// turn, and one std::chrono::steady_clock read every RING_WATCHDOG_SPIN_READS turns in the tight
// spin of the consumer, every RING_WATCHDOG_WAIT_READS turns in the wait of the producer (one turn
// of that one already blocks on the condition variable for a millisecond).
static constexpr int64_t RING_WATCHDOG_US         = 30 * 1000 * 1000;
static constexpr int     RING_WATCHDOG_SPIN_READS = 64 * 1024;
static constexpr int     RING_WATCHDOG_WAIT_READS = 1024;



struct ggml_cuda_expert_ring_watchdog_wait {
    const char * site            = nullptr;
    int          region          = -1;
    int          arc             = -1;
    int64_t      block           = 0;
    int64_t      my_block        = -1;
    int64_t      prev_arc        = -1;
    bool         prev_same_block = false;
    int64_t      need            = -1;
    int64_t      elapsed_us      = 0;
};



static const char * phase_name(int64_t phase) {
    switch (phase) {
        case 0:  return "idle";
        case 1:  return "host_gate";
        case 2:  return "released_event";
        case 3:  return "comm_done";
        case 4:  return "copying";
        default: return "?";
    }
}







static void ggml_cuda_expert_ring_watchdog_dump(ggml_backend_cuda_expert_buffer_context * ctx,
        const ggml_cuda_expert_stream_state & ss, const ggml_cuda_expert_ring_watchdog_wait & w) {
    const ggml_cuda_expert_stage_ring & ring = ss.rings[(size_t) w.region];
    const int64_t issued   = w.arc >= 0 ? ss.ctl->arc_issued[(size_t) w.arc].load(std::memory_order_acquire)   : -1;
    const int64_t consumed = w.arc >= 0 ? ss.ctl->arc_consumed[(size_t) w.arc].load(std::memory_order_acquire) : -1;

    fprintf(stderr, "%s\n", w.site);
    fprintf(stderr, "ring-watchdog: ring=%d device=%d arc=%d layer=%d block=%" PRId64 "\n",
        w.region, ctx->device, w.arc,
        w.arc >= 0 && (size_t) w.arc < ctx->arcs.size() ? ctx->arcs[(size_t) w.arc].layer : -1,
        w.block);
    fprintf(stderr, "ring-watchdog: blocks[ring]=%llu last_layer=%d\n",
        (unsigned long long) ss.ctl->blocks[(size_t) w.region].load(std::memory_order_acquire),
        ring.last_layer);
    fprintf(stderr, "ring-watchdog: arc_issued[arc]=%" PRId64 " arc_consumed[arc]=%" PRId64 "\n",
        issued, consumed);



    int at = -1;
    for (size_t i = 0; i < ring.arcs.size(); ++i) {
        if (ring.arcs[i] == w.arc) {
            at = (int) i;
            break;
        }
    }
    if (at >= 0) {
        const int lo = std::max(0, at - 3);
        const int hi = std::min((int) ring.arcs.size() - 1, at + 3);
        for (int i = lo; i <= hi; ++i) {
            const int a = ring.arcs[(size_t) i];
            const ggml_cuda_expert_stage_arc & arc = ctx->arcs[(size_t) a];
            fprintf(stderr, "ring-watchdog:   arc=%d layer=%d n_host=%" PRId64 " pos_base=%" PRId64
                    " prev_arc=%d issued=%" PRId64 " consumed=%" PRId64 "%s\n",
                a, arc.layer, arc.n_host, arc.pos_base, arc.prev_arc,
                ss.ctl->arc_issued[(size_t) a].load(std::memory_order_acquire),
                ss.ctl->arc_consumed[(size_t) a].load(std::memory_order_acquire),
                i == at ? "   <- the wait" : "");
        }


        const int last = (int) ring.arcs.size() - 1;
        if (last > hi || last < lo) {
            const int a = ring.arcs[(size_t) last];
            const ggml_cuda_expert_stage_arc & arc = ctx->arcs[(size_t) a];
            fprintf(stderr, "ring-watchdog:   arc=%d layer=%d n_host=%" PRId64 " pos_base=%" PRId64
                    " prev_arc=%d issued=%" PRId64 " consumed=%" PRId64 "   <- the last arc of the ring\n",
                a, arc.layer, arc.n_host, arc.pos_base, arc.prev_arc,
                ss.ctl->arc_issued[(size_t) a].load(std::memory_order_acquire),
                ss.ctl->arc_consumed[(size_t) a].load(std::memory_order_acquire));
        }
    } else {
        fprintf(stderr, "ring-watchdog:   arc %d is not in the %zu arcs of ring %d\n",
            w.arc, ring.arcs.size(), w.region);
    }

    if (w.my_block >= 0) {
        fprintf(stderr, "ring-watchdog: my_block=%" PRId64 " prev=%" PRId64 " prev_same_block=%d"
                " need=%" PRId64 " prev_consumed=%" PRId64 "\n",
            w.my_block, w.prev_arc, w.prev_same_block ? 1 : 0, w.need,
            w.prev_arc >= 0 ? ss.ctl->arc_consumed[(size_t) w.prev_arc].load(std::memory_order_acquire) : -1);
    }

    fprintf(stderr, "ring-watchdog: launches=%" PRId64 " stage_hits=%" PRId64 " stage_cold=%" PRId64
            " prefetch_issued=%" PRId64 " waited_ms=%" PRId64 "\n",
        ctx->launches,
        ctx->stage_hits.load(std::memory_order_relaxed),
        ctx->stage_cold.load(std::memory_order_relaxed),
        ctx->prefetch_issued.load(std::memory_order_relaxed),
        w.elapsed_us / 1000);



    fprintf(stderr, "ring-watchdog: producer phase=%s arc=%" PRId64 " block=%" PRId64 "\n",
        phase_name(ss.ctl->producer_phase[(size_t) w.region].load(std::memory_order_relaxed)),
        ss.ctl->producer_arc[(size_t) w.region].load(std::memory_order_relaxed),
        ss.ctl->producer_block[(size_t) w.region].load(std::memory_order_relaxed));
    fflush(stderr);
}








static void ggml_cuda_expert_ring_producer(ggml_backend_cuda_expert_buffer_context * ctx,
        ggml_cuda_expert_stream_state * ss_ptr, int region) {
    ggml_cuda_expert_stream_state & ss = *ss_ptr;
    ggml_cuda_set_device(ctx->device);

    ggml_cuda_expert_stage_ring & ring = ss.rings[(size_t) region];
    std::atomic<uint64_t> & blocks = ss.ctl->blocks[(size_t) region];



    std::atomic<int64_t> & producer_phase = ss.ctl->producer_phase[(size_t) region];
    std::atomic<int64_t> & producer_arc   = ss.ctl->producer_arc[(size_t) region];
    std::atomic<int64_t> & producer_block = ss.ctl->producer_block[(size_t) region];

    uint64_t my_block = 0;


    uint64_t my_generation = ss.ctl->generation.load(std::memory_order_acquire);
    bool init_waited = false;



    static constexpr int COPY_TIMING_RING = 32;
    cudaEvent_t copy_ev[COPY_TIMING_RING][2] = {};
    int copy_head = 0;
    int copy_tail = 0;
    int copy_count = 0;
    for (int i = 0; i < COPY_TIMING_RING; ++i) {




        CUDA_CHECK(cudaEventCreate(&copy_ev[i][0]));
        CUDA_CHECK(cudaEventCreate(&copy_ev[i][1]));
    }

    for (;;) {

        producer_phase.store(0, std::memory_order_relaxed);
        producer_arc.store(-1, std::memory_order_relaxed);
        {
            std::unique_lock<std::mutex> lock(ss.ctl->mu);
            while (!ss.ctl->stop && blocks.load() <= my_block &&
                    ss.ctl->generation.load(std::memory_order_acquire) == my_generation) {
                ss.ctl->cv.wait(lock);
            }
            if (ss.ctl->stop) {
                break;
            }



            const uint64_t generation = ss.ctl->generation.load(std::memory_order_acquire);
            if (generation != my_generation) {
                my_generation = generation;
                my_block      = 0;
                continue;
            }
            my_block = blocks.load();
        }





        std::vector<int>                       block_arcs;
        std::vector<ggml_cuda_expert_stage_arc> arc_table;
        ggml_cuda_expert_stage_ring             block_ring = ring;
        {
            std::lock_guard<std::mutex> lock(ctx->ring_mutex);
            block_arcs = ring.arcs;
            arc_table  = ctx->arcs;
            block_ring = ring;
        }




        producer_block.store((int64_t) my_block, std::memory_order_relaxed);





        if (!init_waited) {
            CUDA_CHECK(cudaStreamWaitEvent(ring.copy_stream, ss.init_ev, 0));
            init_waited = true;
        }
        CUDA_CHECK(cudaStreamWaitEvent(ring.copy_stream, ss.block_barrier, 0));

        for (size_t i = 0; i < block_arcs.size(); ++i) {
            const int arc_index = block_arcs[i];
            const ggml_cuda_expert_stage_arc & arc = arc_table[(size_t) arc_index];
            if (arc.n_host == 0) {
                continue;
            }


            producer_arc.store((int64_t) arc_index, std::memory_order_relaxed);








            if (arc.prev_arc >= 0) {
                const int64_t need = (int64_t) my_block - (arc.prev_same_block ? 0 : 1);





                const auto ring_watchdog_start = std::chrono::steady_clock::now();
                int        ring_watchdog_turns = 0;


                producer_phase.store(1, std::memory_order_relaxed);
                while (ss.ctl->arc_consumed[(size_t) arc.prev_arc].load(std::memory_order_acquire) < need) {
                    std::unique_lock<std::mutex> lock(ss.ctl->mu);
                    ss.ctl->cv.wait_for(lock, std::chrono::milliseconds(1));
                    if (ss.ctl->stop) {
                        break;
                    }
                    if (++ring_watchdog_turns % RING_WATCHDOG_WAIT_READS == 0) {
                        const int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - ring_watchdog_start).count();
                        if (elapsed_us >= RING_WATCHDOG_US) {
                            ggml_cuda_expert_ring_watchdog_wait w;
                            w.site            = "ring-watchdog: producer";
                            w.region          = region;
                            w.arc             = arc_index;
                            w.block           = (int64_t) my_block;
                            w.my_block        = (int64_t) my_block;
                            w.prev_arc        = arc.prev_arc;
                            w.prev_same_block = arc.prev_same_block;
                            w.need            = need;
                            w.elapsed_us      = elapsed_us;
                            ggml_cuda_expert_ring_watchdog_dump(ctx, ss, w);
                            GGML_ABORT("%s: device %d: ring %d: the slot of arc %d was not released for"
                                " %" PRId64 " ms in block %" PRId64 " (arc %d consumed %" PRId64 ", need %" PRId64 ")\n",
                                __func__, ctx->device, region, arc_index, elapsed_us / 1000,
                                (int64_t) my_block, arc.prev_arc,
                                ss.ctl->arc_consumed[(size_t) arc.prev_arc].load(std::memory_order_acquire),
                                need);
                        }
                    }
                }
                if (ss.ctl->stop) {
                    break;
                }

                producer_phase.store(2, std::memory_order_relaxed);
                CUDA_CHECK(cudaStreamWaitEvent(block_ring.copy_stream, ss.arc_released[(size_t) arc.prev_arc], 0));
            }








            if (ss.comm_done != nullptr) {

                producer_phase.store(3, std::memory_order_relaxed);
                CUDA_CHECK(cudaStreamWaitEvent(block_ring.copy_stream, ss.comm_done, 0));
            }




            const int64_t n     = arc.n_host;
            const int64_t s0    = arc.pos_base % block_ring.slots;
            const int64_t first = std::min<int64_t>(n, (int64_t) block_ring.slots - s0);
            CUDA_CHECK(cudaEventRecord(copy_ev[copy_head][0], block_ring.copy_stream));

            producer_phase.store(4, std::memory_order_relaxed);
            ggml_cuda_expert_stage_copy_arc(arc, block_ring, s0, 0, first);
            if (first < n) {
                ggml_cuda_expert_stage_copy_arc(arc, block_ring, 0, first, n - first);
            }
            CUDA_CHECK(cudaEventRecord(copy_ev[copy_head][1], block_ring.copy_stream));
            copy_head = (copy_head + 1) % COPY_TIMING_RING;
            copy_count += 1;
            if (copy_count == COPY_TIMING_RING) {
                if (cudaEventQuery(copy_ev[copy_tail][1]) == cudaSuccess) {
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, copy_ev[copy_tail][0], copy_ev[copy_tail][1]));
                    ctx->stage_copy_us.fetch_add((int64_t) (ms * 1000.0f), std::memory_order_relaxed);
                    ctx->stage_copy_arcs.fetch_add(1, std::memory_order_relaxed);
                    copy_tail = (copy_tail + 1) % COPY_TIMING_RING;
                    copy_count -= 1;
                } else {
                    (void) cudaGetLastError();
                }
            }
            CUDA_CHECK(cudaEventRecord(ss.arc_copied[(size_t) arc_index], block_ring.copy_stream));
            ss.ctl->arc_issued[(size_t) arc_index].store((int64_t) my_block, std::memory_order_release);
            ctx->prefetch_issued.fetch_add(1, std::memory_order_relaxed);
        }
    }

    for (int i = 0; i < COPY_TIMING_RING; ++i) {
        CUDA_CHECK(cudaEventDestroy(copy_ev[i][0]));
        CUDA_CHECK(cudaEventDestroy(copy_ev[i][1]));
    }
}

ggml_cuda_expert_stream_state & ggml_backend_cuda_expert_buffer_context::get_stream(cudaStream_t stream) {



    std::lock_guard<std::mutex> lock(streams_mu);
    const auto it = streams.find(stream);
    if (it != streams.end()) {
        return it->second;
    }

    ggml_cuda_set_device(device);









    ggml_cuda_expert_stream_state ss;
    if (stage_budget > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.stage_dev, stage_budget));


        CUDA_CHECK(cudaMemsetAsync(ss.stage_dev, 0, stage_budget, stream));
        ss.stage_bytes = stage_budget;
    }


    {
        size_t off = 0;
        ss.regions.resize(region_bytes.size());
        for (size_t r = 0; r < region_bytes.size(); ++r) {
            ss.regions[r].base  = ss.stage_dev + off;
            ss.regions[r].bytes = region_bytes[r];
            off += region_bytes[r];
        }
        GGML_ASSERT(off <= stage_budget);
    }

    bool any_stage = false;
    for (const ggml_cuda_expert_tensor_state & s : tensors) {
        any_stage = any_stage || s.has_stage;
    }

    ss.ctl = std::make_unique<ggml_cuda_expert_ring_control>();
    if (any_stage) {
        CUDA_CHECK(cudaMalloc((void **) &ss.fault_dev, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ss.fault_dev, 0, sizeof(int), stream));
        CUDA_CHECK(cudaHostAlloc((void **) &ss.fault_pinned, sizeof(int), cudaHostAllocDefault));
        *ss.fault_pinned = 0;
        CUDA_CHECK(cudaEventCreateWithFlags(&ss.fault_event, cudaEventDisableTiming));
        for (int i = 0; i < ggml_cuda_expert_stream_state::WAIT_TIMING_RING; ++i) {
            CUDA_CHECK(cudaEventCreate(&ss.wait_ev[i][0]));
            CUDA_CHECK(cudaEventCreate(&ss.wait_ev[i][1]));
        }



        CUDA_CHECK(cudaEventCreateWithFlags(&ss.init_ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss.block_barrier, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ss.comm_done, cudaEventDisableTiming));
        ss.rings.resize(region_bytes.size());


        ss.ring_block_open.assign(region_bytes.size(), 0);
        for (size_t r = 0; r < region_bytes.size(); ++r) {
            ggml_cuda_expert_stage_ring & ring = ss.rings[r];
            ring.base  = ss.regions[r].base;
            ring.pitch = region_pitch[r];
            ring.slots = region_slots[r];
            ring.arcs  = region_arcs[r];
            CUDA_CHECK(cudaStreamCreateWithFlags(&ring.copy_stream, cudaStreamNonBlocking));
        }



        ss.ctl->blocks       = std::make_unique<std::atomic<uint64_t>[]>(region_bytes.size());
        ss.ctl->arc_issued   = std::make_unique<std::atomic<int64_t>[]>(arcs.size());
        ss.ctl->arc_consumed = std::make_unique<std::atomic<int64_t>[]>(arcs.size());


        ss.ctl->producer_phase = std::make_unique<std::atomic<int64_t>[]>(region_bytes.size());
        ss.ctl->producer_arc   = std::make_unique<std::atomic<int64_t>[]>(region_bytes.size());
        ss.ctl->producer_block = std::make_unique<std::atomic<int64_t>[]>(region_bytes.size());
        for (size_t r = 0; r < region_bytes.size(); ++r) {
            ss.ctl->blocks[r].store(0, std::memory_order_relaxed);
            ss.ctl->producer_phase[r].store(0, std::memory_order_relaxed);
            ss.ctl->producer_arc[r].store(-1, std::memory_order_relaxed);
            ss.ctl->producer_block[r].store(0, std::memory_order_relaxed);
        }


        ss.arc_copied.resize(arcs.size());
        ss.arc_released.resize(arcs.size());
        for (size_t i = 0; i < arcs.size(); ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ss.arc_copied[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ss.arc_released[i], cudaEventDisableTiming));
            ss.ctl->arc_issued[i].store(0, std::memory_order_relaxed);
            ss.ctl->arc_consumed[i].store(0, std::memory_order_relaxed);
        }
    }
    ss.tables_static.assign(tensors.size(), nullptr);
    ss.tables_static_host.assign(tensors.size(), nullptr);
    ss.tables_decode.assign(tensors.size(), nullptr);
    ss.bitmaps.assign(tensors.size(), nullptr);



    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_cuda_expert_tensor_state & s = tensors[i];
        if (!s.has_stage) {
            continue;
        }
        GGML_ASSERT(s.region >= 0 && (size_t) s.region < ss.regions.size());
        const ggml_cuda_expert_stage_region & R = ss.regions[(size_t) s.region];
        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(store, s.tensor_id);
        GGML_ASSERT(R.bytes >= (size_t) st->host_used);






        const size_t table_bytes = (size_t) s.n_expert * sizeof(char *);
        CUDA_CHECK(cudaHostAlloc((void **) &ss.tables_static_host[i], table_bytes, cudaHostAllocDefault));
        const ggml_cuda_expert_stage_arc  & arc  = arcs[(size_t) s.arc];
        const ggml_cuda_expert_stage_ring & ring = ss.rings[(size_t) s.region];
        for (int64_t e = 0; e < s.n_expert; ++e) {
            if (st->residency[(size_t) e] >= 0) {
                ss.tables_static_host[i][e] = st->table_host[(size_t) e];
                continue;
            }
            const int64_t comp = st->host_offset[(size_t) e] / (int64_t) st->stride;
            const int64_t slot = (arc.pos_base + comp) % ring.slots;
            ss.tables_static_host[i][e] = R.base + (size_t) slot * ring.pitch;
        }
        CUDA_CHECK(cudaMalloc((void **) &ss.tables_static[i], table_bytes));
        CUDA_CHECK(cudaMemcpyAsync(ss.tables_static[i], ss.tables_static_host[i], table_bytes, cudaMemcpyHostToDevice, stream));





        CUDA_CHECK(cudaMalloc((void **) &ss.tables_decode[i], table_bytes));
        CUDA_CHECK(cudaMemsetAsync(ss.tables_decode[i], 0, table_bytes, stream));



        const size_t n_words = ((size_t) s.n_expert + 31) / 32;
        CUDA_CHECK(cudaMalloc((void **) &ss.bitmaps[i], n_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(ss.bitmaps[i], 0, n_words * sizeof(uint32_t), stream));
        ss.hit_cap = std::max(ss.hit_cap, (size_t) s.n_expert);
    }



    if (ss.hit_cap > 0) {
        CUDA_CHECK(cudaMalloc((void **) &ss.hit_list, ss.hit_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &ss.plan_state, 4 * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ss.plan_state, 0, 4 * sizeof(int), stream));
    }



    if (ss.init_ev != nullptr) {
        CUDA_CHECK(cudaEventRecord(ss.init_ev, stream));
    }

    GGML_LOG_DEBUG("%s: device %d: staging state of stream %zu: %.2f MiB staging in %zu regions, %zu tensors\n",
        __func__, device, streams.size(), ss.stage_bytes/1024.0/1024.0, ss.regions.size(), tensors.size());

    ggml_cuda_expert_stream_state & ref = streams.emplace(stream, std::move(ss)).first->second;

    for (size_t r = 0; r < ref.rings.size(); ++r) {
        ref.producers.emplace_back(ggml_cuda_expert_ring_producer, this, &ref, (int) r);
    }
    return ref;
}



static void ggml_cuda_expert_free_stream_state(ggml_cuda_expert_stream_state & ss) {


    if (ss.ctl) {
        {
            std::lock_guard<std::mutex> lock(ss.ctl->mu);
            ss.ctl->stop = true;
        }
        ss.ctl->cv.notify_all();
    }
    for (std::thread & t : ss.producers) {
        if (t.joinable()) {
            t.join();
        }
    }
    ss.producers.clear();
    for (ggml_cuda_expert_stage_ring & ring : ss.rings) {
        if (ring.copy_stream) {
            CUDA_CHECK(cudaStreamSynchronize(ring.copy_stream));
            CUDA_CHECK(cudaStreamDestroy(ring.copy_stream));
            ring.copy_stream = nullptr;
        }
    }
    for (size_t i = 0; i < ss.tables_static.size(); ++i) {
        if (ss.tables_static[i] != nullptr) {
            CUDA_CHECK(cudaFree(ss.tables_static[i]));
        }
        if (ss.tables_static_host[i] != nullptr) {
            CUDA_CHECK(cudaFreeHost(ss.tables_static_host[i]));
        }
        if (ss.tables_decode[i] != nullptr) {
            CUDA_CHECK(cudaFree(ss.tables_decode[i]));
        }
        if (i < ss.bitmaps.size() && ss.bitmaps[i] != nullptr) {
            CUDA_CHECK(cudaFree(ss.bitmaps[i]));
        }
    }
    if (ss.hit_list) {
        CUDA_CHECK(cudaFree(ss.hit_list));
    }
    if (ss.plan_state) {
        CUDA_CHECK(cudaFree(ss.plan_state));
    }
    if (ss.stage_dev) {
        CUDA_CHECK(cudaFree(ss.stage_dev));
    }
    if (ss.fault_dev) {
        CUDA_CHECK(cudaFree(ss.fault_dev));
    }
    if (ss.fault_pinned) {
        CUDA_CHECK(cudaFreeHost(ss.fault_pinned));
    }
    if (ss.fault_event) {
        CUDA_CHECK(cudaEventDestroy(ss.fault_event));
    }
    for (int i = 0; i < ggml_cuda_expert_stream_state::WAIT_TIMING_RING; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (ss.wait_ev[i][j]) {
                CUDA_CHECK(cudaEventDestroy(ss.wait_ev[i][j]));
            }
        }
    }
    for (ggml_cuda_expert_stage_region & R : ss.regions) {
        (void) R;
    }
    for (cudaEvent_t e : ss.arc_copied) {
        if (e) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
    }
    for (cudaEvent_t e : ss.arc_released) {
        if (e) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
    }
    ss.arc_copied.clear();
    ss.arc_released.clear();
    if (ss.block_barrier) {
        CUDA_CHECK(cudaEventDestroy(ss.block_barrier));
        ss.block_barrier = nullptr;
    }
    if (ss.init_ev) {
        CUDA_CHECK(cudaEventDestroy(ss.init_ev));
    }
    if (ss.comm_done) {
        CUDA_CHECK(cudaEventDestroy(ss.comm_done));
        ss.comm_done = nullptr;
    }
}





void ggml_cuda_expert_reset_streams(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        if (ctx->device != buft_ctx->device) {
            continue;
        }
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaDeviceSynchronize());
        {


            std::lock_guard<std::mutex> lock(ctx->streams_mu);
            for (auto & kv : ctx->streams) {
                ggml_cuda_expert_free_stream_state(kv.second);
            }
            ctx->streams.clear();
        }
    }
}









bool ggml_cuda_expert_last_stage_hits(ggml_backend_buffer_type_t buft, size_t * n_hit) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft) || n_hit == nullptr) {
        return false;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    bool found = false;
    *n_hit = 0;
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        if (ctx->device != buft_ctx->device) {
            continue;
        }
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (auto & kv : ctx->streams) {
            const ggml_cuda_expert_stream_state & ss = kv.second;
            if (ss.last_stage_n_host == -2) {


                if (ss.plan_state != nullptr) {
                    int plan[2] = { 0, 0 };
                    CUDA_CHECK(cudaMemcpy(plan, ss.plan_state, sizeof(plan), cudaMemcpyDeviceToHost));
                    *n_hit += (size_t) plan[0];
                }
                found = true;
                continue;
            }
            if (ss.last_stage_n_host < 0) {
                continue;
            }
            *n_hit += (size_t) ss.last_stage_n_host;
            found = true;
        }
    }
    return found;
}







void ggml_cuda_expert_comm_done(int device, cudaStream_t stream) {
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        if (ctx->device != device) {
            continue;
        }







        cudaEvent_t ev = nullptr;
        {
            std::lock_guard<std::mutex> lock(ctx->streams_mu);
            const auto it = ctx->streams.find(stream);
            if (it != ctx->streams.end()) {


                ev = it->second.comm_done;
            }
        }
        if (ev == nullptr) {
            continue;
        }
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaEventRecord(ev, stream));
    }
}

bool ggml_cuda_expert_stage_counters(ggml_backend_buffer_type_t buft, size_t * cold, size_t * hits) {
    if (buft == nullptr || buft->context == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return false;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    size_t c = 0;
    size_t h = 0;
    bool found = false;
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        if (ctx->device != buft_ctx->device) {
            continue;
        }
        c += (size_t) ctx->stage_cold.load(std::memory_order_relaxed);
        h += (size_t) ctx->stage_hits.load(std::memory_order_relaxed);
        found = true;
    }
    if (cold != nullptr) {
        *cold = c;
    }
    if (hits != nullptr) {
        *hits = h;
    }
    return found;
}

ggml_backend_cuda_expert_buffer_context::~ggml_backend_cuda_expert_buffer_context() {
    ggml_cuda_set_device(device);



    if (engine) {
        {
            std::lock_guard<std::mutex> lock(engine->job_mutex);
            engine->job_stop = true;
        }
        engine->job_cv.notify_one();
        if (engine->worker.joinable()) {
            engine->worker.join();
        }
        for (int m = 0; m < 2; ++m) {
            CUDA_CHECK(cudaEventDestroy(engine->counts_ev[m]));
            CUDA_CHECK(cudaFreeHost(engine->counts_host[m]));
        }
        CUDA_CHECK(cudaFree(engine->counts_dev));
        CUDA_CHECK(cudaStreamSynchronize(engine->copy));
        for (auto * q : { &engine->copying, &engine->leaving }) {
            for (const ggml_cuda_expert_xbatch & b : *q) {
                engine->ev_free.push_back(b.ev);
            }
        }
        for (cudaEvent_t ev : engine->ev_free) {
            CUDA_CHECK(cudaEventDestroy(ev));
        }
        CUDA_CHECK(cudaEventDestroy(engine->ev_flip));
        CUDA_CHECK(cudaEventDestroy(engine->ev_done));
        CUDA_CHECK(cudaEventDestroy(engine->ev_bcopy));
        CUDA_CHECK(cudaStreamDestroy(engine->copy));
        CUDA_CHECK(cudaFree(engine->tables_dev));

        swap_in_flight.store(0, std::memory_order_release);
        engine.reset();
    }



    if (store != nullptr) {
        for (ggml_cuda_expert_tensor_state & s : tensors) {
            if (s.counts_dev == nullptr || s.launches_dev == nullptr || s.n_expert <= 0) {
                continue;
            }
            std::vector<uint32_t> counts((size_t) s.n_expert);
            std::vector<uint32_t> launches((size_t) s.n_expert);
            CUDA_CHECK(cudaMemcpy(counts.data(), s.counts_dev, counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(launches.data(), s.launches_dev, launches.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            ggml_expert_store_set_tensor_stats(store, s.tensor_id, counts.data(), launches.data(), counts.size());
        }
        ggml_expert_store_dump_profile(store, ggml_backend_cuda_get_device_count());
    }

    for (auto & kv : streams) {
        ggml_cuda_expert_free_stream_state(kv.second);
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
        if (s.launches_dev) {
            CUDA_CHECK(cudaFree(s.launches_dev));
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
    if (backend_user.slab) {
        CUDA_CHECK(cudaFree(backend_user.slab));
    }
}








void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
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


    GGML_ASSERT(ex_table.expert_start + ex_table.n_expert <= s.n_expert);
    expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
        (const int32_t *) ids->data, n_entries, (int) ids->ne[0], ids->nb[1] / sizeof(int32_t),
        lut, (int) ex_table.expert_start,
        s.residency_dev, (int64_t) ex_table.blob_bytes,
        ctx->usage_dev, s.counts_dev, nullptr);


    CUDA_CHECK(cudaGetLastError());



    const size_t smem = (size_t) (((s.n_expert + 31) / 32 + 3) & ~3) * sizeof(uint32_t);
    GGML_ASSERT(smem <= 48 * 1024);
    expert_launch_count_kernel<<<1, 1024, smem, stream>>>(
        (const int32_t *) ids->data, n_entries, (int) ids->ne[0], ids->nb[1] / sizeof(int32_t),
        lut, (int) ex_table.expert_start, (int) s.n_expert, s.launches_dev);
    CUDA_CHECK(cudaGetLastError());
}








static void ggml_cuda_expert_stats_readback(ggml_backend_cuda_expert_buffer_context * ctx, cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusActive) {




        if (!ctx->usage_readback_pending) {
            CUDA_CHECK(cudaMemcpyAsync(ctx->usage_host, ctx->usage_dev, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(ctx->usage_event, stream));
            ctx->usage_readback_pending = true;
            return;
        }
        if (cudaEventQuery(ctx->usage_event) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        ctx->usage_readback_pending = false;
        const unsigned long long total  = ctx->usage_host[0] + ctx->usage_host[1];
        const double             staged = (double) ctx->stage_host_bytes + (double) ctx->usage_host[3];
        if (total > 0 || staged > 0) {

            const int64_t copy_us  = ctx->stage_copy_us.load(std::memory_order_relaxed);
            const int64_t n_copy   = ctx->stage_copy_arcs.load(std::memory_order_relaxed);
            const double  gbs      = copy_us > 0 && n_copy > 0
                ? (double) ctx->stage_host_bytes / ((double) copy_us * 1e6)
                : 0.0;
            GGML_LOG_DEBUG("%s: device %d: expert hits vram %llu / host %llu (%.2f%% vram), host hit %.2f MiB, staged %.2f MiB (prefill %.2f, decode %.2f), stage copy %.1f ms over %" PRId64 " arcs, %.1f GB/s; ring arcs ready %" PRId64 " / waited %" PRId64 " (issued %" PRId64 "), prefetch wait %.1f ms over %" PRId64 " launches, ring_epoch %lld\n",
                __func__, ctx->device, ctx->usage_host[0], ctx->usage_host[1],
                total > 0 ? 100.0 * ctx->usage_host[0] / total : 0.0, ctx->usage_host[2] / 1024.0 / 1024.0,
                staged / 1024.0 / 1024.0, ctx->stage_host_bytes / 1024.0 / 1024.0, ctx->usage_host[3] / 1024.0 / 1024.0,
                copy_us / 1000.0, n_copy, gbs,
                ctx->stage_hits.load(std::memory_order_relaxed), ctx->stage_cold.load(std::memory_order_relaxed),
                ctx->prefetch_issued.load(std::memory_order_relaxed), ctx->prefetch_wait_ms, ctx->prefetch_wait_timed,
                (long long) ctx->ring_epoch.load(std::memory_order_relaxed));
        }

        CUDA_CHECK(cudaMemcpyAsync(ctx->usage_host, ctx->usage_dev, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ctx->usage_event, stream));
        ctx->usage_readback_pending = true;
    }
}

void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);
    ctx->launches++;
    if (ctx->launches % 64 != 0) {
        return;
    }
    ggml_cuda_expert_stats_readback(ctx, stream);

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table) || !ex_table.has_stage) {
        return;
    }
    const ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];

    ggml_cuda_expert_stream_state * ssp = nullptr;
    {
        std::lock_guard<std::mutex> lock(ctx->streams_mu);
        const auto it = ctx->streams.find(stream);
        if (it != ctx->streams.end()) {
            ssp = &it->second;
        }
    }
    if (ssp == nullptr || ssp->fault_dev == nullptr) {
        return;
    }
    ggml_cuda_expert_stream_state & ss = *ssp;






    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool capturing = cudaStreamIsCapturing(stream, &capture) == cudaSuccess && capture == cudaStreamCaptureStatusActive;
    if (cudaEventQuery(ss.fault_event) == cudaSuccess) {
        const int fault_decode = *ss.fault_pinned;
        if (fault_decode != 0) {
            const ggml_cuda_expert_stage_region & R = ss.regions[(size_t) s.region];
            GGML_ABORT("%s: tensor '%s' on stream %p: decode staging launch %d did not fit its region: "
                "region %.2f MiB holds %zu experts of %.2f MiB, the host complement of the tensor is %" PRId64 " experts\n",
                __func__, s.tensor != nullptr ? s.tensor->name : "?", (void *) stream, fault_decode,
                (double) R.bytes / 1024.0 / 1024.0, R.bytes / s.blob_stride,
                (double) s.blob_stride / 1024.0 / 1024.0, ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id)->n_host);
        }
        if (!capturing) {
            CUDA_CHECK(cudaMemcpyAsync(ss.fault_pinned, ss.fault_dev, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(ss.fault_event, stream));
        }
    } else {
        (void) cudaGetLastError();
    }

    if (ss.last_stage_n_host >= 0) {
        GGML_LOG_DEBUG("%s: device %d tensor %s: last prefill launch of the stream staged %" PRId64 " host experts (%.2f MiB)\n",
            __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?",
            ss.last_stage_n_host, (double) (ss.last_stage_n_host * (int64_t) s.blob_bytes) / 1024.0 / 1024.0);
    }
}



static void ggml_cuda_expert_stage_copy_timing_drain(ggml_backend_cuda_expert_buffer_context * ctx, ggml_cuda_expert_stream_state & ss) {


    while (ss.wait_ring_count > 0) {
        cudaEvent_t * pair = ss.wait_ev[ss.wait_ring_tail];
        if (cudaEventQuery(pair[1]) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pair[0], pair[1]));
        ctx->prefetch_wait_ms += ms;
        ctx->prefetch_wait_timed++;
        ss.wait_ring_tail = (ss.wait_ring_tail + 1) % ggml_cuda_expert_stream_state::WAIT_TIMING_RING;
        ss.wait_ring_count--;
    }
}
















static const char * const * ggml_cuda_expert_stage_prepare_routed(
        const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table)) {
        return nullptr;
    }
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id);

    ggml_cuda_set_device(ctx->device);



    if (!s.has_host) {
        return s.table_main_dev + ex_table.expert_start;
    }

    GGML_ASSERT(ids != nullptr && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(s.has_stage);




    GGML_LOG_DEBUG("%s: device %d: launch of region %d layer %d took the routed copy\n",
        __func__, ctx->device, s.region,
        s.arc >= 0 && (size_t) s.arc < ctx->arcs.size() ? ctx->arcs[(size_t) s.arc].layer : -1);

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
    GGML_ASSERT(!ss.regions.empty());




    ss.launch_mode = GGML_CUDA_EXPERT_MODE_ROUTED;



    GGML_ASSERT(s.region >= 0 && (size_t) s.region < ss.regions.size());
    ggml_cuda_expert_stage_region & R = ss.regions[(size_t) s.region];
















    GGML_ASSERT(R.base != nullptr && R.bytes >= (size_t) st->host_used);

    const int nthreads = 256;
    const int64_t n_entries = ggml_nelements(ids);
    const int n_expert_used = (int) ids->ne[0];
    const int expert_lo = (int) ex_table.expert_start;
    const int expert_hi = expert_lo + (int) ex_table.n_expert;
    const int n_table = (int) s.n_expert;


    const size_t n_words = ((size_t) s.n_expert + 31) / 32;
    uint32_t * bitmap_dev = ss.bitmaps[(size_t) ex_table.tensor_id];
    GGML_ASSERT(bitmap_dev != nullptr);
    GGML_ASSERT(ss.hit_list != nullptr && ss.hit_cap >= (size_t) s.n_expert);
    GGML_ASSERT(ss.plan_state != nullptr);



    const char ** table_stage = ss.tables_decode[(size_t) ex_table.tensor_id];
    GGML_ASSERT(table_stage != nullptr);


    CUDA_CHECK(cudaMemsetAsync(bitmap_dev, 0, n_words * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ss.plan_state, 0, 2 * sizeof(int), stream));




    const int nblocks = (int) ((n_entries + nthreads - 1) / nthreads);
    expert_routing_kernel<<<nblocks, nthreads, 0, stream>>>(
        (const int32_t *) ids->data, n_entries, n_expert_used, ids->nb[1] / sizeof(int32_t),
        lut, expert_lo, s.residency_dev, (int64_t) ex_table.blob_bytes,
        ctx->usage_dev, nullptr, bitmap_dev);
    CUDA_CHECK(cudaGetLastError());


    const int plan_blocks = (int) ((n_table + nthreads - 1) / nthreads);
    expert_stage_plan_kernel<<<plan_blocks, nthreads, 0, stream>>>(
        bitmap_dev, expert_lo, expert_hi, n_table, s.residency_dev, s.table_main_dev,
        table_stage, R.base, s.blob_stride, s.blob_bytes, R.bytes,
        ss.hit_list, ss.hit_cap, ss.plan_state, ss.plan_state + 1, ss.plan_state + 2, nullptr);
    CUDA_CHECK(cudaGetLastError());


















    ggml_cuda_expert_ring_block_end(*ctx, ss, s.region, stream);

    ggml_cuda_expert_exchange_settle(*ctx, stream);
    const int copy_chunks = (int) ((s.blob_bytes + EXPERT_STAGE_COPY_CHUNK - 1) / EXPERT_STAGE_COPY_CHUNK);
    expert_stage_copy_kernel<<<dim3(copy_chunks, expert_hi - expert_lo), nthreads, 0, stream>>>(
        ss.hit_list, ss.plan_state, st->host_dev, s.host_offset_dev,
        R.base, s.blob_stride, s.blob_bytes, expert_lo, expert_hi, table_stage, ss.plan_state + 3);
    CUDA_CHECK(cudaGetLastError());




    ss.last_stage_n_host = -2;

    return table_stage + ex_table.expert_start;
}














const char * const * ggml_cuda_expert_stage_prepare(const ggml_tensor * src0, int64_t n_tokens,
        const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

















    const int launch_mode = n_tokens >= GGML_CUDA_EXPERT_S3PRIME_MIN_TOKENS
        ? GGML_CUDA_EXPERT_MODE_S3P : GGML_CUDA_EXPERT_MODE_ROUTED;
    if (ctx->prefill_mode == 0) {
        ctx->prefill_mode = launch_mode;
        GGML_LOG_DEBUG("%s: device %d: prefill staging mode of the phase opener %s (batch %lld tokens, threshold %d)\n",
            __func__, ctx->device,
            launch_mode == GGML_CUDA_EXPERT_MODE_S3P
                ? "static whole-complement ring (S3')" : "routed copy (parent)",
            (long long) n_tokens, GGML_CUDA_EXPERT_S3PRIME_MIN_TOKENS);
    }
    if (launch_mode == GGML_CUDA_EXPERT_MODE_ROUTED) {
        return ggml_cuda_expert_stage_prepare_routed(src0, ids, lut, stream);
    }

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table)) {
        return nullptr;
    }
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id);

    ggml_cuda_set_device(ctx->device);



    if (!s.has_host) {
        return s.table_main_dev + ex_table.expert_start;
    }




    GGML_ASSERT(s.has_stage);
    GGML_ASSERT(ctx->stage_budget > 0 && ctx->stage_budget >= (size_t) st->host_used);

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);










    ggml_cuda_expert_exchange_settle(*ctx, stream);









    const int64_t epoch = ctx->swap_epoch.load(std::memory_order_acquire);
    if (epoch != ss.swap_epoch) {
        ggml_cuda_expert_ring_rebuild(*ctx, ss, stream, epoch);
    }




    ss.launch_mode = GGML_CUDA_EXPERT_MODE_S3P;
    GGML_ASSERT(s.region >= 0 && (size_t) s.region < ss.regions.size());
    ggml_cuda_expert_stage_region & R = ss.regions[(size_t) s.region];
    GGML_ASSERT(R.base != nullptr && R.bytes >= (size_t) st->host_used);
    GGML_ASSERT(st->host_host != nullptr && st->n_host > 0);
    const char ** table_static = ss.tables_static[(size_t) ex_table.tensor_id];
    GGML_ASSERT(table_static != nullptr);



    GGML_ASSERT(s.arc >= 0 && (size_t) s.arc < ctx->arcs.size());
    const ggml_cuda_expert_stage_arc & arc = ctx->arcs[(size_t) s.arc];
    GGML_ASSERT(arc.n_host == st->n_host);



    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool timing = cudaStreamIsCapturing(stream, &capture) == cudaSuccess && capture == cudaStreamCaptureStatusNone;
    if (timing) {
        ggml_cuda_expert_stage_copy_timing_drain(ctx, ss);
    }















    ggml_cuda_expert_stage_ring & ring = ss.rings[(size_t) s.region];


    GGML_ASSERT(arc.n_host <= (int64_t) ring.slots);
    std::atomic<uint64_t> & blocks = ss.ctl->blocks[(size_t) s.region];
    if (!ss.block_barrier_armed) {

        CUDA_CHECK(cudaEventRecord(ss.block_barrier, stream));
        ss.block_barrier_armed = true;




        std::lock_guard<std::mutex> lock(ss.ctl->mu);
        for (size_t r = 0; r < ss.rings.size(); ++r) {
            ss.ctl->blocks[r].fetch_add(1, std::memory_order_release);


            ggml_cuda_expert_ring_block_open(*ctx, ss, (int) r);
        }
        ring.last_layer = arc.layer;
        ss.ctl->cv.notify_all();
    } else if (arc.layer <= ring.last_layer) {
        CUDA_CHECK(cudaEventRecord(ss.block_barrier, stream));
        ring.last_layer = arc.layer;






        ggml_cuda_expert_ring_block_end(*ctx, ss, s.region, stream);
        {

            std::lock_guard<std::mutex> lock(ss.ctl->mu);
            blocks.fetch_add(1, std::memory_order_release);
        }
        ggml_cuda_expert_ring_block_open(*ctx, ss, s.region);
        ss.ctl->cv.notify_all();
    } else {
        ring.last_layer = arc.layer;
    }






    const int64_t block = (int64_t) blocks.load(std::memory_order_acquire);
    const auto    ring_watchdog_start = std::chrono::steady_clock::now();
    int           ring_watchdog_turns = 0;
    while (ss.ctl->arc_issued[(size_t) s.arc].load(std::memory_order_acquire) < block) {


        if (++ring_watchdog_turns % RING_WATCHDOG_SPIN_READS == 0) {
            const int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - ring_watchdog_start).count();
            if (elapsed_us >= RING_WATCHDOG_US) {
                ggml_cuda_expert_ring_watchdog_wait w;
                w.site       = "ring-watchdog: consumer";
                w.region     = s.region;
                w.arc        = s.arc;
                w.block      = block;
                w.elapsed_us = elapsed_us;
                ggml_cuda_expert_ring_watchdog_dump(ctx, ss, w);
                GGML_ABORT("%s: device %d: the arc of layer %d (arc %d) was not issued for %" PRId64
                    " ms in block %" PRId64 " (issued %" PRId64 ")\n",
                    __func__, ctx->device, arc.layer, s.arc, elapsed_us / 1000, block,
                    ss.ctl->arc_issued[(size_t) s.arc].load(std::memory_order_acquire));
            }
        }
        std::this_thread::yield();
    }


    const bool ready = cudaEventQuery(ss.arc_copied[(size_t) s.arc]) == cudaSuccess;
    if (!ready) {
        (void) cudaGetLastError();
    }



    const bool timed = timing && ss.wait_ring_count < ggml_cuda_expert_stream_state::WAIT_TIMING_RING;
    if (timed) {
        CUDA_CHECK(cudaEventRecord(ss.wait_ev[ss.wait_ring_head][0], stream));
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, ss.arc_copied[(size_t) s.arc], 0));
    if (timed) {
        CUDA_CHECK(cudaEventRecord(ss.wait_ev[ss.wait_ring_head][1], stream));
        ss.wait_ring_head = (ss.wait_ring_head + 1) % ggml_cuda_expert_stream_state::WAIT_TIMING_RING;
        ss.wait_ring_count++;
    }
    if (ready) {
        ctx->stage_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        ctx->stage_cold.fetch_add(1, std::memory_order_relaxed);
    }

    ss.last_stage_n_host  = st->n_host;
    ctx->stage_host_bytes += st->n_host * (int64_t) s.blob_bytes;

    return table_static + ex_table.expert_start;
}







void ggml_cuda_expert_stage_consumed(const ggml_tensor * src0, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table)) {
        return;
    }






    ggml_cuda_expert_stream_state * ssp = nullptr;
    int launch_mode = ctx->prefill_mode;
    {
        std::lock_guard<std::mutex> lock(ctx->streams_mu);
        const auto it = ctx->streams.find(stream);
        if (it != ctx->streams.end()) {
            ssp = &it->second;
            if (ssp->launch_mode != 0) {
                launch_mode = ssp->launch_mode;
            }
        }
    }
    if (launch_mode != GGML_CUDA_EXPERT_MODE_S3P) {
        return;
    }

    const ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    if (!s.has_host || !s.has_stage) {
        return;
    }
    if (ssp == nullptr) {
        return;
    }
    ggml_cuda_expert_stream_state & ss = *ssp;
    GGML_ASSERT(s.arc >= 0 && (size_t) s.arc < ctx->arcs.size());





    CUDA_CHECK(cudaEventRecord(ss.arc_released[(size_t) s.arc], stream));
    {
        std::lock_guard<std::mutex> lock(ss.ctl->mu);
        ss.ctl->arc_consumed[(size_t) s.arc].store((int64_t) ss.ctl->blocks[(size_t) s.region].load(), std::memory_order_release);
    }
    ss.ctl->cv.notify_all();






    const std::vector<int> & ring_arcs = ss.rings[(size_t) s.region].arcs;
    if (!ring_arcs.empty() && ring_arcs.back() == s.arc) {
        ggml_cuda_expert_ring_block_end(*ctx, ss, s.region, stream);
    }
}






const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);




    ctx->prefill_mode = 0;








    {
        ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
        for (int r = 0; r < (int) ss.rings.size(); ++r) {
            ggml_cuda_expert_ring_block_end(*ctx, ss, r, stream);
        }
    }

    ggml_cuda_expert_table ex_table;
    GGML_ASSERT(ggml_cuda_expert_table_get(src0, &ex_table));
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) ex_table.tensor_id];
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, ex_table.tensor_id);

    ggml_cuda_set_device(ctx->device);



    if (!s.has_host) {
        return s.table_main_dev + ex_table.expert_start;
    }




    GGML_ASSERT(ids != nullptr && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ex_table.has_stage);
    GGML_ASSERT(ctx->stage_budget > 0 && ctx->stage_budget >= (size_t) st->host_used);

    ggml_cuda_expert_stream_state & ss = ctx->get_stream(stream);
    GGML_ASSERT(s.region >= 0 && (size_t) s.region < ss.regions.size());
    ggml_cuda_expert_stage_region & R = ss.regions[(size_t) s.region];
    GGML_ASSERT(R.base != nullptr && R.bytes >= s.blob_stride);

    const int n_entries = (int) ggml_nelements(ids);
    const int n_expert_used = (int) ids->ne[0];
    GGML_ASSERT(n_entries > 0 && n_entries <= (int) (32 * 1024 / sizeof(int32_t)));
    GGML_ASSERT((size_t) std::min<int64_t>(n_entries, st->n_host) * s.blob_stride <= R.bytes);



    const char ** table_decode = ss.tables_decode[(size_t) ex_table.tensor_id];
    GGML_ASSERT(table_decode != nullptr);

    const int nthreads = 256;
    const int expert_lo = (int) ex_table.expert_start;
    const int expert_hi = expert_lo + (int) ex_table.n_expert;
    const int n_chunks = (int) ((s.blob_bytes + EXPERT_STAGE_COPY_CHUNK - 1) / EXPERT_STAGE_COPY_CHUNK);











    if (ctx->engine) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->engine->ev_bcopy,
            capture == cudaStreamCaptureStatusNone ? cudaEventWaitDefault : cudaEventWaitExternal));
    }












    {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        for (int r = 0; r < (int) ss.rings.size(); ++r) {
            const std::vector<int> & arcs = ss.rings[(size_t) r].arcs;
            if (arcs.empty()) {
                continue;
            }



            if (ss.ctl->blocks[(size_t) r].load(std::memory_order_relaxed) == 0) {
                continue;
            }
            CUDA_CHECK(cudaStreamWaitEvent(stream, ss.arc_copied[(size_t) arcs.back()],
                capture == cudaStreamCaptureStatusNone ? cudaEventWaitDefault : cudaEventWaitExternal));
        }
    }



    expert_stage_decode_kernel<<<dim3(n_chunks, (unsigned int) s.n_expert), nthreads,
        ((n_entries + 3) & ~3) * sizeof(int32_t), stream>>>(
        (const int32_t *) ids->data, n_entries, n_expert_used, ids->nb[1] / sizeof(int32_t),
        lut, (int) ex_table.expert_start, expert_lo, expert_hi,
        s.residency_dev, st->host_dev, s.host_offset_dev, s.table_main_dev, table_decode,
        R.base, s.blob_stride, s.blob_bytes, R.bytes,
        ++ss.stage_seq, ss.fault_dev, ctx->usage_dev,
        ctx->engine ? ctx->engine->count_ptr[(size_t) ex_table.tensor_id] : nullptr);
    CUDA_CHECK(cudaGetLastError());








    return table_decode + ex_table.expert_start;
}


static bool ggml_cuda_expert_swap_target(const ggml_tensor * weight,
        ggml_backend_cuda_expert_buffer_context ** ctx_out, ggml_cuda_expert_tensor_state ** s_out);






int ggml_cuda_expert_counts_read(const ggml_tensor * weight, uint32_t * out, bool reset, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = nullptr;
    ggml_cuda_expert_tensor_state * sp = nullptr;
    if (weight == nullptr || out == nullptr ||
        !ggml_cuda_expert_swap_target(weight, &ctx, &sp) || !ctx->engine) {
        return -1;
    }
    ggml_cuda_expert_engine & eng = *ctx->engine;
    std::lock_guard<std::mutex> lock(eng.mutex);
    const int row = eng.row_of_tensor[(size_t) sp->tensor_id];
    if (row < 0) {
        return -1;
    }
    ggml_cuda_set_device(ctx->device);

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        return -1;
    }

    const int64_t n       = eng.rows[(size_t) row].n_expert;
    uint32_t *    src     = eng.counts_dev + eng.count_off[(size_t) row];
    const size_t  n_bytes = (size_t) n * sizeof(uint32_t);
    CUDA_CHECK(cudaMemcpyAsync(out, src, n_bytes, cudaMemcpyDeviceToHost, stream));
    if (reset) {


        CUDA_CHECK(cudaMemsetAsync(src, 0, n_bytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return (int) n;
}






static bool ggml_cuda_expert_swap_target(const ggml_tensor * weight,
        ggml_backend_cuda_expert_buffer_context ** ctx_out, ggml_cuda_expert_tensor_state ** s_out) {
    if (weight == nullptr || weight->buffer == nullptr || !ggml_cuda_expert_buft_is_expert(weight->buffer->buft)) {
        return false;
    }
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(weight);
    if (ctx == nullptr || ctx->store == nullptr) {
        return false;
    }
    int64_t view_offset = 0;
    const ggml_tensor * store_tensor = ggml_cuda_expert_trace_view(weight, &view_offset);
    if (store_tensor == nullptr || view_offset != 0) {
        return false;
    }
    const int tensor_id = ggml_expert_store_tensor_id(ctx->store, store_tensor);
    if (tensor_id < 0 || (size_t) tensor_id >= ctx->tensors.size() ||
            (size_t) tensor_id >= ctx->store->tensors.size()) {
        return false;
    }
    *ctx_out = ctx;
    *s_out   = &ctx->tensors[(size_t) tensor_id];
    return true;
}



static ggml_expert_store_tensor * ggml_cuda_expert_engine_st(ggml_backend_cuda_expert_buffer_context & ctx, int t) {
    return &ctx.store->tensors[(size_t) ctx.tensors[(size_t) t].tensor_id];
}


static cudaEvent_t ggml_cuda_expert_engine_event(ggml_cuda_expert_engine & eng) {
    if (!eng.ev_free.empty()) {
        cudaEvent_t ev = eng.ev_free.back();
        eng.ev_free.pop_back();
        return ev;
    }
    cudaEvent_t ev = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    return ev;
}


static void ggml_cuda_expert_engine_copy(cudaStream_t stream, const std::vector<ggml_cuda_expert_xcopy> & copies) {
    for (size_t first = 0; first < copies.size(); first += GGML_CUDA_EXPERT_XCOPY_MAX) {
        const size_t n = std::min<size_t>(GGML_CUDA_EXPERT_XCOPY_MAX, copies.size() - first);
        ggml_cuda_expert_xcopy_arg arg = {};
        std::copy(copies.begin() + first, copies.begin() + first + n, arg.e);
        expert_xcopy_kernel<<<GGML_CUDA_EXPERT_XCOPY_BLOCKS, 256, 0, stream>>>(arg, (int) n);
        CUDA_CHECK(cudaGetLastError());
    }
}


static void ggml_cuda_expert_engine_flip(ggml_cuda_expert_engine & eng, cudaStream_t stream, const std::vector<ggml_cuda_expert_xflip> & flips) {
    for (size_t first = 0; first < flips.size(); first += GGML_CUDA_EXPERT_XFLIP_MAX) {
        const size_t n = std::min<size_t>(GGML_CUDA_EXPERT_XFLIP_MAX, flips.size() - first);
        ggml_cuda_expert_xflip_arg arg = {};
        std::copy(flips.begin() + first, flips.begin() + first + n, arg.e);
        expert_xflip_kernel<<<(unsigned int) ((n + 31) / 32), 32, 0, stream>>>(arg, (int) n, eng.tables_dev);
        CUDA_CHECK(cudaGetLastError());
        eng.n_flips++;
    }
}



static bool ggml_cuda_expert_engine_row_ok(const ggml_backend_cuda_expert_buffer_context & ctx, const ggml_cuda_expert_engine & eng, int row) {
    if (eng.driver == nullptr) {
        return false;
    }
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) row];
    for (int k = 0; k < r.n_tensor; ++k) {
        const ggml_cuda_expert_tensor_state & s = ctx.tensors[(size_t) r.tensor[k]];
        if (s.reader_shared || s.reader != eng.driver) {
            return false;
        }
    }
    return true;
}



static bool ggml_cuda_expert_engine_take_slots(ggml_cuda_expert_engine & eng, ggml_cuda_expert_xchg & x) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        int need = 0;
        for (int j = 0; j < r.n_tensor; ++j) {
            need += r.pool[j] == r.pool[k];
        }
        if ((int) eng.pools[(size_t) r.pool[k]].free.size() < need) {
            return false;
        }
    }
    for (int k = 0; k < r.n_tensor; ++k) {
        ggml_cuda_expert_xpool & pool = eng.pools[(size_t) r.pool[k]];
        x.fresh[k] = pool.free.back();
        pool.free.pop_back();
    }
    eng.state[(size_t) x.row][(size_t) x.e_in]  = GGML_CUDA_EXPERT_XS_BUSY;
    eng.state[(size_t) x.row][(size_t) x.e_out] = GGML_CUDA_EXPERT_XS_BUSY;
    eng.n_inflight++;
    eng.inflight_peak = std::max(eng.inflight_peak, eng.n_inflight);
    return true;
}


static void ggml_cuda_expert_engine_copies_in(ggml_backend_cuda_expert_buffer_context & ctx, const ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xchg & x, std::vector<ggml_cuda_expert_xcopy> & out) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        const ggml_expert_store_tensor * st = ggml_cuda_expert_engine_st(ctx, r.tensor[k]);
        out.push_back({ st->host_dev + st->host_offset[(size_t) x.e_in], x.fresh[k], (unsigned long long) st->stride });
    }
}


static void ggml_cuda_expert_engine_flips_in(ggml_backend_cuda_expert_buffer_context & ctx, const ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xchg & x, std::vector<ggml_cuda_expert_xflip> & out) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        const ggml_expert_store_tensor * st = ggml_cuda_expert_engine_st(ctx, r.tensor[k]);
        out.push_back({ x.fresh[k], -1, r.tensor[k], x.e_in, st->residency[(size_t) x.e_out], 0 });
    }
}


static void ggml_cuda_expert_engine_copies_out(ggml_backend_cuda_expert_buffer_context & ctx, const ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xchg & x, std::vector<ggml_cuda_expert_xcopy> & out) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        const ggml_expert_store_tensor * st = ggml_cuda_expert_engine_st(ctx, r.tensor[k]);
        out.push_back({ st->table_host[(size_t) x.e_out], st->host_dev + st->host_offset[(size_t) x.e_in], (unsigned long long) st->stride });
    }
}


static void ggml_cuda_expert_engine_flips_out(ggml_backend_cuda_expert_buffer_context & ctx, const ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xchg & x, std::vector<ggml_cuda_expert_xflip> & out) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        const ggml_expert_store_tensor * st = ggml_cuda_expert_engine_st(ctx, r.tensor[k]);
        const int64_t off = st->host_offset[(size_t) x.e_in];
        out.push_back({ st->host_dev + off, (long long) off, r.tensor[k], x.e_out, -1, 0 });
    }
}







































static void ggml_cuda_expert_engine_quiesce(ggml_backend_cuda_expert_buffer_context & ctx, bool now) {
    const int64_t open = ctx.ring_blocks_open.load(std::memory_order_acquire);
    if (open == 0) {
        return;
    }
    if (now) {
        GGML_ABORT("%s: device %d: %" PRId64 " prefill blocks of the rings of the device are open, but the exchange has to settle now\n",
            __func__, ctx.device, open);
    }
    GGML_LOG_DEBUG("%s: device %d: a prefill block is open, the exchange waits for it\n",
        __func__, ctx.device);
    int64_t waited_ms = 0;
    while (ctx.ring_blocks_open.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited_ms++;
    }
    GGML_LOG_DEBUG("%s: device %d: the prefill block is over, the exchange goes on (waited %" PRId64 " ms)\n",
        __func__, ctx.device, waited_ms);
}







static void ggml_cuda_expert_engine_commit(ggml_backend_cuda_expert_buffer_context & ctx, ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xchg & x, bool quiesce_now) {


    ggml_cuda_expert_engine_quiesce(ctx, quiesce_now);

    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        ggml_expert_store_tensor * st = ggml_cuda_expert_engine_st(ctx, r.tensor[k]);
        char * old = nullptr;



        const bool swapped = ggml_expert_store_tensor_swap_to(st, x.e_in, x.e_out, x.fresh[k], &old);
        if (!swapped) {
            ctx.swap_in_flight.store(0, std::memory_order_release);
            GGML_ABORT("%s: device %d: the exchange of the expert %lld into the slot of the expert %lld failed\n",
                __func__, ctx.device, (long long) x.e_in, (long long) x.e_out);
        }
        eng.pools[(size_t) r.pool[k]].free.push_back(old);
    }




    ctx.swap_epoch.store(ggml_expert_store_swap_epoch(ctx.store), std::memory_order_release);




    ctx.swap_in_flight.store(0, std::memory_order_release);
    eng.state[(size_t) x.row][(size_t) x.e_in]  = GGML_CUDA_EXPERT_XS_SLOT;
    eng.state[(size_t) x.row][(size_t) x.e_out] = GGML_CUDA_EXPERT_XS_HOST;
    eng.n_done++;
    eng.n_inflight--;
}


static void ggml_cuda_expert_engine_drop(ggml_cuda_expert_engine & eng, const ggml_cuda_expert_xchg & x) {
    const ggml_cuda_expert_xrow & r = eng.rows[(size_t) x.row];
    for (int k = 0; k < r.n_tensor; ++k) {
        eng.pools[(size_t) r.pool[k]].free.push_back(x.fresh[k]);
    }
    eng.state[(size_t) x.row][(size_t) x.e_in]  = GGML_CUDA_EXPERT_XS_HOST;
    eng.state[(size_t) x.row][(size_t) x.e_out] = GGML_CUDA_EXPERT_XS_SLOT;
    eng.n_dropped++;
    eng.n_inflight--;
}



static void ggml_cuda_expert_engine_landed(ggml_cuda_expert_engine & eng, std::deque<ggml_cuda_expert_xbatch> & q,
        std::vector<ggml_cuda_expert_xchg> & out) {
    while (!q.empty()) {
        const cudaError_t err = cudaEventQuery(q.front().ev);
        if (err == cudaErrorNotReady) {
            (void) cudaGetLastError();
            break;
        }
        CUDA_CHECK(err);
        out.insert(out.end(), q.front().xs.begin(), q.front().xs.end());
        eng.ev_free.push_back(q.front().ev);
        q.pop_front();
    }
}



static void ggml_cuda_expert_engine_advance(ggml_backend_cuda_expert_buffer_context & ctx, ggml_cuda_expert_engine & eng, cudaStream_t stream) {
    std::vector<ggml_cuda_expert_xchg> in;
    std::vector<ggml_cuda_expert_xchg> out;
    ggml_cuda_expert_engine_landed(eng, eng.copying, in);
    ggml_cuda_expert_engine_landed(eng, eng.leaving, out);
    if (in.empty() && out.empty()) {
        return;
    }

    std::vector<ggml_cuda_expert_xflip> flips;
    for (const ggml_cuda_expert_xchg & x : in) {
        ggml_cuda_expert_engine_flips_in(ctx, eng, x, flips);
    }
    for (const ggml_cuda_expert_xchg & x : out) {
        ggml_cuda_expert_engine_flips_out(ctx, eng, x, flips);
    }
    ggml_cuda_expert_engine_flip(eng, stream, flips);


    CUDA_CHECK(cudaEventRecord(eng.ev_flip, stream));
    CUDA_CHECK(cudaStreamWaitEvent(eng.copy, eng.ev_flip, 0));

    for (const ggml_cuda_expert_xchg & x : out) {
        ggml_cuda_expert_engine_commit(ctx, eng, x, false);
    }

    if (!in.empty()) {
        std::vector<ggml_cuda_expert_xcopy> copies;
        for (const ggml_cuda_expert_xchg & x : in) {
            ggml_cuda_expert_engine_copies_out(ctx, eng, x, copies);
        }








        ggml_cuda_expert_engine_quiesce(ctx, false);
        ggml_cuda_expert_engine_copy(eng.copy, copies);






        ctx.swap_in_flight.store(1, std::memory_order_release);
        ggml_cuda_expert_xbatch b;
        b.xs = std::move(in);
        b.ev = ggml_cuda_expert_engine_event(eng);
        CUDA_CHECK(cudaEventRecord(b.ev, eng.copy));






        CUDA_CHECK(cudaEventRecord(eng.ev_bcopy, eng.copy));
        eng.leaving.push_back(std::move(b));
    }
}




static void ggml_cuda_expert_engine_issue(ggml_backend_cuda_expert_buffer_context & ctx, ggml_cuda_expert_engine & eng, int64_t cap) {
    std::vector<ggml_cuda_expert_xchg> xs;
    std::vector<ggml_cuda_expert_xcopy> copies;
    while (!eng.todo.empty() && (int64_t) xs.size() < cap) {
        const ggml_cuda_expert_xreq q = eng.todo.front();
        const std::vector<uint8_t> & st = eng.state[(size_t) q.row];
        if (!ggml_cuda_expert_engine_row_ok(ctx, eng, q.row) ||
                st[(size_t) q.e_in] != GGML_CUDA_EXPERT_XS_HOST || st[(size_t) q.e_out] != GGML_CUDA_EXPERT_XS_SLOT) {
            eng.todo.pop_front();
            continue;
        }
        ggml_cuda_expert_xchg x;
        x.row   = q.row;
        x.e_in  = q.e_in;
        x.e_out = q.e_out;
        if (!ggml_cuda_expert_engine_take_slots(eng, x)) {
            break;
        }
        eng.todo.pop_front();
        ggml_cuda_expert_engine_copies_in(ctx, eng, x, copies);
        xs.push_back(x);
    }
    if (xs.empty()) {
        return;
    }
    ggml_cuda_expert_engine_copy(eng.copy, copies);
    ggml_cuda_expert_xbatch b;
    b.ev = ggml_cuda_expert_engine_event(eng);
    CUDA_CHECK(cudaEventRecord(b.ev, eng.copy));
    eng.n_issued += (int64_t) xs.size();
    b.xs = std::move(xs);
    eng.copying.push_back(std::move(b));
}










static void ggml_cuda_expert_engine_drain(ggml_backend_cuda_expert_buffer_context & ctx,
        ggml_cuda_expert_engine & eng, bool quiesce_now) {
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (const ggml_cuda_expert_xbatch & b : eng.copying) {
        for (const ggml_cuda_expert_xchg & x : b.xs) {
            ggml_cuda_expert_engine_drop(eng, x);
        }
        eng.ev_free.push_back(b.ev);
    }
    eng.copying.clear();

    std::vector<ggml_cuda_expert_xchg> out;
    for (const ggml_cuda_expert_xbatch & b : eng.leaving) {
        out.insert(out.end(), b.xs.begin(), b.xs.end());
        eng.ev_free.push_back(b.ev);
    }
    eng.leaving.clear();
    if (!out.empty()) {
        std::vector<ggml_cuda_expert_xflip> flips;
        for (const ggml_cuda_expert_xchg & x : out) {
            ggml_cuda_expert_engine_flips_out(ctx, eng, x, flips);
        }
        ggml_cuda_expert_engine_flip(eng, eng.copy, flips);




        CUDA_CHECK(cudaEventRecord(eng.ev_done, eng.copy));
        CUDA_CHECK(cudaStreamSynchronize(eng.copy));
        for (const ggml_cuda_expert_xchg & x : out) {
            ggml_cuda_expert_engine_commit(ctx, eng, x, quiesce_now);
        }
    } else {


        CUDA_CHECK(cudaEventRecord(eng.ev_done, eng.copy));
    }
    eng.todo.clear();



    ctx.swap_in_flight.store(0, std::memory_order_release);
}

void ggml_cuda_expert_note_reader(const ggml_tensor * src0, int tensor_id, cudaStream_t stream) {
    ggml_backend_cuda_expert_buffer_context * ctx = ggml_cuda_expert_buffer_ctx(src0);
    if (tensor_id < 0 || (size_t) tensor_id >= ctx->tensors.size()) {
        return;
    }
    ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) tensor_id];
    if (s.reader == stream || s.reader_shared) {
        return;
    }
    if (s.reader == nullptr) {
        s.reader = stream;
        return;
    }



    s.reader_shared = true;
    GGML_LOG_DEBUG("%s: device %d: tensor %s is read by a second stream, its layer keeps its expert placement\n",
        __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?");
    if (!ctx->engine) {
        return;
    }
    ggml_cuda_expert_engine & eng = *ctx->engine;
    std::lock_guard<std::mutex> lock(eng.mutex);
    const int row = eng.row_of_tensor[(size_t) tensor_id];
    bool in_flight = false;
    for (const auto * q : { &eng.copying, &eng.leaving }) {
        for (const ggml_cuda_expert_xbatch & b : *q) {
            for (const ggml_cuda_expert_xchg & x : b.xs) {
                in_flight = in_flight || (row >= 0 && x.row == row);
            }
        }
    }
    if (!in_flight) {
        return;
    }




    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        GGML_ABORT("%s: device %d: tensor %s is first read by a second stream while it is being captured, with exchanges of its layer in flight\n",
            __func__, ctx->device, s.tensor != nullptr ? s.tensor->name : "?");
    }
    ggml_cuda_expert_engine_drain(*ctx, eng, false);
}







int ggml_cuda_expert_swap_test(ggml_tensor * weight, const int32_t * pairs, int n_pairs, int32_t * pairs_out) {
    if (n_pairs <= 0) {
        return -1;
    }
    ggml_backend_cuda_expert_buffer_context * ctx = nullptr;
    ggml_cuda_expert_tensor_state * sp = nullptr;
    if (!ggml_cuda_expert_swap_target(weight, &ctx, &sp)) {
        GGML_LOG_ERROR("%s: '%s' is not an expert weight of a store tensor\n", __func__,
            weight != nullptr ? weight->name : "?");
        return -1;
    }
    if (!ctx->engine) {
        GGML_LOG_ERROR("%s: the expert buffer of '%s' has no spare slots, nothing can be exchanged\n", __func__, weight->name);
        return -1;
    }
    ggml_cuda_expert_engine & eng = *ctx->engine;
    std::lock_guard<std::mutex> lock(eng.mutex);
    const int row = eng.row_of_tensor[(size_t) sp->tensor_id];
    if (row < 0) {
        GGML_LOG_ERROR("%s: '%s' has no host tier, nothing can be exchanged\n", __func__, weight->name);
        return -1;
    }
    ggml_cuda_set_device(ctx->device);
    ggml_cuda_expert_engine_drain(*ctx, eng, false);

    std::vector<uint8_t> & state = eng.state[(size_t) row];
    std::vector<int32_t> drawn;
    if (pairs == nullptr) {
        std::vector<int32_t> hosts;
        std::vector<int32_t> slots;
        for (size_t e = 0; e < state.size(); ++e) {
            if (state[e] == GGML_CUDA_EXPERT_XS_HOST) {
                hosts.push_back((int32_t) e);
            } else if (state[e] == GGML_CUDA_EXPERT_XS_SLOT) {
                slots.push_back((int32_t) e);
            }
        }
        const int n = std::min(std::min(n_pairs, (int) hosts.size()), (int) slots.size());
        if (n <= 0) {
            GGML_LOG_ERROR("%s: '%s': %zu host and %zu slot experts, cannot exchange %d pairs\n",
                __func__, weight->name, hosts.size(), slots.size(), n_pairs);
            return -1;
        }
        std::mt19937 rng(0x5eed1234u + (unsigned int) n);
        std::shuffle(hosts.begin(), hosts.end(), rng);
        std::shuffle(slots.begin(), slots.end(), rng);
        drawn.resize((size_t) 2 * n);
        for (int i = 0; i < n; ++i) {
            drawn[(size_t) (2*i)]     = hosts[(size_t) i];
            drawn[(size_t) (2*i + 1)] = slots[(size_t) i];
        }
        pairs   = drawn.data();
        n_pairs = n;
    }

    std::vector<ggml_cuda_expert_xchg> xs;
    std::vector<uint8_t> seen(state.size(), 0);
    bool ok = true;
    for (int i = 0; i < n_pairs && ok; ++i) {
        const int32_t e_in  = pairs[2*i];
        const int32_t e_out = pairs[2*i + 1];
        ok = e_in >= 0 && e_out >= 0 && (size_t) e_in < state.size() && (size_t) e_out < state.size() &&
             e_in != e_out && !seen[(size_t) e_in] && !seen[(size_t) e_out] &&
             state[(size_t) e_in] == GGML_CUDA_EXPERT_XS_HOST && state[(size_t) e_out] == GGML_CUDA_EXPERT_XS_SLOT;
        if (!ok) {
            GGML_LOG_ERROR("%s: '%s': pair %d (%d, %d) is not (host expert, slot expert) of distinct experts\n",
                __func__, weight->name, i, (int) e_in, (int) e_out);
            break;
        }
        seen[(size_t) e_in]  = 1;
        seen[(size_t) e_out] = 1;
        ggml_cuda_expert_xchg x;
        x.row   = row;
        x.e_in  = e_in;
        x.e_out = e_out;
        ok = ggml_cuda_expert_engine_take_slots(eng, x);
        if (!ok) {
            GGML_LOG_ERROR("%s: '%s': no spare slots left for pair %d\n", __func__, weight->name, i);
            break;
        }
        xs.push_back(x);
    }
    if (!ok) {
        for (const ggml_cuda_expert_xchg & x : xs) {
            ggml_cuda_expert_engine_drop(eng, x);
        }
        return -1;
    }


    std::vector<ggml_cuda_expert_xcopy> copies;
    std::vector<ggml_cuda_expert_xflip> flips;
    for (const ggml_cuda_expert_xchg & x : xs) {
        ggml_cuda_expert_engine_copies_in(*ctx, eng, x, copies);
    }
    ggml_cuda_expert_engine_copy(eng.copy, copies);
    CUDA_CHECK(cudaStreamSynchronize(eng.copy));
    for (const ggml_cuda_expert_xchg & x : xs) {
        ggml_cuda_expert_engine_flips_in(*ctx, eng, x, flips);
    }
    ggml_cuda_expert_engine_flip(eng, eng.copy, flips);
    CUDA_CHECK(cudaStreamSynchronize(eng.copy));
    copies.clear();
    for (const ggml_cuda_expert_xchg & x : xs) {
        ggml_cuda_expert_engine_copies_out(*ctx, eng, x, copies);
    }


    ggml_cuda_expert_engine_quiesce(*ctx, false);
    ggml_cuda_expert_engine_copy(eng.copy, copies);
    CUDA_CHECK(cudaStreamSynchronize(eng.copy));


    ctx->swap_in_flight.store(1, std::memory_order_release);
    flips.clear();
    for (const ggml_cuda_expert_xchg & x : xs) {
        ggml_cuda_expert_engine_flips_out(*ctx, eng, x, flips);
    }
    ggml_cuda_expert_engine_flip(eng, eng.copy, flips);
    CUDA_CHECK(cudaStreamSynchronize(eng.copy));
    for (const ggml_cuda_expert_xchg & x : xs) {
        ggml_cuda_expert_engine_commit(*ctx, eng, x, false);
    }
    eng.n_issued += (int64_t) xs.size();

    if (pairs_out != nullptr) {
        memcpy(pairs_out, pairs, (size_t) (2 * n_pairs) * sizeof(int32_t));
    }
    return n_pairs;
}





int ggml_cuda_expert_residency(const ggml_tensor * weight, int32_t * out) {
    ggml_backend_cuda_expert_buffer_context * ctx = nullptr;
    ggml_cuda_expert_tensor_state * sp = nullptr;
    if (weight == nullptr || out == nullptr ||
            !ggml_cuda_expert_swap_target(weight, &ctx, &sp)) {
        return -1;
    }
    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, sp->tensor_id);
    if (st == nullptr) {
        return -1;
    }
    memcpy(out, st->residency.data(), st->residency.size() * sizeof(int32_t));
    return (int) st->residency.size();
}








#define GGML_CUDA_EXPERT_DECODE_MAX_TOKENS 16






static const ggml_tensor * ggml_cuda_expert_graph_first_mmid(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {


        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID) {
            return node;
        }
    }
    return nullptr;
}











struct ggml_cuda_expert_xcand {
    double  gain;
    int     row;
    int32_t e_in;
    int32_t e_out;
};

static std::deque<ggml_cuda_expert_xreq> ggml_cuda_expert_engine_decide(ggml_cuda_expert_engine & eng,
        const ggml_cuda_expert_xjob & job, const ggml_expert_store_adaptive_cfg & cfg) {
    const uint32_t * counts = eng.counts_host[job.mirror];
    std::vector<ggml_cuda_expert_xcand> cand;
    std::vector<int32_t> slots;
    std::vector<int32_t> hosts;

    for (size_t ri = 0; ri < eng.rows.size(); ++ri) {
        const int64_t n_expert = eng.rows[ri].n_expert;
        std::vector<double> & score = eng.score[ri];
        const uint32_t * c = counts + eng.count_off[ri];

        for (int64_t e = 0; e < n_expert; ++e) {
            score[(size_t) e] = cfg.decay * score[(size_t) e] + (double) c[(size_t) e];
        }

        slots.clear();
        hosts.clear();
        const std::vector<uint8_t> & state = job.state[ri];
        for (int64_t e = 0; e < n_expert; ++e) {
            if (state[(size_t) e] == GGML_CUDA_EXPERT_XS_SLOT) {
                slots.push_back((int32_t) e);
            } else if (state[(size_t) e] == GGML_CUDA_EXPERT_XS_HOST) {
                hosts.push_back((int32_t) e);
            }
        }
        if (slots.empty() || hosts.empty()) {
            continue;
        }


        const auto by_count = [&score](int32_t a, int32_t b, bool desc) {
            const double sa = score[(size_t) a];
            const double sb = score[(size_t) b];
            if (sa != sb) {
                return desc ? sa > sb : sa < sb;
            }
            return a < b;
        };
        std::sort(hosts.begin(), hosts.end(), [&](int32_t a, int32_t b) { return by_count(a, b, true); });
        std::sort(slots.begin(), slots.end(), [&](int32_t a, int32_t b) { return by_count(a, b, false); });





        const size_t n_pairs = std::min(slots.size(), hosts.size());
        for (size_t i = 0; i < n_pairs && (int64_t) i < job.budget; ++i) {
            const double s_in  = score[(size_t) hosts[i]];
            const double s_out = score[(size_t) slots[i]];
            if (s_in < cfg.ratio * s_out || s_in - s_out < cfg.min_gain) {
                break;
            }
            cand.push_back({ s_in - s_out, (int) ri, hosts[i], slots[i] });
        }
    }


    std::sort(cand.begin(), cand.end(), [](const ggml_cuda_expert_xcand & a, const ggml_cuda_expert_xcand & b) {
        if (a.gain != b.gain) {
            return a.gain > b.gain;
        }
        if (a.row != b.row) {
            return a.row < b.row;
        }
        return a.e_in < b.e_in;
    });
    std::deque<ggml_cuda_expert_xreq> out;
    for (size_t i = 0; i < cand.size() && (int64_t) i < job.budget; ++i) {
        out.push_back({ cand[i].row, cand[i].e_in, cand[i].e_out });
    }
    return out;
}


static void ggml_cuda_expert_engine_worker(ggml_cuda_expert_engine * eng, int device) {
    ggml_cuda_set_device(device);
    while (true) {
        ggml_cuda_expert_xjob job;
        {
            std::unique_lock<std::mutex> lock(eng->job_mutex);
            eng->job_cv.wait(lock, [eng] { return eng->job_stop || eng->job_ready; });
            if (eng->job_stop) {
                return;
            }
            job = std::move(eng->job);
            eng->job_ready = false;
        }


        CUDA_CHECK(cudaEventSynchronize(eng->counts_ev[job.mirror]));
        std::deque<ggml_cuda_expert_xreq> decided = ggml_cuda_expert_engine_decide(*eng, job, ggml_expert_store_get_adaptive());
        {
            std::lock_guard<std::mutex> lock(eng->job_mutex);
            eng->counts_busy[job.mirror] = false;
            eng->result = std::move(decided);
            eng->result_ready.store(true, std::memory_order_release);
        }
    }
}





static void ggml_cuda_expert_engine_window(ggml_backend_cuda_expert_buffer_context & ctx, ggml_cuda_expert_engine & eng,
        cudaStream_t stream, int64_t budget) {
    const int m = eng.counts_next;
    {
        std::lock_guard<std::mutex> lock(eng.job_mutex);
        if (eng.job_ready || eng.counts_busy[m]) {
            return;
        }
    }

    const size_t n_bytes = eng.counts_n * sizeof(uint32_t);
    CUDA_CHECK(cudaMemcpyAsync(eng.counts_host[m], eng.counts_dev, n_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(eng.counts_dev, 0, n_bytes, stream));
    CUDA_CHECK(cudaEventRecord(eng.counts_ev[m], stream));

    ggml_cuda_expert_xjob job;
    job.mirror = m;
    job.budget = budget;
    job.state  = eng.state;
    for (size_t ri = 0; ri < eng.rows.size(); ++ri) {
        if (!ggml_cuda_expert_engine_row_ok(ctx, eng, (int) ri)) {
            std::fill(job.state[ri].begin(), job.state[ri].end(), (uint8_t) GGML_CUDA_EXPERT_XS_MIXED);
        }
    }
    {
        std::lock_guard<std::mutex> lock(eng.job_mutex);
        eng.counts_busy[m] = true;
        eng.job            = std::move(job);
        eng.job_ready      = true;
    }
    eng.job_cv.notify_one();
    eng.counts_next   = 1 - m;
    eng.window_tokens = 0;
    eng.window_budget = 0;
}










void ggml_cuda_expert_maintain(ggml_backend_cuda_context & cuda_ctx, const ggml_cgraph * cgraph) {
    if (cgraph == nullptr || !ggml_expert_store_adaptive_enabled()) {
        return;
    }
    const ggml_tensor * mmid = ggml_cuda_expert_graph_first_mmid(cgraph);
    if (mmid == nullptr || mmid->ne[2] > GGML_CUDA_EXPERT_DECODE_MAX_TOKENS) {
        return;
    }
    const int64_t n_tokens = mmid->ne[2];
    const int     layer    = ggml_expert_layer_of_name(mmid->src[0]->name);
    cudaStream_t stream = cuda_ctx.stream(cuda_ctx.device, 0);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return;
    }

    const ggml_expert_store_adaptive_cfg cfg = ggml_expert_store_get_adaptive();
    std::lock_guard<std::mutex> lock_buffers(g_expert_buffers_mutex);
    for (ggml_backend_cuda_expert_buffer_context * ctx : g_expert_buffers) {
        if (ctx->device != cuda_ctx.device || !ctx->engine) {
            continue;
        }
        ggml_cuda_expert_engine & eng = *ctx->engine;
        if (layer != eng.first_layer) {
            continue;
        }
        std::lock_guard<std::mutex> lock(eng.mutex);
        if (eng.driver == nullptr) {
            eng.driver = stream;
        }
        if (eng.driver != stream) {
            continue;
        }
        ggml_cuda_set_device(ctx->device);

        if (eng.result_ready.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock_job(eng.job_mutex);
            eng.todo = std::move(eng.result);
            eng.result.clear();
            eng.result_ready.store(false, std::memory_order_relaxed);
        }

        ggml_cuda_expert_engine_advance(*ctx, eng, stream);






        const int64_t cap = std::min<int64_t>(cfg.pairs_step_max, cfg.pairs_per_token * n_tokens);
        eng.window_tokens += n_tokens;
        eng.window_budget += cap;
        if (eng.window_tokens >= cfg.window) {
            ggml_cuda_expert_engine_window(*ctx, eng, stream, eng.window_budget);
        }

        ggml_cuda_expert_engine_issue(*ctx, eng, cap);
    }
}



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

        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }


    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);
    const bool bound = ggml_cuda_numa_bind_cpus(ggml_cuda_expert_host_node(st != nullptr ? st->host_host : nullptr));
    ggml_expert_store_memset(ctx->store, tensor_id, value, store_offset + offset, size);
    if (bound) {
        ggml_cuda_numa_unbind_cpus();
    }
}

static void ggml_backend_cuda_expert_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    int tensor_id = -1;
    size_t store_offset = 0;
    if (!ggml_backend_cuda_expert_buffer_store_target(tensor, ctx, __func__, &tensor_id, &store_offset)) {

        ggml_cuda_set_device(ctx->device);
        ggml_cuda_memcpy_h2d_async((char *) tensor->data + offset, data, size, cudaStreamPerThread);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }
    ggml_cuda_set_device(ctx->device);


    const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);
    const bool bound = ggml_cuda_numa_bind_cpus(ggml_cuda_expert_host_node(st != nullptr ? st->host_host : nullptr));
    ggml_expert_store_write(ctx->store, tensor_id, data, store_offset + offset, size);
    if (bound) {
        ggml_cuda_numa_unbind_cpus();
    }
}

static void ggml_backend_cuda_expert_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_expert_buffer_context * ctx = (ggml_backend_cuda_expert_buffer_context *) buffer->context;
    int tensor_id = -1;
    size_t store_offset = 0;
    if (!ggml_backend_cuda_expert_buffer_store_target(tensor, ctx, __func__, &tensor_id, &store_offset)) {

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


        const size_t blob_pad = ggml_cuda_expert_blob_pad(tensor->type, tensor->ne[0]);
        return (size_t) tensor->ne[2] * ggml_expert_store_stride((size_t) tensor->nb[2], blob_pad, ggml_type_size(tensor->type));
    }

    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;
    return ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(buft_ctx->device), tensor);
}

static size_t ggml_backend_cuda_expert_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        if (tensors[i]->view_src != nullptr) {
            continue;
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



    ggml_cuda_expert_table table;
    return ggml_cuda_expert_table_get(t, &table);
}








struct ggml_cuda_expert_spare {
    size_t  stride = 0;
    int64_t count  = 0;
};

static int64_t ggml_cuda_expert_spare_depth(void) {


    if ((GGML_EXPERT_ADAPTIVE != 0) != ggml_expert_store_adaptive_enabled()) {
        GGML_ABORT("%s: GGML_EXPERT_ADAPTIVE is %d in the CUDA sources but the expert store was built with the other value; "
            "set it for both (cmake -DGGML_EXPERT_ADAPTIVE=ON|OFF)\n", __func__, (int) (GGML_EXPERT_ADAPTIVE != 0));
    }
    const ggml_expert_store_adaptive_cfg cfg = ggml_expert_store_get_adaptive();
    return cfg.enabled ? cfg.pairs_per_token * cfg.inflight_tokens : 0;
}

int64_t ggml_cuda_expert_spare_slots(void) {
    return ggml_cuda_expert_spare_depth();
}

static void ggml_cuda_expert_plan_spares(
        const std::vector<ggml_expert_store_plan_tensor> & tensors, int n_device, int64_t slot_budget, const char * profile_path,
        ggml_expert_store_placement * plan, std::vector<ggml_cuda_expert_spare> * spares, size_t * spare_bytes) {
    ggml_expert_store_plan(tensors.data(), tensors.size(), n_device, slot_budget, profile_path, plan);
    spares->clear();
    *spare_bytes = 0;
    const int64_t depth = ggml_cuda_expert_spare_depth();
    if (depth <= 0 || slot_budget < 0 || plan->host_bytes <= 0) {
        return;
    }


    std::map<size_t, std::map<int, int64_t>> per_stride;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (plan->host_used[i] <= 0) {
            continue;
        }
        const size_t stride = ggml_expert_store_stride(tensors[i].blob_bytes, tensors[i].blob_pad, tensors[i].type_size);
        per_stride[stride][tensors[i].layer]++;
    }
    size_t bytes = 0;
    for (const auto & kv : per_stride) {
        int64_t mult = 0;
        for (const auto & layer : kv.second) {
            mult = std::max(mult, layer.second);
        }
        spares->push_back({ kv.first, depth * mult });
        bytes += (size_t) (depth * mult) * kv.first;
    }






    const int64_t min_host_per_pool = std::max<int64_t>(1, ggml_expert_store_get_adaptive().min_host_per_pool);
    if ((int64_t) bytes >= slot_budget || (int64_t) bytes * min_host_per_pool > plan->host_bytes) {
        spares->clear();
        return;
    }
    ggml_expert_store_plan(tensors.data(), tensors.size(), n_device, slot_budget - (int64_t) bytes, profile_path, plan);
    *spare_bytes = bytes;
}




static void ggml_cuda_expert_engine_init(ggml_backend_cuda_expert_buffer_context * ctx, const std::vector<ggml_cuda_expert_spare> & spares) {
    if (spares.empty()) {
        return;
    }
    std::unique_ptr<ggml_cuda_expert_engine> eng(new ggml_cuda_expert_engine());

    size_t bytes = 0;
    for (const ggml_cuda_expert_spare & sp : spares) {
        ggml_cuda_expert_xpool pool;
        pool.stride   = sp.stride;
        pool.capacity = sp.count;
        char * base = (char *) ggml_cuda_expert_alloc_device(&ctx->backend_user, (size_t) sp.count * sp.stride);
        GGML_ASSERT(base != nullptr);
        CUDA_CHECK(cudaMemset(base, 0, (size_t) sp.count * sp.stride));
        for (int64_t i = 0; i < sp.count; ++i) {
            pool.free.push_back(base + (size_t) i * sp.stride);
        }
        eng->pools.push_back(std::move(pool));
        bytes += (size_t) sp.count * sp.stride;
    }


    std::map<int, std::vector<int>> layers;
    for (size_t i = 0; i < ctx->tensors.size(); ++i) {
        const ggml_cuda_expert_tensor_state & s = ctx->tensors[i];
        if (s.has_host && s.tensor != nullptr) {
            layers[ggml_cuda_expert_tensor_layer(s.tensor)].push_back((int) i);
        }
    }
    eng->first_layer = INT32_MAX;
    for (const ggml_cuda_expert_tensor_state & s : ctx->tensors) {
        if (s.tensor != nullptr) {
            eng->first_layer = std::min(eng->first_layer, ggml_cuda_expert_tensor_layer(s.tensor));
        }
    }
    eng->row_of_tensor.assign(ctx->tensors.size(), -1);
    for (const auto & kv : layers) {
        ggml_cuda_expert_xrow r;
        r.layer = kv.first;
        bool ok = kv.second.size() <= GGML_CUDA_EXPERT_ROW_MAX;
        for (size_t k = 0; ok && k < kv.second.size(); ++k) {
            const ggml_cuda_expert_tensor_state & s = ctx->tensors[(size_t) kv.second[k]];
            int pool = -1;
            for (size_t p = 0; p < eng->pools.size(); ++p) {
                if (eng->pools[p].stride == s.blob_stride) {
                    pool = (int) p;
                }
            }
            ok = pool >= 0 && (r.n_tensor == 0 || s.n_expert == r.n_expert);
            r.tensor[r.n_tensor] = kv.second[k];
            r.pool  [r.n_tensor] = pool;
            r.n_tensor++;
            r.n_expert = s.n_expert;
        }
        if (!ok) {
            GGML_LOG_WARN("%s: device %d layer %d: %zu expert tensors with a host tier do not form one exchange row, the layer keeps its placement\n",
                __func__, ctx->device, kv.first, kv.second.size());
            continue;
        }


        std::vector<uint8_t> state((size_t) r.n_expert, GGML_CUDA_EXPERT_XS_MIXED);
        for (int64_t e = 0; e < r.n_expert; ++e) {
            int n_slot = 0;
            for (int k = 0; k < r.n_tensor; ++k) {
                n_slot += ggml_cuda_expert_engine_st(*ctx, r.tensor[k])->residency[(size_t) e] >= 0;
            }
            state[(size_t) e] = n_slot == r.n_tensor ? GGML_CUDA_EXPERT_XS_SLOT :
                                n_slot == 0          ? GGML_CUDA_EXPERT_XS_HOST : GGML_CUDA_EXPERT_XS_MIXED;
        }
        for (int k = 0; k < r.n_tensor; ++k) {
            eng->row_of_tensor[(size_t) r.tensor[k]] = (int) eng->rows.size();
        }
        eng->rows.push_back(r);
        eng->state.push_back(std::move(state));
    }

    std::vector<ggml_cuda_expert_tables> tables(ctx->tensors.size());
    for (size_t i = 0; i < ctx->tensors.size(); ++i) {
        tables[i] = { ctx->tensors[i].table_main_dev, ctx->tensors[i].residency_dev, ctx->tensors[i].host_offset_dev };
    }
    CUDA_CHECK(cudaMalloc((void **) &eng->tables_dev, tables.size() * sizeof(ggml_cuda_expert_tables)));
    CUDA_CHECK(cudaMemcpy(eng->tables_dev, tables.data(), tables.size() * sizeof(ggml_cuda_expert_tables), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&eng->copy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&eng->ev_flip, cudaEventDisableTiming));


    CUDA_CHECK(cudaEventCreateWithFlags(&eng->ev_done, cudaEventDisableTiming));




    CUDA_CHECK(cudaEventCreateWithFlags(&eng->ev_bcopy, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(eng->ev_bcopy, eng->copy));


    eng->count_off.resize(eng->rows.size());
    eng->count_ptr.assign(ctx->tensors.size(), nullptr);
    eng->score.resize(eng->rows.size());
    for (size_t ri = 0; ri < eng->rows.size(); ++ri) {
        eng->count_off[ri] = eng->counts_n;
        eng->counts_n     += (size_t) eng->rows[ri].n_expert;
        eng->score[ri].assign((size_t) eng->rows[ri].n_expert, 0.0);
    }
    const size_t count_bytes = std::max<size_t>(eng->counts_n, 1) * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc((void **) &eng->counts_dev, count_bytes));
    CUDA_CHECK(cudaMemset(eng->counts_dev, 0, count_bytes));
    for (size_t ri = 0; ri < eng->rows.size(); ++ri) {
        eng->count_ptr[(size_t) eng->rows[ri].tensor[0]] = eng->counts_dev + eng->count_off[ri];
    }
    for (int m = 0; m < 2; ++m) {
        CUDA_CHECK(cudaHostAlloc((void **) &eng->counts_host[m], count_bytes, cudaHostAllocDefault));
        memset(eng->counts_host[m], 0, count_bytes);

        CUDA_CHECK(cudaEventCreateWithFlags(&eng->counts_ev[m], cudaEventDisableTiming | cudaEventBlockingSync));
    }
    eng->worker = std::thread(ggml_cuda_expert_engine_worker, eng.get(), ctx->device);

    GGML_LOG_DEBUG("%s: device %d: adaptive expert placement: %zu layers, %zu spare pools, %.2f MiB of spare slots\n",
        __func__, ctx->device, eng->rows.size(), eng->pools.size(), bytes / 1024.0 / 1024.0);
    ctx->engine = std::move(eng);
}

static ggml_backend_buffer_t ggml_backend_cuda_expert_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {


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





    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    size_t expert_bytes = 0;
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


            const size_t blob_pad = ggml_cuda_expert_blob_pad(t->type, t->ne[0]);
            expert_bytes += (size_t) t->ne[2] * ggml_expert_store_stride((size_t) t->nb[2], blob_pad, ggml_type_size(t->type));
            expert_tensors.push_back(t);
        } else {
            plain_offset[i] = plain_size;
            plain_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
            plain_alloc[i] = true;
        }
    }





    const size_t slot_budget  = ggml_cuda_expert_get_slot_budget(device);
    const size_t stage_budget = ggml_cuda_expert_get_stage_budget(device);
    if (expert_bytes > 0 && slot_budget == 0 && stage_budget == 0) {







        GGML_LOG_ERROR("%s: device %d: no expert VRAM budget was set; the host experts of a prefill batch cannot be staged, refusing the buffer"
                       " (the parameter fit plans the expert placement of a model with routed experts: keep the fit enabled, -fit off is not supported for such a model)\n",
            __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_expert_buffer_context * ctx = new ggml_backend_cuda_expert_buffer_context();
    ctx->device       = device;
    ctx->backend_user.device = device;
    ctx->backend      = ggml_cuda_expert_backend(&ctx->backend_user);
    ctx->store        = ggml_expert_store_init(&ctx->backend, device, nullptr);
    ctx->stage_budget = stage_budget;





    std::vector<ggml_expert_store_plan_tensor> plan_in;
    plan_in.reserve(expert_tensors.size());
    for (const ggml_tensor * t : expert_tensors) {
        plan_in.push_back({ ggml_cuda_expert_tensor_layer(t), t->ne[2], ggml_cuda_expert_get_tensor_base(t),
            (size_t) t->nb[2], ggml_cuda_expert_blob_pad(t->type, t->ne[0]), ggml_type_size(t->type),
            ggml_cuda_expert_tensor_kind(t) });
    }
    ggml_expert_store_placement plan;
    std::vector<ggml_cuda_expert_spare> spares;
    size_t spare_bytes = 0;
    ggml_cuda_expert_plan_spares(plan_in, ggml_backend_cuda_get_device_count(), (int64_t) slot_budget, ctx->store->profile_path.c_str(),
        &plan, &spares, &spare_bytes);




    if ((size_t) plan.host_peak > stage_budget) {
        GGML_LOG_ERROR("%s: device %d: staging budget %.2f MiB is smaller than the %.2f MiB the regions of one prefill batch need, refusing the buffer\n",
            __func__, device, stage_budget / 1024.0 / 1024.0, plan.host_peak / 1024.0 / 1024.0);
        GGML_LOG_ERROR("%s: device %d: the expert buffer was refused, the caller has to stop the operation: the machine has no expert placement for it\n",
            __func__, device);
        delete ctx;
        return nullptr;
    }






    const ggml_cuda_expert_device_bytes device_bytes_before = ggml_cuda_expert_device_bytes_get(device);
    if (!ggml_cuda_expert_register_host_plan(device, (size_t) plan.slot_used + spare_bytes, (size_t) plan.host_bytes,
            stage_budget, (size_t) plan.host_peak)) {
        ggml_cuda_expert_device_bytes_set(device, device_bytes_before);
        delete ctx;
        return nullptr;
    }







    {
        size_t free_now  = 0;
        size_t total_now = 0;
        ggml_cuda_expert_get_mem(device, &free_now, &total_now);
        const size_t need_now = (size_t) plan.slot_used + spare_bytes + stage_budget + plain_size;
        GGML_LOG_DEBUG("%s: device %d: expert buffer needs %.2f MiB (slots %.2f + spare slots %.2f + staging %.2f + dense %.2f), %.2f MiB free of %.2f MiB\n",
            __func__, device, need_now / 1024.0 / 1024.0, plan.slot_used / 1024.0 / 1024.0,
            spare_bytes / 1024.0 / 1024.0, stage_budget / 1024.0 / 1024.0, plain_size / 1024.0 / 1024.0,
            free_now / 1024.0 / 1024.0, total_now / 1024.0 / 1024.0);
        if (need_now > free_now) {
            GGML_LOG_ERROR("%s: device %d: the expert buffer needs %.2f MiB (slots %.2f + spare slots %.2f + staging %.2f + dense %.2f), but only %.2f MiB are free; short by %.2f MiB. The fit budget was planned from a different free-memory state.\n",
                __func__, device, need_now / 1024.0 / 1024.0, plan.slot_used / 1024.0 / 1024.0,
                spare_bytes / 1024.0 / 1024.0, stage_budget / 1024.0 / 1024.0, plain_size / 1024.0 / 1024.0,
                free_now / 1024.0 / 1024.0, (need_now - free_now) / 1024.0 / 1024.0);
            GGML_LOG_ERROR("%s: device %d: the expert buffer was refused, the caller has to stop the operation: the machine has no expert placement for it\n",
                __func__, device);
            ggml_cuda_expert_device_bytes_set(device, device_bytes_before);
            delete ctx;
            return nullptr;
        }
    }



    {
        size_t slab = 0;
        for (size_t k = 0; k < expert_tensors.size(); ++k) {
            const ggml_tensor * t = expert_tensors[k];
            const size_t stride = ggml_expert_store_stride((size_t) t->nb[2], ggml_cuda_expert_blob_pad(t->type, t->ne[0]),
                ggml_type_size(t->type));
            slab += GGML_PAD((size_t) plan.n_slot[k] * stride, GGML_CUDA_EXPERT_SLAB_ALIGN);
        }

        for (const ggml_cuda_expert_spare & sp : spares) {
            slab += GGML_PAD((size_t) sp.count * sp.stride, GGML_CUDA_EXPERT_SLAB_ALIGN);
        }
        if (slab > 0) {
            size_t free_before = 0;
            size_t total       = 0;
            ggml_cuda_expert_get_mem(device, &free_before, &total);
            ggml_cuda_set_device(device);
            const cudaError_t err = cudaMalloc((void **) &ctx->backend_user.slab, slab);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: device %d: cudaMalloc of the %.2f MiB slot block failed (%s), %.2f MiB free of %.2f MiB\n",
                    __func__, device, slab / 1024.0 / 1024.0, cudaGetErrorString(err),
                    free_before / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
                delete ctx;
                return nullptr;
            }
            ctx->backend_user.slab_size = slab;
            GGML_LOG_DEBUG("%s: device %d: slot block cudaMalloc %.2f MiB for %zu tensors, free %.2f MiB before\n",
                __func__, device, slab / 1024.0 / 1024.0, expert_tensors.size(), free_before / 1024.0 / 1024.0);
        }
    }

    CUDA_CHECK(cudaMalloc((void **) &ctx->usage_dev, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx->usage_dev, 0, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **) &ctx->usage_host, 4 * sizeof(unsigned long long), cudaHostAllocDefault));
    memset(ctx->usage_host, 0, 4 * sizeof(unsigned long long));
    CUDA_CHECK(cudaEventCreateWithFlags(&ctx->usage_event, cudaEventDisableTiming));

    ctx->tensors.reserve((size_t) n_tensors);


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
            -1);
        ggml_cuda_expert_clear_tensor_base(t);
        if (tensor_id < 0) {
            GGML_LOG_ERROR("%s: cannot place expert tensor %s\n", __func__, t->name);
            delete ctx;
            return nullptr;
        }

        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, tensor_id);






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
        s.kind         = ggml_cuda_expert_tensor_kind(t);

        CUDA_CHECK(cudaMalloc((void **) &s.table_main_dev, (size_t) n_expert * sizeof(char *)));
        CUDA_CHECK(cudaMemcpy(s.table_main_dev, st->table_host, (size_t) n_expert * sizeof(char *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **) &s.residency_dev, (size_t) n_expert * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(s.residency_dev, st->residency.data(), (size_t) n_expert * sizeof(int32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &s.host_offset_dev, (size_t) n_expert * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(s.host_offset_dev, st->host_offset.data(), (size_t) n_expert * sizeof(int64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &s.counts_dev, (size_t) n_expert * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(s.counts_dev, 0, (size_t) n_expert * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void **) &s.launches_dev, (size_t) n_expert * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(s.launches_dev, 0, (size_t) n_expert * sizeof(uint32_t)));

        s.has_stage = s.has_host && stage_budget > 0;
        if (s.has_stage) {



            size_t r = 0;
            for (; r < ctx->tensors.size(); ++r) {
                if (ctx->tensors[r].has_stage && ctx->tensors[r].kind == s.kind) {
                    break;
                }
            }
            if (r < ctx->tensors.size()) {
                s.region = ctx->tensors[r].region;
            } else {
                s.region = (int) ctx->region_bytes.size();
                ctx->region_bytes.push_back(0);
            }
            ctx->region_bytes[(size_t) s.region] = std::max(ctx->region_bytes[(size_t) s.region], (size_t) st->host_used);
        }



        t->data = st->host_dev != nullptr ? (void *) st->host_dev : (void *) st->slots_dev;

        ggml_expert_store_set_tensor_id(ctx->store, t, tensor_id);
        ctx->tensors.push_back(std::move(s));

        GGML_LOG_DEBUG("%s: device %d tensor %s: %" PRId64 " experts, slots %" PRId64 ", host %" PRId64 ", stage %s\n",
            __func__, device, t->name, n_expert, st->n_slot, st->n_host,
            s.has_stage ? "yes" : "no");
    }






    ggml_cuda_expert_ring_build(*ctx, ggml_cuda_expert_ring_specs(*ctx));
    for (size_t r = 0; r < ctx->region_bytes.size(); ++r) {
        ggml_cuda_expert_ring_log_rebuild(*ctx, (int) r, 0, 0, "init");
    }


    ggml_cuda_expert_engine_init(ctx, spares);

    if (expert_bytes > 0) {





        {
            std::lock_guard<std::mutex> lock(g_expert_budget_mutex);
            if (g_expert_plan_valid[device] &&
                ((size_t) ctx->store->slot_used + spare_bytes != g_expert_plan_slot[device] ||
                 (size_t) ctx->store->host_used != g_expert_plan_host[device])) {
                GGML_ABORT("%s: device %d: expert placement (slots %.2f MiB, host %.2f MiB) differs from the fit plan"
                    " (slots %.2f MiB, host %.2f MiB) for slot budget %.2f MiB\n",
                    __func__, device,
                    (double) (ctx->store->slot_used + spare_bytes) / 1024.0 / 1024.0, (double) ctx->store->host_used / 1024.0 / 1024.0,
                    (double) g_expert_plan_slot[device] / 1024.0 / 1024.0, (double) g_expert_plan_host[device] / 1024.0 / 1024.0,
                    slot_budget / 1024.0 / 1024.0);
            }


            g_expert_plan_valid[device] = false;
        }

        ggml_expert_store_print_plan(ctx->store);
        ggml_cuda_expert_set_allocation(device, (size_t) ctx->store->slot_used + spare_bytes, (size_t) ctx->store->host_used,
            stage_budget, (size_t) plan.host_peak);




        {
            size_t region_sum = 0;
            for (const size_t b : ctx->region_bytes) {
                region_sum += b;
            }
            GGML_ASSERT(region_sum == (size_t) plan.host_peak);
        }
        GGML_LOG_DEBUG("%s: device %d: expert store: slots %.2f MiB, spare slots %.2f MiB, host complement %.2f MiB, staging %.2f MiB in %zu regions\n",
            __func__, device, plan.slot_used / 1024.0 / 1024.0, spare_bytes / 1024.0 / 1024.0,
            plan.host_bytes / 1024.0 / 1024.0, stage_budget / 1024.0 / 1024.0, ctx->region_bytes.size());



        ggml_cuda_numa_log_placement("expert host placement");
    }


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
    /* .is_host             = */ NULL,
};




void ggml_backend_cuda_expert_verify_all(void) {
    std::vector<ggml_backend_cuda_expert_buffer_context *> list;
    {
        std::lock_guard<std::mutex> lock(g_expert_buffers_mutex);
        list = g_expert_buffers;
    }
    for (ggml_backend_cuda_expert_buffer_context * ctx : list) {
        ggml_cuda_set_device(ctx->device);
        int64_t  n_tensors = 0;
        int64_t  n_checked = 0;
        int64_t  n_skipped = 0;
        uint32_t table     = 0;
        bool     tables_differ = false;
        for (ggml_cuda_expert_tensor_state & s : ctx->tensors) {
            if (s.verify_done) {
                continue;
            }
            s.verify_done = true;
            if (!ggml_expert_store_verify(ctx->store, s.tensor_id)) {
                GGML_ABORT("%s: expert store byte check failed for tensor %s\n",
                    __func__, s.tensor != nullptr ? s.tensor->name : "?");
            }


            int64_t checked = 0;
            int64_t skipped = 0;
            if (!ggml_expert_store_verify_placement(ctx->store, s.tensor_id,
                    s.tensor != nullptr ? s.tensor->name : nullptr, &checked, &skipped)) {
                GGML_ABORT("%s: expert placement check failed for tensor %s on device %d\n",
                    __func__, s.tensor != nullptr ? s.tensor->name : "?", ctx->device);
            }
            const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(ctx->store, s.tensor_id);
            if (n_tensors > 0 && st->assignment != table) {
                tables_differ = true;
            }
            table = st->assignment;
            n_tensors++;
            n_checked += checked;
            n_skipped += skipped;
        }
        if (n_tensors > 0) {

            GGML_LOG_DEBUG("%s: device %d: expert placement check: %" PRId64 " tensors, %" PRId64 " experts match the assignment (version %u%s), %" PRId64 " skipped\n",
                __func__, ctx->device, n_tensors, n_checked, table, tables_differ ? ", tables differ" : "", n_skipped);
        }
    }



    ggml_cuda_numa_log_final_placement();
}






void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, const int * kind, int n_expert_used, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each,
        double * routed_share) {
    if (buft == nullptr || !ggml_cuda_expert_buft_is_expert(buft)) {
        return;
    }

    std::vector<ggml_expert_store_plan_tensor> tensors;
    tensors.reserve(n_tensors);
    for (size_t i = 0; i < n_tensors; ++i) {



        const size_t blob_pad = ggml_cuda_expert_blob_pad((ggml_type) type[i], ne0[i]);
        tensors.push_back({ layer[i], n_expert[i], expert_base != nullptr ? expert_base[i] : 0, blob_bytes[i], blob_pad,
            ggml_type_size((ggml_type) type[i]), kind != nullptr ? kind[i] : -1, n_expert_used });
    }




    ggml_expert_store_placement plan;
    std::vector<ggml_cuda_expert_spare> spares;
    size_t spare_bytes = 0;
    ggml_cuda_expert_plan_spares(tensors, ggml_backend_cuda_get_device_count(), (int64_t) slot_budget, nullptr,
        &plan, &spares, &spare_bytes);

    if (slot_bytes != nullptr) {
        *slot_bytes = (size_t) plan.slot_used + spare_bytes;
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
    if (routed_share != nullptr) {



        *routed_share = plan.has_routed_share ? plan.routed_share : -1.0;
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

#else





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

const char * const * ggml_cuda_expert_stage_prepare(const ggml_tensor * src0, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(stream);
    return nullptr;
}

void ggml_cuda_expert_stage_consumed(const ggml_tensor * src0, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(stream);
}

bool ggml_cuda_expert_stage_counters(ggml_backend_buffer_type_t buft, size_t * cold, size_t * hits) {
    GGML_UNUSED(buft);
    GGML_UNUSED(cold);
    GGML_UNUSED(hits);
    return false;
}


void ggml_cuda_expert_comm_done(int device, cudaStream_t stream) {
    GGML_UNUSED(device);
    GGML_UNUSED(stream);
}

const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(lut);
    GGML_UNUSED(stream);
    return nullptr;
}

void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(lut);
    GGML_UNUSED(stream);
}

int ggml_cuda_expert_counts_read(const ggml_tensor * weight, uint32_t * out, bool reset, cudaStream_t stream) {
    GGML_UNUSED(weight);
    GGML_UNUSED(out);
    GGML_UNUSED(reset);
    GGML_UNUSED(stream);
    return -1;
}

void ggml_backend_cuda_expert_verify_all(void) {
}

void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(stream);
}

void ggml_cuda_expert_note_reader(const ggml_tensor * src0, int tensor_id, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(tensor_id);
    GGML_UNUSED(stream);
}

int64_t ggml_cuda_expert_spare_slots(void) {
    return 0;
}

int ggml_cuda_expert_swap_test(ggml_tensor * weight, const int32_t * pairs, int n_pairs, int32_t * pairs_out) {
    GGML_UNUSED(weight);
    GGML_UNUSED(pairs);
    GGML_UNUSED(n_pairs);
    GGML_UNUSED(pairs_out);
    return -1;
}

void ggml_cuda_expert_maintain(ggml_backend_cuda_context & cuda_ctx, const ggml_cgraph * cgraph) {

    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
}

int ggml_cuda_expert_residency(const ggml_tensor * weight, int32_t * out) {
    GGML_UNUSED(weight);
    GGML_UNUSED(out);
    return -1;
}

void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, const int * kind, int n_expert_used, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each,
        double * routed_share) {
    GGML_UNUSED(buft);
    GGML_UNUSED(layer);
    GGML_UNUSED(expert_base);
    GGML_UNUSED(n_expert);    GGML_UNUSED(blob_bytes);
    GGML_UNUSED(type);
    GGML_UNUSED(ne0);
    GGML_UNUSED(kind);
    GGML_UNUSED(n_expert_used);
    GGML_UNUSED(n_tensors);
    GGML_UNUSED(slot_budget);
    GGML_UNUSED(slot_bytes);
    GGML_UNUSED(host_bytes);
    GGML_UNUSED(host_peak);
    GGML_UNUSED(host_bytes_each);
    GGML_UNUSED(routed_share);
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

void ggml_cuda_expert_reset_streams(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
}

bool ggml_cuda_expert_last_stage_hits(ggml_backend_buffer_type_t buft, size_t * n_hit) {
    GGML_UNUSED(buft);
    GGML_UNUSED(n_hit);
    return false;
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

#endif








ggml_cuda_expert_lut ggml_cuda_mul_mat_id_lut(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    GGML_ASSERT(dst != nullptr && dst->op == GGML_OP_MUL_MAT_ID);
    const ggml_tensor * src0 = dst->src[0];

    const int  n_local = (int) src0->ne[2];
    const bool ep      = ggml_mul_mat_id_is_expert_parallel(dst);

    const int     offset  = ep ? ggml_mul_mat_id_get_expert_offset(dst)  : 0;
    const int     n_total = ep ? ggml_mul_mat_id_get_n_expert_total(dst) : n_local;
    const int32_t table   = ep ? ggml_mul_mat_id_get_assignment(dst)     : 0;
    if (n_local <= 0 || n_total < n_local || n_total > INT16_MAX || offset < 0 || offset + n_local > n_total) {
        GGML_ABORT("%s: node %s: expert state offset %d, local %d, total %d is inconsistent\n",
            __func__, dst->name, offset, n_local, n_total);
    }

    uint64_t key;
    if (table > 0) {


        const int device = ggml_mul_mat_id_get_device(dst);
        const int layer  = ggml_mul_mat_id_get_layer(dst);
        if (device != ctx.device) {
            GGML_ABORT("%s: node %s is marked for device %d but runs on device %d\n", __func__, dst->name, device, ctx.device);
        }
        if (layer < 0 || layer > 0xFFFF || device < 0 || device > 0xFF) {
            GGML_ABORT("%s: node %s: layer %d / device %d out of range\n", __func__, dst->name, layer, device);
        }
        key = (uint64_t) 1 << 63 | (uint64_t) table << 24 | (uint64_t) device << 16 | (uint64_t) layer;
    } else if (table == 0) {
        key = (uint64_t) offset << 32 | (uint64_t) n_local << 16 | (uint64_t) n_total;
    } else {

        ggml_expert_store_node_lut(dst);
        GGML_ABORT("%s: node %s has no expert assignment\n", __func__, dst->name);
    }

    auto it = ctx.expert_luts.find(key);
    if (it != ctx.expert_luts.end()) {


        if (it->second.offset != offset || it->second.n_local != n_local || it->second.n != n_total) {
            GGML_ABORT("%s: node %s: slice [%d, %d) of %d experts differs from the slice [%d, %d) of %d its expert map was made for\n",
                __func__, dst->name, offset, offset + n_local, n_total,
                it->second.offset, it->second.offset + it->second.n_local, it->second.n);
        }
    } else {
        const int16_t * host = ggml_expert_store_node_lut(dst);
        GGML_ASSERT(host != nullptr);



        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            GGML_ABORT("%s: the expert map of node %s is first needed while a CUDA graph is being captured\n",
                __func__, dst->name);
        }

        ggml_cuda_set_device(ctx.device);
        ggml_backend_cuda_context::expert_lut_entry entry;
        CUDA_CHECK(cudaMalloc((void **) &entry.dev, (size_t) n_total * sizeof(int16_t)));
        CUDA_CHECK(cudaMemcpy(entry.dev, host, (size_t) n_total * sizeof(int16_t), cudaMemcpyHostToDevice));
        entry.host    = host;
        entry.n       = n_total;
        entry.offset  = offset;
        entry.n_local = n_local;
        it = ctx.expert_luts.emplace(key, entry).first;
    }

    ggml_cuda_expert_lut lut;
    lut.map  = it->second.dev;
    lut.host = it->second.host;
    lut.n    = it->second.n;
    return lut;
}
