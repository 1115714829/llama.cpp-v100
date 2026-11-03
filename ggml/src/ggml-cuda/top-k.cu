#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_reset_counters
            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}




static constexpr int TOP_K_BLOCK_SIZE = 1024;
static constexpr int TOP_K_CHUNK_SIZE = 1024;
static constexpr int TOP_K_MERGE_SIZE = 4096;

static int top_k_next_power_of_2(const int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}


static __device__ __forceinline__ bool top_k_better(const float av, const int ai, const float bv, const int bi) {
    return av > bv || (av == bv && ai < bi);
}


static __device__ __forceinline__ void top_k_compare_exchange(float * sv, int * si, const int a, const int b) {
    const float av = sv[a];
    const int   ai = si[a];
    const float bv = sv[b];
    const int   bi = si[b];
    if (top_k_better(bv, bi, av, ai)) {
        sv[a] = bv;
        si[a] = bi;
        sv[b] = av;
        si[b] = ai;
    }
}

static __global__ void top_k_bitonic_chunk(const float * __restrict__ src,
                                           float * __restrict__ cand_v,
                                           int * __restrict__ cand_i,
                                           const int ncols,
                                           const int k_pad) {
    const int col  = threadIdx.x;
    const int icol = blockIdx.x * TOP_K_CHUNK_SIZE + col;

    __shared__ float sv[TOP_K_CHUNK_SIZE];
    __shared__ int   si[TOP_K_CHUNK_SIZE];

    float v = -INFINITY;
    int   i = INT_MAX;
    if (icol < ncols) {
        v = src[(size_t) blockIdx.y * ncols + icol];

        v = isnan(v) ? -INFINITY : v;
        i = icol;
    }
    sv[col] = v;
    si[col] = i;
    __syncthreads();

    for (int k = 2; k <= TOP_K_CHUNK_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    top_k_compare_exchange(sv, si, col, ixj);
                } else {
                    top_k_compare_exchange(sv, si, ixj, col);
                }
            }
            __syncthreads();
        }
    }

    if (col < k_pad) {
        const size_t off = ((size_t) blockIdx.y * gridDim.x + blockIdx.x) * k_pad + col;
        cand_v[off] = sv[col];
        cand_i[off] = si[col];
    }
}


template<bool FINAL>
static __global__ void top_k_bitonic_merge(const float * __restrict__ cand_v,
                                           const int * __restrict__ cand_i,
                                           float * __restrict__ out_v,
                                           int * __restrict__ out_i,
                                           int * __restrict__ dst,
                                           const int n_cand,
                                           const int n_pad,
                                           const int k,
                                           const int k_pad) {
    const int tid = threadIdx.x;
    const int row = blockIdx.y;

    __shared__ float sv[TOP_K_MERGE_SIZE];
    __shared__ int   si[TOP_K_MERGE_SIZE];

    const float * row_v = cand_v + (size_t) row * n_cand;
    const int *   row_i = cand_i + (size_t) row * n_cand;
    const int     base  = blockIdx.x * TOP_K_MERGE_SIZE;

    for (int i = tid; i < n_pad; i += TOP_K_BLOCK_SIZE) {
        const int ig = base + i;
        if (ig < n_cand) {
            sv[i] = row_v[ig];
            si[i] = row_i[ig];
        } else {
            sv[i] = -INFINITY;
            si[i] = INT_MAX;
        }
    }
    __syncthreads();

    for (int kk = 2; kk <= n_pad; kk *= 2) {
        for (int j = kk / 2; j > 0; j /= 2) {
            for (int i = tid; i < n_pad; i += TOP_K_BLOCK_SIZE) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & kk) == 0) {
                        top_k_compare_exchange(sv, si, i, ixj);
                    } else {
                        top_k_compare_exchange(sv, si, ixj, i);
                    }
                }
            }
            __syncthreads();
        }
    }

    if (FINAL) {
        for (int i = tid; i < k; i += TOP_K_BLOCK_SIZE) {
            dst[(size_t) row * k + i] = si[i];
        }
    } else {
        const size_t off = ((size_t) row * gridDim.x + blockIdx.x) * k_pad;
        for (int i = tid; i < k_pad; i += TOP_K_BLOCK_SIZE) {
            out_v[off + i] = sv[i];
            out_i[off + i] = si[i];
        }
    }
}

static void top_k_bitonic_cuda(ggml_cuda_pool & pool,
                               const float *    src,
                               int *            dst,
                               const int        ncols,
                               const int        nrows,
                               const int        k,
                               cudaStream_t     stream) {
    const int k_pad    = top_k_next_power_of_2(k);
    const int n_chunks = (ncols + TOP_K_CHUNK_SIZE - 1) / TOP_K_CHUNK_SIZE;


    const size_t n_pairs = (size_t) nrows * n_chunks * k_pad;

    ggml_cuda_pool_alloc<float> cand_v_alloc(pool, 2 * n_pairs);
    ggml_cuda_pool_alloc<int>   cand_i_alloc(pool, 2 * n_pairs);
    float * cand_v[2] = { cand_v_alloc.get(), cand_v_alloc.get() + n_pairs };
    int *   cand_i[2] = { cand_i_alloc.get(), cand_i_alloc.get() + n_pairs };

    top_k_bitonic_chunk<<<dim3(n_chunks, nrows), TOP_K_BLOCK_SIZE, 0, stream>>>(
        src, cand_v[0], cand_i[0], ncols, k_pad);

    size_t n_cand_cur = (size_t) n_chunks * k_pad;
    int    cur        = 0;
    while (true) {
        const int n_chunks_merge = (int) ((n_cand_cur + TOP_K_MERGE_SIZE - 1) / TOP_K_MERGE_SIZE);
        const int n_pad = n_cand_cur > TOP_K_MERGE_SIZE ? TOP_K_MERGE_SIZE : top_k_next_power_of_2((int) n_cand_cur);

        if (n_chunks_merge == 1) {
            top_k_bitonic_merge<true><<<dim3(1, nrows), TOP_K_BLOCK_SIZE, 0, stream>>>(
                cand_v[cur], cand_i[cur], nullptr, nullptr, dst, (int) n_cand_cur, n_pad, k, k_pad);
            break;
        }

        top_k_bitonic_merge<false><<<dim3(n_chunks_merge, nrows), TOP_K_BLOCK_SIZE, 0, stream>>>(
            cand_v[cur], cand_i[cur], cand_v[cur ^ 1], cand_i[cur ^ 1], nullptr, (int) n_cand_cur, n_pad, k, k_pad);

        n_cand_cur = (size_t) n_chunks_merge * k_pad;
        cur ^= 1;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();


    if (ncols > 1024 && k <= 64 && nrows <= 65535) {
        top_k_bitonic_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
