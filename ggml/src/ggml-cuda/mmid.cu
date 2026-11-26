// Portions from Strata and its AC922 fork; Copyright (c) 2026 Niko1221 and the Strata contributors (MIT); see licenses/LICENSE-strata.
#include "common.cuh"
#include "mmid.cuh"

#include <algorithm>

static __global__ void mmids_count_kernel(
        const int32_t * __restrict__ ids, const int n_pairs, const ggml_cuda_expert_lut lut,
        const int n_experts, const int n_expert_used, const int si1, int32_t * __restrict__ cnt) {
    for (int p = blockIdx.x*blockDim.x + threadIdx.x; p < n_pairs; p += gridDim.x*blockDim.x) {
        const int e = ggml_cuda_expert_local(lut, ids[(p / n_expert_used)*si1 + p % n_expert_used]);
        if ((unsigned) e < (unsigned) n_experts) {
            atomicAdd(&cnt[e], 1);
        }
    }
}

static __device__ __forceinline__ int mmids_block_excl_sum(int v, int * total, int * sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();
    return r;
}

static __global__ void mmids_scan_kernel(int32_t * __restrict__ bounds, int32_t * __restrict__ fill, const int n_experts) {
    __shared__ int sh[32];
    const int chunk = (n_experts + (int) blockDim.x - 1) / (int) blockDim.x;
    const int e0    = (int) threadIdx.x*chunk;
    const int e1    = min(e0 + chunk, n_experts);
    int own = 0;
    for (int e = e0; e < e1; ++e) {
        own += bounds[e];
    }
    int rows = 0;
    int run  = mmids_block_excl_sum(own, &rows, sh);
    for (int e = e0; e < e1; ++e) {
        const int c = bounds[e];
        bounds[e] = run;
        fill[e]   = run;
        run += c;
    }
    if (threadIdx.x == 0) {
        bounds[n_experts] = rows;
    }
}

static __global__ void mmids_place_kernel(
        const int32_t * __restrict__ ids, const int n_pairs, const ggml_cuda_expert_lut lut,
        const int n_experts, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, int32_t * __restrict__ fill, int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst) {
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n_pairs) {
        return;
    }
    const int it  = p / n_expert_used;
    const int iex = p - it*n_expert_used;
    const int e   = ggml_cuda_expert_local(lut, ids[it*si1 + iex]);
    if ((unsigned) e >= (unsigned) n_experts) {
        return;
    }

    const int c = atomicAdd(&fill[e], 1);
    ids_dst[c] = p;

    if (write_inverse) {
        ids_src1[p] = c;
    } else {
        ids_src1[c] = it*sis1 + iex % nchannels_y;
    }
}

template <typename T>
__global__ void mmid_zero_nonlocal_kernel(
        T * __restrict__ dst, const int32_t * __restrict__ ids,
        const ggml_cuda_expert_lut lut, const int n_expert_used,
        const int64_t ne0, const int64_t ids_stride,
        const int64_t stride_slot, const int64_t stride_token) {
    const int pair  = blockIdx.x;
    const int token = pair / n_expert_used;
    const int slot  = pair % n_expert_used;

    if (ggml_cuda_expert_local(lut, ids[token*ids_stride + slot]) >= 0) {
        return;
    }

    T * dst_slot = dst + token*stride_token + slot*stride_slot;
    for (int64_t r = threadIdx.x; r < ne0; r += blockDim.x) {
        dst_slot[r] = (T) 0.0f;
    }
}

void ggml_cuda_mmid_zero_nonlocal(
        void * dst, const int32_t * ids, const ggml_cuda_expert_lut lut,
        const int64_t n_tokens, const int n_expert_used, const int64_t ne0,
        const int64_t ids_stride, const int64_t stride_slot, const int64_t stride_token,
        cudaStream_t stream, const bool dst_f16) {
    const int n_pairs = (int) (n_tokens * n_expert_used);
    if (n_pairs == 0) {
        return;
    }
    const dim3 num_blocks(n_pairs, 1, 1);
    const dim3 block_size(256, 1, 1);
    if (dst_f16) {
        mmid_zero_nonlocal_kernel<half><<<num_blocks, block_size, 0, stream>>>(
            (half *) dst, ids, lut, n_expert_used, ne0, ids_stride, stride_slot, stride_token);
    } else {
        mmid_zero_nonlocal_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            (float *) dst, ids, lut, n_expert_used, ne0, ids_stride, stride_slot, stride_token);
    }
}

static __global__ void mmids_tile_spec_kernel(
        const int32_t * __restrict__ bounds, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        int32_t * __restrict__ cursor, const int n_experts, const int tile_width) {
    const int expert = blockIdx.x*blockDim.x + threadIdx.x;
    if (expert >= n_experts) {
        return;
    }

    const int col_low_expert  = bounds[expert];
    const int col_high_expert = bounds[expert + 1];
    const int width = col_high_expert - col_low_expert;
    if (width <= 0) {
        return;
    }

    const int n_pieces = (width + tile_width - 1) / tile_width;
    const int base = atomicAdd(cursor, n_pieces);

    int col_low = col_low_expert;
    for (int p = 0; p < n_pieces; ++p) {
        const int col_high = min(col_low + tile_width, col_high_expert);
        tile_spec[base + p] = {expert, col_low, col_high,               0};
        col_low = col_high;
    }
}

void ggml_cuda_mmid_build_tile_spec(
        const int32_t * expert_bounds, ggml_cuda_mm_id_tile_spec * tile_spec, int32_t * tile_spec_cursor,
        const int n_experts, const int tile_width, cudaStream_t stream) {
    if (n_experts == 0 || tile_width <= 0) {
        return;
    }
    GGML_ASSERT((unsigned) n_experts <= 1024u && "one block, one thread per expert");

    const dim3 num_blocks(1, 1, 1);
    const dim3 block_size((unsigned) n_experts, 1, 1);

    mmids_tile_spec_kernel<<<num_blocks, block_size, 0, stream>>>(
        expert_bounds, tile_spec, tile_spec_cursor, n_experts, tile_width);

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, int32_t * __restrict__ expert_fill,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, const ggml_cuda_expert_lut lut, cudaStream_t stream) {
    GGML_ASSERT(n_tokens > 0 && n_expert_used > 0 && "no pairs to group");

    const int n_pairs = n_tokens*n_expert_used;
    GGML_ASSERT((int64_t) n_tokens*n_expert_used == (int64_t) n_pairs && "too many pairs to group");

    CUDA_CHECK(cudaMemsetAsync(expert_bounds, 0, (n_experts + 1)*sizeof(int32_t), stream));

    const int sms = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const unsigned count_blocks = (unsigned) std::min<int64_t>((n_pairs + 255)/256, 4*(int64_t) std::max(sms, 1));
    mmids_count_kernel<<<count_blocks, 256, 0, stream>>>(
        ids, n_pairs, lut, n_experts, n_expert_used, si1, expert_bounds);

    const unsigned scan_threads = (unsigned) std::min<int64_t>(GGML_PAD(n_experts, 32), 1024);
    mmids_scan_kernel<<<1, scan_threads, 0, stream>>>(expert_bounds, expert_fill, n_experts);

    mmids_place_kernel<<<(n_pairs + 255)/256, 256, 0, stream>>>(
        ids, n_pairs, lut, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse, expert_fill,
        ids_src1, ids_dst);

    CUDA_CHECK(cudaGetLastError());
}
