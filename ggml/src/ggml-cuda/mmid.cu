#include "common.cuh"
#include "mmid.cuh"

#include <algorithm>

// ---------------------------------------------------------------------------------------------
// M3 device-side grouping of the (token, slot) pairs, adapted from Strata-AC922
// src/prefill/moe_fused.cu (MIT license, refs/strata-ac922): its count_kernel / block_excl_sum /
// scan_kernel / place_kernel.
//
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact
// src1 tensor sorted by expert; ids_dst describes the same mapping for the dst tensor; the range
// of the ith expert in the compact src1 tensor is expert_bounds[i:i+1]. The experts of the compact
// layout are the local expert numbers of this device: an id whose local number (lut, see
// ggml_cuda_expert_lut) is e belongs to expert e, the ids of the other devices map to -1 and are
// left out of every count. The compact ranges are ordered by local number, which is the order of
// the weight slice and of the expert store tables.
//
// The whole grouping is three passes over the pairs (count, place) plus one block that scans the
// counts, so its cost follows the number of pairs instead of the number of experts times the pairs.
// The rows of one expert come out in no fixed order; every consumer addresses a compact row
// through ids_src1 and ids_dst, and each compact row is one (token, slot) output element, so no
// output depends on that order (the property Strata's group() also relies on).
// ---------------------------------------------------------------------------------------------

// one thread per pair: count the pairs of every local expert
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

// exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
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
    __syncthreads();   // sh is reused by the next call
    return r;
}

// one block: turn the counts into the bounds and the placement cursors. The experts are split into
// one contiguous run per thread, so the block covers any expert count with blockDim threads and a
// thread's own run only has to accumulate its own counts (the second pass re-reads what the thread
// itself owns, no other thread writes those elements, which is what lets the counts be overwritten
// in place).
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

// one thread per pair: place the pair in its expert's range and write the two maps
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
    // the cursors start at the bounds, so the old value the atomic returns is the compact row
    const int c = atomicAdd(&fill[e], 1);
    ids_dst[c] = p;
    // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
    if (write_inverse) {
        ids_src1[p] = c;
    } else {
        ids_src1[c] = it*sis1 + iex % nchannels_y;
    }
}

// Expert parallelism: the MMQ and grouped kernels only write the (token, slot) entries whose
// expert is local, so the entries routed to other devices must be zeroed here. One block per
// (token, slot) pair, the threads of the block cover the output row. T is the output element
// type: the fused-expert SwiGLU epilogue produces F16 rows, every other path F32.
template <typename T>
__global__ void mmid_zero_nonlocal_kernel(
        T * __restrict__ dst, const int32_t * __restrict__ ids,
        const ggml_cuda_expert_lut lut, const int n_expert_used,
        const int64_t ne0, const int64_t ids_stride,
        const int64_t stride_slot, const int64_t stride_token) {
    const int pair  = blockIdx.x;
    const int token = pair / n_expert_used;
    const int slot  = pair % n_expert_used;

    // the same map as mm_ids_helper: a local expert keeps its (computed) entry
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

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, int32_t * __restrict__ expert_fill,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, const ggml_cuda_expert_lut lut, cudaStream_t stream) {
    GGML_ASSERT(n_tokens > 0 && n_expert_used > 0 && "no pairs to group");

    const int n_pairs = n_tokens*n_expert_used;
    GGML_ASSERT((int64_t) n_tokens*n_expert_used == (int64_t) n_pairs && "too many pairs to group");

    // the counts of the experts this device holds; the scan turns them into the bounds in place
    CUDA_CHECK(cudaMemsetAsync(expert_bounds, 0, (n_experts + 1)*sizeof(int32_t), stream));

    // the count grid follows the pair count and the device size, never the routing, so a captured
    // launch replays the same geometry
    const int sms = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const unsigned count_blocks = (unsigned) std::min<int64_t>((n_pairs + 255)/256, 4*(int64_t) std::max(sms, 1));
    mmids_count_kernel<<<count_blocks, 256, 0, stream>>>(
        ids, n_pairs, lut, n_experts, n_expert_used, si1, expert_bounds);

    // one block covers the experts with a grid-stride loop inside, so its width only has to be a
    // multiple of the warp size (block_excl_sum) and at most 1024, whatever the expert count is
    const unsigned scan_threads = (unsigned) std::min<int64_t>(GGML_PAD(n_experts, 32), 1024);
    mmids_scan_kernel<<<1, scan_threads, 0, stream>>>(expert_bounds, expert_fill, n_experts);

    mmids_place_kernel<<<(n_pairs + 255)/256, 256, 0, stream>>>(
        ids, n_pairs, lut, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse, expert_fill,
        ids_src1, ids_dst);

    CUDA_CHECK(cudaGetLastError());
}
