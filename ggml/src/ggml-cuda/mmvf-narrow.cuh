#pragma once

#include "common.cuh"

// ---- narrow F32/BF16 GEMV: one warp of the split kernel ----
//
// mul_mat_vec_f runs one 256-thread block per output row: thread t adds columns t, t + 256, ...
// (float2 steps), each warp reduces its sums with a butterfly and the block reduces the 8 warp sums
// the same way. mul_mat_vec_f_split (mmvf.cu) runs each of the 8 warps of a row as its own CTA:
// the warp stores its reduced sums to a scratch, the last of the 8 (one atomic counter per row)
// loads them into lanes 0..7 (lanes 8..31 add zero, as the zeroed shared buffer does), reduces
// them with the same butterfly and resets the counter, so the results are bitwise those of
// mul_mat_vec_f. The function below is that warp: the split kernel, the kernel for several narrow
// weights of one input (mmvf.cu), the CTAs that other kernels add for the narrow projections of
// their input (q8-skinny.cu, the qwen4exp shared expert gate next to its gate/up pair) and the
// hyper-connection norm kernel that computes the previous mixer's inject (norm.cu) all run it, so the
// arithmetic is the same everywhere.

static constexpr int mmvf_narrow_block_size  = 256;
static constexpr int mmvf_narrow_split_warps = mmvf_narrow_block_size / WARP_SIZE;

template <typename T, int ncols_dst>
static __device__ __forceinline__ void mul_mat_vec_f_narrow_accumulate(
        const T * __restrict__ x, const float2 * __restrict__ y2, const int ncols2, const int stride_col_y2,
        const int tid, float (&sumf)[ncols_dst]) {
    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
#pragma unroll 4
        for (int col2 = tid; col2 < ncols2; col2 += mmvf_narrow_block_size) {
            const float2 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    } else {
        static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported type");
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
#pragma unroll 4
        for (int col2 = tid; col2 < ncols2; col2 += mmvf_narrow_block_size) {
            const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    }
}

// Warp `warp` (0..7) of row `x_row`: dst points at the row's element of column 0, row_scratch at the
// row's [warp][column] scratch, row_counter at the row's counter (zero between launches).
template <typename T, int ncols_dst>
static __device__ __forceinline__ void mul_mat_vec_f_split_warp(
        const T * __restrict__ x_row, const float2 * __restrict__ y2, const int ncols2, const int stride_col_y2,
        float * __restrict__ dst, const int stride_col_dst, float * __restrict__ row_scratch,
        unsigned int * __restrict__ row_counter, const int warp, const int lane) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    float sumf[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = 0.0f;
    }
    mul_mat_vec_f_narrow_accumulate<T, ncols_dst>(x_row, y2, ncols2, stride_col_y2, warp*WARP_SIZE + lane, sumf);

    // [warp][column]
    float * warp_sums = row_scratch + warp*ncols_dst;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
        if (lane == 0) {
            warp_sums[j] = sumf[j];
        }
    }
    __threadfence();
    __syncwarp();
    unsigned int last = 0;
    if (lane == 0) {
        last = atomicAdd(row_counter, 1u) == (unsigned int) (mmvf_narrow_split_warps - 1);
    }
    last = __shfl_sync(0xffffffffU, last, 0);
    if (!last) {
        return;
    }
    __threadfence();

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float v = lane < mmvf_narrow_split_warps ? __ldcg(row_scratch + lane*ncols_dst + j) : 0.0f;
        v = warp_reduce_sum<WARP_SIZE>(v);
        if (lane == j) {
            dst[j*stride_col_dst] = v;
        }
    }
    if (lane == 0) {
        *row_counter = 0;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(x_row, y2, ncols2, stride_col_y2, dst, stride_col_dst, row_scratch, row_counter, warp, lane);
#endif
}

// Narrow rows that a kernel runs next to its own work, one virtual warp per (row, warp) as in
// mul_mat_vec_f_split: F32 weights [nrows][ncols] (row stride stride_row), F32 input y [ncols_dst][ncols]
// (column stride stride_col_y), dst [ncols_dst][nrows] (column stride stride_col_dst).
struct mmvf_narrow_side_rows {
    const float *  x;
    const float *  y;
    float *        dst;
    float *        scratch;   // nrows * 8 * ncols_dst floats
    unsigned int * counters;  // nrows counters, zero between launches
    int64_t        stride_row;
    int            ncols2;
    int            stride_col_y2;
    int            stride_col_dst;
    int            nrows;
    int            ncols_dst; // 1..4 (the F32 GEMV takes mul_mat_vec_f up to 4 columns on Volta)
};

template <int ncols_dst>
static __device__ __forceinline__ void mmvf_narrow_side_rows_warp_n(const mmvf_narrow_side_rows & s, const int vwarp,
                                                                    const int lane) {
    const int row  = vwarp / mmvf_narrow_split_warps;
    const int warp = vwarp % mmvf_narrow_split_warps;
    mul_mat_vec_f_split_warp<float, ncols_dst>(s.x + row*s.stride_row, (const float2 *) s.y, s.ncols2, s.stride_col_y2,
                                               s.dst + row, s.stride_col_dst,
                                               s.scratch + (int64_t) row*mmvf_narrow_split_warps*ncols_dst,
                                               s.counters + row, warp, lane);
}

// Virtual warp vwarp in [0, nrows * 8) of the side rows.
static __device__ __forceinline__ void mmvf_narrow_side_rows_warp(const mmvf_narrow_side_rows & s, const int vwarp,
                                                                  const int lane) {
    if (vwarp >= s.nrows * mmvf_narrow_split_warps) {
        return;
    }
    switch (s.ncols_dst) {
        case 1: mmvf_narrow_side_rows_warp_n<1>(s, vwarp, lane); break;
        case 2: mmvf_narrow_side_rows_warp_n<2>(s, vwarp, lane); break;
        case 3: mmvf_narrow_side_rows_warp_n<3>(s, vwarp, lane); break;
        case 4: mmvf_narrow_side_rows_warp_n<4>(s, vwarp, lane); break;
        default: break;
    }
}
