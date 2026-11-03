#pragma once

#include "common.cuh"















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




struct mmvf_narrow_side_rows {
    const float *  x;
    const float *  y;
    float *        dst;
    float *        scratch;
    unsigned int * counters;
    int64_t        stride_row;
    int            ncols2;
    int            stride_col_y2;
    int            stride_col_dst;
    int            nrows;
    int            ncols_dst;
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
