






#include "fused-expert-scale.cuh"
#include "mmq.cuh"

#define FES_BLOCK 256




__global__ void fused_expert_x16_scaled_kernel(
        const float * __restrict__ x, __half * __restrict__ y, int32_t * __restrict__ row_k,
        const int64_t ne0, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne1, const int64_t ne2, const int64_t ne3) {
    const int64_t row = blockIdx.x;
    const int64_t i03 = row / (ne1 * ne2);
    const int64_t i02 = (row - i03 * ne1 * ne2) / ne1;
    const int64_t i01 = row % ne1;

    const float * xr = x + i03 * s03 + i02 * s02 + i01 * s01;

    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < ne0; i += FES_BLOCK) {
        amax = fmaxf(amax, fabsf(xr[i]));
    }
    __shared__ float red[FES_BLOCK / 32];
    __shared__ int   k_shared;
    amax = warp_reduce_max<32>(amax);
    const int lane = threadIdx.x & 31;
    const int widp = threadIdx.x >> 5;
    if (lane == 0) {
        red[widp] = amax;
    }
    __syncthreads();
    if (widp == 0) {
        float m = (lane < FES_BLOCK / 32) ? red[lane] : 0.0f;
        m = warp_reduce_max<FES_BLOCK / 32>(m);
        if (lane == 0) {



            int k = 0;
            if (isfinite(m) && m > 65504.0f) {
                int e;
                frexpf(m, &e);
                k = e - 15;
            }
            k_shared = k;
            row_k[row] = k;
        }
    }
    __syncthreads();
    const int k = k_shared;
    __half * yr = y + row * ne0;
    if (k == 0) {


        for (int64_t i = threadIdx.x; i < ne0; i += FES_BLOCK) {
            yr[i] = __float2half_rn(xr[i] * 1.0f);
        }
    } else {
        for (int64_t i = threadIdx.x; i < ne0; i += FES_BLOCK) {
            yr[i] = __float2half_rn(ldexpf(xr[i], -k));
        }
    }
}

void ggml_cuda_fused_expert_x16_scaled(
        const float * x, __half * y, int32_t * row_k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    const int64_t rows = ne1 * ne2 * ne3;
    if (rows == 0 || ne0 == 0) {
        return;
    }
    fused_expert_x16_scaled_kernel<<<(unsigned) rows, FES_BLOCK, 0, stream>>>(
        x, y, row_k, ne0, s01, s02, s03, ne1, ne2, ne3);
    CUDA_CHECK(cudaGetLastError());
}
