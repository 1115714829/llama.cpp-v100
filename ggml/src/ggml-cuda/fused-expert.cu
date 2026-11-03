#include "fused-expert.cuh"

#include "convert.cuh"
#include "fused-expert-scale.cuh"
#include "ggml-cuda-expert.cuh"
#include "mmid.cuh"
#include "moe-grouped.cuh"
#include "unary.cuh"

#include <cuda_fp16.h>
#include <cstdint>

// W4A16 fused-expert MUL_MAT_ID for Volta (sm_70), ported 1:1 from Strata-AC922's
// src/kernels/cuda/fused_expert.cu (MIT, Copyright (c) 2026 Niko1221 and the Strata
// contributors, see licenses/LICENSE-strata). The tile shape, the XOR swizzle, the per-step
// independent accumulators and the Q4_K/Q5_K/Q5_1/Q8_0 dequantizers are theirs; the expert
// addressing uses our grouped tables (expert_bounds / ids_src1 / ids_dst).
//
// Two output layouts, one per SWIGLU instantiation of the 256 kernel: plain F32 (the SwiGLU
// epilogue stays in the graph) or, when the MUL_MAT_ID carries the SwiGLU epilogue
// (ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU), the F16 H written by
// the paired-row epilogue below, as in Strata's gu_kernel.
// The weights are read row interleaved in that case: row 2c of the tile is the gate row of pair
// c, row 2c+1 its up row, so one BN-row weight tile is exactly BN/2 (gate, up) pairs.

namespace {




constexpr int TM = 64, BN = 64, BK = 64, LDE = BN + 4, NT = 128, TPR = NT / BN;



__device__ __forceinline__ int swz(int r) { return (r & 3) | (((r >> 3) & 1) << 2); }
__device__ __forceinline__ int soff(int r, int c) { return r * BK + ((c ^ swz(r)) << 3); }
constexpr int SMEM = (TM + BN) * BK * 2 > TM * LDE * 4 ? (TM + BN) * BK * 2 : TM * LDE * 4;





__device__ __forceinline__ float fused_expert_row_unscale(const int32_t * __restrict__ row_k, const int32_t row) {
    if (row_k == nullptr) {
        return 1.0f;
    }
    const int32_t k = row_k[row];
    return k == 0 ? 1.0f : ldexpf(1.0f, k);
}
using Acc = float[8];




#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING
#define FUSED_EXPERT_VOLTA_MMA 1
#else
#define FUSED_EXPERT_VOLTA_MMA 0
#endif



__device__ __forceinline__ int sbyte(uint32_t s0, uint32_t s1, uint32_t s2, int b) {
    const uint32_t w = b < 4 ? s0 : (b < 8 ? s1 : s2);
    return (int) ((w >> (8 * (b & 3))) & 0xFF);
}
__device__ __forceinline__ void scale_min_k4(int j, uint32_t s0, uint32_t s1, uint32_t s2, int& d, int& m) {
    if (j < 4) { d = sbyte(s0, s1, s2, j) & 63; m = sbyte(s0, s1, s2, j + 4) & 63; }
    else {
        const int a = sbyte(s0, s1, s2, j + 4), lo = sbyte(s0, s1, s2, j - 4), hi = sbyte(s0, s1, s2, j);
        d = (a & 0x0F) | ((lo >> 6) << 4); m = (a >> 4) | ((hi >> 6) << 4);
    }
}





struct XRegs { uint4 v[TM * BK / 8 / NT]; };
__device__ __forceinline__ void fetch_x(XRegs& x, const __half* __restrict__ X, const int32_t* __restrict__ rows,
                                        int64_t ld, int ne, int64_t k0) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8), c = (i % (BK / 8)) * 8;
        if (r < ne) {
            x.v[j] = *(const uint4*) (X + (int64_t) rows[r] * ld + k0 + c);
        } else {
            x.v[j] = make_uint4(0, 0, 0, 0);
        }
    }
}
__device__ __forceinline__ void commit_x(__half* Xs, const XRegs& x) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8);
        *(uint4*) &Xs[soff(r, i % (BK / 8))] = x.v[j];
    }
}
__device__ __forceinline__ void mma884(Acc& d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if FUSED_EXPERT_VOLTA_MMA
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, b0, b1);
    NO_DEVICE_CODE;
#endif
}



__device__ __forceinline__ void qp_coords(int& q, int& t) {
    const int lane = threadIdx.x & 31;
    q = (lane >> 2) & 3;
    t = (lane & 3) + ((lane >> 4) << 2);
}




__device__ __forceinline__ void mma_tile(const __half* Xs, const __half* Ws, Acc (&tot)[2][2], int ne) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2;
    if (32 * wm >= ne) return;
    const bool f1 = 32 * wm + 16 < ne;
    int q, t;
    qp_coords(q, t);
    const int ra0 = 32 * wm + 8 * (q & 1) + t, ra1 = ra0 + 16, rb0 = 32 * wn + 16 * (q >> 1) + t, rb1 = rb0 + 8;
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;

    if (f1) {
#pragma unroll
        for (int c = 0; c < BK / 8; ++c) {
            const uint4 b0 = *(const uint4*) &Ws[soff(rb0, c)], b1 = *(const uint4*) &Ws[soff(rb1, c)];
            const uint4 a0 = *(const uint4*) &Xs[soff(ra0, c)], a1 = *(const uint4*) &Xs[soff(ra1, c)];
            mma884(acc[0][0], a0.x, a0.y, b0.x, b0.y);
            mma884(acc[0][1], a0.x, a0.y, b1.x, b1.y);
            mma884(acc[1][0], a1.x, a1.y, b0.x, b0.y);
            mma884(acc[1][1], a1.x, a1.y, b1.x, b1.y);
            mma884(acc[0][0], a0.z, a0.w, b0.z, b0.w);
            mma884(acc[0][1], a0.z, a0.w, b1.z, b1.w);
            mma884(acc[1][0], a1.z, a1.w, b0.z, b0.w);
            mma884(acc[1][1], a1.z, a1.w, b1.z, b1.w);
        }
    } else {
#pragma unroll
        for (int c = 0; c < BK / 8; ++c) {
            const uint4 b0 = *(const uint4*) &Ws[soff(rb0, c)], b1 = *(const uint4*) &Ws[soff(rb1, c)];
            const uint4 a0 = *(const uint4*) &Xs[soff(ra0, c)];
            mma884(acc[0][0], a0.x, a0.y, b0.x, b0.y);
            mma884(acc[0][1], a0.x, a0.y, b1.x, b1.y);
            mma884(acc[0][0], a0.z, a0.w, b0.z, b0.w);
            mma884(acc[0][1], a0.z, a0.w, b1.z, b1.w);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) tot[i][j][e] += acc[i][j][e];
}


__device__ __forceinline__ void store_acc(float (*E)[LDE], Acc (&acc)[2][2]) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2, lane = threadIdx.x & 31;
    int q, t;
    qp_coords(q, t);
    (void) t;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r = 32 * wm + 8 * ((q & 1) + 2 * i) + (lane & 1) + (e & 2) + ((lane >> 4) << 2);
                const int c = 32 * wn + 8 * (2 * (q >> 1) + j) + (e & 4) + (lane & 2) + (e & 1);
                E[r][c] = acc[i][j][e];
            }
}





template<int GT, bool SWIGLU>
__device__ __forceinline__ void fused_expert_256_tile(
        unsigned char* sm, const char* __restrict__ wbase, const __half* __restrict__ X, const int32_t* __restrict__ ids_dst,
        void* __restrict__ D, const int32_t* __restrict__ rows, int ne, int rt, int col_start,
        int64_t K, int64_t nb01, int64_t Xld, int64_t dst_stride_col, const int32_t * __restrict__ row_k) {
    constexpr int BB = GT == 12 ? 144 : 176, QS = GT == 12 ? 16 : 48;
    __half* Xs = (__half*) sm;
    __half* Ws = (__half*) (sm + TM * BK * 2);
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;


    constexpr int NV = 4 / TPR;
    const int wr = threadIdx.x / TPR, hi = threadIdx.x % 2, v0 = (threadIdx.x % TPR) / 2 * NV;
    const uint8_t* rowp = (const uint8_t*) wbase + (size_t) (rt * BN + wr) * nb01;
    XRegs xr;
    uint4 qr[NV], hr, qhr[GT == 13 ? NV : 1];
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 256) * BB;
        const int jj = (int) (k0 % 256) / 64;
        hr = *(const uint4*) b;
#pragma unroll
        for (int v = 0; v < NV; ++v) qr[v] = ((const uint4*) (b + QS + 32 * jj))[v0 + v];
        if constexpr (GT == 13) {
#pragma unroll
            for (int v = 0; v < NV; ++v) qhr[v] = ((const uint4*) (b + 16))[v0 + v];
        }
    };
    auto commit_w = [&](int64_t k0) {
        const int jj = (int) (k0 % 256) / 64;
        const float2 dm = __half22float2(*(const __half2*) &hr.x);
        int s, m;
        scale_min_k4(2 * jj + hi, hr.y, hr.z, hr.w, s, m);

        const float d1 = dm.x * (uint8_t) s, m1 = dm.y * (uint8_t) m;
        const int sh = 4 * hi;

#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const uint8_t* q = (const uint8_t*) &qr[v];
            __align__(16) __half o[16];
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                if constexpr (GT == 13) {
                    const uint8_t* qh = (const uint8_t*) &qhr[v];
                    const uint8_t hm = (uint8_t) (1u << (2 * jj + hi));
                    o[l] = __float2half(d1 * (((q[l] >> sh) & 0xF) + (qh[l] & hm ? 16 : 0)) - m1);
                } else {
                    o[l] = __float2half(d1 * ((q[l] >> sh) & 0xF) - m1);
                }
            }
            *(uint4*) &Ws[soff(wr, hi * 4 + 2 * (v0 + v))] = *(const uint4*) &o[0];
            *(uint4*) &Ws[soff(wr, hi * 4 + 2 * (v0 + v) + 1)] = *(const uint4*) &o[8];
        }
    };
    fetch_x(xr, X, rows, Xld, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        commit_x(Xs, xr);
        commit_w(k0);
        __syncthreads();
        if (k0 + BK < K) { fetch_x(xr, X, rows, Xld, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc, ne);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    if constexpr (SWIGLU) {



        __half * H = (__half *) D;
        for (int i = threadIdx.x; i < TM * (BN / 2); i += NT) {
            const int r = i / (BN / 2), c = i % (BN / 2);
            if (r < ne) {




                const float u = fused_expert_row_unscale(row_k, rows[r]);
                const float h = ggml_cuda_op_silu_single(E[r][2 * c] * u) * (E[r][2 * c + 1] * u);
                H[(size_t) ids_dst[col_start + r] * dst_stride_col + rt * (BN / 2) + c] = __float2half(h);
            }
        }
    } else {
        for (int i = threadIdx.x; i < TM * BN; i += NT) {
            const int r = i / BN, c = i % BN;
            if (r < ne) ((float *) D)[(size_t) ids_dst[col_start + r] * dst_stride_col + rt * BN + c] = E[r][c] * fused_expert_row_unscale(row_k, rows[r]);
        }
    }
}


template<int DT>
__device__ __forceinline__ void fused_expert_32_tile(
        unsigned char* sm, const char* __restrict__ wbase, const __half* __restrict__ X, const int32_t* __restrict__ ids_dst,
        float* __restrict__ D, const int32_t* __restrict__ rows, int ne, int rt, int col_start,
        int64_t K, int64_t nb01, int64_t Xld, int64_t dst_stride_col, const int32_t * __restrict__ row_k) {
    __half* Xs = (__half*) sm;
    __half* Ws = (__half*) (sm + TM * BK * 2);
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;


    constexpr int NI = 32 / (TPR / 2);
    const int wr = threadIdx.x / TPR, hi = threadIdx.x % 2, i0 = (threadIdx.x % TPR) / 2 * NI;
    constexpr int BB = DT == 7 ? 24 : 34;
    const uint8_t* rowp = (const uint8_t*) wbase + (size_t) (rt * BN + wr) * nb01;
    XRegs xr;
    uint2 w5[3];
    uint16_t w8[17];
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 32 + hi) * BB;
        if constexpr (DT == 7) {
            w5[0] = ((const uint2*) b)[0]; w5[1] = ((const uint2*) b)[1]; w5[2] = ((const uint2*) b)[2];
        } else {
#pragma unroll
            for (int i = 0; i < 17; ++i) w8[i] = ((const uint16_t*) b)[i];
        }
    };
    auto commit_w = [&]() {
        __align__(16) __half y[NI];
        if constexpr (DT == 7) {


            const float2 dm = __half22float2(*(const __half2*) &w5[0].x);
            const uint32_t qh = w5[0].y;
            const uint8_t* q = (const uint8_t*) &w5[1];
            const int b0 = i0 / 2;
#pragma unroll
            for (int l = 0; l < NI / 2; ++l) {
                const int i = b0 + l;
                const int xh_0 = ((qh >> (i + 0)) << 4) & 0x10, xh_1 = ((qh >> (i + 12))) & 0x10;
                y[l] = __float2half((float) ((q[i] & 0xf) | xh_0) * dm.x + dm.y);
                y[l + NI / 2] = __float2half((float) ((q[i] >> 4) | xh_1) * dm.x + dm.y);
            }
#pragma unroll
            for (int l = 0; l < NI / 16; ++l) {
                *(uint4*) &Ws[soff(wr, hi * 4 + b0 / 8 + l)] = *(const uint4*) &y[8 * l];
                *(uint4*) &Ws[soff(wr, hi * 4 + 2 + b0 / 8 + l)] = *(const uint4*) &y[NI / 2 + 8 * l];
            }
        } else {
            const float d = __half2float(*(const __half*) &w8[0]);
            const int8_t* q = (const int8_t*) &w8[1];
#pragma unroll
            for (int i = 0; i < NI; ++i) y[i] = __float2half((float) q[i0 + i] * d);
#pragma unroll
            for (int i = 0; i < NI / 8; ++i) *(uint4*) &Ws[soff(wr, hi * 4 + i0 / 8 + i)] = *(const uint4*) &y[8 * i];
        }
    };
    fetch_x(xr, X, rows, Xld, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        commit_x(Xs, xr);
        commit_w();
        __syncthreads();
        if (k0 + BK < K) { fetch_x(xr, X, rows, Xld, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc, ne);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * BN; i += NT) {
        const int r = i / BN, c = i % BN;
        if (r < ne) D[(size_t) ids_dst[col_start + r] * dst_stride_col + rt * BN + c] = E[r][c] * fused_expert_row_unscale(row_k, rows[r]);
    }
}




template<int GT, bool SWIGLU>
__global__ void __launch_bounds__(NT) fused_expert_256_kernel(
        const char* __restrict__ vx, const int32_t* __restrict__ expert_bounds,
        const int32_t* __restrict__ ids_src1, const int32_t* __restrict__ ids_dst,
        const __half* __restrict__ X, void* __restrict__ D,
        const int64_t K, const int64_t nb01, const int64_t nb02, const int64_t Xld, const int64_t dst_stride_col,
        const char* const* __restrict__ expert_ptrs, const int32_t * __restrict__ row_k) {
    const int e = blockIdx.y;
    const int col_start = expert_bounds[e];
    const int ne_total = expert_bounds[e + 1] - col_start;
    if (ne_total <= 0) {
        return;
    }
    extern __shared__ __align__(16) unsigned char sm[];
    const int rt = blockIdx.x;
    const char* wbase = expert_ptrs != nullptr ? expert_ptrs[e] : vx + (size_t) e * nb02;
    for (int tt = 0; tt * TM < ne_total; ++tt) {
        const int ne = min(TM, ne_total - tt * TM);
        const int32_t* rows = ids_src1 + col_start + tt * TM;
        fused_expert_256_tile<GT, SWIGLU>(sm, wbase, X, ids_dst, D, rows, ne, rt, col_start + tt * TM,
            K, nb01, Xld, dst_stride_col, row_k);
        __syncthreads();
    }
}

template<int DT>
__global__ void __launch_bounds__(NT) fused_expert_32_kernel(
        const char* __restrict__ vx, const int32_t* __restrict__ expert_bounds,
        const int32_t* __restrict__ ids_src1, const int32_t* __restrict__ ids_dst,
        const __half* __restrict__ X, float* __restrict__ D,
        const int64_t K, const int64_t nb01, const int64_t nb02, const int64_t Xld, const int64_t dst_stride_col,
        const char* const* __restrict__ expert_ptrs, const int32_t * __restrict__ row_k) {
    const int e = blockIdx.y;
    const int col_start = expert_bounds[e];
    const int ne_total = expert_bounds[e + 1] - col_start;
    if (ne_total <= 0) {
        return;
    }
    extern __shared__ __align__(16) unsigned char sm[];
    const int rt = blockIdx.x;
    const char* wbase = expert_ptrs != nullptr ? expert_ptrs[e] : vx + (size_t) e * nb02;
    for (int tt = 0; tt * TM < ne_total; ++tt) {
        const int ne = min(TM, ne_total - tt * TM);
        const int32_t* rows = ids_src1 + col_start + tt * TM;
        fused_expert_32_tile<DT>(sm, wbase, X, ids_dst, D, rows, ne, rt, col_start + tt * TM,
            K, nb01, Xld, dst_stride_col, row_k);
        __syncthreads();
    }
}

}  // namespace

static void fused_expert_launch(const ggml_tensor * dst, const char * const * expert_ptrs,
        const int32_t * expert_bounds, const int32_t * ids_src1, const int32_t * ids_dst,
        const __half * X, const int64_t Xld, const bool swiglu, cudaStream_t stream, const int32_t * row_k) {
    const ggml_tensor * src0 = dst->src[0];

#if !defined(GGML_USE_HIP)

    static const bool smem_set = [] {
        cudaFuncSetAttribute(fused_expert_256_kernel<12, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(fused_expert_256_kernel<13, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(fused_expert_256_kernel<12, true >, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(fused_expert_256_kernel<13, true >, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(fused_expert_32_kernel<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(fused_expert_32_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void) smem_set;
#endif

    const dim3 grid((unsigned) (src0->ne[1] / BN), (unsigned) src0->ne[2]);
    const dim3 block(NT);

    const int64_t K = src0->ne[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    void * D = dst->data;

    const int64_t dst_stride_col = dst->nb[1] / (swiglu ? sizeof(__half) : sizeof(float));

    if (swiglu) {

        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                fused_expert_256_kernel<12, true><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            case GGML_TYPE_Q5_K:
                fused_expert_256_kernel<13, true><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            default:
                GGML_ABORT("fatal error: no SwiGLU epilogue for %s", ggml_type_name(src0->type));
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                fused_expert_256_kernel<12, false><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            case GGML_TYPE_Q5_K:
                fused_expert_256_kernel<13, false><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            case GGML_TYPE_Q5_1:
                fused_expert_32_kernel<7><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, (float *) D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            case GGML_TYPE_Q8_0:
                fused_expert_32_kernel<8><<<grid, block, SMEM, stream>>>((const char *) src0->data, expert_bounds,
                    ids_src1, ids_dst, X, (float *) D, K, nb01, nb02, Xld, dst_stride_col, expert_ptrs, row_k);
                break;
            default:
                GGML_ABORT("fatal error");
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_fused_expert_mul_mat_id_supported(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];





    const bool swiglu = ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU;




    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }
    if (dst->type != (swiglu ? GGML_TYPE_F16 : GGML_TYPE_F32) || ids->type != GGML_TYPE_I32) {
        return false;
    }




    if (dst->ne[2] <= MOE_GROUPED_MAX_TOKENS) {
        return false;
    }


    if (swiglu && src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) {
        return false;
    }



    switch (src0->type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            if (src0->ne[0] % 256 != 0 || src0->nb[1] % 16 != 0) {
                return false;
            }
            break;
        case GGML_TYPE_Q5_1:
            if (src0->ne[0] % BK != 0 || src0->nb[1] % 8 != 0) {
                return false;
            }
            break;
        case GGML_TYPE_Q8_0:
            if (src0->ne[0] % BK != 0 || src0->nb[1] % 2 != 0) {
                return false;
            }
            break;
        default:
            return false;
    }

    if (src0->ne[1] % BN != 0) {
        return false;
    }

    if (src0->extra != nullptr) {
        return false;
    }
    if (src0->ne[2] <= 0 || src0->ne[2] > 65535) {
        return false;
    }
    if (src1->ne[3] != 1 || src0->ne[3] != 1) {
        return false;
    }



    if (src1->nb[0] != ggml_type_size(src1->type) || src1->nb[1] % 16 != 0) {
        return false;
    }
    if (src1->nb[2] % src1->nb[1] != 0) {
        return false;
    }


    if (dst->ne[0] != (swiglu ? src0->ne[1] / 2 : src0->ne[1]) || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) {
        return false;
    }
    if (src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || dst->nb[1] % ggml_type_size(dst->type) != 0) {
        return false;
    }
    return true;
}

bool ggml_cuda_fused_expert_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc)) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;



    const bool swiglu = ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU;

    cudaStream_t stream = ctx.stream();

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = ne12 * n_expert_used;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    ggml_cuda_pool_alloc<int32_t> expert_fill(ctx.pool(), ne02);




    const ggml_cuda_expert_lut lut = ggml_cuda_mul_mat_id_lut(ctx, dst);



    if (ggml_mul_mat_id_is_expert_parallel(dst) && ctx.moe_reduce_local_slots != dst) {


        const size_t ts = swiglu ? sizeof(__half) : sizeof(float);
        ggml_cuda_mmid_zero_nonlocal(dst->data, (const int32_t *) ids->data, lut,
            ne12, (int) n_expert_used, dst->ne[0], ids->nb[1] / (int64_t) sizeof(int32_t),
            dst->nb[1] / ts, dst->nb[2] / ts, stream, swiglu);
    }

    {
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        const int si1  = ids->nb[1] / ggml_type_size(ids->type);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(),
            expert_bounds.get(), expert_fill.get(), ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/false,
            lut, stream);
        CUDA_CHECK(cudaGetLastError());
    }











    if (src1->type == GGML_TYPE_F16) {
        fused_expert_launch(dst, expert_ptrs, expert_bounds.get(), ids_src1.get(), ids_dst.get(),
            (const __half *) src1->data, nb11 / (int64_t) sizeof(__half), swiglu, stream, nullptr);
    } else if (cc == GGML_CUDA_CC_VOLTA) {
        ggml_cuda_pool_alloc<__half> X16(ctx.pool(), ggml_nelements(src1));
        ggml_cuda_pool_alloc<int32_t> row_k(ctx.pool(), (size_t) ne11 * ne12 * ne13);
        ggml_cuda_fused_expert_x16_scaled((const float *) src1->data, X16.get(), row_k.get(),
            ne00, ne11, ne12, ne13, nb11 / (int64_t) sizeof(float), nb12 / (int64_t) sizeof(float),
            nb13 / (int64_t) sizeof(float), stream);

        fused_expert_launch(dst, expert_ptrs, expert_bounds.get(), ids_src1.get(), ids_dst.get(),
            X16.get(), ne00, swiglu, stream, row_k.get());
    } else {
        ggml_cuda_pool_alloc<__half> X16(ctx.pool(), ggml_nelements(src1));
        if (ggml_is_contiguous(src1)) {
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, X16.get(), ggml_nelements(src1), stream);
        } else {
            ggml_get_to_fp16_nc_cuda(GGML_TYPE_F32)(src1->data, X16.get(), ne00, ne11, ne12, ne13,
                nb11 / (int64_t) sizeof(float), nb12 / (int64_t) sizeof(float), nb13 / (int64_t) sizeof(float), stream);
        }
        CUDA_CHECK(cudaGetLastError());

        fused_expert_launch(dst, expert_ptrs, expert_bounds.get(), ids_src1.get(), ids_dst.get(), X16.get(), ne00, swiglu, stream, nullptr);
    }

    return true;
}
