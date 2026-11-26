// Portions from Strata and its AC922 fork; Copyright (c) 2026 Niko1221 and the Strata contributors (MIT); see licenses/LICENSE-strata.

#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <mma.h>
namespace wmma = nvcuda::wmma;

#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);

    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        __syncthreads();

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        __syncthreads();

        qk_shared[0][h][k] = sum;

        __syncthreads();

        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        __syncthreads();
    }

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif
#endif

#define LIGHTNING_INDEXER_TILE_TOKENS 8

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING
#define LIGHTNING_INDEXER_VOLTA_MMA 1
#else
#define LIGHTNING_INDEXER_VOLTA_MMA 0
#endif

struct lightning_indexer_scores_volta_smem {
    half  qh[4][16][136];
    half  ql[4][16][136];
    half  kh[4][16][24];
    half  kl[4][16][24];
    float ep[4][16][20];
};

static __device__ __forceinline__ void lightning_indexer_split16(const float x, half & hi, half & lo) {
    hi = __float2half_rn(x);
    lo = __float2half_rn((x - __half2float(hi))*2048.0f);
}

template <ggml_type TYPE_K>
static __device__ __forceinline__ void lightning_indexer_read8(const char * __restrict__ row, const int d0, float4 & v0, float4 & v1) {
    if constexpr (TYPE_K == GGML_TYPE_F32) {
        v0 = reinterpret_cast<const float4 *>(row)[d0/4];
        v1 = reinterpret_cast<const float4 *>(row)[d0/4 + 1];
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
        dequantize_k(row, &v0, d0);
        dequantize_k(row, &v1, d0 + 4);
    }
}

template <ggml_type TYPE_K, bool reg_heads, int iterations, bool masked>
static __global__ void __launch_bounds__(128) lightning_indexer_scores_volta_kernel(
        const float * __restrict__ Q, const char * __restrict__ K, const int32_t * __restrict__ cells, const float w,
        float * __restrict__ dst, const size_t nbq1, const size_t nbq2, const size_t nbk2, const size_t nb1,
        const int64_t n_batch, const int64_t n_kv, const int32_t * ranges, int64_t stride, int top) {
#if LIGHTNING_INDEXER_VOLTA_MMA
    constexpr int   VQ       = 16;
    constexpr int   VQS      = 136;
    constexpr int   VKS      = 24;
    constexpr float LO_SCALE = 2048.0f;

    extern __shared__ __align__(16) unsigned char smraw[];
    lightning_indexer_scores_volta_smem & S = *reinterpret_cast<lightning_indexer_scores_volta_smem *>(smraw);

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int64_t q0 = (int64_t) blockIdx.y*VQ;

    if constexpr (masked) {
        const int qr = t & 31;
        bool needed = false;
        if (qr < VQ && q0 + qr < n_batch) {
            const int32_t * r = ranges + (q0 + qr)*stride;
            needed = (int64_t) r[1] - r[0] > top;
        }
        if (!__any_sync(0xffffffff, needed)) { return; }
    }

    for (int i = t; i < VQ*4*(128/4); i += 128) {
        const int row = i / (128/4);
        const int c   = i % (128/4);
        const int qr  = row / 4;
        const int h   = row % 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + qr < n_batch) {
            v = *reinterpret_cast<const float4 *>((const char *) Q + (q0 + qr)*nbq2 + h*nbq1 + c*sizeof(float4));
        }
        const float x[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lightning_indexer_split16(x[j], S.qh[h][qr][c*4 + j], S.ql[h][qr][c*4 + j]);
        }
    }

    __syncthreads();

    for (int it = 0; it < iterations; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x*iterations + it)*4 + warp)*16;
        if (b0 >= n_kv) {
            break;
        }

        const int kr = lane >> 1;
        const int kc = (lane & 1)*8;
        const int64_t row = b0 + kr;
        const bool rv = row < n_kv;
        const char * kp = K + (rv ? (int64_t) cells[row] : 0)*nbk2;

        float4 n0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 n1 = n0;
        if (rv) {
            lightning_indexer_read8<TYPE_K>(kp, kc, n0, n1);
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4], cor[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            wmma::fill_fragment(acc[h], 0.0f);
            wmma::fill_fragment(cor[h], 0.0f);
        }

#pragma unroll 1
        for (int kk = 0; kk < 128/16; ++kk) {
            const float x[8] = { n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w };
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                lightning_indexer_split16(x[j], S.kh[warp][kr][kc + j], S.kl[warp][kr][kc + j]);
            }
            __syncwarp();
            if (kk + 1 < 128/16 && rv) {
                lightning_indexer_read8<TYPE_K>(kp, kc + (kk + 1)*16, n0, n1);
            }

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Ah, Al;
            wmma::load_matrix_sync(Ah, &S.kh[warp][0][0], VKS);
            wmma::load_matrix_sync(Al, &S.kl[warp][0][0], VKS);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Bh, Bl;
                wmma::load_matrix_sync(Bh, &S.qh[h][0][kk*16], VQS);
                wmma::load_matrix_sync(Bl, &S.ql[h][0][kk*16], VQS);

                wmma::fragment<wmma::accumulator, 16, 16, 16, float> st;
                wmma::fill_fragment(st, 0.0f);
                wmma::mma_sync(st, Ah, Bh, st);
#pragma unroll
                for (int e = 0; e < st.num_elements; ++e) {
                    acc[h].x[e] += st.x[e];
                }
                wmma::mma_sync(cor[h], Ah, Bl, cor[h]);
                wmma::mma_sync(cor[h], Al, Bh, cor[h]);
            }
        }

        float s_out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_out[j] = 0.0f;
        }
        if constexpr (reg_heads) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> combined;
            wmma::fill_fragment(combined, 0.0f);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
#pragma unroll
                for (int e = 0; e < combined.num_elements; ++e) {
                    const float d = acc[h].x[e] + cor[h].x[e]*(1.0f/LO_SCALE);
                    combined.x[e] += (d > 0.0f ? d : 0.0f)*w;
                }
            }
            wmma::store_matrix_sync(&S.ep[warp][0][0], combined, 20, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = lane + 32*j;
                s_out[j] = S.ep[warp][i % 16][i / 16];
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
#pragma unroll
                for (int e = 0; e < sc.num_elements; ++e) {
                    const float d = acc[h].x[e] + cor[h].x[e]*(1.0f/LO_SCALE);
                    sc.x[e] = d > 0.0f ? d : 0.0f;
                }
                wmma::store_matrix_sync(&S.ep[warp][0][0], sc, 16 + 4, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int i = lane + 32*j;
                    s_out[j] += S.ep[warp][i % 16][i / 16]*w;
                }
                __syncwarp();
            }

        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i  = lane + 32*j;
            const int qr = i / 16;
            const int br = i % 16;
            const int64_t b = b0 + br;
            if (q0 + qr < n_batch && b < n_kv) {
                ((float *) ((char *) dst + (q0 + qr)*nb1))[b] = s_out[j];
            }
        }
    }
#endif
}

#define QSA_VOLTA_LAUNCH(type, reg, count, mask) \
    lightning_indexer_scores_volta_kernel<type, reg, count, mask><<<grid, block, smem, stream>>>( \
        q_d, k_d, cells_d, head_scale, dst_d, nbq1, nbq2, k->nb[2], nb1, n_batch, n_pool, ranges, stride, top)
#define QSA_VOLTA_LAUNCH_REG(type, count, mask) \
    do { \
        if (combine_heads) { QSA_VOLTA_LAUNCH(type, true, count, mask); } \
        else { QSA_VOLTA_LAUNCH(type, false, count, mask); } \
    } while (0)
#define LIGHTNING_INDEXER_SCORES_VOLTA_CASE(type_K) \
    if (k->type == (type_K)) { \
        if (iterations == 8) { \
            if (ranges) { QSA_VOLTA_LAUNCH_REG(type_K, 8, true); } else { QSA_VOLTA_LAUNCH_REG(type_K, 8, false); } \
        } else { \
            if (ranges) { QSA_VOLTA_LAUNCH_REG(type_K, 4, true); } else { QSA_VOLTA_LAUNCH_REG(type_K, 4, false); } \
        } \
    } else

#endif

#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
#else
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
#endif

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool HAS_MASK = true>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {

#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {

        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            float m_val = 0.0f;
            if constexpr (HAS_MASK) {
                const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
                m_val = __half2float(m_base[i_kv]);
            }
            dst_base[i_kv] = dst_shared[tid] + m_val;
        }
    }
}

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K, bool HAS_MASK = true>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;
    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll
    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {

        if (h0 > 0) {
            __syncthreads();
        }

#pragma unroll
        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
            const int c4 = i % (N_EMBD / 4);

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_batch + r < n_batch) {
                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
            }

            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
        }

        __syncthreads();

#pragma unroll 8
        for (int c = 0; c < N_EMBD_H2; ++c) {
            float2 k_val[KEYS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_PASS; ++h) {
                const float2 q_val = q_shared[h][tl][c];
#pragma unroll
                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
                }
            }
        }
    }

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            float m_val = 0.0f;
            if constexpr (HAS_MASK) {
                const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
                m_val = __half2float(m_base[i_kv]);
            }
            dst_base[i_kv] = score[j] + m_val;
        }
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {

            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else
        {
#endif

            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {

            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else
        {
#endif

            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {

        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else if (n_embd == 128 && n_head == 4) {

        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("fatal error");
    }
}

#define QSA_SV_THREADS 256
#define QSA_SV_TOKENS  8
#define QSA_SV_KPW     4

static __device__ __forceinline__ float qsa_reduce_scatter16(const float (&p)[16], const int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? p[i + 8] : p[i]) + __shfl_xor_sync(0xffffffff, hi ? p[i] : p[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffff, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffff, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool  hi = (lane & 2) != 0;
    const float r1 = (hi ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffff, hi ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffff, r1, 1);
}

template <ggml_type TYPE_K>
static __global__ void __launch_bounds__(QSA_SV_THREADS) qsa_scores_vec_kernel(
        const float * __restrict__ Q, const size_t nbq1, const size_t nbq2, const int n_batch,
        const char * __restrict__ K, const size_t nbk2, const int32_t * __restrict__ cells, const int64_t n_kv,
        const float w, float * __restrict__ dst, const size_t nb1, const int32_t * ranges, int64_t stride, int top) {
    if (ranges) {
        bool needed = false;
        for (int q = 0; q < n_batch; ++q) {
            const int32_t * r = ranges + q*stride;
            needed |= (int64_t) r[1] - r[0] > top;
        }
        if (!needed) { return; }
    }
    constexpr int WARPS = QSA_SV_THREADS/WARP_SIZE;
    constexpr int KPW   = QSA_SV_KPW;

    __shared__ float4 sq[QSA_SV_TOKENS][4][WARP_SIZE];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    for (int i = threadIdx.x; i < n_batch*4*WARP_SIZE; i += QSA_SV_THREADS) {
        const int t = i / (4*WARP_SIZE);
        const int h = i / WARP_SIZE % 4;
        const int c = i % WARP_SIZE;
        sq[t][h][c] = *(const float4 *) ((const char *) Q + t*nbq2 + h*nbq1 + c*sizeof(float4));
    }

    __syncthreads();

    for (int64_t k0 = ((int64_t) blockIdx.x*WARPS + warp)*KPW; k0 < n_kv; k0 += (int64_t) gridDim.x*WARPS*KPW) {

        float4 kr[KPW];
#pragma unroll
        for (int k = 0; k < KPW; ++k) {
            kr[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (k0 + k < n_kv) {
                const char * row = K + (int64_t) cells[k0 + k]*nbk2;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    kr[k] = ((const float4 *) row)[lane];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(row, &kr[k], lane*4);
                }
            }
        }

        for (int t = 0; t < n_batch; ++t) {
            float p[KPW*4];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float4 q_vec = sq[t][h][lane];
#pragma unroll
                for (int k = 0; k < KPW; ++k) {
                    float qk = 0.0f;
                    ggml_cuda_mad(qk, q_vec.x, kr[k].x);
                    ggml_cuda_mad(qk, q_vec.y, kr[k].y);
                    ggml_cuda_mad(qk, q_vec.z, kr[k].z);
                    ggml_cuda_mad(qk, q_vec.w, kr[k].w);
                    p[k*4 + h] = qk;
                }
            }

            const float s0 = qsa_reduce_scatter16(p, lane);
            const float s1 = __shfl_down_sync(0xffffffff, s0, 2);
            const float s2 = __shfl_down_sync(0xffffffff, s0, 4);
            const float s3 = __shfl_down_sync(0xffffffff, s0, 6);

            const int64_t b = k0 + lane/8;
            if (lane % 8 == 0 && b < n_kv) {
                const float sh[4] = { s0, s1, s2, s3 };
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    float sum = sh[h];

                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score += sum * w;
                }
                ((float *) ((char *) dst + t*nb1))[b] = score + 0.0f;
            }
        }
    }
}

#define QSA_SCORES_VEC_CASE(type_K)                                                        \
    if (k->type == (type_K)) {                                                             \
        qsa_scores_vec_kernel<type_K><<<grid, QSA_SV_THREADS, 0, stream>>>(                \
            q_d + t0*(nbq2/sizeof(float)), nbq1, nbq2, (int) nt, k_d, k->nb[2], cells_d,   \
            n_pool, head_scale, (float *) ((char *) dst_d + t0*nb1), nb1, ranges ? ranges + t0*stride : nullptr, stride, top);                 \
    } else

void ggml_cuda_qsa_range_scores(
        const float * q_d, size_t nbq1, size_t nbq2, int64_t n_batch,
        const ggml_tensor * k, const int32_t * cells_d, int64_t n_pool, float head_scale,
        float * dst_d, size_t nb1, cudaStream_t stream, const int32_t * ranges, int64_t stride, int top) {
    GGML_ASSERT(k->ne[0] == 128 && k->ne[1] == 1);
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type));
    GGML_ASSERT(nbq2 % sizeof(float) == 0);

    const char * k_d = (const char *) k->data;
    const int    id  = ggml_cuda_get_device();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if (n_batch >= LIGHTNING_INDEXER_TILE_TOKENS && volta_mma_available(ggml_cuda_info().devices[id].cc)) {
        constexpr int VQ = 16;
        static const bool reuse = true;
        const int iterations = reuse && (n_pool/512)*(n_batch/16) >= ggml_cuda_info().devices[id].nsm ? 8 : 4;
        const int KEYS_PER_BLOCK = iterations*4*16;

        dim3 block(128);
        dim3 grid((unsigned) ((n_pool + KEYS_PER_BLOCK - 1)/KEYS_PER_BLOCK),
                  (unsigned) ((n_batch + VQ - 1)/VQ), 1);
        const int smem = (int) sizeof(lightning_indexer_scores_volta_smem);

        static const bool disable_combine = false;
        const bool combine_heads = !disable_combine && n_batch >= VQ;
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_F16)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_BF16)
        LIGHTNING_INDEXER_SCORES_VOLTA_CASE(GGML_TYPE_F32)
        GGML_ABORT("fatal error");
        return;
    }
#endif

    const int64_t keys_per_block = QSA_SV_KPW*(QSA_SV_THREADS/WARP_SIZE);
    const int64_t n_groups = (n_pool + keys_per_block - 1)/keys_per_block;
    const dim3 grid((unsigned) std::max<int64_t>(1, std::min<int64_t>(n_groups, 8*(int64_t) ggml_cuda_info().devices[id].nsm)));

    for (int64_t t0 = 0; t0 < n_batch; t0 += QSA_SV_TOKENS) {
        const int64_t nt = std::min<int64_t>(QSA_SV_TOKENS, n_batch - t0);

        QSA_SCORES_VEC_CASE(GGML_TYPE_F16)
        QSA_SCORES_VEC_CASE(GGML_TYPE_Q4_0)
        QSA_SCORES_VEC_CASE(GGML_TYPE_Q4_1)
        QSA_SCORES_VEC_CASE(GGML_TYPE_Q5_0)
        QSA_SCORES_VEC_CASE(GGML_TYPE_Q5_1)
        QSA_SCORES_VEC_CASE(GGML_TYPE_Q8_0)
        QSA_SCORES_VEC_CASE(GGML_TYPE_BF16)
        QSA_SCORES_VEC_CASE(GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_lightning_indexer_scores(const float * q, size_t nbq1, size_t nbq2, int64_t n,
        const ggml_tensor * k, const int32_t * cells, int64_t pools, float scale, float * dst, size_t nb1, cudaStream_t stream) {
    ggml_cuda_qsa_range_scores(q, nbq1, nbq2, n, k, cells, pools, scale, dst, nb1, stream, nullptr, 0, 0);
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
        return false;
    }

    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
