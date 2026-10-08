// The Volta (sm_70) scoring path for QSA_SELECT is ported from Strata (https://github.com/Niko1221/strata,
// MIT License, see licenses/LICENSE-strata): scores_volta_kernel in src/kernels/cuda/qsa_select.cu.
#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// TODO add support for AMD cards via rocWMMA
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

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
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

    // phase 2 - load (and dequantize if needed) K to shared mem

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

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
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

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

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

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// ---------------------------------------------------------------------------------------------------------------------
// QSA_SELECT scoring on Volta tensor cores

// Ported from Strata (https://github.com/Niko1221/strata, MIT License, see licenses/LICENSE-strata): the
// scores_volta_kernel in src/kernels/cuda/qsa_select.cu. FP16 wmma m16n16k16 with both operands split into
// hi = fp16(x) and lo = fp16((x - hi)*2^11); Ah*Bh + Ah*Bl + Al*Bh keeps FP32-level accuracy. Volta's tensor cores
// truncate when accumulating, so Ah*Bh goes through a fresh fragment per 16-dim step and is added with FP32 adds.
// The Q8_0 keys are dequantized during the stage and split the same way. 16 tokens share each key read.
// Capability branch, not a fallback: only sm_70 runs this, other GPUs keep the SIMT tile kernel.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING
#define LIGHTNING_INDEXER_VOLTA_MMA 1
#else
#define LIGHTNING_INDEXER_VOLTA_MMA 0
#endif

struct lightning_indexer_scores_volta_smem {
    half  qh[4][16][136]; // queries hi, [head][token][dim], 136 halves per row for the wmma loads
    half  ql[4][16][136]; // queries lo
    half  kh[4][16][24];  // per warp: the current 16-dim key slice hi, [warp][key][dim]
    half  kl[4][16][24];  // the same slice lo
    float ep[4][16][20];  // per warp: the epilogue tile [key][token]
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

template <ggml_type TYPE_K>
static __global__ void __launch_bounds__(128) lightning_indexer_scores_volta_kernel(
        const float * __restrict__ Q, const char * __restrict__ K, const float * __restrict__ W, float * __restrict__ dst,
        const size_t nbq1, const size_t nbq2, const size_t nbk2, const size_t nb1,
        const int64_t n_batch, const int64_t n_kv) {
#if LIGHTNING_INDEXER_VOLTA_MMA
    constexpr int   VQ       = 16;   // tokens per block (one m16 tile)
    constexpr int   VITER    = 4;    // 16-key tiles per warp
    constexpr int   VQS      = 136;  // halves per query row in the shared queries
    constexpr int   VKS      = 24;   // halves per key row of a warp's 16-dim slice
    constexpr float LO_SCALE = 2048.0f;

    extern __shared__ __align__(16) unsigned char smraw[];
    lightning_indexer_scores_volta_smem & S = *reinterpret_cast<lightning_indexer_scores_volta_smem *>(smraw);

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int64_t q0 = (int64_t) blockIdx.y*VQ;

    // stage the queries of every head, split into hi/lo halves
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

    for (int it = 0; it < VITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x*VITER + it)*4 + warp)*16;
        if (b0 >= n_kv) {
            break; // warp-uniform
        }

        // this lane's share of the key slice: row lane/2, 8 dims at (lane%2)*8
        const int kr = lane >> 1;
        const int kc = (lane & 1)*8;
        const int64_t row = b0 + kr;
        const bool rv = row < n_kv;
        const char * kp = K + (rv ? row : 0)*nbk2;

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
            if (kk + 1 < 128/16 && rv) { // the next slice's loads behind this slice's products
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
                // hi*hi in a fresh fragment, added with FP32 adds (the tensor cores truncate when accumulating)
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

        // relu per head, weighted by the head, added in order; this lane holds tokens/keys at (i/16, i%16)
        float s_out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_out[j] = 0.0f;
        }
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
#pragma unroll
            for (int e = 0; e < sc.num_elements; ++e) {
                const float d = acc[h].x[e] + cor[h].x[e]*(1.0f/LO_SCALE);
                sc.x[e] = d > 0.0f ? d : 0.0f;
            }
            wmma::store_matrix_sync(&S.ep[warp][0][0], sc, 16 + 4, wmma::mem_row_major); // [key][token]
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = lane + 32*j;
                s_out[j] += S.ep[warp][i % 16][i / 16]*W[h];
            }
            __syncwarp();
        }

        // masked write [token][key]
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
#endif // LIGHTNING_INDEXER_VOLTA_MMA
}

#define LIGHTNING_INDEXER_SCORES_VOLTA_CASE(type_K)                                       \
    if (k->type == (type_K)) {                                                            \
        lightning_indexer_scores_volta_kernel<type_K><<<grid, block, smem, stream>>>(     \
            q_d, k_d, w_d, dst_d, nbq1, nbq2, k->nb[2], nb1, n_batch, n_kv);              \
    } else

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

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

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
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
        // dequantize remaining types to float
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

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

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

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
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

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

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

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
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

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[N_HEAD][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

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

    // phase 2 - stage the queries and weights of every head, tokens past n_batch are zero

#pragma unroll
    for (int i = tid; i < N_HEAD * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
        const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
        const int c4 = i % (N_EMBD / 4);

        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (start_batch + r < n_batch) {
            v = *(const float4 *) ((const char *) Q + h*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
        }

        q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
        q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
    }

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    __syncthreads();

    // phase 3 - float products of the widened keys for every head, ReLU, weight

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll 8
    for (int c = 0; c < N_EMBD_H2; ++c) {
        float2 k_val[KEYS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
        }
#pragma unroll
        for (int h = 0; h < N_HEAD; ++h) {
            const float2 q_val = q_shared[h][tl][c];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                qk[h][j] = fmaf(k_val[j].x, q_val.x, qk[h][j]);
                qk[h][j] = fmaf(k_val[j].y, q_val.y, qk[h][j]);
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

    // phase 4 - add the mask and write, consecutive threads write consecutive keys

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
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

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

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
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
            // use wmma kernel
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
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
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
            // use wmma kernel
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
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
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
#ifndef GGML_USE_MUSA
    // MUSA archs 21 and 22 cap static shared memory at 28 KB, below what the tile kernel stages
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
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
#endif // GGML_USE_MUSA
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, or MUSA, use vector kernel
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

// The unmasked scores of n_batch query tokens against every key of k, for QSA_SELECT: q points at
// the first token of the tile, w at one row of N_HEAD head weights shared by all tokens (nbw1 = 0).
#define LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel, type_K)                       \
    if (k->type == (type_K)) {                                                                \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 4, type_K, false>    \
            <<<grid, block, 0, stream>>>(                                                     \
            q_d, k_d, w_d, nullptr, dst_d,                                                    \
            1, n_batch, n_kv,                                                                 \
            nb1, nb1*n_batch, nb1*n_batch,                                                    \
            nbq1, nbq2, nbq2*n_batch,                                                         \
            k->nb[1], k->nb[2], k->nb[3],                                                     \
            0, 0, 0,                                                                          \
            0, 0, 0,                                                                          \
            1                                                                                 \
        );                                                                                    \
    } else

void ggml_cuda_lightning_indexer_scores(
        const float * q_d, size_t nbq1, size_t nbq2, int64_t n_batch,
        const ggml_tensor * k, const float * w_d, float * dst_d, size_t nb1, cudaStream_t stream) {
    GGML_ASSERT(k->ne[0] == 128 && k->ne[1] == 1);
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type));

    const char * k_d  = (const char *) k->data;
    const int    n_kv = k->ne[2];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // capability branch, not a fallback: the wmma scorer is written for Volta's m16n16k16, other GPUs run the
    // SIMT tile kernel for every batch size
    if (n_batch >= LIGHTNING_INDEXER_TILE_TOKENS &&
            volta_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        constexpr int VQ = 16;
        constexpr int KEYS_PER_BLOCK = 4*4*16;

        dim3 block(128);
        dim3 grid((unsigned) ((n_kv + KEYS_PER_BLOCK - 1)/KEYS_PER_BLOCK),
                  (unsigned) ((n_batch + VQ - 1)/VQ), 1);
        const int smem = (int) sizeof(lightning_indexer_scores_volta_smem);

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
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#ifndef GGML_USE_MUSA
    if (n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((n_kv + K_VECS_PER_BLOCK - 1)/K_VECS_PER_BLOCK,
                  (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1)/LIGHTNING_INDEXER_TILE_TOKENS, 1);

        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_F16)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_tile, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
        return;
    }
#endif // GGML_USE_MUSA
    {
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((n_kv + K_VECS_PER_BLOCK - 1)/K_VECS_PER_BLOCK, n_batch, 1);

        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_F16)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_SCORES_CASE(lightning_indexer_kernel_vec, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

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

    // alignment checks
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
