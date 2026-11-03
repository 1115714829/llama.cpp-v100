// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// SM70 (Volta) D256 Split-D prefill attention, host side: dispatch predicate,
// scratch layout, K/V dequant (F16 read in place, Q8_0 or Q4_0 mirrored to f16),
// Q staging, mask pre-scan, kernel launch and O scatter. Device kernel
// provenance: fattn-sm70-d256-kernel.cuh.



#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "ggml-backend-impl.h"
#include "fattn-common.cuh"
#include "fattn-sm70-d256.cuh"
#include "fattn-sm70-d256-kernel.cuh"

#include <climits>

// NOTE: no anonymous namespace in this file. The CuTe vendor headers
// (cute/atom/mma_traits_sm70.hpp) open their own anonymous namespaces; a second
// one in this TU makes cudafe's _GLOBAL__N__<hash> symbol mangling ambiguous.


#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634
#endif

constexpr int SM70_D256_BLOCK_M = 64;
constexpr int SM70_D256_D       = 256;






#define SM70_D256_KV_WINDOW 131072
static_assert(SM70_D256_KV_WINDOW % SM70_D256_BLOCK_M == 0, "window is not a multiple of the q block");

static_assert(SM70_D256_KV_WINDOW == GGML_BACKEND_META_ASSIST_KV_WINDOW, "assist window mismatch");









constexpr int SM70_D256_KV_SPLITS = 2;
constexpr int SM70_D256_SPLIT_MIN_BLOCKS = 32;




constexpr int SM70_D256_NSM = 80;





__global__ void sm70_d256_stage_q_kernel(
        const float2 * __restrict__ src, half2 * __restrict__ dst,
        const int q_len, const int hkv, const int gqa, const int q_pad,
        const int64_t src_row, const int64_t src_head, const int64_t src_batch) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;
    const int c  = blockIdx.z;
    if (r >= q_len) {
        return;
    }
    const int j = bj % hkv;
    const int b = bj / hkv;
    const int head_q = j * gqa + c;
    const int heads_q = hkv * gqa;
    const float2 v = src[threadIdx.x
        + (int64_t) r * src_row
        + (int64_t) head_q * src_head
        + (int64_t) b * src_batch];
    dst[threadIdx.x
        + (int64_t) b * heads_q * q_pad * (SM70_D256_D/2)
        + (int64_t) head_q * q_pad * (SM70_D256_D/2)
        + (int64_t) r * (SM70_D256_D/2)] = __float22half2_rn(v);
}








template <bool Normalize = false>
__global__ void sm70_d256_scatter_kernel(
        const float2 * __restrict__ src, float2 * __restrict__ dst,
        const float * __restrict__ acc_sum,
        const int heads_q, const int q_pad,
        const int64_t dst_row, const int64_t dst_head, const int64_t dst_batch) {
    const int r  = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / heads_q;
    const int head_q = bh % heads_q;
    float2 v = src[threadIdx.x
        + (int64_t) b * q_pad * heads_q * (SM70_D256_D/2)
        + (int64_t) r * heads_q * (SM70_D256_D/2)
        + (int64_t) head_q * (SM70_D256_D/2)];
    if constexpr (Normalize) {
        const int64_t row = (int64_t) b * q_pad * heads_q
            + (int64_t) r * heads_q + head_q;
        const float sum = acc_sum[row];
        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        if (sum > 0.0f) {
            v.x *= inv_sum;
            v.y *= inv_sum;
        } else {
            v = make_float2(0.0f, 0.0f);
        }
    }
    dst[threadIdx.x
        + (int64_t) r * dst_row
        + (int64_t) head_q * dst_head
        + (int64_t) b * dst_batch] = v;
}













static constexpr int SM70_D256_MASK_BLOCK_N = 32;
static constexpr int SM70_D256_MASK_SPLIT   = 8192;
static_assert(SM70_D256_MASK_BLOCK_N == FLASH_NAMESPACE::Sm70D256SplitDTraits::kBlockN,
              "mask bounds are in units of the kernel's KV block size");

__global__ void sm70_d256_mask_bounds_kernel(
        const __half * __restrict__ mask, int2 * __restrict__ bounds,
        const int q_len, const int kv_len,
        const int64_t mask_row_stride, const int64_t mask_batch_stride,
        const bool vec_ok) {
    const int q_block = blockIdx.x;
    const int batch   = blockIdx.y;
    const int split   = blockIdx.z;
    const int row_begin = q_block * SM70_D256_BLOCK_M;
    const int row_end = (row_begin + SM70_D256_BLOCK_M) < q_len
        ? (row_begin + SM70_D256_BLOCK_M) : q_len;
    const int col_begin = split * SM70_D256_MASK_SPLIT;
    const int col_end = (col_begin + SM70_D256_MASK_SPLIT) < kv_len
        ? (col_begin + SM70_D256_MASK_SPLIT) : kv_len;

    const __half * batch_mask = mask + batch * mask_batch_stride;
    int first_nz = kv_len;
    int last_fin = -1;
    for (int row = row_begin; row < row_end; ++row) {
        const __half * mask_row = batch_mask + (int64_t) row * mask_row_stride;




#pragma unroll
        for (int j = 0; j < SM70_D256_MASK_SPLIT / (8 * 256); ++j) {
            const int col = col_begin + 8 * threadIdx.x + j * (8 * 256);
            if (col >= col_end) {
                continue;
            }
            if (vec_ok && col + 8 <= col_end) {
                const uint4 raw = *reinterpret_cast<const uint4 *>(mask_row + col);
                const __half * h = reinterpret_cast<const __half *>(&raw);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = __half2float(h[i]);
                    if (v != 0.0f && col + i < first_nz) {
                        first_nz = col + i;
                    }
                    if (v > -INFINITY && col + i > last_fin) {
                        last_fin = col + i;
                    }
                }
            } else {
                const int col_last = col + 8 < col_end ? col + 8 : col_end;
                for (int c = col; c < col_last; ++c) {
                    const float v = __half2float(mask_row[c]);
                    if (v != 0.0f && c < first_nz) {
                        first_nz = c;
                    }
                    if (v > -INFINITY && c > last_fin) {
                        last_fin = c;
                    }
                }
            }
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        const int other_first = __shfl_down_sync(0xffffffffu, first_nz, off);
        const int other_last = __shfl_down_sync(0xffffffffu, last_fin, off);
        if (other_first < first_nz) {
            first_nz = other_first;
        }
        if (other_last > last_fin) {
            last_fin = other_last;
        }
    }
    __shared__ int smem_first[8];
    __shared__ int smem_last[8];
    if ((threadIdx.x & 31) == 0) {
        smem_first[threadIdx.x / 32] = first_nz;
        smem_last[threadIdx.x / 32] = last_fin;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) {
            if (smem_first[w] < first_nz) {
                first_nz = smem_first[w];
            }
            if (smem_last[w] > last_fin) {
                last_fin = smem_last[w];
            }
        }
        int2 & b = bounds[(int64_t) batch * gridDim.x + q_block];


        if (first_nz < kv_len) {
            atomicMax(&b.x, kv_len - first_nz);
        }
        if (last_fin >= 0) {
            atomicMax(&b.y, last_fin + 1);
        }
    }
}




__global__ void sm70_d256_range_bounds_kernel(
        const int4 * __restrict__ range, int2 * __restrict__ bounds,
        int * __restrict__ kv_limit,
        const int q_len, const int kv_len,
        const int64_t range_row_stride, const int64_t range_batch_stride) {
    const int q_block = blockIdx.x;
    const int batch   = blockIdx.y;
    const int row     = q_block * SM70_D256_BLOCK_M + threadIdx.x;


    int first_nz = kv_len;
    int last_fin = -1;
    if (row < q_len) {
        const int4 r = range[(int64_t) batch * range_batch_stride + (int64_t) row * range_row_stride];
        const int lo0 = r.x < 0 ? 0 : (r.x > kv_len ? kv_len : r.x);
        const int hi0 = r.y < 0 ? 0 : (r.y > kv_len ? kv_len : r.y);
        const int lo1 = r.z < 0 ? 0 : (r.z > kv_len ? kv_len : r.z);
        const int hi1 = r.w < 0 ? 0 : (r.w > kv_len ? kv_len : r.w);



        const bool seg0 = hi0 > lo0;
        const bool seg1 = hi1 > lo1;
        int cover_end = 0;
        bool covered0 = false;
        if (seg0 && lo0 == 0) { cover_end = hi0; covered0 = true; }
        if (seg1 && lo1 == 0) { if (hi1 > cover_end) { cover_end = hi1; } covered0 = true; }
        if (seg0 && covered0 && lo0 <= cover_end && hi0 > cover_end) { cover_end = hi0; }
        if (seg1 && covered0 && lo1 <= cover_end && hi1 > cover_end) { cover_end = hi1; }
        first_nz = covered0 && cover_end < kv_len ? cover_end : (covered0 ? kv_len : 0);

        int hi_max = 0;
        if (seg0) { hi_max = hi0; }
        if (seg1 && hi1 > hi_max) { hi_max = hi1; }
        last_fin = hi_max - 1;
    }

    for (int off = 16; off > 0; off >>= 1) {
        const int other_first = __shfl_down_sync(0xffffffffu, first_nz, off);
        const int other_last = __shfl_down_sync(0xffffffffu, last_fin, off);
        if (other_first < first_nz) {
            first_nz = other_first;
        }
        if (other_last > last_fin) {
            last_fin = other_last;
        }
    }
    __shared__ int smem_first[SM70_D256_BLOCK_M / 32];
    __shared__ int smem_last[SM70_D256_BLOCK_M / 32];
    if ((threadIdx.x & 31) == 0) {
        smem_first[threadIdx.x / 32] = first_nz;
        smem_last[threadIdx.x / 32] = last_fin;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < SM70_D256_BLOCK_M / 32; ++w) {
            if (smem_first[w] < first_nz) {
                first_nz = smem_first[w];
            }
            if (smem_last[w] > last_fin) {
                last_fin = smem_last[w];
            }
        }

        bounds[(int64_t) batch * gridDim.x + q_block] = make_int2(kv_len - first_nz, last_fin + 1);



        const int rows = ((last_fin + 1 + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N) * SM70_D256_MASK_BLOCK_N;
        atomicMax(kv_limit, rows < kv_len ? rows : kv_len);
    }
}



















#define SM70_D256_ASSIST_ROW_COST_NUM 9
#define SM70_D256_ASSIST_ROW_COST_DEN 7
__global__ void sm70_d256_assist_split_kernel(int * __restrict__ kv_limit, ggml_backend_meta_assist_split * __restrict__ split,
        const int idle_load) {
    if (threadIdx.x != 0) {
        return;
    }
    const int kv_end = *kv_limit;
    const int64_t num = (int64_t) idle_load * SM70_D256_ASSIST_ROW_COST_NUM;
    const int64_t den = SM70_D256_ASSIST_ROW_COST_DEN + num;
    const int s = kv_end < GGML_BACKEND_META_ASSIST_MIN_KV
        ? kv_end
        : (int) (((int64_t) kv_end * num / den) & ~(int64_t) (SM70_D256_BLOCK_M - 1));
    split->kv_end = kv_end;
    split->s = s;
    *kv_limit = s;
}



















struct sm70_d256_scratch {
    size_t total;
    size_t qs_offset;
    size_t os_offset;
    size_t bounds_offset;
    size_t kv_limit_offset;
    size_t p_out_offset;
    size_t acc_max_offset;
    size_t acc_sum_offset;
    size_t p_max_offset;
    size_t p_sum_offset;
    size_t n_q;
    size_t rows;
    int q_pad;
    int n_q_blocks;
    bool windowed;
    bool partial;
    bool need_f16_k;
    bool need_f16_v;
    bool v_is_k_view;
    ggml_cuda_flash_attn_ext_f16_extra_data f16_extra;
};

static sm70_d256_scratch sm70_d256_get_scratch(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool v_is_k_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const bool need_f16_k = K->type != GGML_TYPE_F16;

    const bool need_f16_v = !(v_is_k_view && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16;


    const bool windowed = mask != nullptr && mask->type == GGML_TYPE_I32
        && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0)
        && K->ne[1] > SM70_D256_KV_WINDOW;




    const int q_pad = (int) GGML_PAD(Q->ne[1], SM70_D256_BLOCK_M);
    const int64_t kv_len = mask == nullptr ? 0
        : (mask->type == GGML_TYPE_I32 ? K->ne[1] : mask->ne[0]);
    const int64_t launch_ctas =
        (int64_t) (q_pad / SM70_D256_BLOCK_M) * Q->ne[3] * Q->ne[2];
    const int64_t kv_blocks =
        (kv_len + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N;
    const bool split_dense = !windowed
        && launch_ctas < 4 * SM70_D256_NSM
        && kv_blocks >= 2 * SM70_D256_SPLIT_MIN_BLOCKS;
    const bool partial = windowed || split_dense;

    sm70_d256_scratch s = {};
    s.windowed = windowed;
    s.partial = partial;
    s.need_f16_k = need_f16_k;
    s.need_f16_v = need_f16_v;
    s.v_is_k_view = v_is_k_view;

    const char * base = (const char *) dst->data + ggml_nbytes(dst);

    if (windowed) {



        uintptr_t end = (uintptr_t) base;
        if (need_f16_k) {
            end = GGML_PAD(end, 128);
            s.f16_extra.K = end;
            end += (size_t) K->ne[0] * SM70_D256_KV_WINDOW * K->ne[2] * K->ne[3] * sizeof(half);
        }
        if (need_f16_v) {
            if (v_is_k_view) {
                s.f16_extra.V = s.f16_extra.K;
            } else {
                end = GGML_PAD(end, 128);
                s.f16_extra.V = end;
                end += (size_t) V->ne[0] * SM70_D256_KV_WINDOW * V->ne[2] * V->ne[3] * sizeof(half);
            }
        }
        s.f16_extra.end = end;
    } else {
        s.f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_k, need_f16_v);
    }

    const size_t f16_bytes = GGML_PAD((size_t) (s.f16_extra.end - (uintptr_t) base), 128);
    s.q_pad = q_pad;
    s.n_q = (size_t) Q->ne[2] * s.q_pad * SM70_D256_D * Q->ne[3];
    s.rows = s.n_q / SM70_D256_D;
    s.qs_offset = f16_bytes;
    s.os_offset = s.qs_offset + s.n_q * sizeof(half);
    s.n_q_blocks = s.q_pad / SM70_D256_BLOCK_M;
    s.bounds_offset = GGML_PAD(s.os_offset + s.n_q * sizeof(float), 128);
    s.kv_limit_offset = GGML_PAD(s.bounds_offset + (size_t) Q->ne[3] * s.n_q_blocks * sizeof(int2), 128);
    if (partial) {
        s.p_out_offset   = GGML_PAD(s.kv_limit_offset + sizeof(int), 128);
        s.p_max_offset   = GGML_PAD(s.p_out_offset + SM70_D256_KV_SPLITS * s.n_q * sizeof(float), 128);
        s.p_sum_offset   = GGML_PAD(s.p_max_offset + SM70_D256_KV_SPLITS * s.rows * sizeof(float), 128);
        s.acc_max_offset = GGML_PAD(s.p_sum_offset + SM70_D256_KV_SPLITS * s.rows * sizeof(float), 128);
        s.acc_sum_offset = GGML_PAD(s.acc_max_offset + s.rows * sizeof(float), 128);
        s.total = ggml_nbytes(dst) + GGML_PAD(s.acc_sum_offset + s.rows * sizeof(float), 128);
    } else {
        s.total = ggml_nbytes(dst) + s.kv_limit_offset + sizeof(int);
    }
    return s;
}







static void sm70_d256_dequant_kv_tensor(
        const ggml_tensor * t, half * dst, cudaStream_t stream) {
    const size_t ts = ggml_type_size(t->type);
    const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
    to_fp16(t->data, dst, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[1] / ts, t->nb[2] / ts, t->nb[3] / ts, stream);
}



template <ggml_type type>
struct sm70_d256_dequant_block;

template <>
struct sm70_d256_dequant_block<GGML_TYPE_Q8_0> {
    static constexpr int kBlockBytes = sizeof(block_q8_0);


    static __device__ __forceinline__ void decode(const char * block, const int iq, const float d, half2 * h) {
        const int8_t * qs = reinterpret_cast<const int8_t *>(block) + sizeof(__half);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            h[i] = __floats2half2_rn(d * qs[iq + 2*i], d * qs[iq + 2*i + 1]);
        }
    }
};

template <>
struct sm70_d256_dequant_block<GGML_TYPE_Q4_0> {
    static constexpr int kBlockBytes = sizeof(block_q4_0);

    static __device__ __forceinline__ __half2 pair(const uint32_t packed, const int shift, const float d) {
        return __floats2half2_rn(d * ((int) ((packed >> shift) & 0xf) - 8),
                                 d * ((int) ((packed >> (8 + shift)) & 0xf) - 8));
    }


    static __device__ __forceinline__ void decode(const char * block, const int iq, const float d, half2 * h) {


        const char * codes = block + sizeof(__half) + (iq & 15);
        const uint32_t w0 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes)
                          | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 2) << 16;
        const uint32_t w1 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 4)
                          | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 6) << 16;
        const int shift = iq >= 16 ? 4 : 0;
        h[0] = pair(w0,      shift, d);
        h[1] = pair(w0 >> 16, shift, d);
        h[2] = pair(w1,      shift, d);
        h[3] = pair(w1 >> 16, shift, d);
    }
};








template <ggml_type type>
static __global__ void sm70_d256_dequant_rows(
        const char * __restrict__ src, half * __restrict__ dst,
        const int ne0, const int ne2,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int * __restrict__ kv_lo,
        const int * __restrict__ kv_limit,
        const int row_lo, const int win_rows) {
    static_assert(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0, "unsupported KV type");

    const int i1_local = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i1_local >= win_rows) {
        return;
    }
    const int i1 = row_lo + i1_local;
    if (i1 >= *kv_limit || (kv_lo != nullptr && i1 < *kv_lo)) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;


    const int ib = lane / 4;
    const int iq = (lane % 4) * 8;

    const char * src_row = src + (int64_t) i3*nb3 + (int64_t) i2*nb2 + (int64_t) i1*nb1;
    half * dst_row = dst + (((int64_t) i3*ne2 + i2)*win_rows + i1_local)*ne0;

    const char * src_block = src_row + ib * sm70_d256_dequant_block<type>::kBlockBytes;
    const float d = __half2float(*reinterpret_cast<const __half *>(src_block));
    half2 h[4];
    sm70_d256_dequant_block<type>::decode(src_block, iq, d, h);
    uint4 out;
    memcpy(&out.x, &h[0], sizeof(uint32_t));
    memcpy(&out.y, &h[1], sizeof(uint32_t));
    memcpy(&out.z, &h[2], sizeof(uint32_t));
    memcpy(&out.w, &h[3], sizeof(uint32_t));
    *reinterpret_cast<uint4 *>(dst_row + ib*32 + iq) = out;
}




static void sm70_d256_dequant_kv_window(
        const ggml_tensor * t, half * dst, const int * kv_limit,
        const int row_lo, const int win_rows, cudaStream_t stream) {
    GGML_ASSERT(t->ne[0] == SM70_D256_D);
    GGML_ASSERT(row_lo >= 0 && row_lo % 8 == 0 && win_rows > 0);
    const dim3 grid((unsigned) ((win_rows + 7) / 8), (unsigned) t->ne[2], (unsigned) t->ne[3]);
    if (t->type == GGML_TYPE_Q8_0) {
        sm70_d256_dequant_rows<GGML_TYPE_Q8_0><<<grid, 256, 0, stream>>>(
            (const char *) t->data, dst,
            (int) t->ne[0], (int) t->ne[2],
            t->nb[1], t->nb[2], t->nb[3], nullptr, kv_limit, row_lo, win_rows);
    } else {
        sm70_d256_dequant_rows<GGML_TYPE_Q4_0><<<grid, 256, 0, stream>>>(
            (const char *) t->data, dst,
            (int) t->ne[0], (int) t->ne[2],
            t->nb[1], t->nb[2], t->nb[3], nullptr, kv_limit, row_lo, win_rows);
    }
    CUDA_CHECK(cudaGetLastError());
}


static void sm70_d256_dequant_kv_partial(
        const ggml_tensor * t, half * dst, const int * kv_limit, cudaStream_t stream) {
    sm70_d256_dequant_kv_window(t, dst, kv_limit, 0, (int) t->ne[1], stream);
}



static bool sm70_d256_kv_type_ok(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        return t->nb[0] == sizeof(half) && t->nb[1] % 16 == 0 &&
               t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    }
    if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        const size_t ts = ggml_type_size(t->type);
        return t->nb[0] == ts && t->nb[1] % ts == 0 &&
               t->nb[2] % ts == 0 && t->nb[3] % ts == 0;
    }
    return false;
}

bool ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA || !volta_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != SM70_D256_D || K->ne[0] != SM70_D256_D || V->ne[0] != SM70_D256_D) {
        return false;
    }

    if (Q->ne[1] < 17) {
        return false;
    }

    if (K->ne[2] == 0 || Q->ne[2] == 0) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || K->ne[2] != V->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }


    const bool mask_range = mask && mask->type == GGML_TYPE_I32;
    if (!mask || (mask->type != GGML_TYPE_F16 && !mask_range) || mask->ne[2] != 1) {
        return false;
    }
    if (mask_range) {

        if (mask->ne[0] != 4 || mask->nb[0] != sizeof(int32_t) || mask->nb[1] != sizeof(int4) || (uintptr_t) mask->data % sizeof(int4) != 0) {
            return false;
        }
    } else if (mask->nb[0] != sizeof(half)) {
        return false;
    }
    if (mask->ne[1] != Q->ne[1]) {
        return false;
    }
    if (mask->ne[3] != Q->ne[3] && mask->ne[3] != 1) {
        return false;
    }

    const int64_t kv_len = mask_range ? K->ne[1] : mask->ne[0];
    if (kv_len < Q->ne[1] || kv_len > K->ne[1] || kv_len > V->ne[1]) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) {
        return false;
    }
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    if (!(scale > 0.0f)) {
        return false;
    }




    if (Q->nb[0] != sizeof(float) ||
        Q->nb[1] % sizeof(float2) != 0 || Q->nb[2] % sizeof(float2) != 0 ||
        Q->nb[3] % sizeof(float2) != 0 || (uintptr_t) Q->data % 8 != 0) {
        return false;
    }
    if (!sm70_d256_kv_type_ok(K) || !sm70_d256_kv_type_ok(V)) {
        return false;
    }
    return true;
}














template <bool Partial>
static void sm70_d256_launch_dense(
        const void * q, const void * k, const void * v, void * out,
        const void * mask, const int2 * mask_bounds,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_outer_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_outer_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t mask_row_stride, int64_t mask_batch_stride, bool mask_is_range,
        int q_pad, int kv_len, int heads_q, int heads_kv, int batch, int kv_offset,
        float softmax_scale, float mask_scale,
        int win_block_lo, int win_block_hi, int kv_splits,
        float * partial_out, float * partial_max, float * partial_sum,
        const int * visit_lo, const int * visit_hi,
        cudaStream_t stream) {
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    using El = cutlass::half_t;

    GGML_ASSERT(q_pad % Traits::kBlockM == 0);
    GGML_ASSERT(kv_len - kv_offset >= 1 && kv_len - kv_offset <= q_pad);
    if constexpr (Partial) {
        GGML_ASSERT(win_block_lo % Traits::kBlockN == 0);
        GGML_ASSERT(kv_splits >= 1 && kv_splits <= SM70_D256_KV_SPLITS);
        GGML_ASSERT(partial_out != nullptr && partial_max != nullptr && partial_sum != nullptr);
    } else {
        GGML_ASSERT(kv_splits == 1);
    }

    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / Traits::kBlockM, batch * kv_splits, heads_q);



    if (kv_splits > 1) {
        auto kernel_split = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Traits, El, float, false, Partial, SM70_D256_KV_SPLITS>;
        CUDA_SET_SHARED_MEMORY_LIMIT((const void *) kernel_split, Traits::kSmemBytes);
        kernel_split<<<grid, block, Traits::kSmemBytes, stream>>>(
            (const El *) q, (const El *) k, (const El *) v, (float *) out,
            (const __half *) mask,
            (int) q_batch_stride, (int) q_row_stride, (int) q_head_stride,
            (int) k_outer_stride, (int) k_row_stride, (int) k_head_stride,
            (int) v_outer_stride, (int) v_row_stride, (int) v_head_stride,
            mask_row_stride, mask_batch_stride, mask_is_range, (const int2 *) mask_bounds,
            q_pad, kv_len, heads_q, heads_kv, kv_offset,
            softmax_scale * float(M_LOG2E), mask_scale,
            partial_out, partial_max, partial_sum,
            win_block_lo, win_block_hi,
            visit_lo, visit_hi);
    } else {
        auto kernel_plain = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Traits, El, float, false, Partial, 1>;
        CUDA_SET_SHARED_MEMORY_LIMIT((const void *) kernel_plain, Traits::kSmemBytes);
        kernel_plain<<<grid, block, Traits::kSmemBytes, stream>>>(
            (const El *) q, (const El *) k, (const El *) v, (float *) out,
            (const __half *) mask,
            (int) q_batch_stride, (int) q_row_stride, (int) q_head_stride,
            (int) k_outer_stride, (int) k_row_stride, (int) k_head_stride,
            (int) v_outer_stride, (int) v_row_stride, (int) v_head_stride,
            mask_row_stride, mask_batch_stride, mask_is_range, (const int2 *) mask_bounds,
            q_pad, kv_len, heads_q, heads_kv, kv_offset,
            softmax_scale * float(M_LOG2E), mask_scale,
            partial_out, partial_max, partial_sum,
            win_block_lo, win_block_hi,
            visit_lo, visit_hi);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_sm70_d256_launch_raw(
        const void * q, const void * k, const void * v, void * out,
        const void * mask, const int2 * mask_bounds,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_outer_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_outer_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t mask_row_stride, int64_t mask_batch_stride, bool mask_is_range,
        int q_pad, int kv_len, int heads_q, int heads_kv, int batch, int kv_offset,
        float softmax_scale, float mask_scale, cudaStream_t stream) {
    sm70_d256_launch_dense<false>(
        q, k, v, out, mask, mask_bounds,
        q_batch_stride, q_row_stride, q_head_stride,
        k_outer_stride, k_row_stride, k_head_stride,
        v_outer_stride, v_row_stride, v_head_stride,
        mask_row_stride, mask_batch_stride, mask_is_range,
        q_pad, kv_len, heads_q, heads_kv, batch, kv_offset,
        softmax_scale, mask_scale,
        0, INT_MAX, 1,
        nullptr, nullptr, nullptr,
        nullptr, nullptr,
        stream);
}

// ---------------------------------------------------------------------------
// Meta assist: cross-device epochs, idle-rank workspace and idle-rank run
// ---------------------------------------------------------------------------




static __global__ void sm70_d256_assist_signal(uint32_t * epoch) {
    __threadfence_system();
    atomicAdd(epoch, 1);
}








static __global__ void sm70_d256_assist_wait(volatile uint32_t * epoch, uint32_t * gen, uint32_t bump, uint32_t lag) {
    if (threadIdx.x != 0) {
        return;
    }
    __threadfence_system();
    const uint32_t g = *gen + bump;
    *gen = g;
    const uint32_t target = g - lag;
    while ((int32_t) (*epoch - target) < 0) {
    }
    __threadfence_system();
}

static void sm70_d256_assist_signal_launch(uint32_t * epoch, cudaStream_t stream) {
    sm70_d256_assist_signal<<<1, 1, 0, stream>>>(epoch);
    CUDA_CHECK(cudaGetLastError());
}

static void sm70_d256_assist_wait_launch(volatile uint32_t * epoch, uint32_t * gen, uint32_t bump, uint32_t lag, cudaStream_t stream) {
    sm70_d256_assist_wait<<<1, 1, 0, stream>>>(epoch, gen, bump, lag);
    CUDA_CHECK(cudaGetLastError());
}



__global__ void sm70_d256_assist_read_split(volatile const ggml_backend_meta_assist_split * __restrict__ src,
        ggml_backend_meta_assist_split * __restrict__ dst) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        const int32_t kv_end = src->kv_end;
        const int32_t split  = src->s;
        __threadfence_system();
        dst->kv_end = kv_end;
        dst->s      = split;
    }
}





__global__ void sm70_d256_fetch_rows(
        const char * __restrict__ src, char * __restrict__ dst,
        const ggml_backend_meta_assist_split * __restrict__ split,
        const int win_lo, const int win_rows, const int row_bytes) {
    const int chunks = row_bytes / 16;
    const int row_lo = win_lo > split->s ? win_lo : split->s;
    const int win_hi = win_lo + win_rows;
    const int row_hi = win_hi < split->kv_end ? win_hi : split->kv_end;
    if (row_lo >= row_hi) {
        return;
    }
    const int n = (row_hi - row_lo) * chunks;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        const int lrow = row_lo - win_lo + c / chunks;
        const int coff = (c % chunks) * 16;
        const uint4 v = *reinterpret_cast<const uint4 *>(src + (int64_t) (win_lo + lrow) * row_bytes + coff);
        *reinterpret_cast<uint4 *>(dst + (int64_t) lrow * row_bytes + coff) = v;
    }
}



__global__ void sm70_d256_assist_clear_partial(
        float * __restrict__ out, float * __restrict__ pmax, float * __restrict__ psum,
        const int64_t rows) {
    const int64_t n = rows * SM70_D256_D;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        out[i] = 0.0f;
    }
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += (int64_t) gridDim.x * blockDim.x) {
        pmax[i] = -INFINITY;
        psum[i] = 0.0f;
    }
}







struct sm70_d256_assist_workspace {
    size_t total;
    size_t stage_k;
    size_t stage_v;
    size_t mirror_k;
    size_t mirror_v;
    size_t qs;
    size_t bounds;
    size_t kv_limit;
    size_t split;
    size_t win_out;
    size_t win_max;
    size_t win_sum;
    int64_t rows;
    int q_pad;
    int q_blocks;
};

static sm70_d256_assist_workspace sm70_d256_assist_get_workspace(int q_pad, int heads_q, int kv_type) {
    sm70_d256_assist_workspace w = {};
    const size_t row_bytes = ggml_row_size((enum ggml_type) kv_type, SM70_D256_D);
    const size_t staging = (size_t) SM70_D256_KV_WINDOW * row_bytes;
    const size_t mirror = (size_t) SM70_D256_KV_WINDOW * SM70_D256_D * sizeof(half);
    w.rows = (int64_t) q_pad * heads_q;
    w.q_pad = q_pad;
    w.q_blocks = q_pad / SM70_D256_BLOCK_M;

    size_t off = 0;
    off = GGML_PAD(off, 128); w.stage_k  = off; off += staging;
    off = GGML_PAD(off, 128); w.stage_v  = off; off += staging;
    off = GGML_PAD(off, 128); w.mirror_k = off; off += mirror;
    off = GGML_PAD(off, 128); w.mirror_v = off; off += mirror;
    off = GGML_PAD(off, 128); w.qs       = off; off += (size_t) w.rows * SM70_D256_D * sizeof(half);
    off = GGML_PAD(off, 128); w.bounds   = off; off += (size_t) w.q_blocks * sizeof(int2);
    off = GGML_PAD(off, 128); w.kv_limit = off; off += sizeof(int);
    off = GGML_PAD(off, 128); w.split    = off; off += sizeof(ggml_backend_meta_assist_split);
    off = GGML_PAD(off, 128); w.win_out  = off; off += (size_t) w.rows * SM70_D256_D * sizeof(float);
    off = GGML_PAD(off, 128); w.win_max  = off; off += (size_t) w.rows * sizeof(float);
    off = GGML_PAD(off, 128); w.win_sum  = off; off += (size_t) w.rows * sizeof(float);
    w.total = GGML_PAD(off, 128);
    return w;
}

size_t ggml_cuda_sm70_d256_assist_workspace_size(int q_pad, int heads_q) {

    return sm70_d256_assist_get_workspace(q_pad, heads_q, GGML_TYPE_Q8_0).total;
}




template <ggml_type type>
static void sm70_d256_dequant_staging_window(
        const char * src, half * dst, const int * kv_lo, const int * kv_limit,
        const int win_lo, const int win_rows, const int64_t row_bytes, cudaStream_t stream) {
    const dim3 grid((unsigned) ((win_rows + 7) / 8), 1, 1);
    sm70_d256_dequant_rows<type><<<grid, 256, 0, stream>>>(
        src, dst, SM70_D256_D, 1, row_bytes, 0, 0, kv_lo, kv_limit, win_lo, win_rows);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_sm70_d256_assist_run(void * backend, const ggml_backend_meta_assist_rank * d) {
    if (backend == nullptr || d == nullptr || d->role != 1 || d->work == nullptr) {
        return false;
    }
    if (d->n_owners < 1 || d->n_owners > 2 || d->batch != 1 || d->heads_kv != 1 || d->mask == nullptr || !d->mask_is_range) {
        return false;
    }
    if (d->q_len < GGML_BACKEND_META_ASSIST_MIN_Q || d->kv_view < GGML_BACKEND_META_ASSIST_MIN_KV) {
        return false;
    }
    if (d->kv_window != SM70_D256_KV_WINDOW) {
        return false;
    }
    if (d->q_pad <= 0 || d->q_pad % SM70_D256_BLOCK_M != 0 || d->heads_q <= 0) {
        return false;
    }
    if (d->kv_type != GGML_TYPE_Q8_0 && d->kv_type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (d->split_local == nullptr) {
        return false;
    }
    ggml_backend_t be = (ggml_backend_t) backend;
    if (!ggml_backend_is_cuda(be)) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) be->context;
    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    const sm70_d256_assist_workspace w = sm70_d256_assist_get_workspace(d->q_pad, d->heads_q, d->kv_type);
    GGML_ASSERT(d->work_bytes >= w.total);
    GGML_ASSERT((uintptr_t) d->work % 128 == 0);

    char * const work = (char *) d->work;
    char * const stage_k  = work + w.stage_k;
    char * const stage_v  = work + w.stage_v;
    half * const mirror_k = (half *) (work + w.mirror_k);
    half * const mirror_v = (half *) (work + w.mirror_v);
    half * const qs       = (half *) (work + w.qs);
    int2 * const bounds   = (int2 *) (work + w.bounds);
    int  * const kv_limit = (int *) (work + w.kv_limit);
    ggml_backend_meta_assist_split * const split = (ggml_backend_meta_assist_split *) (work + w.split);
    float * const win_out = (float *) (work + w.win_out);
    float * const win_max = (float *) (work + w.win_max);
    float * const win_sum = (float *) (work + w.win_sum);

    const int64_t row_bytes = ggml_row_size((enum ggml_type) d->kv_type, SM70_D256_D);
    const int64_t qs_bytes = (int64_t) d->heads_q * d->q_pad * SM70_D256_D * sizeof(half);
    const int n_win = (int) ((d->kv_view + SM70_D256_KV_WINDOW - 1) / SM70_D256_KV_WINDOW);
    const unsigned n_merge = (unsigned) ((w.rows
        + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
        / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
    const unsigned fetch_blocks = (unsigned) (((int64_t) SM70_D256_KV_WINDOW * (row_bytes / 16) + 255) / 256);
    const unsigned clear_blocks = (unsigned) std::min<int64_t>((w.rows * SM70_D256_D + 255) / 256, 65535);




    for (int j = 0; j < d->n_owners; ++j) {
        GGML_ASSERT(d->k_src[j] != nullptr && d->v_src[j] != nullptr && d->qs_src[j] != nullptr);
        GGML_ASSERT(d->p_out_dst[j] != nullptr && d->p_max_dst[j] != nullptr && d->p_sum_dst[j] != nullptr);
        GGML_ASSERT(d->epoch_kv[j] != nullptr && d->epoch_partial[j] != nullptr && d->epoch_free[j] != nullptr);
        GGML_ASSERT(d->gen[j] != nullptr && d->free_lag[j] <= 1 && d->split_src[j] != nullptr);


        sm70_d256_assist_wait_launch(d->epoch_kv[j], d->gen[j], 1, 0, stream);
        sm70_d256_assist_read_split<<<1, 32, 0, stream>>>(d->split_src[j], split);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyPeerAsync(qs, ctx->device, d->qs_src[j], d->owner_dev[j], (size_t) qs_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));
        {
            const dim3 grid(w.q_blocks, d->batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int4 *) d->mask, bounds, kv_limit,
                d->q_len, d->kv_view, d->mask_row_stride, d->mask_batch_stride);
            CUDA_CHECK(cudaGetLastError());
        }





        sm70_d256_assist_wait_launch(d->epoch_free[j], d->gen[j], 0, d->free_lag[j], stream);
        sm70_d256_assist_clear_partial<<<clear_blocks, 256, 0, stream>>>(
            d->p_out_dst[j], d->p_max_dst[j], d->p_sum_dst[j], w.rows);
        CUDA_CHECK(cudaGetLastError());

        for (int wi = 0; wi < n_win; ++wi) {
            const int win_lo = wi * SM70_D256_KV_WINDOW;
            const int rows_left = d->kv_view - win_lo;
            const int win_rows = rows_left < SM70_D256_KV_WINDOW ? rows_left : SM70_D256_KV_WINDOW;


            sm70_d256_fetch_rows<<<fetch_blocks, 256, 0, stream>>>(
                (const char *) d->k_src[j], stage_k, split, win_lo, win_rows, (int) row_bytes);
            sm70_d256_fetch_rows<<<fetch_blocks, 256, 0, stream>>>(
                (const char *) d->v_src[j], stage_v, split, win_lo, win_rows, (int) row_bytes);
            CUDA_CHECK(cudaGetLastError());

            const int * const kv_lo = &split->s;
            const int * const kv_hi = &split->kv_end;
            if (d->kv_type == GGML_TYPE_Q8_0) {
                sm70_d256_dequant_staging_window<GGML_TYPE_Q8_0>(
                    stage_k - win_lo * row_bytes, mirror_k, kv_lo, kv_hi, win_lo, win_rows, row_bytes, stream);
                sm70_d256_dequant_staging_window<GGML_TYPE_Q8_0>(
                    stage_v - win_lo * row_bytes, mirror_v, kv_lo, kv_hi, win_lo, win_rows, row_bytes, stream);
            } else {
                sm70_d256_dequant_staging_window<GGML_TYPE_Q4_0>(
                    stage_k - win_lo * row_bytes, mirror_k, kv_lo, kv_hi, win_lo, win_rows, row_bytes, stream);
                sm70_d256_dequant_staging_window<GGML_TYPE_Q4_0>(
                    stage_v - win_lo * row_bytes, mirror_v, kv_lo, kv_hi, win_lo, win_rows, row_bytes, stream);
            }

            const half * const K_win = mirror_k - win_lo * SM70_D256_D;
            const half * const V_win = mirror_v - win_lo * SM70_D256_D;
            const int64_t kv_head_stride = (int64_t) win_rows * SM70_D256_D;
            const int64_t kv_outer_stride = (int64_t) d->heads_kv * win_rows * SM70_D256_D;

            sm70_d256_launch_dense<true>(
                qs, K_win, V_win, win_out, d->mask, bounds,
                (int64_t) d->heads_q * d->q_pad * SM70_D256_D,
                SM70_D256_D,
                (int64_t) d->q_pad * SM70_D256_D,
                kv_outer_stride, SM70_D256_D, kv_head_stride,
                kv_outer_stride, SM70_D256_D, kv_head_stride,
                d->mask_row_stride, d->mask_batch_stride, true,
                d->q_pad, d->kv_view, d->heads_q, d->heads_kv, d->batch, d->kv_view - d->q_len,
                d->scale, 1.0f / d->scale,
                (int) (win_lo / SM70_D256_MASK_BLOCK_N),
                (int) ((win_lo + win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N),
                1,
                win_out, win_max, win_sum,
                kv_lo, kv_hi,
                stream);




            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                d->p_out_dst[j], d->p_max_dst[j], d->p_sum_dst[j],
                win_out, win_max, win_sum,
                w.rows, false, d->scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        }

        sm70_d256_assist_signal_launch(d->epoch_partial[j], stream);
    }
    return true;
}

size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst) {
    return sm70_d256_get_scratch(dst).total;
}




bool ggml_cuda_sm70_d256_assist_scratch_layout(const ggml_tensor * dst, ggml_backend_meta_assist_scratch * layout) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT || layout == nullptr) {
        return false;
    }
    const sm70_d256_scratch scratch = sm70_d256_get_scratch(dst);
    if (!scratch.windowed) {
        return false;
    }
    layout->qs_offset    = scratch.qs_offset;
    layout->p_out_offset = scratch.p_out_offset;
    layout->p_max_offset = scratch.p_max_offset;
    layout->p_sum_offset = scratch.p_sum_offset;
    layout->q_pad        = scratch.q_pad;
    return true;
}

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool mask_range = mask->type == GGML_TYPE_I32;

    const int q_len   = (int) Q->ne[1];
    const int kv_len  = mask_range ? (int) K->ne[1] : (int) mask->ne[0];
    const int heads_q = (int) Q->ne[2];
    const int hkv     = (int) K->ne[2];
    if (heads_q == 0 || hkv == 0) {
        return;
    }
    const int gqa     = heads_q / hkv;
    const int batch   = (int) Q->ne[3];
    const int q_pad   = (int) GGML_PAD(q_len, SM70_D256_BLOCK_M);
    const int kv_offset = kv_len - q_len;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const sm70_d256_scratch scratch = sm70_d256_get_scratch(dst);
    char * base = (char *) dst->data + ggml_nbytes(dst);
    half  * Qs = (half  *) (base + scratch.qs_offset);
    float * Os = (float *) (base + scratch.os_offset);
    int2  * mask_bounds = (int2 *) (base + scratch.bounds_offset);
    int   * kv_limit = (int *) (base + scratch.kv_limit_offset);

    cudaStream_t stream = ctx.stream();





    const ggml_backend_meta_assist_rank * const assist = ctx.assist;
    const bool assist_owner = assist != nullptr && assist->role == 0;
    if (assist_owner && !scratch.windowed) {

        GGML_ABORT("%s: assist owner requires the windowed partial path", __func__);
    }

    if (scratch.windowed) {




        if (assist_owner) {
            GGML_ASSERT(assist->batch == 1 && assist->q_pad == q_pad && assist->q_len == q_len);
            GGML_ASSERT(assist->heads_q == heads_q && assist->kv_view == kv_len);
            GGML_ASSERT(assist->split_local != nullptr);
            GGML_ASSERT(assist->idle_load >= 1);
            GGML_ASSERT(assist->p_out_local != nullptr && assist->p_max_local != nullptr && assist->p_sum_local != nullptr);
            GGML_ASSERT(assist->epoch_kv_local != nullptr && assist->epoch_partial_local != nullptr);
            GGML_ASSERT(assist->gen_local != nullptr && assist->epoch_free_local != nullptr);
        }
        CUDA_CHECK(cudaMemsetAsync(Qs, 0, scratch.n_q * sizeof(half), stream));

        CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));

        {
            const dim3 grid(q_pad, batch * hkv, gqa);
            sm70_d256_stage_q_kernel<<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Q->data, (half2 *) Qs,
                q_len, hkv, gqa, q_pad,
                Q->nb[1] / sizeof(float2), Q->nb[2] / sizeof(float2), Q->nb[3] / sizeof(float2));
            CUDA_CHECK(cudaGetLastError());
        }




        {
            const dim3 grid(scratch.n_q_blocks, batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int4 *) mask->data, mask_bounds, kv_limit,
                q_len, kv_len,
                mask->nb[1] / sizeof(int4),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int4));
            CUDA_CHECK(cudaGetLastError());
        }

        if (assist_owner) {
            sm70_d256_assist_split_kernel<<<1, 1, 0, stream>>>(
                kv_limit, assist->split_local, assist->idle_load);
            CUDA_CHECK(cudaGetLastError());

            sm70_d256_assist_signal_launch(assist->epoch_kv_local, stream);
        }

        const half * const K_mirror = (const half *) scratch.f16_extra.K;
        GGML_ASSERT(K_mirror != nullptr);
        float * const p_out   = (float *) (base + scratch.p_out_offset);
        float * const acc_max = (float *) (base + scratch.acc_max_offset);
        float * const acc_sum = (float *) (base + scratch.acc_sum_offset);
        float * const p_max   = (float *) (base + scratch.p_max_offset);
        float * const p_sum   = (float *) (base + scratch.p_sum_offset);





        const int m_blocks = q_pad / SM70_D256_BLOCK_M;
        const int window_ctas = m_blocks * batch * heads_q;
        const bool split_window = window_ctas < 4 * SM70_D256_NSM;




        const int64_t n_kv_rows = K->ne[1];
        const int n_win = (int) ((n_kv_rows + SM70_D256_KV_WINDOW - 1) / SM70_D256_KV_WINDOW);
        for (int w = 0; w < n_win; ++w) {
            const int64_t row_lo = (int64_t) w * SM70_D256_KV_WINDOW;
            const int64_t rows_left = n_kv_rows - row_lo;
            const int win_rows = (int) (rows_left < SM70_D256_KV_WINDOW ? rows_left : SM70_D256_KV_WINDOW);





            const int win_blocks =
                (win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N;
            const int kv_splits = split_window
                && win_blocks >= 2 * SM70_D256_SPLIT_MIN_BLOCKS
                ? SM70_D256_KV_SPLITS : 1;


            sm70_d256_dequant_kv_window(K, (half *) K_mirror, kv_limit, (int) row_lo, win_rows, stream);
            if (scratch.need_f16_v && !scratch.v_is_k_view) {
                sm70_d256_dequant_kv_window(V, (half *) scratch.f16_extra.V, kv_limit, (int) row_lo, win_rows, stream);
            }




            const half * const K_win = K_mirror - row_lo * SM70_D256_D;
            const int64_t k_row_stride   = SM70_D256_D;
            const int64_t k_head_stride  = (int64_t) win_rows * SM70_D256_D;
            const int64_t k_outer_stride = (int64_t) K->ne[2] * win_rows * SM70_D256_D;

            const half * V_win;
            int64_t v_row_stride, v_head_stride, v_outer_stride;
            if (scratch.v_is_k_view) {
                V_win = K_win;
                v_row_stride   = k_row_stride;
                v_head_stride  = k_head_stride;
                v_outer_stride = k_outer_stride;
            } else if (V->type == GGML_TYPE_F16) {

                V_win = (const half *) V->data;
                v_row_stride   = V->nb[1] / sizeof(half);
                v_head_stride  = V->nb[2] / sizeof(half);
                v_outer_stride = V->nb[3] / sizeof(half);
            } else {
                V_win = (const half *) scratch.f16_extra.V - row_lo * SM70_D256_D;
                v_row_stride   = SM70_D256_D;
                v_head_stride  = (int64_t) win_rows * SM70_D256_D;
                v_outer_stride = (int64_t) V->ne[2] * win_rows * SM70_D256_D;
            }




            const bool direct_acc = w == 0 && kv_splits == 1;
            float * const win_out = direct_acc ? Os : p_out;
            float * const win_max = direct_acc ? acc_max : p_max;
            float * const win_sum = direct_acc ? acc_sum : p_sum;

            sm70_d256_launch_dense<true>(
                Qs, K_win, V_win, Os, mask->data, mask_bounds,
                (int64_t) heads_q * q_pad * SM70_D256_D,
                SM70_D256_D,
                (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                mask->nb[1] / sizeof(int4),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int4),
                true,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale,
                (int) (row_lo / SM70_D256_MASK_BLOCK_N),
                (int) ((row_lo + win_rows + SM70_D256_MASK_BLOCK_N - 1) / SM70_D256_MASK_BLOCK_N),
                kv_splits,
                win_out, win_max, win_sum,
                nullptr,
                assist_owner ? &assist->split_local->s : nullptr,
                stream);

            if (!direct_acc) {
                const unsigned n_merge = (unsigned) ((scratch.rows
                    + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                    / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
                const int64_t slice_out = (int64_t) scratch.rows * SM70_D256_D;



                FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                    Os, acc_max, acc_sum, win_out, win_max, win_sum,
                    (int64_t) scratch.rows, w == 0, scale * float(M_LOG2E));
                if (kv_splits > 1) {
                    FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                        Os, acc_max, acc_sum,
                        win_out + slice_out, win_max + scratch.rows, win_sum + scratch.rows,
                        (int64_t) scratch.rows, false, scale * float(M_LOG2E));
                }
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (assist_owner) {



            sm70_d256_assist_wait_launch(assist->epoch_partial_local, assist->gen_local, 1, 0, stream);
            const unsigned n_merge = (unsigned) ((scratch.rows
                + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum,
                assist->p_out_local, assist->p_max_local, assist->p_sum_local,
                (int64_t) scratch.rows, false, scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        }
    } else {


        const bool k_partial = mask_range && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0);
        const bool v_partial = mask_range && !scratch.v_is_k_view &&
                               (V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0);


        CUDA_CHECK(cudaMemsetAsync(Qs, 0, scratch.n_q * sizeof(half), stream));

        if (k_partial || v_partial) {

            CUDA_CHECK(cudaMemsetAsync(kv_limit, 0, sizeof(int), stream));
        }
        if (scratch.need_f16_k && !k_partial) {
            GGML_ASSERT(scratch.f16_extra.K != 0);
            sm70_d256_dequant_kv_tensor(K, (half *) scratch.f16_extra.K, stream);
        }


        if (scratch.need_f16_v && !scratch.v_is_k_view && !v_partial) {
            GGML_ASSERT(scratch.f16_extra.V != 0);
            sm70_d256_dequant_kv_tensor(V, (half *) scratch.f16_extra.V, stream);
        }



        const half * K_h2;
        const half * V_h2;
        int64_t k_outer_stride, k_row_stride, k_head_stride;
        int64_t v_outer_stride, v_row_stride, v_head_stride;
        if (K->type == GGML_TYPE_F16) {
            K_h2 = (const half *) K->data;
            k_row_stride   = K->nb[1] / sizeof(half);
            k_head_stride  = K->nb[2] / sizeof(half);
            k_outer_stride = K->nb[3] / sizeof(half);
        } else {
            K_h2 = (const half *) scratch.f16_extra.K;
            k_row_stride   = K->ne[0];
            k_head_stride  = K->ne[1] * K->ne[0];
            k_outer_stride = K->ne[2] * K->ne[1] * K->ne[0];
        }
        if (scratch.v_is_k_view) {
            V_h2 = K_h2;
            v_row_stride   = k_row_stride;
            v_head_stride  = k_head_stride;
            v_outer_stride = k_outer_stride;
        } else if (V->type == GGML_TYPE_F16) {
            V_h2 = (const half *) V->data;
            v_row_stride   = V->nb[1] / sizeof(half);
            v_head_stride  = V->nb[2] / sizeof(half);
            v_outer_stride = V->nb[3] / sizeof(half);
        } else {
            V_h2 = (const half *) scratch.f16_extra.V;
            v_row_stride   = V->ne[0];
            v_head_stride  = V->ne[1] * V->ne[0];
            v_outer_stride = V->ne[2] * V->ne[1] * V->ne[0];
        }

        {
            const dim3 grid(q_pad, batch * hkv, gqa);
            sm70_d256_stage_q_kernel<<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Q->data, (half2 *) Qs,
                q_len, hkv, gqa, q_pad,
                Q->nb[1] / sizeof(float2), Q->nb[2] / sizeof(float2), Q->nb[3] / sizeof(float2));
            CUDA_CHECK(cudaGetLastError());
        }

        if (mask_range) {

            const dim3 grid(scratch.n_q_blocks, batch);
            sm70_d256_range_bounds_kernel<<<grid, SM70_D256_BLOCK_M, 0, stream>>>(
                (const int4 *) mask->data, mask_bounds, kv_limit,
                q_len, kv_len,
                mask->nb[1] / sizeof(int4),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(int4));
            CUDA_CHECK(cudaGetLastError());

            if (k_partial) {

                sm70_d256_dequant_kv_partial(K, (half *) scratch.f16_extra.K, kv_limit, stream);
            }
            if (v_partial) {
                sm70_d256_dequant_kv_partial(V, (half *) scratch.f16_extra.V, kv_limit, stream);
            }
        } else {
            const int n_splits = (kv_len + SM70_D256_MASK_SPLIT - 1) / SM70_D256_MASK_SPLIT;
            const dim3 grid(scratch.n_q_blocks, batch, n_splits);


            CUDA_CHECK(cudaMemsetAsync(mask_bounds, 0,
                (size_t) batch * scratch.n_q_blocks * sizeof(int2), stream));


            const bool vec_ok = mask->nb[1] % 16 == 0
                && (mask->ne[3] == 1 || mask->nb[3] % 16 == 0)
                && (uintptr_t) mask->data % 16 == 0;
            sm70_d256_mask_bounds_kernel<<<grid, 256, 0, stream>>>(
                (const __half *) mask->data, mask_bounds,
                q_len, kv_len,
                mask->nb[1] / sizeof(half),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / sizeof(half),
                vec_ok);
            CUDA_CHECK(cudaGetLastError());
        }

        if (scratch.partial) {



            float * const p_out   = (float *) (base + scratch.p_out_offset);
            float * const p_max   = (float *) (base + scratch.p_max_offset);
            float * const p_sum   = (float *) (base + scratch.p_sum_offset);
            float * const acc_max = (float *) (base + scratch.acc_max_offset);
            float * const acc_sum = (float *) (base + scratch.acc_sum_offset);

            sm70_d256_launch_dense<true>(
                Qs, K_h2, V_h2, Os, mask->data, mask_bounds,
                (int64_t) heads_q * q_pad * SM70_D256_D,
                SM70_D256_D,
                (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                mask_range ? mask->nb[1] / sizeof(int4)   : mask->nb[1] / sizeof(half),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / (mask_range ? sizeof(int4) : sizeof(half)),
                mask_range,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale,
                0, INT_MAX,
                SM70_D256_KV_SPLITS,
                p_out, p_max, p_sum,
                nullptr, nullptr,
                stream);

            const unsigned n_merge = (unsigned) ((scratch.rows
                + FLASH_NAMESPACE::kWindowMergeRowsPerCta - 1)
                / FLASH_NAMESPACE::kWindowMergeRowsPerCta);
            const int64_t slice_out = (int64_t) scratch.rows * SM70_D256_D;
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum, p_out, p_max, p_sum,
                (int64_t) scratch.rows, true, scale * float(M_LOG2E));
            FLASH_NAMESPACE::sm70_d256_window_merge_kernel<<<n_merge, 256, 0, stream>>>(
                Os, acc_max, acc_sum,
                p_out + slice_out, p_max + scratch.rows, p_sum + scratch.rows,
                (int64_t) scratch.rows, false, scale * float(M_LOG2E));
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_sm70_d256_launch_raw(
                Qs, K_h2, V_h2, Os, mask->data, mask_bounds,
                (int64_t) heads_q * q_pad * SM70_D256_D,
                SM70_D256_D,
                (int64_t) q_pad * SM70_D256_D,
                k_outer_stride, k_row_stride, k_head_stride,
                v_outer_stride, v_row_stride, v_head_stride,
                mask_range ? mask->nb[1] / sizeof(int4)   : mask->nb[1] / sizeof(half),
                mask->ne[3] == 1 ? 0 : (int64_t) mask->nb[3] / (mask_range ? sizeof(int4) : sizeof(half)),
                mask_range,
                q_pad, kv_len, heads_q, hkv, batch, kv_offset,
                scale, 1.0f / scale, stream);
        }
    }



    {
        const dim3 grid(q_len, batch * heads_q);
        if (scratch.partial) {


            sm70_d256_scatter_kernel<true><<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Os, (float2 *) dst->data,
                (const float *) (base + scratch.acc_sum_offset),
                heads_q, q_pad,
                dst->nb[2] / sizeof(float2),
                dst->nb[1] / sizeof(float2),
                dst->nb[3] / sizeof(float2));
        } else {
            sm70_d256_scatter_kernel<false><<<grid, SM70_D256_D/2, 0, stream>>>(
                (const float2 *) Os, (float2 *) dst->data, nullptr,
                heads_q, q_pad,
                dst->nb[2] / sizeof(float2),
                dst->nb[1] / sizeof(float2),
                dst->nb[3] / sizeof(float2));
        }
        CUDA_CHECK(cudaGetLastError());
    }



    if (assist_owner) {
        sm70_d256_assist_signal_launch(assist->epoch_free_local, stream);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
