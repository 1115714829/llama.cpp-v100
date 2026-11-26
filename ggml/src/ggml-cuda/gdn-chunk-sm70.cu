// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
// Portions from TileLang, FlashQLA and 1Cat-vLLM; see licenses/LICENSE-tilelang, licenses/LICENSE-FlashQLA and licenses/LICENSE-1Cat-vLLM.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "gdn-chunk-sm70.cuh"

#include "gdn-chunk-sm70/gdn-chunk-cumsum.cuh"
#include "gdn-chunk-sm70/gdn-chunk-cumsum-hv12.cuh"
#include "gdn-chunk-sm70/gdn-chunk-fwd.cuh"
#include "gdn-chunk-sm70/gdn-chunk-kkt.cuh"

namespace gdn_chunk_sm70 {
template <int HV_CT, int HK_CT>
__global__ void __launch_bounds__(128, 1) gdn_chunk_direct_kernel(const half_t* __restrict__ a, const float* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const half_t* __restrict__ k, float* __restrict__ o, const half_t* __restrict__ q, const half_t* __restrict__ v, int batch_size, int num_tokens, int raw_batch_size, int Hv_rt, int Hk_rt) {
  const int Hv = HV_CT > 0 ? HV_CT : Hv_rt;
  const int Hk = HK_CT > 0 ? HK_CT : Hk_rt;
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* kt_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 59376));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 67568));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 75760));
  void* vd_shared = ((void*)((char*)buf_dyn_shmem + 80864));
  void* vn_shared = ((void*)((char*)buf_dyn_shmem + 85968));
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int state_slot = 0;
  int num_iters = 0;
  float h_fragment[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  __shared__ __align__(16) float g_exp_shared[64];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  float p_fragment[32];
  float g_fragment[32];
  float a_fragment[32];
  float u_fragment[16];
  float v_fragment[16];
  float o_fragment[16];
  float g_last_local[1];
  batch_idx = 0;
  const int hv4 = Hv * 4;
  const int blk = ((int) blockIdx.x) % hv4;
  const int h   = blk >> 2;
  const int batch = ((int) blockIdx.x) / hv4;
  const int qk_head = h % Hk;
  const int64_t qk_tok_stride  = (int64_t) Hk * 128;
  const int64_t qk_iter_stride = 4 * qk_tok_stride;
  const int64_t v_tok_stride   = (int64_t) Hv * 128;
  const int64_t v_iter_stride  = 32 * v_tok_stride;
  const int64_t a_tok_stride   = (int64_t) Hv * 64;
  const int64_t a_iter_stride  = 8 * a_tok_stride;
  const int64_t g_tok_stride   = Hv;
  seq_start_idx = cu_seqlens[batch];
  seq_end_idx = cu_seqlens[(int64_t) batch + 1];
  chunk_start_idx = chunk_offsets[batch];
  raw_batch_idx = batch;
  raw_seq_end_idx = seq_end_idx;
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  state_slot = batch;
  num_iters = (((seq_end_idx + 63) - seq_start_idx) >> 6);
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    h_fragment[i] = h0[((((((((((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)2048)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)2) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)3) * (int64_t)512)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) >> (int64_t)1) * (int64_t)256)) + ((((int64_t)i) & (int64_t)1) * (int64_t)128)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)7) >> (int64_t)2) * (int64_t)8)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4) * (int64_t)4)) + (((((int64_t)i) & (int64_t)3) >> (int64_t)1) * (int64_t)2)) + (((int64_t)((int)threadIdx.x)) & (int64_t)1))];
  }
  for (int i_s = 0; i_s < num_iters; ++i_s) {
    int left = ((i_s * 64) + seq_start_idx);
    if ((left + 64) <= seq_end_idx) {
      __syncthreads();
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        half_t broadcast_var = half_t(0x0p+0f );
        uint2 condval;
        if ((((((((i_1 * 4) + (((int)threadIdx.x) >> 5)) + left) < num_tokens) && (0 <= (((i_1 * 4) + (((int)threadIdx.x) >> 5)) + left))) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval = *(uint2*)(q + ((int64_t)i_1 * qk_iter_stride + (int64_t)(threadIdx.x >> 5) * qk_tok_stride + (int64_t)left * qk_tok_stride + (int64_t)batch_idx * num_tokens * qk_tok_stride + (int64_t)qk_head * 128 + (int64_t)(threadIdx.x & 31) * 4));
        } else {
          condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        }
        *(uint2*)(((half_t*)q_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_1 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 7) >> 2)) + ((i_1 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 4))) = condval;
      }
      __syncthreads();
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        half_t broadcast_var_1 = half_t(0x0p+0f );
        uint2 condval_1;
        if ((((((((i_2 * 4) + (((int)threadIdx.x) >> 5)) + left) < num_tokens) && (0 <= (((i_2 * 4) + (((int)threadIdx.x) >> 5)) + left))) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval_1 = *(uint2*)(k + ((int64_t)i_2 * qk_iter_stride + (int64_t)(threadIdx.x >> 5) * qk_tok_stride + (int64_t)left * qk_tok_stride + (int64_t)batch_idx * num_tokens * qk_tok_stride + (int64_t)qk_head * 128 + (int64_t)(threadIdx.x & 31) * 4));
        } else {
          condval_1 = make_uint2(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
        }
        *(uint2*)(((half_t*)k_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_2 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 7) >> 2)) + ((i_2 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 4))) = condval_1;
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        half_t broadcast_var_2 = half_t(0x0p+0f );
        uint4 condval_2;
        if ((((((((i_3 * 32) + (((int)threadIdx.x) >> 2)) + left) < num_tokens) && (0 <= (((i_3 * 32) + (((int)threadIdx.x) >> 2)) + left))) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval_2 = *(uint4*)(v + ((int64_t)i_3 * v_iter_stride + (int64_t)(threadIdx.x >> 2) * v_tok_stride + (int64_t)left * v_tok_stride + (int64_t)batch_idx * num_tokens * v_tok_stride + (int64_t)blk * 32 + (int64_t)(threadIdx.x & 3) * 8));
        } else {
          condval_2 = make_uint4(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
        }
        *(uint4*)(((half_t*)v_shared) + (((i_3 * 1280) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = condval_2;
      }
      __syncthreads();
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        half_t broadcast_var_3 = half_t(0x0p+0f );
        uint2 condval_3;
        if ((((((((i_4 * 8) + (((int)threadIdx.x) >> 4)) + left) < num_tokens) && (0 <= (((i_4 * 8) + (((int)threadIdx.x) >> 4)) + left))) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval_3 = *(uint2*)(a + ((int64_t)i_4 * a_iter_stride + (int64_t)(threadIdx.x >> 4) * a_tok_stride + (int64_t)left * a_tok_stride + (int64_t)batch_idx * num_tokens * a_tok_stride + (int64_t)h * 64 + (int64_t)(threadIdx.x & 15) * 4));
        } else {
          condval_3 = make_uint2(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
        }
        *(uint2*)(((half_t*)a_shared) + ((((((((int)threadIdx.x) & 15) * 256) + ((i_4 >> 1) * 64)) + (((((int)threadIdx.x) & 63) >> 4) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) + (i_4 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))) = condval_3;
      }
      if (((int)threadIdx.x) < 64) {
        float condval_4;
        if (((((0 <= (left + ((int)threadIdx.x))) && ((left + ((int)threadIdx.x)) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval_4 = g[((int64_t)left + threadIdx.x) * g_tok_stride + (int64_t)batch_idx * num_tokens * g_tok_stride + h];
        } else {
          condval_4 = 0x0p+0f ;
        }
        g_shared[((int)threadIdx.x)] = condval_4;
        float condval_5;
        if (((((0 <= (left + ((int)threadIdx.x))) && ((left + ((int)threadIdx.x)) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
          condval_5 = b[((int64_t)left + threadIdx.x) * g_tok_stride + (int64_t)batch_idx * num_tokens * g_tok_stride + h];
        } else {
          condval_5 = 0x0p+0f ;
        }
        b_shared[((int)threadIdx.x)] = condval_5;
      }
    } else {
      __syncthreads();
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        if ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + left) < seq_end_idx) {
          half_t broadcast_var_4 = half_t(0x0p+0f );
          uint2 condval_6;
          if (((((0 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + left)) && ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + left) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_6 = *(uint2*)(q + ((int64_t)i_5 * qk_iter_stride + (int64_t)(threadIdx.x >> 5) * qk_tok_stride + (int64_t)left * qk_tok_stride + (int64_t)batch_idx * num_tokens * qk_tok_stride + (int64_t)qk_head * 128 + (int64_t)(threadIdx.x & 31) * 4));
          } else {
            condval_6 = make_uint2(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
          }
          *(uint2*)(((half_t*)q_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_5 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 4))) = condval_6;
          half_t broadcast_var_5 = half_t(0x0p+0f );
          uint2 condval_7;
          if (((((0 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + left)) && ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + left) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_7 = *(uint2*)(k + ((int64_t)i_5 * qk_iter_stride + (int64_t)(threadIdx.x >> 5) * qk_tok_stride + (int64_t)left * qk_tok_stride + (int64_t)batch_idx * num_tokens * qk_tok_stride + (int64_t)qk_head * 128 + (int64_t)(threadIdx.x & 31) * 4));
          } else {
            condval_7 = make_uint2(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
          }
          *(uint2*)(((half_t*)k_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_5 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 4))) = condval_7;
        } else {
          half_t broadcast_var_6 = half_t(0x0p+0f );
          *(uint2*)(((half_t*)q_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_5 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 4))) = make_uint2(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
          half_t broadcast_var_7 = half_t(0x0p+0f );
          *(uint2*)(((half_t*)k_shared) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_5 >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 4))) = make_uint2(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
        }
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        if ((((i_6 * 32) + (((int)threadIdx.x) >> 2)) + left) < seq_end_idx) {
          half_t broadcast_var_8 = half_t(0x0p+0f );
          uint4 condval_8;
          if (((((0 <= (((i_6 * 32) + (((int)threadIdx.x) >> 2)) + left)) && ((((i_6 * 32) + (((int)threadIdx.x) >> 2)) + left) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_8 = *(uint4*)(v + ((int64_t)i_6 * v_iter_stride + (int64_t)(threadIdx.x >> 2) * v_tok_stride + (int64_t)left * v_tok_stride + (int64_t)batch_idx * num_tokens * v_tok_stride + (int64_t)blk * 32 + (int64_t)(threadIdx.x & 3) * 8));
          } else {
            condval_8 = make_uint4(__pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8));
          }
          *(uint4*)(((half_t*)v_shared) + (((i_6 * 1280) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = condval_8;
        } else {
          half_t broadcast_var_9 = half_t(0x0p+0f );
          *(uint4*)(((half_t*)v_shared) + (((i_6 * 1280) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9));
        }
      }
      __syncthreads();
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        if ((((i_7 * 8) + (((int)threadIdx.x) >> 4)) + left) < seq_end_idx) {
          half_t broadcast_var_10 = half_t(0x0p+0f );
          uint2 condval_9;
          if (((((0 <= (((i_7 * 8) + (((int)threadIdx.x) >> 4)) + left)) && ((((i_7 * 8) + (((int)threadIdx.x) >> 4)) + left) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_9 = *(uint2*)(a + ((int64_t)i_7 * a_iter_stride + (int64_t)(threadIdx.x >> 4) * a_tok_stride + (int64_t)left * a_tok_stride + (int64_t)batch_idx * num_tokens * a_tok_stride + (int64_t)h * 64 + (int64_t)(threadIdx.x & 15) * 4));
          } else {
            condval_9 = make_uint2(__pack_half2(broadcast_var_10, broadcast_var_10), __pack_half2(broadcast_var_10, broadcast_var_10));
          }
          *(uint2*)(((half_t*)a_shared) + ((((((((int)threadIdx.x) & 15) * 256) + ((i_7 >> 1) * 64)) + (((((int)threadIdx.x) & 63) >> 4) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) + (i_7 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))) = condval_9;
        } else {
          half_t broadcast_var_11 = half_t(0x0p+0f );
          *(uint2*)(((half_t*)a_shared) + ((((((((int)threadIdx.x) & 15) * 256) + ((i_7 >> 1) * 64)) + (((((int)threadIdx.x) & 63) >> 4) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) + (i_7 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))) = make_uint2(__pack_half2(broadcast_var_11, broadcast_var_11), __pack_half2(broadcast_var_11, broadcast_var_11));
        }
      }
      if (((int)threadIdx.x) < 64) {
        if ((left + ((int)threadIdx.x)) < seq_end_idx) {
          float condval_10;
          if (((((0 <= (left + ((int)threadIdx.x))) && ((left + ((int)threadIdx.x)) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_10 = g[((int64_t)left + threadIdx.x) * g_tok_stride + (int64_t)batch_idx * num_tokens * g_tok_stride + h];
          } else {
            condval_10 = 0x0p+0f ;
          }
          g_shared[((int)threadIdx.x)] = condval_10;
          float condval_11;
          if (((((0 <= (left + ((int)threadIdx.x))) && ((left + ((int)threadIdx.x)) < num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_11 = b[((int64_t)left + threadIdx.x) * g_tok_stride + (int64_t)batch_idx * num_tokens * g_tok_stride + h];
          } else {
            condval_11 = 0x0p+0f ;
          }
          b_shared[((int)threadIdx.x)] = condval_11;
        } else {
          float condval_12;
          if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
            condval_12 = g[((int64_t)seq_end_idx - 1) * g_tok_stride + (int64_t)batch_idx * num_tokens * g_tok_stride + h];
          } else {
            condval_12 = 0x0p+0f ;
          }
          g_shared[((int)threadIdx.x)] = condval_12;
          b_shared[((int)threadIdx.x)] = 0x0p+0f ;
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      ((half_t*)kt_shared)[(((((((i_8 >> 2) * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((i_8 & 31) >> 4) + ((((int)threadIdx.x) & 31) >> 4)) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((i_8 & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (i_8 & 3))] = ((half_t*)k_shared)[(((((((((int)threadIdx.x) >> 2) * 256) + ((i_8 >> 4) * 64)) + ((i_8 & 3) * 16)) + ((((((i_8 & 31) >> 4) + ((((int)threadIdx.x) & 31) >> 4)) + ((i_8 & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_8 & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))];
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      half_t h_shared_local_cast[2];
      uint1 __1;
      float2 v_ = *(float2*)(h_fragment + (i_9 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(h_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)h_shared) + ((((((((((((((int)threadIdx.x) & 63) >> 5) * 2560) + ((i_9 >> 2) * 640)) + (((((int)threadIdx.x) & 7) >> 2) * 320)) + (((((int)threadIdx.x) & 31) >> 4) * 160)) + ((i_9 & 1) * 80)) + ((((int)threadIdx.x) & 1) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_9 & 3) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(h_shared_local_cast + 0);
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      g_exp_shared[((int)threadIdx.x)] = exp2f((g_shared[((int)threadIdx.x)] * 0x1.715476p+0f ));
      float condval_13;
      if (((left + ((int)threadIdx.x)) < seq_end_idx)) {
        condval_13 = exp2f(((g_shared[63] - g_shared[((int)threadIdx.x)]) * 0x1.715476p+0f ));
      } else {
        condval_13 = 0x0p+0f ;
      }
      g_rev_exp_shared[((int)threadIdx.x)] = condval_13;
    }
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[8];
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        float broadcast_var_12 = 0x0p+0f ;
        *(float4*)(p_fragment + (i_10 * 4)) = make_float4(broadcast_var_12, broadcast_var_12, broadcast_var_12, broadcast_var_12);
      }
      for (int ki = 0; ki < 32; ++ki) {
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          *(uint2*)(A_local + (i_11 * 4)) = *(uint2*)(((half_t*)q_shared) + ((((((ki * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (i_11 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + i_11) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki & 3) >> 1)) & 1) * 4)));
        }
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          *(uint2*)(B_local + (i_12 * 4)) = *(uint2*)(((half_t*)k_shared) + ((((((ki * 256) + ((((int)threadIdx.x) >> 6) * 128)) + (i_12 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((((int)threadIdx.x) & 31) >> 4) + ((ki & 7) >> 2)) + i_12) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + ((ki & 3) >> 1)) & 1) * 4)));
        }
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, true>(reinterpret_cast<float*>(p_fragment + ((i_13 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_13 * 4)), reinterpret_cast<const unsigned*>(B_local + (j * 4)));
          }
        }
      }
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      float2 __2;
        float2 v__1 = make_float2(g_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_14 >> 3) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_14 & 1) * 2)) + (((int)threadIdx.x) & 1))], g_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_14 >> 3) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_14 & 1) * 2)) + (((int)threadIdx.x) & 1))]);
        float2 v__2 = *(float2*)(g_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_14 & 7) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)));
        __2.x = (v__1.x-v__2.x);
        __2.y = (v__1.y-v__2.y);
      *(float2*)(g_fragment + (i_14 * 2)) = __2;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      if (((((((((int)threadIdx.x) >> 6) * 32) + (((i_15 & 15) >> 2) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)) + (i_15 & 1)) <= ((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_15 >> 4) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + (((i_15 & 3) >> 1) * 2)) + (((int)threadIdx.x) & 1))) {
        g_fragment[i_15] = exp2f((g_fragment[i_15] * 0x1.715476p+0f ));
      } else {
        g_fragment[i_15] = 0x0p+0f ;
      }
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 16; ++i_16) {
      half_t a_shared_local_cast_1[2];
      *(uint1*)(a_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)a_shared) + (((((((((((((int)threadIdx.x) >> 6) * 2048) + (((i_16 & 7) >> 1) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + ((i_16 >> 3) * 64)) + ((i_16 & 1) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((((((((i_16 & 7) >> 1) * 8) + (((((int)threadIdx.x) & 15) >> 3) * 4)) >> 4) + (i_16 >> 3)) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_16 & 3) >> 1)) & 1) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)));
      float2 __3;
      uint1 v__3 = *(uint1*)(a_shared_local_cast_1 + 0);
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(a_fragment + (i_16 * 2)) = __3;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 32; ++i_17) {
      a_fragment[i_17] = (a_fragment[i_17] * g_fragment[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      float2 __4;
        float2 v__4 = *(float2*)(a_fragment + (i_18 * 2));
        float2 v__5 = *(float2*)(b_shared + (((((((int)threadIdx.x) >> 6) * 32) + (((i_18 & 7) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)));
        __4.x = (v__4.x*v__5.x);
        __4.y = (v__4.y*v__5.y);
      *(float2*)(a_fragment + (i_18 * 2)) = __4;
    }
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      half_t a_shared_local_cast_2[2];
      uint1 __5;
      float2 v__6 = *(float2*)(a_fragment + (i_19 * 2));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__6))[0]);
      *(uint1*)(a_shared_local_cast_2 + 0) = __5;
      *(uint1*)(((half_t*)a_shared) + (((((((((((((int)threadIdx.x) >> 6) * 2048) + (((i_19 & 7) >> 1) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + ((i_19 >> 3) * 64)) + ((i_19 & 1) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((((((((i_19 & 7) >> 1) * 8) + (((((int)threadIdx.x) & 15) >> 3) * 4)) >> 4) + (i_19 >> 3)) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_19 & 3) >> 1)) & 1) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(a_shared_local_cast_2 + 0);
    }
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[4];
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        float broadcast_var_13 = 0x0p+0f ;
        *(float4*)(u_fragment + (i_20 * 4)) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
      }
      for (int ki_1 = 0; ki_1 < 32; ++ki_1) {
        for (int i_21 = 0; i_21 < 2; ++i_21) {
          *(uint2*)(A_local_1 + (i_21 * 4)) = *(uint2*)(((half_t*)k_shared) + ((((((ki_1 * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (i_21 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki_1 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + i_21) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki_1 & 3) >> 1)) & 1) * 4)));
        }
        *(uint2*)(B_local_1 + 0) = *(uint2*)(((half_t*)h_shared) + ((((ki_1 * 160) + ((((int)threadIdx.x) & 3) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((((int)threadIdx.x) & 31) >> 3) * 4)));
        for (int i_22 = 0; i_22 < 2; ++i_22) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(u_fragment + (i_22 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_22 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        }
      }
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      float2 __6;
        float2 v__7 = *(float2*)(u_fragment + (i_23 * 2));
        float2 v__8 = make_float2((g_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_23 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_23 & 1) * 2)) + (((int)threadIdx.x) & 1))] * -0x1p+0f ), (g_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_23 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_23 & 1) * 2)) + (((int)threadIdx.x) & 1))] * -0x1p+0f ));
        __6.x = (v__7.x*v__8.x);
        __6.y = (v__7.y*v__8.y);
      *(float2*)(u_fragment + (i_23 * 2)) = __6;
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      half_t v_shared_local_cast_3[2];
      *(uint1*)(v_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)v_shared) + ((((((((((((((int)threadIdx.x) & 63) >> 5) * 1280) + ((i_24 >> 2) * 640)) + (((((int)threadIdx.x) & 7) >> 2) * 320)) + (((((int)threadIdx.x) & 31) >> 4) * 160)) + ((i_24 & 1) * 80)) + ((((int)threadIdx.x) & 1) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_24 & 3) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)));
      float2 __7;
        float2 v__9 = *(float2*)(u_fragment + (i_24 * 2));
        float2 __8;
        uint1 v__10 = *(uint1*)(v_shared_local_cast_3 + 0);
        ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__10))[0]);
        __7.x = (v__9.x+__8.x);
        __7.y = (v__9.y+__8.y);
      *(float2*)(u_fragment + (i_24 * 2)) = __7;
    }
    __syncthreads();
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      half_t v_shared_local_cast_4[2];
      uint1 __9;
      float2 v__11 = *(float2*)(u_fragment + (i_25 * 2));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__11))[0]);
      *(uint1*)(v_shared_local_cast_4 + 0) = __9;
      *(uint1*)(((half_t*)v_shared) + ((((((((((((((int)threadIdx.x) & 63) >> 5) * 1280) + ((i_25 >> 2) * 640)) + (((((int)threadIdx.x) & 7) >> 2) * 320)) + (((((int)threadIdx.x) & 31) >> 4) * 160)) + ((i_25 & 1) * 80)) + ((((int)threadIdx.x) & 1) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_25 & 3) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(v_shared_local_cast_4 + 0);
    }
    __syncthreads();
    {
      half_t A_local_2[8];
      half_t B_local_2[4];
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        float broadcast_var_14 = 0x0p+0f ;
        *(float4*)(v_fragment + (i_26 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
      }
      for (int ki_2 = 0; ki_2 < 16; ++ki_2) {
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          *(uint2*)(A_local_2 + (i_27 * 4)) = *(uint2*)(((half_t*)a_shared) + ((((((ki_2 * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (i_27 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki_2 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + i_27) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki_2 & 3) >> 1)) & 1) * 4)));
        }
        *(uint2*)(B_local_2 + 0) = *(uint2*)(((half_t*)v_shared) + ((((ki_2 * 160) + ((((int)threadIdx.x) & 3) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((((int)threadIdx.x) & 31) >> 3) * 4)));
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(v_fragment + (i_28 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + (i_28 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
        }
      }
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      half_t vd_shared_local_cast_5[2];
      uint1 __10;
      float2 v__12 = *(float2*)(v_fragment + (i_29 * 2));
      ((half2*)(&__10))[0] = __float22half2_rn(((float2*)(&v__12))[0]);
      *(uint1*)(vd_shared_local_cast_5 + 0) = __10;
      *(uint1*)(((half_t*)vd_shared) + ((((((((((((((int)threadIdx.x) & 63) >> 5) * 1280) + ((i_29 >> 2) * 640)) + (((((int)threadIdx.x) & 7) >> 2) * 320)) + (((((int)threadIdx.x) & 31) >> 4) * 160)) + ((i_29 & 1) * 80)) + ((((int)threadIdx.x) & 1) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_29 & 3) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(vd_shared_local_cast_5 + 0);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      float2 __11;
        float2 v__13 = *(float2*)(v_fragment + (i_30 * 2));
        float2 v__14 = make_float2(g_rev_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_30 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_30 & 1) * 2)) + (((int)threadIdx.x) & 1))], g_rev_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_30 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_30 & 1) * 2)) + (((int)threadIdx.x) & 1))]);
        __11.x = (v__13.x*v__14.x);
        __11.y = (v__13.y*v__14.y);
      *(float2*)(v_fragment + (i_30 * 2)) = __11;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 8; ++i_31) {
      half_t vn_shared_local_cast_6[2];
      uint1 __12;
      float2 v__15 = *(float2*)(v_fragment + (i_31 * 2));
      ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&v__15))[0]);
      *(uint1*)(vn_shared_local_cast_6 + 0) = __12;
      *(uint1*)(((half_t*)vn_shared) + ((((((((((((((int)threadIdx.x) & 63) >> 5) * 1280) + ((i_31 >> 2) * 640)) + (((((int)threadIdx.x) & 7) >> 2) * 320)) + (((((int)threadIdx.x) & 31) >> 4) * 160)) + ((i_31 & 1) * 80)) + ((((int)threadIdx.x) & 1) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_31 & 3) >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(vn_shared_local_cast_6 + 0);
    }
    __syncthreads();
    {
      half_t A_local_3[8];
      half_t B_local_3[4];
      #pragma unroll
      for (int i_32 = 0; i_32 < 4; ++i_32) {
        float broadcast_var_15 = 0x0p+0f ;
        *(float4*)(o_fragment + (i_32 * 4)) = make_float4(broadcast_var_15, broadcast_var_15, broadcast_var_15, broadcast_var_15);
      }
      for (int ki_3 = 0; ki_3 < 32; ++ki_3) {
        for (int i_33 = 0; i_33 < 2; ++i_33) {
          *(uint2*)(A_local_3 + (i_33 * 4)) = *(uint2*)(((half_t*)q_shared) + ((((((ki_3 * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (i_33 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki_3 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + i_33) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki_3 & 3) >> 1)) & 1) * 4)));
        }
        *(uint2*)(B_local_3 + 0) = *(uint2*)(((half_t*)h_shared) + ((((ki_3 * 160) + ((((int)threadIdx.x) & 3) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((((int)threadIdx.x) & 31) >> 3) * 4)));
        for (int i_34 = 0; i_34 < 2; ++i_34) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(o_fragment + (i_34 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + (i_34 * 4)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        }
      }
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 32; ++i_35) {
      p_fragment[i_35] = (p_fragment[i_35] * (0x1.6a09e6p-4f * g_fragment[i_35]));
    }
    __syncthreads();
    #pragma unroll
    for (int i_36 = 0; i_36 < 16; ++i_36) {
      half_t p_shared_local_cast_7[2];
      uint1 __13;
      float2 v__16 = *(float2*)(p_fragment + (i_36 * 2));
      ((half2*)(&__13))[0] = __float22half2_rn(((float2*)(&v__16))[0]);
      *(uint1*)(p_shared_local_cast_7 + 0) = __13;
      *(uint1*)(((half_t*)p_shared) + (((((((((((((int)threadIdx.x) >> 6) * 2048) + (((i_36 & 7) >> 1) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + ((i_36 >> 3) * 64)) + ((i_36 & 1) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((((((((i_36 & 7) >> 1) * 8) + (((((int)threadIdx.x) & 15) >> 3) * 4)) >> 4) + (i_36 >> 3)) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_36 & 3) >> 1)) & 1) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(p_shared_local_cast_7 + 0);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      float2 __14;
        float2 v__17 = *(float2*)(o_fragment + (i_37 * 2));
        float2 v__18 = make_float2((0x1.6a09e6p-4f * g_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_37 & 1) * 2)) + (((int)threadIdx.x) & 1))]), (0x1.6a09e6p-4f * g_exp_shared[((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_37 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_37 & 1) * 2)) + (((int)threadIdx.x) & 1))]));
        __14.x = (v__17.x*v__18.x);
        __14.y = (v__17.y*v__18.y);
      *(float2*)(o_fragment + (i_37 * 2)) = __14;
    }
    __syncthreads();
    {
      half_t A_local_4[8];
      half_t B_local_4[4];
      for (int ki_4 = 0; ki_4 < 16; ++ki_4) {
        for (int i_38 = 0; i_38 < 2; ++i_38) {
          *(uint2*)(A_local_4 + (i_38 * 4)) = *(uint2*)(((half_t*)p_shared) + ((((((ki_4 * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + (i_38 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki_4 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + i_38) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki_4 & 3) >> 1)) & 1) * 4)));
        }
        *(uint2*)(B_local_4 + 0) = *(uint2*)(((half_t*)vd_shared) + ((((ki_4 * 160) + ((((int)threadIdx.x) & 3) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((((int)threadIdx.x) & 31) >> 3) * 4)));
        for (int i_39 = 0; i_39 < 2; ++i_39) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(o_fragment + (i_39 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + (i_39 * 4)), reinterpret_cast<const unsigned*>(B_local_4 + 0));
        }
      }
    }
    if ((left + 64) <= seq_end_idx) {
      #pragma unroll
      for (int i_40 = 0; i_40 < 8; ++i_40) {
        half_t o_local_cast_8[2];
        if (((((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_40 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_40 & 1) * 2)) + left) + (((int)threadIdx.x) & 1)) < num_tokens) && (0 <= (((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_40 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_40 & 1) * 2)) + left) + (((int)threadIdx.x) & 1)))) {
          uint1 __15;
          float2 v__19 = *(float2*)(o_fragment + (i_40 * 2));
          ((half2*)(&__15))[0] = __float22half2_rn(((float2*)(&v__19))[0]);
          *(uint1*)(o_local_cast_8 + 0) = __15;
          if (0 <= batch_idx) {
            if (batch_idx < 1) {
              *reinterpret_cast<float2 *>(o + ((int64_t)((((int)threadIdx.x & 63) >> 5) * 32 + ((int)i_40 >> 2) * 16 + (((int)threadIdx.x & 7) >> 2) * 8 + (((int)threadIdx.x & 31) >> 4) * 4 + ((int)i_40 & 1) * 2 + ((int)threadIdx.x & 1)) + left + batch_idx * (int64_t)num_tokens) * v_tok_stride + ((int64_t)blk * 32 + ((int64_t)threadIdx.x >> 6) * 16 + (int64_t)(((int)i_40 & 3) >> 1) * 8 + (int64_t)(((int)threadIdx.x & 15) >> 3) * 4 + (int64_t)(((int)threadIdx.x & 3) >> 1) * 2)) = __half22float2(*reinterpret_cast<const half2 *>(o_local_cast_8));
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int i_41 = 0; i_41 < 8; ++i_41) {
        half_t o_local_cast_9[2];
        if ((((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_41 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_41 & 1) * 2)) + left) + (((int)threadIdx.x) & 1)) < seq_end_idx) {
          uint1 __16;
          float2 v__20 = *(float2*)(o_fragment + (i_41 * 2));
          ((half2*)(&__16))[0] = __float22half2_rn(((float2*)(&v__20))[0]);
          *(uint1*)(o_local_cast_9 + 0) = __16;
        }
        if ((((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_41 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_41 & 1) * 2)) + left) + (((int)threadIdx.x) & 1)) < seq_end_idx) {
          if (0 <= (((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_41 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_41 & 1) * 2)) + left) + (((int)threadIdx.x) & 1))) {
            if ((((((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_41 >> 2) * 16)) + (((((int)threadIdx.x) & 7) >> 2) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + ((i_41 & 1) * 2)) + left) + (((int)threadIdx.x) & 1)) < num_tokens) {
              if (0 <= batch_idx) {
                if (batch_idx < 1) {
                  *reinterpret_cast<float2 *>(o + ((int64_t)((((int)threadIdx.x & 63) >> 5) * 32 + ((int)i_41 >> 2) * 16 + (((int)threadIdx.x & 7) >> 2) * 8 + (((int)threadIdx.x & 31) >> 4) * 4 + ((int)i_41 & 1) * 2 + ((int)threadIdx.x & 1)) + left + batch_idx * (int64_t)num_tokens) * v_tok_stride + ((int64_t)blk * 32 + ((int64_t)threadIdx.x >> 6) * 16 + (int64_t)(((int)i_41 & 3) >> 1) * 8 + (int64_t)(((int)threadIdx.x & 15) >> 3) * 4 + (int64_t)(((int)threadIdx.x & 3) >> 1) * 2)) = __half22float2(*reinterpret_cast<const half2 *>(o_local_cast_9));
                }
              }
            }
          }
        }
      }
    }
    g_last_local[0] = g_exp_shared[63];
    #pragma unroll
    for (int i_42 = 0; i_42 < 32; ++i_42) {
      h_fragment[i_42] = (h_fragment[i_42] * g_last_local[0]);
    }
    {
      half_t A_local_5[16];
      half_t B_local_5[4];
      for (int ki_5 = 0; ki_5 < 16; ++ki_5) {
        for (int i_43 = 0; i_43 < 4; ++i_43) {
          *(uint2*)(A_local_5 + (i_43 * 4)) = *(uint2*)(((half_t*)kt_shared) + ((((((ki_5 * 512) + (((((int)threadIdx.x) & 63) >> 5) * 256)) + (i_43 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((ki_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) + (i_43 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki_5 & 3) >> 1)) & 1) * 4)));
        }
        *(uint2*)(B_local_5 + 0) = *(uint2*)(((half_t*)vn_shared) + ((((ki_5 * 160) + ((((int)threadIdx.x) & 3) * 40)) + ((((int)threadIdx.x) >> 6) * 16)) + (((((int)threadIdx.x) & 31) >> 3) * 4)));
        for (int i_44 = 0; i_44 < 4; ++i_44) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(h_fragment + (i_44 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + (i_44 * 4)), reinterpret_cast<const unsigned*>(B_local_5 + 0));
        }
      }
    }
  }
  if ((bool)need_store_final_state) {
    if (0 <= raw_batch_idx) {
      #pragma unroll
      for (int i_45 = 0; i_45 < 32; ++i_45) {
        if (raw_batch_idx < raw_batch_size) {
          ht[(int64_t)raw_batch_idx * ((int64_t) 128 * 128 * Hv) + blk * 4096 + ((int64_t)((int)threadIdx.x) >> 6) * 2048 + ((((int64_t)i_45 & 7) >> 2) * 1024) + ((((int64_t)((int)threadIdx.x) & 15) >> 3) * 512) + ((((int64_t)((int)threadIdx.x) & 3) >> 1) * 256) + ((int64_t)i_45 & 1) * 128 + ((((int64_t)((int)threadIdx.x) & 63) >> 5) * 64) + ((int64_t)i_45 >> 3) * 16 + ((((int64_t)((int)threadIdx.x) & 7) >> 2) * 8) + ((((int64_t)((int)threadIdx.x) & 31) >> 4) * 4) + ((((int64_t)i_45 & 3) >> 1) * 2) + ((int64_t)((int)threadIdx.x) & 1)] = h_fragment[i_45];
        }
      }
    }
  }
}
}

using namespace gdn_chunk_sm70;

constexpr int GDN_CHUNK_SM70_D     = 128;
constexpr int GDN_CHUNK_SM70_CHUNK = 64;
constexpr int GDN_CHUNK_SM70_WARP  = 32;

constexpr int GDN_CHUNK_SM70_SMEM_CUMSUM_HV12 = 6400;
constexpr int GDN_CHUNK_SM70_SMEM_KKT = 16384;
constexpr int GDN_CHUNK_SM70_SMEM_FWD = 91072;

__global__ void gdn_chunk_meta_kernel(
        int * __restrict__ cu_seqlens, int * __restrict__ chunk_offsets, int * __restrict__ chunk_indices,
        const int n_tokens, const int num_chunks) {
    if (threadIdx.x == 0) {
        cu_seqlens[0]    = 0;
        cu_seqlens[1]    = n_tokens;
        chunk_offsets[0] = 0;
        chunk_offsets[1] = num_chunks;
    }
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        chunk_indices[2*i + 0] = 0;
        chunk_indices[2*i + 1] = i;
    }
}

__global__ void gdn_chunk_prep_kernel(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        half_t * __restrict__ q16, half_t * __restrict__ k16, half_t * __restrict__ v16,
        const int64_t sq1, const int64_t sq2,
        const int64_t sk1, const int64_t sk2,
        const int64_t sv1, const int64_t sv2,
        const int n, const bool l2norm, const bool standalone_rms_order,
        const float eps_q, const float scale_q, const float eps_k, const float scale_k,
        const int H_k, const int H_v) {
    const int64_t n_q = (int64_t) n * H_k;
    const int64_t n_k = n_q + (int64_t) n * H_k;

    const int64_t warp_id = (int64_t) blockIdx.x * (blockDim.x / GDN_CHUNK_SM70_WARP) + threadIdx.x / GDN_CHUNK_SM70_WARP;
    if (warp_id >= n_k + (int64_t) n * H_v) {
        return;
    }
    const int lane = threadIdx.x % GDN_CHUNK_SM70_WARP;

    const float * src;
    half_t      * dst;
    float         eps   = 0.0f;
    float         scale = 0.0f;
    if (warp_id < n_q) {
        const int64_t t = warp_id / H_k;
        const int64_t h = warp_id % H_k;
        src   = q + t * sq2 + h * sq1;
        dst   = q16 + warp_id * GDN_CHUNK_SM70_D;
        eps   = eps_q;
        scale = scale_q;
    } else if (warp_id < n_k) {
        const int64_t row = warp_id - n_q;
        const int64_t t   = row / H_k;
        const int64_t h   = row % H_k;
        src   = k + t * sk2 + h * sk1;
        dst   = k16 + row * GDN_CHUNK_SM70_D;
        eps   = eps_k;
        scale = scale_k;
    } else {
        const int64_t row = warp_id - n_k;
        const int64_t t   = row / H_v;
        const int64_t h   = row % H_v;
        src = v + t * sv2 + h * sv1;
        dst = v16 + row * GDN_CHUNK_SM70_D;
    }

    float x[GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP];
#pragma unroll
    for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
        x[r] = src[r * GDN_CHUNK_SM70_WARP + lane];
    }
    if (l2norm && warp_id < n_k) {
        if (standalone_rms_order) {
            static_assert(GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP == 4, "exact 4-warp rms_norm_f32<256> reduction tree");
            const float w0 = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(__fmul_rn(x[0], x[0]));
            const float w1 = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(__fmul_rn(x[1], x[1]));
            const float w2 = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(__fmul_rn(x[2], x[2]));
            const float w3 = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(__fmul_rn(x[3], x[3]));
            const float ss = __fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3));
            const float nrm = rsqrtf(__fadd_rn(ss / GDN_CHUNK_SM70_D, eps));
#pragma unroll
            for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
                x[r] = __fmul_rn(scale, __fmul_rn(nrm, x[r]));
            }
        } else {
            float ss = 0.0f;
#pragma unroll
            for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
                ss += x[r] * x[r];
            }
            ss = warp_reduce_sum<GDN_CHUNK_SM70_WARP>(ss);
            const float nrm = rsqrtf(ss / GDN_CHUNK_SM70_D + eps) * scale;
#pragma unroll
            for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
                x[r] *= nrm;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < GDN_CHUNK_SM70_D / GDN_CHUNK_SM70_WARP; r++) {
        dst[r * GDN_CHUNK_SM70_WARP + lane] = half_t(x[r]);
    }
}

__global__ void gdn_chunk_store_kernel(const half2 * __restrict__ src, float2 * __restrict__ dst, const int64_t n2) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        dst[i] = __half22float2(src[i]);
    }
}

template <int HV, int HK>
static void gdn_chunk_launch_kkt_fwd(half_t * a, const ggml_cuda_gdn_chunk_args & args, int * chunk_indices, int * chunk_offsets,
        int * cu_seqlens, float * gcum, half_t * k16, half_t * o16, half_t * q16, half_t * v16,
        const int num_chunks, const int n, const int H, const int Hk, cudaStream_t stream) {
    tilelang_kkt_solve_kernel_kernel<HV, HK><<<num_chunks * H, 128, GDN_CHUNK_SM70_SMEM_KKT, stream>>>(
        a, args.beta, chunk_indices, cu_seqlens, k16, num_chunks, n, 1, H, Hk);
    CUDA_CHECK(cudaGetLastError());

    if (!o16) {
        CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_direct_kernel<HV, HK>), GDN_CHUNK_SM70_SMEM_FWD);
        gdn_chunk_direct_kernel<HV, HK><<<H * 4, 128, GDN_CHUNK_SM70_SMEM_FWD, stream>>>(
            a, args.beta, chunk_offsets, cu_seqlens, gcum, args.s0, args.h_out,
            k16, args.dst, q16, v16, 1, n, 1, H, Hk);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((tilelang_fused_chunk_gdr_fwd_kernel_kernel<HV, HK>), GDN_CHUNK_SM70_SMEM_FWD);
        tilelang_fused_chunk_gdr_fwd_kernel_kernel<HV, HK><<<H * 4, 128, GDN_CHUNK_SM70_SMEM_FWD, stream>>>(
            a, args.beta, chunk_offsets, cu_seqlens, gcum, args.s0, args.h_out,
            k16, o16, q16, v16, 1, n, 1, H, Hk);
    }
    CUDA_CHECK(cudaGetLastError());
}

int64_t ggml_cuda_gdn_chunk_sm70_n_tokens(int cc, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3, bool kda, int K) {
    if (cc != GGML_CUDA_CC_VOLTA || kda || S_v != GDN_CHUNK_SM70_D || H < 1 || H_k < 1 || H % H_k != 0 ||
            n_seqs != 1 || rq3 != 1) {
        return 0;
    }
    GGML_ASSERT(K >= 1 && K <= GDN_CHUNK_SM70_CHUNK);

    int64_t tail = n_tokens % GDN_CHUNK_SM70_CHUNK;
    if (tail < K) {
        tail += GDN_CHUNK_SM70_CHUNK;
    }
    const int64_t n_c = n_tokens - tail;
    return n_c >= GDN_CHUNK_SM70_CHUNK ? n_c : 0;
}

void ggml_cuda_gdn_chunk_sm70(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & args) {
    const int64_t n = args.n_tokens;
    GGML_ASSERT(n > 0 && n % GDN_CHUNK_SM70_CHUNK == 0 && n <= INT32_MAX);

    const int H  = (int) args.H;
    const int Hk = (int) args.H_k;
    GGML_ASSERT(H >= 1 && Hk >= 1 && H % Hk == 0 && H <= INT32_MAX && Hk <= INT32_MAX);

    const int num_chunks = (int) (n / GDN_CHUNK_SM70_CHUNK);
    cudaStream_t stream  = ctx.stream();

    ggml_cuda_pool & pool = ctx.pool();
    const size_t n_qk = (size_t) n * Hk * GDN_CHUNK_SM70_D;
    const size_t n_v  = (size_t) n * H * GDN_CHUNK_SM70_D;

    ggml_cuda_pool_alloc<half_t> q16(pool, n_qk);
    ggml_cuda_pool_alloc<half_t> k16(pool, n_qk);
    ggml_cuda_pool_alloc<half_t> v16(pool, n_v);
    ggml_cuda_pool_alloc<float>  gcum(pool, (size_t) n * H);
    ggml_cuda_pool_alloc<half_t> a(pool, (size_t) n * H * GDN_CHUNK_SM70_CHUNK);
    static const bool direct = true;
    ggml_cuda_pool_alloc<half_t> o16(pool);
    if (!direct) { o16.alloc(n_v); }
    ggml_cuda_pool_alloc<int>    meta(pool, 4 + 2 * (size_t) num_chunks);

    int * cu_seqlens    = meta.get();
    int * chunk_offsets = cu_seqlens + 2;
    int * chunk_indices = chunk_offsets + 2;

    gdn_chunk_meta_kernel<<<1, 128, 0, stream>>>(cu_seqlens, chunk_offsets, chunk_indices, (int) n, num_chunks);
    CUDA_CHECK(cudaGetLastError());

    const int64_t n_rows  = n * (Hk + Hk + H);
    const unsigned int n_warps = (unsigned int) ((n_rows + 3) / 4);
    gdn_chunk_prep_kernel<<<n_warps, 128, 0, stream>>>(
        args.q, args.k, args.v, q16.get(), k16.get(), v16.get(),
        args.sq1, args.sq2, args.sk1, args.sk2, args.sv1, args.sv2,
        (int) n, args.l2norm, args.standalone_rms_order, args.eps_q, args.scale_q, args.eps_k, args.scale_k, Hk, H);
    CUDA_CHECK(cudaGetLastError());

    if (H == 12 && Hk == 4) {

        tilelang_chunk_local_cumsum_kernel_kernel<<<num_chunks, 128, GDN_CHUNK_SM70_SMEM_CUMSUM_HV12, stream>>>(
            chunk_indices, cu_seqlens, gcum.get(), args.g, num_chunks, (int) n, 1);
    } else {
        gdn_chunk_cumsum_kernel<<<num_chunks, 128, 0, stream>>>(
            chunk_indices, cu_seqlens, gcum.get(), args.g, (int) n, 1, H);
    }
    CUDA_CHECK(cudaGetLastError());

    if (H == 12 && Hk == 4) {
        gdn_chunk_launch_kkt_fwd<12, 4>(a.get(), args, chunk_indices, chunk_offsets, cu_seqlens, gcum.get(),
            k16.get(), o16.get(), q16.get(), v16.get(), num_chunks, (int) n, H, Hk, stream);
    } else {
        gdn_chunk_launch_kkt_fwd<0, 0>(a.get(), args, chunk_indices, chunk_offsets, cu_seqlens, gcum.get(),
            k16.get(), o16.get(), q16.get(), v16.get(), num_chunks, (int) n, H, Hk, stream);
    }

    if (direct) { return; }
    const int64_t n2 = n * H * (GDN_CHUNK_SM70_D / 2);
    const unsigned int n_blocks = (unsigned int) ((n2 + 255) / 256);
    gdn_chunk_store_kernel<<<n_blocks, 256, 0, stream>>>(
        reinterpret_cast<const half2 *>(o16.get()), reinterpret_cast<float2 *>(args.dst), n2);
    CUDA_CHECK(cudaGetLastError());
}

#endif
