// Copyright (c) 2025, D.Skryabin. Portions from ai-bond/flash-attention-v100 (BSD-3-Clause) and 1Cat-vLLM (Apache-2.0); see licenses/.

#pragma once

#include "common.cuh"

#if defined(VOLTA_MMA_AVAILABLE)
namespace ggml_sm70_wmma {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <>
struct fragment<matrix_a, 16, 16, 16, half, row_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, col_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, row_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};

template <>
struct fragment<accumulator, 16, 16, 16, float> {
  float x[8];
  static constexpr int num_elements = 8;
};

template <int M, int N, int K>
__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, M, N, K, float>& frag, float value) {
#pragma unroll
  for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, half, row_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, row_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.b.sync.aligned.row.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, half, col_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr, const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m16n16k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  } else {
    asm volatile(
        "wmma.store.d.sync.aligned.col.m16n16k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  }
}

#define VOLTA_WMMA_MMA_F32(M, N, K, ALAY, BLAY)                            \
  __device__ __forceinline__ void mma_sync(                                \
      fragment<accumulator, M, N, K, float>& d,                            \
      const fragment<matrix_a, M, N, K, half, ALAY##_major>& a,            \
      const fragment<matrix_b, M, N, K, half, BLAY##_major>& b,            \
      const fragment<accumulator, M, N, K, float>& c) {                    \
    asm volatile(                                                          \
        "wmma.mma.sync.aligned." #ALAY "." #BLAY ".m" #M "n" #N "k" #K     \
        ".f32.f32 "                                                        \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "                                      \
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "                                \
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "                              \
        "{%24,%25,%26,%27,%28,%29,%30,%31};"                               \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),          \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])           \
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(a.x[4]), \
          "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]), "r"(b.x[0]), "r"(b.x[1]), \
          "r"(b.x[2]), "r"(b.x[3]), "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), \
          "r"(b.x[7]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));             \
  }

VOLTA_WMMA_MMA_F32(16, 16, 16, row, col)
VOLTA_WMMA_MMA_F32(16, 16, 16, row, row)

#undef VOLTA_WMMA_MMA_F32

}
#endif

constexpr float kXQANegInf = -1.0e30f;

constexpr int kGroupedVerifyHeadDim      = 256;
constexpr int kGroupedVerifyRows         = 48;
constexpr int kGroupedVerifyBlockN       = 64;
constexpr int kGroupedVerifyQStride      = 264;
constexpr int kGroupedVerifyKVStride     = 264;
constexpr int kGroupedVerifyScoreStride  = 68;
constexpr int kGroupedVerifyProbStride   = 72;
constexpr int kGroupedVerifyKVQ8RowBytes = 272;
constexpr int kGroupedVerifyKVQ4RowBytes = 144;
constexpr int kGroupedVerifyThreads      = 512;
constexpr int kGroupedVerifyWarps        = kGroupedVerifyThreads / WARP_SIZE;
constexpr int kGroupedVerifyQKWarps      = (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles  = (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp = kGroupedVerifyOutputTiles / kGroupedVerifyWarps;
constexpr int kGroupedVerifyRowsPerWarp  = kGroupedVerifyRows / kGroupedVerifyWarps;

constexpr int kGroupedVerifyColsPerLane  = kGroupedVerifyBlockN / WARP_SIZE;

constexpr int kGroupedVerifyStageRowsPerWarp = kGroupedVerifyBlockN / kGroupedVerifyWarps;

static_assert(kGroupedVerifyBlockN % 16 == 0, "the KV tile must be a multiple of the WMMA N");
static_assert(kGroupedVerifyQKWarps <= kGroupedVerifyWarps, "QK tiles must fit into the warps");
static_assert(kGroupedVerifyColsPerLane == 2, "the softmax handles two adjacent columns per lane");
static_assert(kGroupedVerifyRows % kGroupedVerifyWarps == 0, "softmax rows must divide evenly");
static_assert(kGroupedVerifyWarps == kGroupedVerifyHeadDim / 16, "one warp per V output tile");
static_assert(kGroupedVerifyOutputTilesPerWarp == kGroupedVerifyRows / 16, "one accumulator per M tile");
static_assert(kGroupedVerifyBlockN % kGroupedVerifyWarps == 0, "staging rows must divide evenly");

template <int MAX_QUERY_TOKENS, int HEADS>
struct GroupedVerifyTraits {
    static_assert(MAX_QUERY_TOKENS == 8 || MAX_QUERY_TOKENS == 16, "grouped verify supports 8 and 16 query tokens");
    static_assert(HEADS == 2 || HEADS == 4 || HEADS == 6, "grouped verify supports GQA 2, 4 and 6");
    static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
    static constexpr int kHeadGroups  = (HEADS + kHeadsPerCta - 1) / kHeadsPerCta;
    static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows, "the CTA must keep 48 rows");
};

struct alignas(256) GroupedVerifySmem {
    union {
        struct {
            alignas(16) __half  q[kGroupedVerifyRows * kGroupedVerifyQStride];
            alignas(16) __half  kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
            alignas(16) float   scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
            alignas(16) __half  probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
            alignas(16) uint8_t kv_stage[kGroupedVerifyBlockN * kGroupedVerifyKVQ8RowBytes];
        } compute;
        alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
    } storage;
    alignas(16) float row_max[kGroupedVerifyRows];
    alignas(16) float row_sum[kGroupedVerifyRows];
    alignas(16) float row_scale[kGroupedVerifyRows];
};

static_assert(sizeof(GroupedVerifySmem) <= 96 * 1024, "grouped verify must fit Volta's 96 KiB opt-in budget");
static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0, "output tiles must divide evenly across warps");

#if defined(VOLTA_MMA_AVAILABLE)

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <ggml_type type_KV>
struct GroupedVerifyKVRegs {
    static constexpr int kRowBytes  = type_KV == GGML_TYPE_F16  ? kGroupedVerifyHeadDim * (int) sizeof(__half)
                                     : type_KV == GGML_TYPE_Q4_0 ? kGroupedVerifyKVQ4RowBytes
                                                                 : kGroupedVerifyKVQ8RowBytes;
    static constexpr int kVecsPerRow  = kRowBytes / 16;
    static constexpr int kVecsPerWarp = kGroupedVerifyStageRowsPerWarp * kVecsPerRow;
    static constexpr int kVecsPerThread = (kVecsPerWarp + WARP_SIZE - 1) / WARP_SIZE;
    static_assert(kVecsPerThread <= 4, "the prefetch must stay small");
    uint4 vec[kVecsPerThread];
};

__device__ __forceinline__ __half2 grouped_verify_q8_pair_half2(const uint32_t packed, const __half2 d2) {
    const __half2 q = __halves2half2(
        __short2half_rn((int8_t) (uint8_t)  packed),
        __short2half_rn((int8_t) (uint8_t) (packed >> 8)));
    return __hmul2(d2, q);
}

__device__ __forceinline__ void grouped_verify_q4_quad_half2(
        const uint32_t packed, const bool hi_half, const __half2 d2, __half2 out[2]) {
    const uint32_t nib = hi_half ? ((packed >> 4) & 0x0F0F0F0F) : (packed & 0x0F0F0F0F);
    const unsigned bias_bits = 0x64086408u;
    const __half2 bias = *reinterpret_cast<const __half2 *>(&bias_bits);
#pragma unroll
    for (int i = 0; i < 2; ++i) {

        const unsigned packed_pair = __byte_perm(nib, 0, 0x4140u + 0x0202u*i) | 0x64006400u;
        out[i] = __hmul2(d2, __hsub2(*reinterpret_cast<const __half2 *>(&packed_pair), bias));
    }
}

__device__ __forceinline__ uint32_t grouped_verify_half2_uint(const __half2 h) {
    uint32_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_prefetch_kv(
        GroupedVerifyKVRegs<type_KV> & regs,
        const char * __restrict__ KV, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int seq, const int kv_head, const int tile_start, const int n_kv) {
    static_assert(type_KV == GGML_TYPE_F16 || type_KV == GGML_TYPE_Q8_0 || type_KV == GGML_TYPE_Q4_0,
                  "unsupported KV type");
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerWarp = GroupedVerifyKVRegs<type_KV>::kVecsPerWarp;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
    const char * tile_base = KV + int64_t(tile_start)*nb11 + kv_head*nb12 + int64_t(seq)*nb13;
#pragma unroll
    for (int i = 0; i < GroupedVerifyKVRegs<type_KV>::kVecsPerThread; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        if (idx >= kVecsPerWarp) {
            continue;
        }
        const int row     = row_base + idx / kVecsPerRow;
        const int vec_col = idx % kVecsPerRow;
        if (tile_start + row < n_kv) {
            regs.vec[i] = __ldg(reinterpret_cast<const uint4 *>(tile_base + row*nb11 + vec_col*16));
        } else {
            regs.vec[i] = make_uint4(0, 0, 0, 0);
        }
    }
}

template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_store_kv(
        __half * shared_kv, uint8_t * kv_stage, const GroupedVerifyKVRegs<type_KV> & regs) {
    constexpr int kVecsPerRow  = GroupedVerifyKVRegs<type_KV>::kVecsPerRow;
    constexpr int kVecsPerWarp = GroupedVerifyKVRegs<type_KV>::kVecsPerWarp;
    constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
#pragma unroll
    for (int i = 0; i < GroupedVerifyKVRegs<type_KV>::kVecsPerThread; ++i) {
        const int idx = lane_id + i * WARP_SIZE;
        if (idx >= kVecsPerWarp) {
            continue;
        }
        const int row     = row_base + idx / kVecsPerRow;
        const int vec_col = idx % kVecsPerRow;
        if constexpr (type_KV == GGML_TYPE_F16) {
            reinterpret_cast<uint4 *>(shared_kv)[row * kSharedStrideVec + vec_col] = regs.vec[i];
        } else {
            reinterpret_cast<uint4 *>(kv_stage)[row * kVecsPerRow + vec_col] = regs.vec[i];
        }
    }
}

template <ggml_type type_KV>
__device__ __forceinline__ void flash_attn_sm70_grouped_dequant_kv(
        __half * shared_kv, const uint8_t * kv_stage) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        return;
    }
    constexpr int kColsPerItem = 8;
    constexpr int kGroupsPerRow = kGroupedVerifyHeadDim / kColsPerItem;
    constexpr int kItemsPerWarp = kGroupedVerifyStageRowsPerWarp * kGroupsPerRow;
    constexpr int kBlockBytes   = type_KV == GGML_TYPE_Q4_0 ? 18 : 34;
    constexpr int kRowBytes     = type_KV == GGML_TYPE_Q4_0 ? kGroupedVerifyKVQ4RowBytes
                                                            : kGroupedVerifyKVQ8RowBytes;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int row_base = warp_id * kGroupedVerifyStageRowsPerWarp;
#pragma unroll
    for (int idx = lane_id; idx < kItemsPerWarp; idx += WARP_SIZE) {
        const int row = row_base + idx / kGroupsPerRow;
        const int c   = (idx % kGroupsPerRow) * kColsPerItem;
        const int blk = c / 32;
        const int base = row * kRowBytes + blk * kBlockBytes;
        const __half d = *reinterpret_cast<const __half *>(kv_stage + base);
        const __half2 d2 = __half2half2(d);
        uint4 out;
        if constexpr (type_KV == GGML_TYPE_Q4_0) {

            const uint8_t * codes = kv_stage + base + 2 + (c & 15);
            const uint32_t w0 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes)
                              | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 2) << 16;
            const uint32_t w1 = (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 4)
                              | (uint32_t) *reinterpret_cast<const uint16_t *>(codes + 6) << 16;
            const bool hi_half = (c % 32) >= 16;
            __half2 q4_pairs[2];
            grouped_verify_q4_quad_half2(w0, hi_half, d2, q4_pairs);
            out.x = grouped_verify_half2_uint(q4_pairs[0]);
            out.y = grouped_verify_half2_uint(q4_pairs[1]);
            grouped_verify_q4_quad_half2(w1, hi_half, d2, q4_pairs);
            out.z = grouped_verify_half2_uint(q4_pairs[0]);
            out.w = grouped_verify_half2_uint(q4_pairs[1]);
        } else {

            const uint16_t * packed = reinterpret_cast<const uint16_t *>(kv_stage + base + 2 + (c % 32));
            out.x = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[0], d2));
            out.y = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[1], d2));
            out.z = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[2], d2));
            out.w = grouped_verify_half2_uint(grouped_verify_q8_pair_half2(packed[3], d2));
        }
        __half * out_ptr = reinterpret_cast<__half *>(shared_kv + row * kGroupedVerifyKVStride) + c;
        *reinterpret_cast<uint4 *>(out_ptr) = out;
    }
}

__device__ __forceinline__ void grouped_verify_qk(
        const __half * __restrict__ shared_q, const __half * __restrict__ shared_k,
        float * __restrict__ shared_scores, const float qk_scale) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id >= kGroupedVerifyQKWarps) {
        return;
    }
    const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
    const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_a, 16, 16, 16, half, ggml_sm70_wmma::row_major> q_fragment;
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_b, 16, 16, 16, half, ggml_sm70_wmma::col_major> k_fragment;
    ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float> score_fragment;
    ggml_sm70_wmma::fill_fragment(score_fragment, 0.0f);

#pragma unroll
    for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
        ggml_sm70_wmma::load_matrix_sync(
            q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset, kGroupedVerifyQStride);

        ggml_sm70_wmma::load_matrix_sync(
            k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset, kGroupedVerifyKVStride);
        ggml_sm70_wmma::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
    }
#pragma unroll
    for (int i = 0; i < score_fragment.num_elements; ++i) {
        score_fragment.x[i] *= qk_scale;
    }
    ggml_sm70_wmma::store_matrix_sync(
        shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
        score_fragment, kGroupedVerifyScoreStride, ggml_sm70_wmma::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
        ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float>& fragment,
        const float * __restrict__ row_scale, const int tile_row_start) {
    const int lane = threadIdx.x & 31;
    const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
    const float first_scale  = row_scale[tile_row_start + row];
    const float second_scale = row_scale[tile_row_start + row + 2];
    fragment.x[0] *= first_scale;
    fragment.x[1] *= first_scale;
    fragment.x[2] *= second_scale;
    fragment.x[3] *= second_scale;
    fragment.x[4] *= first_scale;
    fragment.x[5] *= first_scale;
    fragment.x[6] *= second_scale;
    fragment.x[7] *= second_scale;
}

struct GroupedVerifyMaskPair {
    float value[2];
    bool  visible[2];
};

struct GroupedVerifyMaskRange {
    int lo0;
    int hi0;
    int lo1;
    int hi1;
};

__device__ __forceinline__ GroupedVerifyMaskRange grouped_verify_load_mask_range(
        const char * __restrict__ mask, const int64_t seq_mask_off, const int64_t nb31,
        const int token_idx, const int n_q) {
    GroupedVerifyMaskRange range = {0, 0, 0, 0};
    if (token_idx < n_q) {
        const int32_t * r = reinterpret_cast<const int32_t *>(mask + seq_mask_off + nb31 * token_idx);
        range.lo0 = r[0];
        range.hi0 = r[1];
        range.lo1 = r[2];
        range.hi1 = r[3];
    }
    return range;
}

__device__ __forceinline__ GroupedVerifyMaskPair grouped_verify_load_mask_pair(
        const char * __restrict__ mask, const GroupedVerifyMaskRange range,
        const int64_t seq_mask_off, const int64_t nb31,
        const int token_idx, const int n_q, const int kv_idx, const int n_kv,
        const bool mask_is_range) {
    GroupedVerifyMaskPair pair;
    pair.value[0]   = 0.0f;
    pair.value[1]   = 0.0f;
    pair.visible[0] = false;
    pair.visible[1] = false;
    if (token_idx >= n_q || kv_idx >= n_kv) {
        return pair;
    }
    if (mask == nullptr) {
        pair.visible[0] = true;
        pair.visible[1] = kv_idx + 1 < n_kv;
        return pair;
    }
    if (mask_is_range) {
        pair.visible[0] = (kv_idx >= range.lo0 && kv_idx < range.hi0) || (kv_idx >= range.lo1 && kv_idx < range.hi1);
        pair.visible[1] = kv_idx + 1 < n_kv &&
            ((kv_idx + 1 >= range.lo0 && kv_idx + 1 < range.hi0) || (kv_idx + 1 >= range.lo1 && kv_idx + 1 < range.hi1));
        return pair;
    }
    const __half * mask_row = reinterpret_cast<const __half *>(mask + seq_mask_off + nb31 * token_idx);
    pair.value[0] = __half2float(mask_row[kv_idx]);
    pair.visible[0] = pair.value[0] > -INFINITY;
    if (kv_idx + 1 < n_kv) {
        pair.value[1] = __half2float(mask_row[kv_idx + 1]);
        pair.visible[1] = pair.value[1] > -INFINITY;
    }
    return pair;
}

__device__ __forceinline__ void grouped_verify_softmax_tile(
        const float * __restrict__ shared_scores, __half * __restrict__ shared_probs,
        float * __restrict__ row_max, float * __restrict__ row_sum, float * __restrict__ row_scale,
        const GroupedVerifyMaskPair * __restrict__ row_mask, const int warp_id, const int lane_id) {
    const int col = kGroupedVerifyColsPerLane * lane_id;
    float score0[kGroupedVerifyRowsPerWarp];
    float score1[kGroupedVerifyRowsPerWarp];
    float tile_max[kGroupedVerifyRowsPerWarp];
    float old_max[kGroupedVerifyRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
        const int row = warp_id + i * kGroupedVerifyWarps;
        const float2 score_pair = *reinterpret_cast<const float2 *>(
            shared_scores + row * kGroupedVerifyScoreStride + col);
        score0[i]  = row_mask[i].visible[0] ? score_pair.x + row_mask[i].value[0] : kXQANegInf;
        score1[i]  = row_mask[i].visible[1] ? score_pair.y + row_mask[i].value[1] : kXQANegInf;
        tile_max[i] = fmaxf(score0[i], score1[i]);
        old_max[i]  = row_max[row];
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            tile_max[i] = fmaxf(tile_max[i], __shfl_down_sync(0xffffffffu, tile_max[i], offset));
        }
    }
    float new_max[kGroupedVerifyRowsPerWarp];
    float tile_sum[kGroupedVerifyRowsPerWarp];
    float probability0[kGroupedVerifyRowsPerWarp];
    float probability1[kGroupedVerifyRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
        tile_max[i] = __shfl_sync(0xffffffffu, tile_max[i], 0);
        new_max[i]  = fmaxf(old_max[i], tile_max[i]);
        probability0[i] = row_mask[i].visible[0] ? __expf(fmaxf(score0[i] - new_max[i], -80.0f)) : 0.0f;
        probability1[i] = row_mask[i].visible[1] ? __expf(fmaxf(score1[i] - new_max[i], -80.0f)) : 0.0f;
        tile_sum[i] = probability0[i] + probability1[i];
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            tile_sum[i] += __shfl_down_sync(0xffffffffu, tile_sum[i], offset);
        }
    }
    float exp_diff[kGroupedVerifyRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
        const int row = warp_id + i * kGroupedVerifyWarps;
        tile_sum[i] = __shfl_sync(0xffffffffu, tile_sum[i], 0);
        exp_diff[i] = tile_sum[i] > 0.0f ? __expf(fmaxf(old_max[i] - new_max[i], -80.0f)) : 1.0f;
        *reinterpret_cast<__half2 *>(shared_probs + row * kGroupedVerifyProbStride + col) =
            __floats2half2_rn(probability0[i], probability1[i]);
    }

    __syncwarp();
    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            const int row = warp_id + i * kGroupedVerifyWarps;
            if (tile_sum[i] > 0.0f) {
                row_sum[row] = row_sum[row] * exp_diff[i] + tile_sum[i];
                row_max[row] = new_max[i];
            }
            row_scale[row] = exp_diff[i];
        }
    }
}

#endif

template <int MAX_QUERY_TOKENS, int HEADS, bool NONUNIFORM, ggml_type type_K, ggml_type type_V>
__launch_bounds__(kGroupedVerifyThreads, 1)
static __global__ void flash_attn_ext_sm70_grouped(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        float      * dst_partial_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const int32_t n_q,
        const int32_t n_kv,
        const int32_t n_heads,
        const int32_t n_splits,
        const int32_t ne33,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33,
        const int32_t mask_is_range,
        const int32_t q_head_boundary) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(VOLTA_MMA_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q            = Q_ptr;
    const char * GGML_CUDA_RESTRICT K            = K_ptr;
    const char * GGML_CUDA_RESTRICT V            = V_ptr;
    const char * GGML_CUDA_RESTRICT mask         = mask_ptr;
    float       * GGML_CUDA_RESTRICT dst_partial = dst_partial_ptr;
    float2      * GGML_CUDA_RESTRICT dst_meta    = dst_meta_ptr;

    using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS, HEADS>;
    constexpr int kHeadsPerCta = Traits::kHeadsPerCta;
    constexpr int kHeadGroups  = Traits::kHeadGroups;

    const int kv_head    = blockIdx.x / kHeadGroups;
    const int head_group = blockIdx.x % kHeadGroups;
    const int split_id   = blockIdx.y;
    const int seq        = blockIdx.z;

    const int head_base  = NONUNIFORM ? (kv_head == 0 ? 0 : (int) q_head_boundary) : kv_head * HEADS;
    const int head_count = NONUNIFORM ? (kv_head == 0 ? (int) q_head_boundary : n_heads - (int) q_head_boundary) : HEADS;

    const int64_t seq_mask_off = mask ? nb33 * (seq % ne33) : 0;

    int kv_end = n_kv;
    if (mask_is_range) {
        int hi_max = 0;
        for (int t = 0; t < n_q; ++t) {
            const int32_t * r = reinterpret_cast<const int32_t *>(mask + seq_mask_off + nb31 * t);
            hi_max = max(hi_max, max(r[1], r[3]));
        }
        kv_end = min(n_kv, max(hi_max, 0));
    }

    const int total_tiles    = (kv_end + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
    const int base_tiles     = total_tiles / n_splits;
    const int extra_tiles    = total_tiles % n_splits;
    const int split_tile_start = split_id * base_tiles + min(split_id, extra_tiles);
    const int split_tiles    = base_tiles + (split_id < extra_tiles ? 1 : 0);
    const int split_start    = split_tile_start * kGroupedVerifyBlockN;
    const int split_end      = min(kv_end, split_start + split_tiles * kGroupedVerifyBlockN);

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char grouped_verify_smem_raw[];
    GroupedVerifySmem& smem = *reinterpret_cast<GroupedVerifySmem*>(grouped_verify_smem_raw);
    __half * shared_q      = smem.storage.compute.q;
    __half * shared_kv     = smem.storage.compute.kv;
    float  * shared_scores = smem.storage.compute.scores;
    __half * shared_probs  = smem.storage.compute.probs;
    uint8_t * kv_stage     = smem.storage.compute.kv_stage;

    for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim; idx += kGroupedVerifyThreads) {
        const int row        = idx / kGroupedVerifyHeadDim;
        const int d          = idx % kGroupedVerifyHeadDim;
        const int token_idx  = row / kHeadsPerCta;
        const int local_head = row % kHeadsPerCta;
        const int head_index = head_group * kHeadsPerCta + local_head;
        const bool head_used = NONUNIFORM ? head_index < head_count
                                          : (HEADS % kHeadsPerCta == 0 || head_index < HEADS);
        const int head = head_base + head_index;
        __half * dst = shared_q + row * kGroupedVerifyQStride + d;
        if (!head_used) {
            *dst = __float2half_rn(0.0f);
            continue;
        }
        if (token_idx < n_q) {
            const float val = *(const float *) (Q + token_idx*nb01 + head*nb02 + int64_t(seq)*nb03 + d*sizeof(float));
            *dst = __float2half_rn(val);
        } else {
            *dst = __float2half_rn(0.0f);
        }
    }
    if (tid < kGroupedVerifyRows) {
        smem.row_max[tid]   = kXQANegInf;
        smem.row_sum[tid]   = 0.0f;
        smem.row_scale[tid] = 1.0f;
    }
    __syncthreads();

    ggml_sm70_wmma::fragment<ggml_sm70_wmma::accumulator, 16, 16, 16, float>
        output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        ggml_sm70_wmma::fill_fragment(output_fragments[fragment_idx], 0.0f);
    }

    GroupedVerifyKVRegs<type_K> k_regs;
    GroupedVerifyKVRegs<type_V> v_regs;
    if (split_start < split_end) {
        flash_attn_sm70_grouped_prefetch_kv<type_K>(
            k_regs, K, nb11, nb12, nb13, seq, kv_head, split_start, n_kv);
        flash_attn_sm70_grouped_prefetch_kv<type_V>(
            v_regs, V, nb21, nb22, nb23, seq, kv_head, split_start, n_kv);
    }

    GroupedVerifyMaskRange row_range[kGroupedVerifyRowsPerWarp] = {};
    if (mask_is_range) {
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            const int row = warp_id + i * kGroupedVerifyWarps;
            row_range[i] = grouped_verify_load_mask_range(mask, seq_mask_off, nb31, row / kHeadsPerCta, n_q);
        }
    }

    for (int tile_start = split_start; tile_start < split_end; tile_start += kGroupedVerifyBlockN) {
        const bool has_next = tile_start + kGroupedVerifyBlockN < split_end;

        if constexpr (type_K == GGML_TYPE_F16) {
            __syncthreads();
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs);
            __syncthreads();
        } else {
            flash_attn_sm70_grouped_store_kv<type_K>(shared_kv, kv_stage, k_regs);
            __syncthreads();
            flash_attn_sm70_grouped_dequant_kv<type_K>(shared_kv, kv_stage);
            __syncthreads();
        }

        if (has_next) {
            flash_attn_sm70_grouped_prefetch_kv<type_K>(
                k_regs, K, nb11, nb12, nb13, seq, kv_head, tile_start + kGroupedVerifyBlockN, n_kv);
        }
        if constexpr (type_V != GGML_TYPE_F16) {

            flash_attn_sm70_grouped_store_kv<type_V>(shared_kv, kv_stage, v_regs);
            if (has_next) {
                flash_attn_sm70_grouped_prefetch_kv<type_V>(
                    v_regs, V, nb21, nb22, nb23, seq, kv_head, tile_start + kGroupedVerifyBlockN, n_kv);
            }
        }

        GroupedVerifyMaskPair row_mask[kGroupedVerifyRowsPerWarp];
#pragma unroll
        for (int i = 0; i < kGroupedVerifyRowsPerWarp; ++i) {
            const int row = warp_id + i * kGroupedVerifyWarps;
            row_mask[i] = grouped_verify_load_mask_pair(
                mask, row_range[i], seq_mask_off, nb31, row / kHeadsPerCta, n_q,
                tile_start + kGroupedVerifyColsPerLane * lane_id, n_kv,
                mask_is_range != 0);
        }

        grouped_verify_qk(shared_q, shared_kv, shared_scores, scale);
        __syncthreads();

        if constexpr (type_V == GGML_TYPE_F16) {

            flash_attn_sm70_grouped_store_kv<type_V>(shared_kv, kv_stage, v_regs);
            if (has_next) {
                flash_attn_sm70_grouped_prefetch_kv<type_V>(
                    v_regs, V, nb21, nb22, nb23, seq, kv_head, tile_start + kGroupedVerifyBlockN, n_kv);
            }
        }

        grouped_verify_softmax_tile(
            shared_scores, shared_probs, smem.row_max, smem.row_sum, smem.row_scale, row_mask, warp_id, lane_id);
        if constexpr (type_V != GGML_TYPE_F16) {
            flash_attn_sm70_grouped_dequant_kv<type_V>(shared_kv, kv_stage);
        }

        __syncthreads();

#pragma unroll
        for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
            grouped_verify_scale_output_fragment(output_fragments[fragment_idx], smem.row_scale, fragment_idx * 16);
        }

#pragma unroll
        for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
            ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_b, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                value_fragment;
            ggml_sm70_wmma::load_matrix_sync(
                value_fragment,
                shared_kv + k_offset * kGroupedVerifyKVStride + warp_id * 16,
                kGroupedVerifyKVStride);
#pragma unroll
            for (int m_tile = 0; m_tile < kGroupedVerifyOutputTilesPerWarp; ++m_tile) {
                ggml_sm70_wmma::fragment<ggml_sm70_wmma::matrix_a, 16, 16, 16, half, ggml_sm70_wmma::row_major>
                    probability_fragment;
                ggml_sm70_wmma::load_matrix_sync(
                    probability_fragment,
                    shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
                    kGroupedVerifyProbStride);
                ggml_sm70_wmma::mma_sync(
                    output_fragments[m_tile], probability_fragment, value_fragment, output_fragments[m_tile]);
            }
        }
    }

    __syncthreads();
    float* shared_output = smem.storage.output;
#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        ggml_sm70_wmma::store_matrix_sync(
            shared_output + fragment_idx * 16 * kGroupedVerifyHeadDim + warp_id * 16,
            output_fragments[fragment_idx], kGroupedVerifyHeadDim, ggml_sm70_wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim; idx += kGroupedVerifyThreads) {
        const int row        = idx / kGroupedVerifyHeadDim;
        const int d          = idx % kGroupedVerifyHeadDim;
        const int token_idx  = row / kHeadsPerCta;
        const int local_head = row % kHeadsPerCta;
        const int head_index = head_group * kHeadsPerCta + local_head;
        const bool head_used = NONUNIFORM ? head_index < head_count
                                          : (HEADS % kHeadsPerCta == 0 || head_index < HEADS);
        if (token_idx >= n_q || !head_used) {
            continue;
        }
        const int head = head_base + head_index;
        const int64_t j = (int64_t(seq) * n_q + token_idx) * n_heads + head;

        dst_partial[(j * n_splits + split_id) * kGroupedVerifyHeadDim + d] = shared_output[idx];
    }
    if (tid < kGroupedVerifyRows) {
        const int token_idx  = tid / kHeadsPerCta;
        const int local_head = tid % kHeadsPerCta;
        const int head_index = head_group * kHeadsPerCta + local_head;
        const bool head_filled = NONUNIFORM ? head_index < head_count
                                            : (HEADS % kHeadsPerCta == 0 || head_index < HEADS);
        if (token_idx < n_q && head_filled) {
            const int head = head_base + head_index;
            const int64_t j = (int64_t(seq) * n_q + token_idx) * n_heads + head;
            const float sum = smem.row_sum[tid];
            dst_meta[j * n_splits + split_id] = make_float2(sum > 0.0f ? smem.row_max[tid] : kXQANegInf, sum);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, dst_partial_ptr, dst_meta_ptr, scale,
        n_q, n_kv, n_heads, n_splits, ne33,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        nb31, nb33, q_head_boundary);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_flash_attn_ext_sm70_grouped_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_flash_attn_ext_sm70_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
