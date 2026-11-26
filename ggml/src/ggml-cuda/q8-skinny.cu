// Portions from 1Cat-vLLM, v100-skinny and haohervchb/sglang-V100; see licenses/LICENSE-1Cat-vLLM and licenses/LICENSE-v100-skinny.

#include "common.cuh"
#include "convert.cuh"
#include "mmvf-narrow.cuh"
#include "q-skinny-common.cuh"
#include "q8-skinny.cuh"
#include "unary.cuh"

#include <algorithm>
#include <vector>

static char q8_skinny_marker = 0;

static int q8_skinny_split_k(const int64_t k) {
    if (k <= 0) {
        return 0;
    }
    for (int split_k : {16, 8, 4}) {
        if ((k / 16) % split_k == 0) {
            return split_k;
        }
    }
    return 0;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

__device__ __forceinline__ void s8x8_to_half2x4(uint2 q, half2 out[4]) {
    const unsigned x = q.x ^ 0x80808080u;
    const unsigned y = q.y ^ 0x80808080u;
    const unsigned bias_bits = 0x64806480u;
    const half2 bias = *reinterpret_cast<const half2 *>(&bias_bits);
#pragma unroll
    for (int i = 0; i < 4; ++i) {

        const unsigned packed = (__byte_perm(x, y, 0x0400u + 0x0101u*i) & 0x00FF00FFu) | 0x64006400u;
        out[i] = __hsub2(*reinterpret_cast<const half2 *>(&packed), bias);
    }
}

#define Q8_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                       \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__global__ void q8_skinny_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ codes,
                                        half * __restrict__ scales, int n, int k, int tile0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    const int blocks_k = k >> 5;
    const int tile = tile0 + blockIdx.x;
    uint8_t * codes_tile = codes + (size_t) blockIdx.x * 32 * k;
    for (int i = threadIdx.x; i < 32 * blocks_k; i += blockDim.x) {
        const int row = i / blocks_k;
        const int kb  = i - row * blocks_k;
        const uint8_t * block = src + ((size_t) blockIdx.x * 32 + row) * blocks_k * 34 + (size_t) kb * 34;
        const half d = *reinterpret_cast<const half *>(block);
        const uint8_t * qs = block + 2;
        const int lane   = qpn8_lane_from_col(row & 31);
        const int group0 = kb * 2;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int physical_k = qpn8_physical_k(j);
            codes_tile[((size_t) (group0 + 0) * 32 + lane) * 16 + physical_k] = qs[j];
            codes_tile[((size_t) (group0 + 1) * 32 + lane) * 16 + physical_k] = qs[16 + j];
        }
        scales[(size_t) kb * n + tile * 32 + lane] = d;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(src, codes, scales, n, k, tile0);
#endif
}

__global__ void q8_skinny_to_f16_kernel(half * __restrict__ output, const uint8_t * __restrict__ codes,
                                        const half * __restrict__ scales, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

    __shared__ __align__(8) half2 tile_smem[32][8 * 8 + 4];

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int tile   = blockIdx.y;
    const int groups = k >> 4;
    const int g0     = blockIdx.x * 8;
    const int group  = g0 + warp;

    if (group < groups) {
        const uint4 packed =
            reinterpret_cast<const uint4 *>(codes)[((size_t) tile * groups + group) * 32 + lane];
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half scale   = __ldg(scales + (size_t) (group >> 1) * n + tile * 32 + lane);
        const half2 scale2 = __halves2half2(scale, scale);
        half2 * smem_row = tile_smem[qpn8_col_from_lane(lane)];
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            smem_row[warp * 8 + pair] = __hmul2(weights[pair], scale2);
        }
    }
    __syncthreads();

    const int groups_here = groups - g0 < 8 ? groups - g0 : 8;
    half * out_tile = output + (size_t) tile * 32 * k + g0 * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = warp * 4 + i;
        if (2 * lane + 1 < groups_here * 8) {
            const uint2 value = *reinterpret_cast<const uint2 *>(&tile_smem[row][2 * lane]);
            *reinterpret_cast<uint2 *>(out_tile + (size_t) row * k + 4 * lane) = value;
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, scales, n, k);
#endif
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ scales,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_scale_group = -1;
    half loaded_scale = __float2half(0.0f);
    uint4 prefetched = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
        prefetched = __ldcs(code_ptr + (size_t) group_begin * 32);
    }

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed =
            PrefetchCodes ? prefetched
                          : __ldcs(code_ptr + (size_t) group * 32);
        uint4 next = make_uint4(0, 0, 0, 0);
        if constexpr (PrefetchCodes) {
            if (group + 1 < group_begin + groups_per_warp) {
                next = __ldcs(code_ptr + (size_t) (group + 1) * 32);
            }
        }
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weights[i] = __hmul2(weights[i], scale2);
        }

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
        }
        if constexpr (PrefetchCodes) {
            prefetched = next;
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, scales, input, output, n, k, m);
#endif
}

template <int SplitK, int NAcc, bool M1Only, int RowTiles, int HC>
__global__ void __launch_bounds__(32 * SplitK * HC) q8_skinny_hc_up_pre_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ scales,
    const half * __restrict__ input, float * __restrict__ gate,
    int n, int k, int m,
    const float * __restrict__ xn, float * __restrict__ mixed, half * __restrict__ mixed_f16, float pre_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny M=1 specialization uses one row tile");
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    __shared__ float partials[HC][SplitK][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp_all = threadIdx.x >> 5;
    const int stream = warp_all / SplitK;
    const int warp = warp_all - stream * SplitK;
    const int n_embd = n / HC;
    const int tile = stream * (n_embd / 32) + blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_scale_group = -1;
    half loaded_scale = __float2half(0.0f);

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed = __ldcs(code_ptr + (size_t) group * 32);
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weights[i] = __hmul2(weights[i], scale2);
        }

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[stream][warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[stream][warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < HC * kOutputElements; e += blockDim.x) {
        const int s       = e / kOutputElements;
        const int element = e - s * kOutputElements;
        const int tile_s  = s * (n_embd / 32) + blockIdx.x;
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[s][k_warp][element];
        }
        if constexpr (M1Only) {
            gate[tile_s * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                gate[(size_t) output_row * n + tile_s * 32 + output_col] = value;
            }
        }
        partials[s][0][element] = value;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < 32 * m; e += blockDim.x) {
        const int t = e >> 5;
        const int d = blockIdx.x * 32 + (e & 31);
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float xv = xn[(size_t) t * n + c * n_embd + d];
            const float wv = 1.0f / (1.0f + expf(-partials[c][0][M1Only ? (e & 31) : e]));
            sum += xv * wv;
        }
        const float value = pre_scale * sum;
        mixed[(size_t) t * n_embd + d] = value;
        if (mixed_f16 != nullptr) {
            mixed_f16[(size_t) t * n_embd + d] = __float2half(value);
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, scales, input, gate, n, k, m, xn, mixed, mixed_f16, pre_scale);
#endif
}

struct q8_skinny_multi_seg {
    const uint8_t * codes;
    const half *    scales;
    int             n;
    float *         dst;
    int             first_tile;
};

struct q8_skinny_multi_dot {
    const uint8_t * rows;
    int             n_rows;
    float *         dst;
    int             first_row;
};

struct q8_skinny_multi_params {
    q8_skinny_multi_seg seg[4];
    q8_skinny_multi_dot dot[4];
    int n_seg;
    int n_dot;
    int n_main_tiles;
    int n_dot_rows;
    const half * input;
    int k;
    int m;
};

template <int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_multi_kernel(const q8_skinny_multi_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny multi supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny multi M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if ((int) blockIdx.x >= p.n_main_tiles) {

        const int row_all = (int) blockIdx.x - p.n_main_tiles;
        int seg = 0;
        while (seg + 1 < p.n_dot && row_all >= p.dot[seg + 1].first_row) {
            ++seg;
        }
        const int row = row_all - p.dot[seg].first_row;
        const block_q8_0 * wrow = reinterpret_cast<const block_q8_0 *>(p.dot[seg].rows) + (size_t) row * (p.k / QK8_0);

        float accum[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            accum[t] = 0.0f;
        }
#pragma unroll 2
        for (int i = threadIdx.x; i < p.k; i += blockDim.x) {
            const block_q8_0 * block = wrow + i / QK8_0;
            const float weight = __half2float(block->d) * (float) block->qs[i % QK8_0];
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (t < p.m) {
                    accum[t] = fmaf(weight, __half2float(p.input[(size_t) t * p.k + i]), accum[t]);
                }
            }
        }

        float * red = &partials[0][0];
        const int n_warps = blockDim.x >> 5;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t < p.m) {
                float v = accum[t];
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_down_sync(0xffffffffU, v, offset);
                }
                if (lane == 0) {
                    red[warp * 16 + t] = v;
                }
            }
        }
        __syncthreads();
        if ((int) threadIdx.x < p.m) {
            float v = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                v += red[w * 16 + threadIdx.x];
            }
            p.dot[seg].dst[(size_t) threadIdx.x * p.dot[seg].n_rows + row] = v;
        }
        return;
    }

    int seg = 0;
    while (seg + 1 < p.n_seg && (int) blockIdx.x >= p.seg[seg + 1].first_tile) {
        ++seg;
    }
    const int n = p.seg[seg].n;
    const int k = p.k;
    const int m = p.m;
    const int tile = (int) blockIdx.x - p.seg[seg].first_tile;
    const uint8_t * codes = p.seg[seg].codes;
    const half * scales = p.seg[seg].scales;
    const half * input = p.input;
    float * output = p.seg[seg].dst;

    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_scale_group = -1;
    half loaded_scale = __float2half(0.0f);

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed = __ldcs(code_ptr + (size_t) group * 32);
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weights[i] = __hmul2(weights[i], scale2);
        }

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED(p);
#endif
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only = false, int RowTiles = 1>
__global__ void q8_skinny_gated_kernel(
    const uint8_t * __restrict__ gate_codes, const half * __restrict__ gate_scales,
    const uint8_t * __restrict__ up_codes, const half * __restrict__ up_scales,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q8 skinny gated pair supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q8 skinny gated M=1 specialization uses one row tile");
    __shared__ float partials[2][SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int projection = warp_in_block / SplitK;
    const int warp = warp_in_block - projection * SplitK;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const uint8_t * codes = projection ? up_codes : gate_codes;
    const half * scales = projection ? up_scales : gate_scales;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_scale_group = -1;
    half loaded_scale = __float2half(0.0f);
    uint4 prefetched = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
        prefetched = __ldcs(code_ptr + (size_t) group_begin * 32);
    }

#pragma unroll 4
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const int scale_group = group >> 1;
        if (scale_group != loaded_scale_group) {
            loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
            loaded_scale_group = scale_group;
        }

        const uint4 packed =
            PrefetchCodes ? prefetched
                          : __ldcs(code_ptr + (size_t) group * 32);
        uint4 next = make_uint4(0, 0, 0, 0);
        if constexpr (PrefetchCodes) {
            if (group + 1 < group_begin + groups_per_warp) {
                next = __ldcs(code_ptr + (size_t) (group + 1) * 32);
            }
        }
        half2 weights[8];
        s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
        s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

        const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            weights[i] = __hmul2(weights[i], scale2);
        }

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
        }
        if constexpr (PrefetchCodes) {
            prefetched = next;
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[projection][warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[projection][warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            gate += partials[0][k_warp][element];
            up += partials[1][k_warp][element];
        }
        const float silu = gate / (1.0f + __expf(-gate));
        if constexpr (M1Only) {
            output[tile * 32 + element] = silu * up;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = silu * up;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, gate_scales, up_codes, up_scales, input, output, n, k, m);
#endif
}

struct q8_skinny_config {
    int  split_k;
    int  n_acc;
    bool prefetch;
};

static q8_skinny_config q8_skinny_config_for(const int64_t n, const int64_t k) {
    if (k == 4352 || k == 1536) {
        return {16, 1, false};
    }
    if (k == 5120) {
        if (n == 4352) {

            return {16, 1, true};
        }
        return {16, 2, false};
    }
    return {q8_skinny_split_k(k), k >= 4096 ? 2 : 1, false};
}

static q8_skinny_config q8_skinny_multi_config_for(const int64_t k) {
    if (k == 4352 || k == 1536) {
        return {16, 1, false};
    }
    if (k == 5120) {
        return {16, 2, false};
    }
    return {q8_skinny_split_k(k), k >= 4096 ? 2 : 1, false};
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool M1Only, int RowTiles>
static void q8_skinny_launch(const uint8_t * codes, const half * scales, const half * input,
                             float * output, int n, int k, int m, cudaStream_t stream) {
    q8_skinny_kernel<SplitK, NAcc, PrefetchCodes, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, scales, input, output, n, k, m);
}

#define Q8_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                            \
    do {                                                                                    \
        switch (config.split_k) {                                                           \
            case 16: q8_skinny_launch<16, NAcc, false, M1Only, RowTiles>(codes, scales,     \
                         input, output, n, k, m, stream); break;                            \
            case 8:  q8_skinny_launch<8, NAcc, false, M1Only, RowTiles>(codes, scales,      \
                         input, output, n, k, m, stream); break;                            \
            default: q8_skinny_launch<4, NAcc, false, M1Only, RowTiles>(codes, scales,      \
                         input, output, n, k, m, stream); break;                            \
        }                                                                                   \
    } while (0)

static void q8_skinny_mul_mat_launch(const uint8_t * codes, const half * scales, const half * input,
                                     float * output, int n, int k, int m, cudaStream_t stream) {
    const q8_skinny_config config = q8_skinny_config_for(n, k);
    if (config.prefetch) {

        GGML_ASSERT(config.split_k == 16 && config.n_acc == 1);
        if (m == 1) {
            q8_skinny_launch<16, 1, true, true, 1>(codes, scales, input, output, n, k, m, stream);
        } else if (m <= 8) {
            q8_skinny_launch<16, 1, true, false, 1>(codes, scales, input, output, n, k, m, stream);
        } else {
            q8_skinny_launch<16, 1, true, false, 2>(codes, scales, input, output, n, k, m, stream);
        }
    } else if (config.n_acc == 2) {
        if (m == 1) { Q8_SKINNY_LAUNCH(2, true, 1); } else if (m <= 8) { Q8_SKINNY_LAUNCH(2, false, 1); }
        else { Q8_SKINNY_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q8_SKINNY_LAUNCH(1, true, 1); } else if (m <= 8) { Q8_SKINNY_LAUNCH(1, false, 1); }
        else { Q8_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q8_SKINNY_LAUNCH

template <int SplitK>
__global__ void q8_skinny_m32_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ scales,
    const half * __restrict__ input, float * __restrict__ output,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(SplitK == 12 || SplitK == 16,
                  "Q8 skinny M=32 two-phase kernel supports split-12 or split-16");
    constexpr int kPhysicalWarps = SplitK / 2;
    constexpr int kRowTiles = 4;
    constexpr int kOutputElements = kRowTiles * 256;
    __shared__ float reduction_storage[kPhysicalWarps + 1][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const uint4 * code_ptr = reinterpret_cast<const uint4 *>(codes) +
                             (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr = scales + tile * 32 + lane;

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        float accum[kRowTiles][2][8];
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
                for (int index = 0; index < 8; ++index) {
                    accum[row_tile][chain][index] = 0.0f;
                }
            }
        }

        const int logical_warp = warp + phase * kPhysicalWarps;
        const int group_begin = logical_warp * groups_per_warp;
        int loaded_scale_group = -1;
        half loaded_scale = __float2half(0.0f);
#pragma unroll 4
        for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
            const int scale_group = group >> 1;
            if (scale_group != loaded_scale_group) {
                loaded_scale = __ldg(scale_ptr + (size_t) scale_group * n);
                loaded_scale_group = scale_group;
            }

            const uint4 packed = __ldcs(code_ptr + (size_t) group * 32);
            half2 weights[8];
            s8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
            s8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);

            const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
            for (int index = 0; index < 8; ++index) {
                weights[index] = __hmul2(weights[index], scale2);
            }

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
                uint4 input01 = make_uint4(0, 0, 0, 0);
                uint4 input23 = make_uint4(0, 0, 0, 0);
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                    input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                }
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
            }
        }

#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int index = 0; index < 8; ++index) {
                accum[row_tile][0][index] += accum[row_tile][1][index];
                const int output_row =
                    row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
                reduction_storage[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][index];
            }
        }
        __syncthreads();

        for (int element = threadIdx.x; element < kOutputElements;
             element += blockDim.x) {
            float value = phase == 0 ? 0.0f : reduction_storage[kPhysicalWarps][element];
#pragma unroll
            for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
                value += reduction_storage[k_warp][element];
            }
            if (phase == 0) {
                reduction_storage[kPhysicalWarps][element] = value;
            } else {
                const int output_row = element >> 5;
                const int output_col = element & 31;
                if (output_row < m) {
                    output[(size_t) output_row * n + tile * 32 + output_col] = value;
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, scales, input, output, n, k, m);
#endif
}

static void q8_skinny_m32_mul_mat_launch(const uint8_t * codes, const half * scales,
                                         const half * input, float * output, int n, int k, int m,
                                         cudaStream_t stream) {
    constexpr int kSplitK = 16;
    q8_skinny_m32_kernel<kSplitK><<<n / 32, 32 * (kSplitK / 2), 0, stream>>>(
        codes, scales, input, output, n, k, m);
    CUDA_CHECK(cudaGetLastError());
}

static bool q8_skinny_mul_mat_dispatch(const uint8_t * codes, const half * scales,
                                       const half * input, float * output, int n, int k, int m,
                                       cudaStream_t stream) {
    GGML_ASSERT(m >= 1 && m <= 64);
    if (m <= 16) {
        q8_skinny_mul_mat_launch(codes, scales, input, output, n, k, m, stream);
        return true;
    }
    if (q8_skinny_config_for(n, k).split_k != 16) {
        return false;
    }
    if (m <= 32) {
        q8_skinny_m32_mul_mat_launch(codes, scales, input, output, n, k, m, stream);
    } else {
        q8_skinny_m32_mul_mat_launch(codes, scales, input, output, n, k, 32, stream);
        q8_skinny_m32_mul_mat_launch(codes, scales, input + (size_t) 32 * k,
                                     output + (size_t) 32 * n, n, k, m - 32, stream);
    }
    return true;
}

template <bool M1Only, int RowTiles>
static void q8_skinny_gated_launch(const uint8_t * gate_codes, const half * gate_scales,
                                   const uint8_t * up_codes, const half * up_scales,
                                   const half * input, float * output, int n, int k, int m,
                                   cudaStream_t stream) {
    q8_skinny_gated_kernel<8, 2, true, M1Only, RowTiles><<<n / 32, 64 * 8, 0, stream>>>(
        gate_codes, gate_scales, up_codes, up_scales, input, output, n, k, m);
}

static void q8_skinny_gated_mul_mat_launch(const uint8_t * gate_codes, const half * gate_scales,
                                           const uint8_t * up_codes, const half * up_scales,
                                           const half * input, float * output, int n, int k, int m,
                                           cudaStream_t stream) {
    if (m == 1) {
        q8_skinny_gated_launch<true, 1>(gate_codes, gate_scales, up_codes, up_scales,
                                        input, output, n, k, m, stream);
    } else if (m <= 8) {
        q8_skinny_gated_launch<false, 1>(gate_codes, gate_scales, up_codes, up_scales,
                                         input, output, n, k, m, stream);
    } else {
        q8_skinny_gated_launch<false, 2>(gate_codes, gate_scales, up_codes, up_scales,
                                         input, output, n, k, m, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

__global__ void q8_skinny_swiglu_kernel(float * __restrict__ gate, const float * __restrict__ up,
                                        int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + __expf(-g)) * up[i];
    }
}

template <int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q8_skinny_multi_launch(const q8_skinny_multi_params & p, cudaStream_t stream) {

    const int n_tiles = p.n_main_tiles + p.n_dot_rows;
    q8_skinny_multi_kernel<SplitK, NAcc, M1Only, RowTiles><<<n_tiles, 32 * SplitK, 0, stream>>>(p);
}

#define Q8_SKINNY_MULTI_LAUNCH(NAcc, M1Only, RowTiles)                                      \
    do {                                                                                    \
        switch (config.split_k) {                                                           \
            case 16: q8_skinny_multi_launch<16, NAcc, M1Only, RowTiles>(p, stream); break;  \
            case 8:  q8_skinny_multi_launch<8, NAcc, M1Only, RowTiles>(p, stream); break;   \
            default: q8_skinny_multi_launch<4, NAcc, M1Only, RowTiles>(p, stream); break;   \
        }                                                                                   \
    } while (0)

static void q8_skinny_multi_mul_mat_launch(const q8_skinny_multi_params & p, cudaStream_t stream) {
    const q8_skinny_config config = q8_skinny_multi_config_for(p.k);
    GGML_ASSERT(!config.prefetch);
    if (config.n_acc == 2) {
        if (p.m == 1) { Q8_SKINNY_MULTI_LAUNCH(2, true, 1); } else if (p.m <= 8) { Q8_SKINNY_MULTI_LAUNCH(2, false, 1); }
        else { Q8_SKINNY_MULTI_LAUNCH(2, false, 2); }
    } else {
        if (p.m == 1) { Q8_SKINNY_MULTI_LAUNCH(1, true, 1); } else if (p.m <= 8) { Q8_SKINNY_MULTI_LAUNCH(1, false, 1); }
        else { Q8_SKINNY_MULTI_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q8_SKINNY_MULTI_LAUNCH

template <int NAcc, int RowTiles, int Depth, int InDepth>
static __device__ __forceinline__ void q8_skinny_warp_loop_ahead(
        const uint4 * __restrict__ code_ptr, const half * __restrict__ scale_ptr, const int n,
        const half * __restrict__ input, const int k, const int m, const int row,
        const int group_begin, const int group_end, float (&accum)[RowTiles][NAcc][8]) {
    static_assert(Depth >= 1 && InDepth >= 1 && Depth % InDepth == 0,
                  "Q8 skinny: the input depth must divide the code depth");

    const unsigned short * scale_bits = reinterpret_cast<const unsigned short *>(scale_ptr);
    const int group_last = group_end - 1;

    uint4        codes[Depth];
    unsigned int scales[Depth];
    uint4        in01[InDepth][RowTiles];
    uint4        in23[InDepth][RowTiles];

#pragma unroll
    for (int slot = 0; slot < Depth; ++slot) {
        const int group = min(group_begin + slot, group_last);
        codes[slot]  = __ldcs(code_ptr + (size_t) group * 32);
        scales[slot] = __ldg(scale_bits + (size_t) (group >> 1) * n);
    }
#pragma unroll
    for (int slot = 0; slot < InDepth; ++slot) {
        const int group = min(group_begin + slot, group_last);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            in01[slot][row_tile] = make_uint4(0, 0, 0, 0);
            in23[slot][row_tile] = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                in01[slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                in23[slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
            }
        }
    }

    int base = group_begin;
    for (; base + Depth <= group_end; base += Depth) {
#pragma unroll
        for (int slot = 0; slot < Depth; ++slot) {
            const int group = base + slot;

            const int in_slot     = slot % InDepth;
            const int group_code  = min(group + Depth, group_last);
            const int group_input = min(group + InDepth, group_last);

            half2 weights[8];
            s8x8_to_half2x4(make_uint2(codes[slot].x, codes[slot].y), weights);
            s8x8_to_half2x4(make_uint2(codes[slot].z, codes[slot].w), weights + 4);
            const half  scale  = __ushort_as_half((unsigned short) scales[slot]);
            const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                weights[i] = __hmul2(weights[i], scale2);
            }
            codes[slot]  = __ldcs(code_ptr + (size_t) group_code * 32);
            scales[slot] = __ldg(scale_bits + (size_t) (group_code >> 1) * n);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in01[in_slot][row_tile]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in23[in_slot][row_tile]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            }
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    in01[in_slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group_input * 16);
                    in23[in_slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group_input * 16 + 8);
                }
            }
        }
    }
    if (base < group_end) {
#pragma unroll
        for (int slot = 0; slot < Depth; ++slot) {
            const int group = base + slot;
            if (group >= group_end) {
                break;
            }

            const int in_slot     = slot % InDepth;
            const int group_code  = min(group + Depth, group_last);
            const int group_input = min(group + InDepth, group_last);

            half2 weights[8];
            s8x8_to_half2x4(make_uint2(codes[slot].x, codes[slot].y), weights);
            s8x8_to_half2x4(make_uint2(codes[slot].z, codes[slot].w), weights + 4);
            const half  scale  = __ushort_as_half((unsigned short) scales[slot]);
            const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                weights[i] = __hmul2(weights[i], scale2);
            }
            codes[slot]  = __ldcs(code_ptr + (size_t) group_code * 32);
            scales[slot] = __ldg(scale_bits + (size_t) (group_code >> 1) * n);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in01[in_slot][row_tile]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in23[in_slot][row_tile]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q8_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            }
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    in01[in_slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group_input * 16);
                    in23[in_slot][row_tile] = *reinterpret_cast<const uint4 *>(input_row + group_input * 16 + 8);
                }
            }
        }
    }
}

enum q8_skinny_epilogue {
    Q8_SKINNY_EPI_F32,
    Q8_SKINNY_EPI_SCALE_SILU_F16,
    Q8_SKINNY_EPI_F16,
};

template <int SplitK, int NAcc, int Parts, int Depth, int InDepth, bool Gated, bool M1Only, int RowTiles,
          int Epilogue = Q8_SKINNY_EPI_F32>
__global__ void __launch_bounds__((Gated ? 64 : 32) * (SplitK / Parts)) q8_skinny_split_kernel(
    const uint8_t * __restrict__ codes0, const half * __restrict__ scales0,
    const uint8_t * __restrict__ codes1, const half * __restrict__ scales1,
    const half * __restrict__ input, float * __restrict__ output,
    float * __restrict__ scratch, unsigned int * __restrict__ counters,
    int n, int k, int m, half * __restrict__ output_f16 = nullptr, float epi_scale = 0.0f,
    const mmvf_narrow_side_rows side = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static_assert(Parts > 1 && SplitK % Parts == 0, "Q8 skinny split: the parts must divide split-K");
    if constexpr (Epilogue == Q8_SKINNY_EPI_F16) {

        const int n_main_ctas = (n / 32) * Parts;
        if ((int) blockIdx.x >= n_main_ctas) {
            const int vwarp = ((int) blockIdx.x - n_main_ctas) * (int) (blockDim.x / WARP_SIZE) + (int) (threadIdx.x / WARP_SIZE);
            mmvf_narrow_side_rows_warp(side, vwarp, (int) (threadIdx.x % WARP_SIZE));
            return;
        }
    }
    static_assert(RowTiles == 1 || RowTiles == 2, "Q8 skinny split supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1, "Q8 skinny split M=1 specialization uses one row tile");
    constexpr int kWarps    = SplitK / Parts;
    constexpr int kProj     = Gated ? 2 : 1;
    constexpr int kElements = M1Only ? 32 : RowTiles * 256;
    __shared__ bool last_cta;

    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int projection    = warp_in_block / kWarps;
    const int part          = blockIdx.x % Parts;
    const int tile          = blockIdx.x / Parts;

    const int warp          = part * kWarps + (warp_in_block - projection * kWarps);
    const int quadpair      = (lane >> 2) & 3;
    const int row           = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16    = k >> 4;
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin   = warp * groups_per_warp;
    const uint8_t * codes   = projection ? codes1 : codes0;
    const half * scales     = projection ? scales1 : scales0;
    const uint4 * code_ptr  = reinterpret_cast<const uint4 *>(codes) + (size_t) tile * groups_k16 * 32 + lane;
    const half * scale_ptr  = scales + tile * 32 + lane;

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    q8_skinny_warp_loop_ahead<NAcc, RowTiles, Depth, InDepth>(code_ptr, scale_ptr, n, input, k, m, row,
                                                              group_begin, group_begin + groups_per_warp, accum);

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    float * warp_partials = scratch + ((size_t) (tile * kProj + projection) * SplitK + warp) * kElements;
    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    warp_partials[quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                warp_partials[output_row * 32 + quadpair * 8 + output_col] = accum[row_tile][0][i];
            }
        }
    }

    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last_cta = atomicAdd(counters + tile, 1u) == (unsigned int) (Parts - 1);
    }
    __syncthreads();
    if (!last_cta) {
        return;
    }
    __threadfence();
    const float * tile_partials = scratch + (size_t) tile * kProj * SplitK * kElements;
    auto partial = [&](const int index) {
        return __ldcg(tile_partials + index);
    };

    for (int element = threadIdx.x; element < kElements; element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partial(k_warp * kElements + element);
        }
        if constexpr (Gated) {
            float up = 0.0f;
#pragma unroll
            for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
                up += partial((SplitK + k_warp) * kElements + element);
            }
            const float gate = value;
            const float silu = gate / (1.0f + __expf(-gate));
            value = silu * up;
        }
        int out_index = tile * 32 + element;
        bool out_valid = true;
        if constexpr (!M1Only) {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            out_index = output_row * n + tile * 32 + output_col;
            out_valid = output_row < m;
        }
        if (out_valid) {
            if constexpr (Epilogue == Q8_SKINNY_EPI_SCALE_SILU_F16) {

                const float v = epi_scale * value;
                output_f16[out_index] = __float2half(ggml_cuda_op_silu_single(v));
            } else if constexpr (Epilogue == Q8_SKINNY_EPI_F16) {

                output_f16[out_index] = __float2half(value);
            } else {
                output[out_index] = value;
            }
        }
    }
    if (threadIdx.x == 0) {
        counters[tile] = 0;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes0, scales0, codes1, scales1, input, output, scratch, counters, n, k, m, output_f16, epi_scale, side);
#endif
}

static constexpr int q8_skinny_split_max_tiles = 32;

static int q8_skinny_split_parts(const int tiles, const int split_k) {
    int parts = 2;
    while (parts * 2 <= split_k && parts * 2 <= 8 && tiles * parts * 2 <= 80) {
        parts *= 2;
    }
    return parts;
}

static size_t q8_skinny_split_scratch_floats(const int n, const int split_k, const bool gated, const int m) {
    const int elements = m == 1 ? 32 : (m <= 8 ? 256 : 512);
    return (size_t) (n / 32) * (gated ? 2 : 1) * split_k * elements;
}

struct q8_skinny_split_out {
    float * output;
    half *  output_f16;
    float   epi_scale;
    int     epilogue;

    mmvf_narrow_side_rows side;
};

template <int SplitK, int NAcc, int Parts, bool Gated, bool M1Only, int RowTiles, int depth, int in_depth>
static void q8_skinny_split_launch_one_d(const uint8_t * codes0, const half * scales0, const uint8_t * codes1,
                                       const half * scales1, const half * input, const q8_skinny_split_out & out,
                                       float * scratch, unsigned int * counters, int n, int k, int m, cudaStream_t stream) {
    const dim3 grid((n / 32) * Parts);
    const dim3 block((Gated ? 64 : 32) * (SplitK / Parts));
    if (out.epilogue == Q8_SKINNY_EPI_SCALE_SILU_F16) {
        if constexpr (!Gated) {
            GGML_ASSERT(out.side.nrows == 0);
            q8_skinny_split_kernel<SplitK, NAcc, Parts, depth, in_depth, Gated, M1Only, RowTiles, Q8_SKINNY_EPI_SCALE_SILU_F16>
                <<<grid, block, 0, stream>>>(codes0, scales0, codes1, scales1, input, nullptr, scratch, counters, n, k, m,
                                             out.output_f16, out.epi_scale);
        } else {
            GGML_ABORT("Q8 skinny split: the gated pair has no scale/silu epilogue");
        }
    } else if (out.epilogue == Q8_SKINNY_EPI_F16) {
        if constexpr (Gated) {
            constexpr int warps_per_cta = 2 * (SplitK / Parts);
            const int side_ctas = (out.side.nrows * mmvf_narrow_split_warps + warps_per_cta - 1) / warps_per_cta;
            const dim3 grid_side(grid.x + side_ctas);
            q8_skinny_split_kernel<SplitK, NAcc, Parts, depth, in_depth, Gated, M1Only, RowTiles, Q8_SKINNY_EPI_F16>
                <<<grid_side, block, 0, stream>>>(codes0, scales0, codes1, scales1, input, nullptr, scratch, counters, n, k, m,
                                                  out.output_f16, 0.0f, out.side);
        } else {
            GGML_ABORT("Q8 skinny split: the F16 epilogue is only used by the gated pair");
        }
    } else {
        q8_skinny_split_kernel<SplitK, NAcc, Parts, depth, in_depth, Gated, M1Only, RowTiles>
            <<<grid, block, 0, stream>>>(codes0, scales0, codes1, scales1, input, out.output, scratch, counters, n, k, m);
    }
}

template <int SplitK, int NAcc, int Parts, bool Gated, bool M1Only, int RowTiles>
static void q8_skinny_split_launch_one(const uint8_t * codes0, const half * scales0, const uint8_t * codes1,
                                       const half * scales1, const half * input, const q8_skinny_split_out & out,
                                       float * scratch, unsigned int * counters, int n, int k, int m, cudaStream_t stream) {
    constexpr int depth    = RowTiles == 1 && !Gated ? 8 : 4;
    constexpr int in_depth = RowTiles == 1 ? 4 : 2;
    q8_skinny_split_launch_one_d<SplitK, NAcc, Parts, Gated, M1Only, RowTiles, depth, in_depth>(
        codes0, scales0, codes1, scales1, input, out, scratch, counters, n, k, m, stream);
}

template <int SplitK, int NAcc, bool Gated, bool M1Only, int RowTiles>
static void q8_skinny_split_launch_parts(int parts, const uint8_t * codes0, const half * scales0, const uint8_t * codes1,
                                         const half * scales1, const half * input, const q8_skinny_split_out & out,
                                         float * scratch, unsigned int * counters, int n, int k, int m, cudaStream_t stream) {
    switch (parts) {
        case 2:
            q8_skinny_split_launch_one<SplitK, NAcc, 2, Gated, M1Only, RowTiles>(
                codes0, scales0, codes1, scales1, input, out, scratch, counters, n, k, m, stream);
            break;
        case 4:
            q8_skinny_split_launch_one<SplitK, NAcc, 4, Gated, M1Only, RowTiles>(
                codes0, scales0, codes1, scales1, input, out, scratch, counters, n, k, m, stream);
            break;
        default:
            if constexpr (SplitK >= 8) {
                GGML_ASSERT(parts == 8);
                q8_skinny_split_launch_one<SplitK, NAcc, 8, Gated, M1Only, RowTiles>(
                    codes0, scales0, codes1, scales1, input, out, scratch, counters, n, k, m, stream);
            } else {
                GGML_ABORT("Q8 skinny split: %d parts for split-K %d", parts, SplitK);
            }
            break;
    }
}

template <int SplitK, int NAcc, bool Gated>
static void q8_skinny_split_launch_m(const uint8_t * codes0, const half * scales0, const uint8_t * codes1,
                                     const half * scales1, const half * input, const q8_skinny_split_out & out,
                                     float * scratch, unsigned int * counters, int n, int k, int m, cudaStream_t stream) {
    const int parts = q8_skinny_split_parts(n / 32, SplitK);
    if (m == 1) {
        q8_skinny_split_launch_parts<SplitK, NAcc, Gated, true, 1>(parts, codes0, scales0, codes1, scales1, input, out,
                                                                  scratch, counters, n, k, m, stream);
    } else if (m <= 8) {
        q8_skinny_split_launch_parts<SplitK, NAcc, Gated, false, 1>(parts, codes0, scales0, codes1, scales1, input, out,
                                                                   scratch, counters, n, k, m, stream);
    } else {
        q8_skinny_split_launch_parts<SplitK, NAcc, Gated, false, 2>(parts, codes0, scales0, codes1, scales1, input, out,
                                                                   scratch, counters, n, k, m, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void q8_skinny_split_launch_single(const q8_skinny_config & config, const uint8_t * codes, const half * scales,
                                          const half * input, const q8_skinny_split_out & out, float * scratch,
                                          unsigned int * counters, int n, int k, int m, cudaStream_t stream) {
    switch (config.split_k * 10 + config.n_acc) {
        case 162: q8_skinny_split_launch_m<16, 2, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        case 161: q8_skinny_split_launch_m<16, 1, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        case  82: q8_skinny_split_launch_m<8,  2, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        case  81: q8_skinny_split_launch_m<8,  1, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        case  42: q8_skinny_split_launch_m<4,  2, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        case  41: q8_skinny_split_launch_m<4,  1, false>(codes, scales, nullptr, nullptr, input, out, scratch, counters, n, k, m, stream); break;
        default: GGML_ABORT("Q8 skinny split: split-K %d with %d chains", config.split_k, config.n_acc);
    }
}

#endif

bool ggml_cuda_q8_skinny_is_repacked(const ggml_tensor * t) {
    return t->extra == (const void *) &q8_skinny_marker;
}

bool ggml_cuda_q8_skinny_can_repack(const ggml_tensor * t) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(t);
    return false;
#else
    if (t->type != GGML_TYPE_Q8_0 || t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
        return false;
    }
    if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ggml_cuda_get_device())) {
        return false;
    }
    if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (!ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];

    return k >= 32 && k % 32 == 0 && n % 32 == 0 && n >= 32 && q8_skinny_split_k(k) != 0;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    const int64_t ntiles = n / 32;
    GGML_ASSERT(k >= 32 && ntiles >= 1);
    const size_t tile_src_bytes = 34 * (size_t) k;
    const size_t tile_dst_bytes = 32 * (size_t) k;
    const size_t scales_bytes = (size_t) n * k / 16;

    constexpr size_t staging_max = 32ull * 1024 * 1024;
    size_t tiles_per_chunk = staging_max / tile_src_bytes;
    if (tiles_per_chunk < 1) {
        tiles_per_chunk = 1;
    }
    if (tiles_per_chunk > (size_t) ntiles) {
        tiles_per_chunk = (size_t) ntiles;
    }

    cudaStream_t stream = ctx.stream();
    uint8_t * data = (uint8_t *) t->data;

    void * staging = nullptr;
    void * scales_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, tiles_per_chunk * tile_src_bytes));
    CUDA_CHECK(cudaMalloc(&scales_tmp, scales_bytes));

    for (int64_t t0 = 0; t0 < ntiles; t0 += (int64_t) tiles_per_chunk) {
        const int64_t t1 = std::min(t0 + (int64_t) tiles_per_chunk, ntiles);
        CUDA_CHECK(cudaMemcpyAsync(staging, data + t0 * tile_src_bytes, (t1 - t0) * tile_src_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        q8_skinny_repack_kernel<<<(unsigned) (t1 - t0), 256, 0, stream>>>(
            (const uint8_t *) staging, data + t0 * tile_dst_bytes, (half *) scales_tmp,
            (int) n, (int) k, (int) t0);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(data + n * k, scales_tmp, scales_bytes, cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(scales_tmp));

    t->extra = &q8_skinny_marker;
}

void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {

    if (ctx.q8_skinny_idle_scans >= 1024) {
        return;
    }

    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        ggml_tensor * src0 = node->src[0];
        if (!ggml_cuda_q8_skinny_can_repack(src0) || ggml_cuda_q8_skinny_is_repacked(src0)) {
            continue;
        }
        if (std::find(candidates.begin(), candidates.end(), src0) == candidates.end()) {
            candidates.push_back(src0);
        }
    }
    if (candidates.empty()) {
        ++ctx.q8_skinny_idle_scans;
        return;
    }
    ctx.q8_skinny_idle_scans = 0;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes && !candidates.empty(); ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->view_src != nullptr) {
            candidates.erase(std::remove(candidates.begin(), candidates.end(), node->view_src),
                             candidates.end());
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            if (src->view_src != nullptr) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src->view_src),
                                 candidates.end());
            }
            if (node->op != GGML_OP_MUL_MAT || j != 0) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src),
                                 candidates.end());
            }
        }
    }

    for (ggml_tensor * t : candidates) {
        ggml_cuda_q8_skinny_repack_inplace(ctx, t);
    }
}

static bool q8_skinny_mul_mat_covers(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int split_k = q8_skinny_split_k(k);
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || split_k == 0 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) {
        return false;
    }
    if (m > 16 && q8_skinny_config_for(n, k).split_k != 16) {
        return false;
    }
    return true;
}

bool ggml_cuda_q8_skinny_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_cuda_q8_skinny_is_repacked(src0) && q8_skinny_mul_mat_covers(src0, src1, dst);
}

bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst, const half * input_f16) {
    if (!q8_skinny_mul_mat_covers(src0, src1, dst)) {
        return false;
    }
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];

    ggml_cuda_pool_alloc<half> input(ctx.pool());
    if (input_f16 == nullptr) {
        input.alloc(m * k);
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(src1->data, input.get(), m * k, ctx.stream());
        input_f16 = input.get();
    }

    const uint8_t * data = (const uint8_t *) src0->data;
    if (m <= 16 && n / 32 <= q8_skinny_split_max_tiles) {
        GGML_ASSERT(ctx.tile_counters != nullptr && n / 32 <= GGML_CUDA_MAX_SPLIT_TILES);
        const q8_skinny_config config = q8_skinny_config_for(n, k);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), q8_skinny_split_scratch_floats((int) n, config.split_k, false, (int) m));
        const q8_skinny_split_out out = { (float *) dst->data, nullptr, 0.0f, Q8_SKINNY_EPI_F32, {} };
        q8_skinny_split_launch_single(config, data, (const half *) (data + n * k), input_f16, out,
                                      scratch.get(), ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES,
                                      (int) n, (int) k, (int) m, ctx.stream());
        return true;
    }
    return q8_skinny_mul_mat_dispatch(data, (const half *) (data + n * k), input_f16,
                                      (float *) dst->data, (int) n, (int) k, (int) m,
                                      ctx.stream());
}

bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                       const ggml_tensor * up_w, const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    if (!ggml_cuda_q8_skinny_is_repacked(gate_w) || !ggml_cuda_q8_skinny_is_repacked(up_w)) {
        return false;
    }
    const int64_t k = gate_w->ne[0];
    const int64_t n = gate_w->ne[1];
    const int64_t m = src1->ne[1];
    if (!ggml_are_same_shape(gate_w, up_w) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            m < 1 || m > 64 ||
            k % 16 != 0 || (k / 16) % 8 != 0 || n % 32 != 0) {
        return false;
    }
    if (m > 16 && q8_skinny_config_for(n, k).split_k != 16) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    const uint8_t * gate_data = (const uint8_t *) gate_w->data;
    const uint8_t * up_data   = (const uint8_t *) up_w->data;
    if (m <= 16 && n / 32 <= q8_skinny_split_max_tiles) {
        GGML_ASSERT(ctx.tile_counters != nullptr && n / 32 <= GGML_CUDA_MAX_SPLIT_TILES);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), q8_skinny_split_scratch_floats((int) n, 8, true, (int) m));
        const q8_skinny_split_out out = { (float *) dst->data, nullptr, 0.0f, Q8_SKINNY_EPI_F32, {} };
        q8_skinny_split_launch_m<8, 2, true>(gate_data, (const half *) (gate_data + n * k), up_data,
                                             (const half *) (up_data + n * k), input.get(), out,
                                             scratch.get(), ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES,
                                             (int) n, (int) k, (int) m, ctx.stream());
        return true;
    }
    if (m <= 16) {
        q8_skinny_gated_mul_mat_launch(gate_data, (const half *) (gate_data + n * k),
                                       up_data, (const half *) (up_data + n * k),
                                       input.get(), (float *) dst->data, (int) n, (int) k, (int) m,
                                       ctx.stream());
        return true;
    }

    float * gate = (float *) dst->data;
    ggml_cuda_pool_alloc<float> up(ctx.pool(), n * m);
    const bool gate_ok = q8_skinny_mul_mat_dispatch(gate_data, (const half *) (gate_data + n * k),
                                                    input.get(), gate, (int) n, (int) k, (int) m,
                                                    ctx.stream());
    GGML_ASSERT(gate_ok);
    const bool up_ok = q8_skinny_mul_mat_dispatch(up_data, (const half *) (up_data + n * k),
                                                  input.get(), up.get(), (int) n, (int) k, (int) m,
                                                  ctx.stream());
    GGML_ASSERT(up_ok);

    const int count = (int) (n * m);
    q8_skinny_swiglu_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(gate, up.get(), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_q8_skinny_hc_down_up_supported(const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t m) {
    if (!ggml_cuda_q8_skinny_is_repacked(w_down) || !ggml_cuda_q8_skinny_is_repacked(w_up)) {
        return false;
    }
    const int64_t k_down = w_down->ne[0];
    const int64_t n_down = w_down->ne[1];
    return m >= 1 && m <= 16 && n_down / 32 <= q8_skinny_split_max_tiles && n_down / 32 <= GGML_CUDA_MAX_SPLIT_TILES &&
           q8_skinny_split_k(k_down) != 0 && w_up->ne[0] == n_down && q8_skinny_split_k(n_down) != 0;
}

static void q8_skinny_side_rows_set(mmvf_narrow_side_rows & side, ggml_cuda_pool_alloc<float> & side_scratch,
                                    unsigned int * counters, int first_counter, const ggml_tensor * w,
                                    const ggml_tensor * src1, ggml_tensor * dst, int64_t k, int64_t m) {
    const int64_t nrows = w->ne[1];
    GGML_ASSERT(w->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[0] == k && src1->ne[0] == k && src1->ne[1] == m && m >= 1 && m <= 4);
    GGML_ASSERT(nrows >= 1 && first_counter + nrows <= GGML_CUDA_MAX_SPLIT_TILES);
    side_scratch.alloc((size_t) nrows * mmvf_narrow_split_warps * m);
    side.x              = (const float *) w->data;
    side.y              = (const float *) src1->data;
    side.dst            = (float *) dst->data;
    side.scratch        = side_scratch.get();
    side.counters       = counters + first_counter;
    side.stride_row     = (int64_t) (w->nb[1] / sizeof(float));
    side.ncols2         = (int) (k / 2);
    side.stride_col_y2  = (int) (src1->nb[1] / sizeof(float) / 2);
    side.stride_col_dst = (int) (dst->nb[1] / sizeof(float));
    side.nrows          = (int) nrows;
    side.ncols_dst      = (int) m;
}

void ggml_cuda_q8_skinny_hc_down_up(ggml_backend_cuda_context & ctx, const ggml_tensor * w_down, const ggml_tensor * w_up,
                                    const half * input, int64_t m, float scale, float * gate,
                                    const float * pre_xn, float * pre_mixed, half * pre_mixed_f16, float pre_scale) {
    GGML_ASSERT(ggml_cuda_q8_skinny_hc_down_up_supported(w_down, w_up, m));
    GGML_ASSERT(ctx.tile_counters != nullptr);
    const int64_t k_down = w_down->ne[0];
    const int64_t n_down = w_down->ne[1];
    const int64_t n_up   = w_up->ne[1];
    unsigned int * counters = ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES;

    const q8_skinny_config config = q8_skinny_config_for(n_down, k_down);
    ggml_cuda_pool_alloc<half>  lo(ctx.pool(), m * n_down);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), q8_skinny_split_scratch_floats((int) n_down, config.split_k, false, (int) m));
    const uint8_t * down = (const uint8_t *) w_down->data;
    const q8_skinny_split_out out = { nullptr, lo.get(), scale, Q8_SKINNY_EPI_SCALE_SILU_F16, {} };
    q8_skinny_split_launch_single(config, down, (const half *) (down + n_down * k_down), input, out, scratch.get(),
                                  counters, (int) n_down, (int) k_down, (int) m, ctx.stream());

    if (pre_mixed != nullptr) {
        ggml_cuda_q8_skinny_hc_up_pre(ctx, w_up, lo.get(), m, gate, pre_xn, pre_mixed, pre_mixed_f16, pre_scale);
        return;
    }
    const uint8_t * up = (const uint8_t *) w_up->data;
    const bool ok = q8_skinny_mul_mat_dispatch(up, (const half *) (up + n_up * n_down), lo.get(), gate,
                                               (int) n_up, (int) n_down, (int) m, ctx.stream());
    GGML_ASSERT(ok);
}

bool ggml_cuda_q8_skinny_hc_up_pre_supported(const ggml_tensor * w_up, int64_t m, int64_t hc) {
    if (!ggml_cuda_q8_skinny_is_repacked(w_up) || hc != 4 || m < 1 || m > 16) {
        return false;
    }
    const int64_t k = w_up->ne[0];
    const int64_t n = w_up->ne[1];
    if (n % (hc * 32) != 0 || q8_skinny_split_k(k) == 0) {
        return false;
    }
    const q8_skinny_config config = q8_skinny_config_for(n, k);
    return config.split_k == 4 && config.n_acc == 1 && !config.prefetch;
}

void ggml_cuda_q8_skinny_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const half * lo, int64_t m,
                                   float * gate, const float * xn, float * mixed, half * mixed_f16, float pre_scale) {
    GGML_ASSERT(ggml_cuda_q8_skinny_hc_up_pre_supported(w_up, m, 4));
    constexpr int HC = 4;
    const int64_t k = w_up->ne[0];
    const int64_t n = w_up->ne[1];
    const uint8_t * codes  = (const uint8_t *) w_up->data;
    const half *    scales = (const half *) (codes + n * k);
    const dim3 grid((unsigned) (n / HC / 32));
    const dim3 block(32 * 4 * HC);
    if (m == 1) {
        q8_skinny_hc_up_pre_kernel<4, 1, true, 1, HC><<<grid, block, 0, ctx.stream()>>>(
            codes, scales, lo, gate, (int) n, (int) k, (int) m, xn, mixed, mixed_f16, pre_scale);
    } else if (m <= 8) {
        q8_skinny_hc_up_pre_kernel<4, 1, false, 1, HC><<<grid, block, 0, ctx.stream()>>>(
            codes, scales, lo, gate, (int) n, (int) k, (int) m, xn, mixed, mixed_f16, pre_scale);
    } else {
        q8_skinny_hc_up_pre_kernel<4, 1, false, 2, HC><<<grid, block, 0, ctx.stream()>>>(
            codes, scales, lo, gate, (int) n, (int) k, (int) m, xn, mixed, mixed_f16, pre_scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_q8_skinny_gated_down(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                    const ggml_tensor * src1, const ggml_tensor * down_w, ggml_tensor * dst,
                                    const ggml_tensor * side_w, ggml_tensor * side_dst) {
    if (!ggml_cuda_q8_skinny_is_repacked(gate_w) || !ggml_cuda_q8_skinny_is_repacked(up_w) ||
            !ggml_cuda_q8_skinny_is_repacked(down_w)) {
        return false;
    }
    const int64_t k = gate_w->ne[0];
    const int64_t n = gate_w->ne[1];
    const int64_t m = src1->ne[1];
    const int64_t n_down = down_w->ne[1];

    if (!ggml_are_same_shape(gate_w, up_w) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != n_down || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            m < 1 || m > 16 || k % 16 != 0 || (k / 16) % 8 != 0 || n % 32 != 0 ||
            n / 32 > q8_skinny_split_max_tiles || n / 32 > GGML_CUDA_MAX_SPLIT_TILES ||
            down_w->ne[0] != n || q8_skinny_split_k(n) == 0) {
        return false;
    }
    GGML_ASSERT(ctx.tile_counters != nullptr);

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    unsigned int * counters = ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES;
    ggml_cuda_pool_alloc<half>  act(ctx.pool(), m * n);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), q8_skinny_split_scratch_floats((int) n, 8, true, (int) m));
    const uint8_t * gate_data = (const uint8_t *) gate_w->data;
    const uint8_t * up_data   = (const uint8_t *) up_w->data;
    q8_skinny_split_out out = { nullptr, act.get(), 0.0f, Q8_SKINNY_EPI_F16, {} };
    ggml_cuda_pool_alloc<float> side_scratch(ctx.pool());
    if (side_w != nullptr) {
        q8_skinny_side_rows_set(out.side, side_scratch, counters, (int) (n / 32), side_w, src1, side_dst, k, m);
    }
    q8_skinny_split_launch_m<8, 2, true>(gate_data, (const half *) (gate_data + n * k), up_data,
                                         (const half *) (up_data + n * k), input.get(), out,
                                         scratch.get(), counters, (int) n, (int) k, (int) m, ctx.stream());

    const uint8_t * down = (const uint8_t *) down_w->data;
    if (n_down / 32 <= q8_skinny_split_max_tiles) {
        const q8_skinny_config config = q8_skinny_config_for(n_down, n);
        ggml_cuda_pool_alloc<float> scratch_down(ctx.pool(), q8_skinny_split_scratch_floats((int) n_down, config.split_k, false, (int) m));
        const q8_skinny_split_out out_down = { (float *) dst->data, nullptr, 0.0f, Q8_SKINNY_EPI_F32, {} };
        q8_skinny_split_launch_single(config, down, (const half *) (down + n_down * n), act.get(), out_down,
                                      scratch_down.get(), ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES,
                                      (int) n_down, (int) n, (int) m, ctx.stream());
    } else {
        const bool ok = q8_skinny_mul_mat_dispatch(down, (const half *) (down + n_down * n), act.get(),
                                                   (float *) dst->data, (int) n_down, (int) n, (int) m, ctx.stream());
        GGML_ASSERT(ok);
    }
    return true;
}

static bool q8_skinny_multi_prepare(const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                    int n_nodes, const ggml_tensor * src1, q8_skinny_multi_params & p) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64 ||
            k % 32 != 0 || q8_skinny_split_k(k) == 0) {
        return false;
    }
    if (m > 16 && q8_skinny_multi_config_for(k).split_k != 16) {
        return false;
    }

    p.k = (int) k;
    p.m = (int) m;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        ggml_tensor * dst = dsts[i];
        const int64_t n = w->ne[1];
        if (w->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
        if (ggml_cuda_q8_skinny_is_repacked(w)) {
            if (p.n_seg >= 4) {
                return false;
            }
            q8_skinny_multi_seg & seg = p.seg[p.n_seg++];
            seg.codes = (const uint8_t *) w->data;
            seg.scales = (const half *) (seg.codes + n * k);
            seg.n = (int) n;
            seg.dst = (float *) dst->data;
        } else {

            if (m > 16 || w->type != GGML_TYPE_Q8_0 || w->view_src != nullptr ||
                    w->op != GGML_OP_NONE || n % 32 == 0) {
                return false;
            }
            if (p.n_dot >= 4) {
                return false;
            }
            q8_skinny_multi_dot & dot = p.dot[p.n_dot++];
            dot.rows = (const uint8_t *) w->data;
            dot.n_rows = (int) n;
            dot.dst = (float *) dst->data;
        }
    }

    for (int s = 0; s < p.n_seg; ++s) {
        p.seg[s].first_tile = p.n_main_tiles;
        p.n_main_tiles += p.seg[s].n / 32;
    }
    for (int s = 0; s < p.n_dot; ++s) {
        p.dot[s].first_row = p.n_dot_rows;
        p.n_dot_rows += p.dot[s].n_rows;
    }
    return true;
}

bool ggml_cuda_q8_skinny_mul_mat_multi_supported(const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                                 int n_nodes, const ggml_tensor * src1) {
    q8_skinny_multi_params p = {};
    return q8_skinny_multi_prepare(src0s, dsts, n_nodes, src1, p);
}

bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                       int n_nodes, const ggml_tensor * src1, const half * input_f16) {
    q8_skinny_multi_params p = {};
    if (!q8_skinny_multi_prepare(src0s, dsts, n_nodes, src1, p)) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];

    ggml_cuda_pool_alloc<half> input(ctx.pool());
    if (input_f16 == nullptr) {
        input.alloc(m * k);
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(src1->data, input.get(), m * k, ctx.stream());
        input_f16 = input.get();
    }
    p.input = input_f16;

    if (m <= 16) {
        q8_skinny_multi_mul_mat_launch(p, ctx.stream());
        return true;
    }

    for (int s = 0; s < p.n_seg; ++s) {
        const q8_skinny_multi_seg & seg = p.seg[s];
        const bool ok = q8_skinny_mul_mat_dispatch(seg.codes, seg.scales, p.input, seg.dst,
                                                   seg.n, p.k, (int) m, ctx.stream());
        GGML_ASSERT(ok);
    }
    return true;
}

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int n = (int) src0->ne[1];
    const int k = (int) src0->ne[0];
    const uint8_t * data = (const uint8_t *) src0->data;
    const dim3 grid((unsigned) ((k / 16 + 7) / 8), (unsigned) (n / 32), 1);
    q8_skinny_to_f16_kernel<<<grid, 256, 0, stream>>>(
        dst, data, (const half *) (data + n * k), n, k);
    CUDA_CHECK(cudaGetLastError());
}

#else

void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_q8_skinny_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(src0, src1, dst);
    return false;
}

bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst, const half * input_f16) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst, input_f16);
    return false;
}

bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                       const ggml_tensor * up_w, const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(gate_w);
    GGML_UNUSED(up_w);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_q8_skinny_hc_down_up_supported(const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t m) {
    GGML_UNUSED_VARS(w_down, w_up, m);
    return false;
}

bool ggml_cuda_q8_skinny_gated_down(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                    const ggml_tensor * src1, const ggml_tensor * down_w, ggml_tensor * dst,
                                    const ggml_tensor * side_w, ggml_tensor * side_dst) {
    GGML_UNUSED_VARS(ctx, gate_w, up_w, src1, down_w, dst, side_w, side_dst);
    return false;
}

void ggml_cuda_q8_skinny_hc_down_up(ggml_backend_cuda_context & ctx, const ggml_tensor * w_down, const ggml_tensor * w_up,
                                    const half * input, int64_t m, float scale, float * gate,
                                    const float * pre_xn, float * pre_mixed, half * pre_mixed_f16, float pre_scale) {
    GGML_UNUSED_VARS(ctx, w_down, w_up, input, m, scale, gate, pre_xn, pre_mixed, pre_mixed_f16, pre_scale);
    GGML_ABORT("Q8 skinny: not available");
}

bool ggml_cuda_q8_skinny_hc_up_pre_supported(const ggml_tensor * w_up, int64_t m, int64_t hc) {
    GGML_UNUSED_VARS(w_up, m, hc);
    return false;
}

void ggml_cuda_q8_skinny_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const half * lo, int64_t m,
                                   float * gate, const float * xn, float * mixed, half * mixed_f16, float pre_scale) {
    GGML_UNUSED_VARS(ctx, w_up, lo, m, gate, xn, mixed, mixed_f16, pre_scale);
    GGML_ABORT("Q8 skinny: not available");
}

bool ggml_cuda_q8_skinny_mul_mat_multi_supported(const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                                 int n_nodes, const ggml_tensor * src1) {
    GGML_UNUSED_VARS(src0s, dsts, n_nodes, src1);
    return false;
}

bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                       int n_nodes, const ggml_tensor * src1, const half * input_f16) {
    GGML_UNUSED_VARS(ctx, src0s, dsts, n_nodes, src1, input_f16);
    return false;
}

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif
