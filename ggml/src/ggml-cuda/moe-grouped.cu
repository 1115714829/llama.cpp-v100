#include "moe-grouped.cuh"

#include "mmid.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"
#include "ggml-cuda-expert.cuh"

#include <cstdint>

// Grouped-expert MUL_MAT_ID kernel for Volta (sm_70).
//
// The kernel structure follows Strata's src/kernels/cuda/s2_expert_grouped.cu (MIT license):
// the activation rows of the (token, expert) entries routed to one expert are staged into
// shared memory, then a warp computes one output row against all staged entries. No column
// is padded to a fixed tile, unlike the MMQ MUL_MAT_ID path which pads every expert to a J
// wide tile. The weight row is read once per staged group and reused across the group.
//
// Both decode (T <= 8) and prefill use this kernel. The staged group width is picked at
// launch time (MOE_GROUPED_CT_DECODE / MOE_GROUPED_CT_PREFILL); the column loop keeps a hot
// expert correct when it receives more columns than the group width.

template <ggml_type type>
struct moe_grouped_traits;

#define MOE_GROUPED_TRAITS(type_, qk_, qi_, vdr_, block_, dot_)                                               \
    template <> struct moe_grouped_traits<type_> {                                                            \
        static constexpr int qk  = qk_;                                                                       \
        static constexpr int qi  = qi_;                                                                       \
        static constexpr int vdr = vdr_;                                                                      \
        static constexpr int bs  = sizeof(block_);                                                            \
        static __device__ __forceinline__ float dot(const void * vbq, const block_q8_1 * bq8_1,               \
                                                    const int & kbx, const int & iqs) {                       \
            return dot_(vbq, bq8_1, kbx, iqs);                                                                \
        }                                                                                                     \
    };

MOE_GROUPED_TRAITS(GGML_TYPE_Q4_K, QK_K,  QI4_K, VDR_Q4_K_Q8_1_MMVQ, block_q4_K, vec_dot_q4_K_q8_1)
MOE_GROUPED_TRAITS(GGML_TYPE_Q5_K, QK_K,  QI5_K, VDR_Q5_K_Q8_1_MMVQ, block_q5_K, vec_dot_q5_K_q8_1)
MOE_GROUPED_TRAITS(GGML_TYPE_Q6_K, QK_K,  QI6_K, VDR_Q6_K_Q8_1_MMVQ, block_q6_K, vec_dot_q6_K_q8_1)
MOE_GROUPED_TRAITS(GGML_TYPE_Q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, block_q5_1, vec_dot_q5_1_q8_1)
MOE_GROUPED_TRAITS(GGML_TYPE_Q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, block_q8_0, vec_dot_q8_0_q8_1)

// One block handles ROWS output rows of one expert. The expert's entries are processed in
// groups of at most CT, each group staged into shared memory. Warp w takes rows w, w+nwarps, ...
// For a row the lane loads its weight blocks once into registers and dots them against every
// staged entry, so the weight row is read from global memory once per group.
template <ggml_type type, int ROWS, int CT, bool expert_store>
__launch_bounds__(256, 1)
static __global__ void moe_grouped_kernel(
        const char * __restrict__ vx, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ ids_src1,
        const block_q8_1 * __restrict__ xq, float * __restrict__ dst,
        const int nrows_x, const int nblocks_weight, const int nblocks_row_q8,
        const int64_t nb01, const int64_t nb02, const int64_t dst_stride_col,
        const char * const * __restrict__ expert_ptrs) {
    constexpr int qk  = moe_grouped_traits<type>::qk;
    constexpr int qi  = moe_grouped_traits<type>::qi;
    constexpr int vdr = moe_grouped_traits<type>::vdr;
    constexpr int bs  = moe_grouped_traits<type>::bs;

    extern __shared__ block_q8_1 sh_q[];

    const int expert = blockIdx.y;
    const int col_start = expert_bounds[expert];
    const int col_diff  = expert_bounds[expert + 1] - col_start;
    if (col_diff <= 0) {
        return;
    }

    // Expert store: the expert base comes from the device pointer table, which may point at
    // a VRAM slot or at the mapped host arena. Read it once per block, not per row group.
    const char * expert_base;
    if constexpr (expert_store) {
        expert_base = expert_ptrs[expert];
    } else {
        expert_base = vx + (size_t) expert * nb02;
    }

    const int lane   = threadIdx.x % 32;
    const int warp   = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;
    const int row0   = blockIdx.x * ROWS;

    // Number of weight blocks one lane covers per iteration, and the lane's first block.
    constexpr int blocks_per_iter = vdr * 32 / qi;
    const int     kbx0            = lane / (qi / vdr);
    const int     kqs0            = vdr * (lane % (qi / vdr));

    for (int cg = 0; cg < col_diff; cg += CT) {
        const int cg_n = min(CT, col_diff - cg);

        __syncthreads();
        for (int i = threadIdx.x; i < cg_n * nblocks_row_q8; i += blockDim.x) {
            const int e = i / nblocks_row_q8;
            const int b = i - e * nblocks_row_q8;
            const int row = ids_src1[col_start + cg + e];
            // block_q8_1 holds a half2 union, so it is copied as raw bytes.
            memcpy(sh_q + e * nblocks_row_q8 + b, xq + (size_t) row * nblocks_row_q8 + b, sizeof(block_q8_1));
        }
        __syncthreads();

        for (int rr = warp; rr < ROWS; rr += nwarps) {
            const int r = row0 + rr;
            if (r >= nrows_x) {
                break;
            }
            const char * wrow = expert_base + (size_t) r * nb01;

            float acc[CT] = {0.0f};
            for (int kbx = kbx0; kbx < nblocks_weight; kbx += blocks_per_iter) {
                // The weight block is the same for every entry in the group, so its address is
                // computed once and the reads hit L1 on the entries after the first.
                const char * wb = wrow + (size_t) kbx * bs;
#pragma unroll
                for (int e = 0; e < CT; ++e) {
                    if (e < cg_n) {
                        const block_q8_1 * y = sh_q + e * nblocks_row_q8;
                        acc[e] += moe_grouped_traits<type>::dot(wb, &y[kbx * (qk / QK8_1)], 0, kqs0);
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < CT; ++e) {
                if (e < cg_n) {
                    const float sum = warp_reduce_sum<32>(acc[e]);
                    if (lane == 0) {
                        const int c = col_start + cg + e;
                        dst[(size_t) ids_dst[c] * dst_stride_col + r] = sum;
                    }
                }
            }
        }
    }
}

// Admission predicate, also used to keep CUDA graphs enabled for this path.
bool ggml_cuda_moe_grouped_mul_mat_id_supported(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }
    // T = 1..8 is the decode and verify window, n > 8 is prefill. The entries of one expert are
    // staged into shared memory in groups and the weight row is read once per group, so the same
    // kernel serves both ranges; the group width follows the average columns per expert. There is
    // no upper token bound: the kernel loops over the columns of one expert, a hot expert with
    // more columns than the group width stays correct.
    if (dst->ne[2] < 1) {
        return false;
    }
    if (src1->ne[3] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->nb[2] % src1->nb[1] != 0) {
        return false;
    }
    // The activation is quantized to block_q8_1 rows; K must be a whole number of blocks.
    if (src1->ne[0] % QK8_1 != 0 || src1->ne[0] % ggml_blck_size(src0->type) != 0) {
        return false;
    }
    if (src0->ne[0] % QK8_1 != 0 || src0->ne[0] % ggml_blck_size(src0->type) != 0) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0) {
        return false;
    }
    // The weights are read in their original GGUF layout, a repacked tensor would be misinterpreted.
    if (src0->extra != nullptr) {
        return false;
    }
    if (src0->ne[2] <= 0 || src0->ne[2] > 65535) {
        return false;
    }
    // Shared memory for the decode group width must fit; the prefill width is checked at launch.
    const int64_t nblocks_row_q8 = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) / QK8_1;
    if (MOE_GROUPED_CT_DECODE * nblocks_row_q8 * (int64_t) sizeof(block_q8_1) > (int64_t) ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        return false;
    }
    return true;
}

template <ggml_type type, int ROWS, int CT, bool expert_store>
static void moe_grouped_launch_ct(
        const char * vx, const int32_t * expert_bounds, const int32_t * ids_dst, const int32_t * ids_src1,
        const block_q8_1 * xq, float * dst, const int64_t nrows_x, const int64_t nblocks_weight, const int64_t nblocks_row_q8,
        const int64_t nb01, const int64_t nb02, const int64_t dst_stride_col, const int n_experts, cudaStream_t stream,
        const char * const * expert_ptrs) {
    const size_t nbytes_shared = (size_t) CT * nblocks_row_q8 * sizeof(block_q8_1);

    const dim3 grid((nrows_x + ROWS - 1) / ROWS, n_experts);
    const dim3 block(256);

    // nbytes_shared depends on K, so the opt-in limit is raised on every launch rather than once.
#if !defined(GGML_USE_HIP)
    CUDA_CHECK(cudaFuncSetAttribute((moe_grouped_kernel<type, ROWS, CT, expert_store>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes_shared));
#endif

    moe_grouped_kernel<type, ROWS, CT, expert_store><<<grid, block, nbytes_shared, stream>>>(
        vx, expert_bounds, ids_dst, ids_src1, xq, dst,
        (int) nrows_x, (int) nblocks_weight, (int) nblocks_row_q8, nb01, nb02, dst_stride_col, expert_ptrs);
}

template <ggml_type type, bool expert_store>
static void moe_grouped_launch(
        const char * vx, const int32_t * expert_bounds, const int32_t * ids_dst, const int32_t * ids_src1,
        const block_q8_1 * xq, float * dst, const int64_t nrows_x, const int64_t nblocks_weight, const int64_t nblocks_row_q8,
        const int64_t nb01, const int64_t nb02, const int64_t dst_stride_col, const int n_experts, cudaStream_t stream,
        const char * const * expert_ptrs, const int64_t ncols_avg) {
    // Decode and verify stage one group of MOE_GROUPED_CT_DECODE entries, which covers the whole
    // routing of a small batch. Prefill stages a wider group and more output rows per block, so
    // the weight row of an expert is read once for more of its columns and the staged activation
    // is reused over more rows. The wider group needs more shared memory, so it is used only when
    // it fits. A hot expert with more columns than the capacity keeps looping, so both widths
    // stay correct. The width choice is a shared-memory capability limit, not a fallback.
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    const bool prefill = ncols_avg > MOE_GROUPED_CT_DECODE &&
        (size_t) MOE_GROUPED_CT_PREFILL * nblocks_row_q8 * sizeof(block_q8_1) <= smpbo;

    if (prefill) {
        moe_grouped_launch_ct<type, 64, MOE_GROUPED_CT_PREFILL, expert_store>(vx, expert_bounds, ids_dst, ids_src1, xq, dst,
            nrows_x, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, n_experts, stream, expert_ptrs);
    } else {
        moe_grouped_launch_ct<type, 32, MOE_GROUPED_CT_DECODE, expert_store>(vx, expert_bounds, ids_dst, ids_src1, xq, dst,
            nrows_x, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, n_experts, stream, expert_ptrs);
    }
}

// Dispatch to the pointer-table kernel for expert store weights, to the src0 kernel otherwise.
// The choice is made here so the kernel does not test expert_ptrs at run time.
template <ggml_type type>
static void moe_grouped_launch_dispatch(
        const char * vx, const int32_t * expert_bounds, const int32_t * ids_dst, const int32_t * ids_src1,
        const block_q8_1 * xq, float * dst, const int64_t nrows_x, const int64_t nblocks_weight, const int64_t nblocks_row_q8,
        const int64_t nb01, const int64_t nb02, const int64_t dst_stride_col, const int n_experts, cudaStream_t stream,
        const char * const * expert_ptrs, const int64_t ncols_avg) {
    if (expert_ptrs != nullptr) {
        moe_grouped_launch<type, true>(vx, expert_bounds, ids_dst, ids_src1, xq, dst, nrows_x, nblocks_weight,
            nblocks_row_q8, nb01, nb02, dst_stride_col, n_experts, stream, expert_ptrs, ncols_avg);
    } else {
        moe_grouped_launch<type, false>(vx, expert_bounds, ids_dst, ids_src1, xq, dst, nrows_x, nblocks_weight,
            nblocks_row_q8, nb01, nb02, dst_stride_col, n_experts, stream, expert_ptrs, ncols_avg);
    }
}

bool ggml_cuda_moe_grouped_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_moe_grouped_mul_mat_id_supported(dst, cc)) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src1 = ggml_type_size(src1->type);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = ne12 * n_expert_used;

    cudaStream_t stream = ctx.stream();

    // Shared memory budget for the staged activation group (the decode width, the narrowest
    // group). The prefill width is picked at launch time only when it fits.
    const int64_t nblocks_row_q8 = GGML_PAD(ne10, MATRIX_ROW_PADDING) / QK8_1;
    const int64_t nbytes_shared  = MOE_GROUPED_CT_DECODE * nblocks_row_q8 * (int64_t) sizeof(block_q8_1);
    if (nbytes_shared > (int64_t) ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        return false;
    }

    // Average columns one expert receives. The grouped kernel picks its staged-entry width from
    // it: a small batch stays on the decode width, prefill uses the wider group.
    const int64_t ncols_avg = (ne12 * n_expert_used + ne02 - 1) / ne02;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // an expert store view shifts the weights (and the local expert numbers) by its start
    int32_t expert_offset = ggml_mul_mat_id_get_expert_offset(dst);
    ggml_cuda_expert_table ex_table;
    if (ggml_cuda_expert_table_get(src0, &ex_table)) {
        expert_offset += (int32_t) ex_table.expert_start;
    }

    if (ggml_mul_mat_id_is_expert_parallel(dst)) {
        // the grouped kernel only writes the local entries, zero the others
        ggml_cuda_mmid_zero_nonlocal((float *) dst->data, (const int32_t *) ids->data, expert_offset, (int) ne02,
            ne12, (int) n_expert_used, dst->ne[0], ids->nb[1] / (int64_t) sizeof(int32_t),
            dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), stream);
    }

    {
        GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
        const int si1  = ids->nb[1] / ggml_type_size(ids->type);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/false,
            expert_offset, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // Quantize the flattened activation rows once, to the standard block_q8_1 used by MMVQ.
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne11*ne12 * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type,
            ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t nblocks_weight = ne00 / ggml_blck_size(src0->type);
    const int64_t dst_stride_col = dst->nb[1] / sizeof(float);

    const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get();

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            moe_grouped_launch_dispatch<GGML_TYPE_Q4_K>((const char *) src0->data, expert_bounds.get(), ids_dst.get(), ids_src1.get(),
                xq, (float *) dst->data, ne01, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, ne02, stream, expert_ptrs, ncols_avg);
            break;
        case GGML_TYPE_Q5_K:
            moe_grouped_launch_dispatch<GGML_TYPE_Q5_K>((const char *) src0->data, expert_bounds.get(), ids_dst.get(), ids_src1.get(),
                xq, (float *) dst->data, ne01, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, ne02, stream, expert_ptrs, ncols_avg);
            break;
        case GGML_TYPE_Q6_K:
            moe_grouped_launch_dispatch<GGML_TYPE_Q6_K>((const char *) src0->data, expert_bounds.get(), ids_dst.get(), ids_src1.get(),
                xq, (float *) dst->data, ne01, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, ne02, stream, expert_ptrs, ncols_avg);
            break;
        case GGML_TYPE_Q5_1:
            moe_grouped_launch_dispatch<GGML_TYPE_Q5_1>((const char *) src0->data, expert_bounds.get(), ids_dst.get(), ids_src1.get(),
                xq, (float *) dst->data, ne01, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, ne02, stream, expert_ptrs, ncols_avg);
            break;
        case GGML_TYPE_Q8_0:
            moe_grouped_launch_dispatch<GGML_TYPE_Q8_0>((const char *) src0->data, expert_bounds.get(), ids_dst.get(), ids_src1.get(),
                xq, (float *) dst->data, ne01, nblocks_weight, nblocks_row_q8, nb01, nb02, dst_stride_col, ne02, stream, expert_ptrs, ncols_avg);
            break;
        default:
            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());

    return true;
}
