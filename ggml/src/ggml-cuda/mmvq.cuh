#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// MUL_MAT_ID uses a separate kernel (mul_mat_vec_q_moe) that gives one warp per token, so its
// block is ncols_dst warps wide. Volta allows 1024 threads per block, so the cheap types can go
// past MMVQ_MAX_BATCH_SIZE. 16 keeps the block at 512 threads and lets n = 9..16 use the mmid
// path instead of the padded MMQ tile. Types with a heavier vec_dot stay at MMVQ_MAX_BATCH_SIZE.
#define MMVQ_MMID_MAX_BATCH_VOLTA 16

// Upper bound of get_mmvq_mmid_max_batch() over all architectures, used for asserts.
#define MMVQ_MMID_MAX_BATCH_SIZE 16

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    const char * const * expert_ptrs = nullptr);

// The down projection of a decode/verify MUL_MAT_ID node (no epilogue, F32 or F16 activation)
// followed by the weighted sum over its slots, the moe_weighted_reduction pattern:
// dst[token][row] = sum_slot (down[token][slot][row] * expert_scale) * weights[token][slot], written
// by one compact MMVQ kernel that also quantizes the activation itself, with the results of the
// separate steps (bit identical). The caller checks that ggml_cuda_mul_mat_id would run the node on
// the MMVQ mmid kernel.
bool ggml_cuda_mmvq_moe_reduce_supported(const ggml_tensor * down, const ggml_tensor * weights,
    const ggml_tensor * expert_scale, const ggml_tensor * dst);

// expert_ptrs: the staging table of the launch (ggml_cuda_expert_stage_prepare_decode) or nullptr.
// act_h: the activation in a private buffer, [token][slot][ne10] F16 (the fused MoE FFN), nullptr to
// read down->src[1] itself. The kernel reads the activation, ids, weights and expert_scale while it
// writes dst.
void ggml_cuda_mmvq_moe_down_reduce(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
    const ggml_tensor * weights, const ggml_tensor * expert_scale, ggml_tensor * dst,
    const char * const * expert_ptrs, const half * act_h = nullptr);

// The gate_up MUL_MAT_ID node with the SwiGLU epilogue of a decode/verify batch (F32 activation),
// written into a private buffer h = [token][slot][ne0] F16 for the local slots only (the slots of
// other devices are left unwritten: the down projection of the fused MoE FFN reads no other).
// src1_q8: the q8_1 activation in the layout of quantize_row_q8_1_cuda (rows of
// GGML_PAD(ne10, MATRIX_ROW_PADDING) values) when the caller already has it, nullptr to quantize here.
bool ggml_cuda_mmvq_moe_gate_up_h_supported(const ggml_tensor * gate_up);
void ggml_cuda_mmvq_moe_gate_up_h(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_up,
    half * h, const char * const * expert_ptrs, const void * src1_q8 = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
