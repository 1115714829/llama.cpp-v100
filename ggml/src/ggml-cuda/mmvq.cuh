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

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
