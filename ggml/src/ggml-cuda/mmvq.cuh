#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.





#define MMVQ_MMID_MAX_BATCH_VOLTA 16


#define MMVQ_MMID_MAX_BATCH_SIZE 16

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    const char * const * expert_ptrs = nullptr);







bool ggml_cuda_mmvq_moe_reduce_supported(const ggml_tensor * down, const ggml_tensor * weights,
    const ggml_tensor * expert_scale, const ggml_tensor * dst);





void ggml_cuda_mmvq_moe_down_reduce(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
    const ggml_tensor * weights, const ggml_tensor * expert_scale, ggml_tensor * dst,
    const char * const * expert_ptrs, const half * act_h = nullptr);






bool ggml_cuda_mmvq_moe_gate_up_h_supported(const ggml_tensor * gate_up);
void ggml_cuda_mmvq_moe_gate_up_h(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_up,
    half * h, const char * const * expert_ptrs, const void * src1_q8 = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
