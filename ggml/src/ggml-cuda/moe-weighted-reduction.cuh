#include "common.cuh"

// mm_id is the MUL_MAT_ID that wrote experts, or nullptr when experts is the result of another op.
// Under expert parallelism its slots of the other devices are skipped by the kernel itself, so the
// node does not have to zero them.
void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         mm_id);
