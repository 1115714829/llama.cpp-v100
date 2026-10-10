#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// DSV4_HC_POST (gated by SCALE -> SIGMOID -> SCALE, identity comb) followed by RMS_NORM -> MUL in
// one kernel; returns false if the shapes are not covered, the caller then runs the nodes unfused
bool ggml_cuda_op_dsv4_hc_post_rms_norm_mul_fused(ggml_backend_cuda_context & ctx,
                                                  ggml_tensor *               hc_post,
                                                  ggml_tensor *               scale_inv_hc,
                                                  ggml_tensor *               scale_gate,
                                                  ggml_tensor *               rms_norm,
                                                  ggml_tensor *               mul);

void ggml_cuda_op_rms_norm_mul_silu_mul_fused(ggml_backend_cuda_context & ctx,
                                              ggml_tensor *               rms_norm,
                                              ggml_tensor *               mul,
                                              ggml_tensor *               silu,
                                              ggml_tensor *               mul2);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               add,
                                     ggml_tensor *               rms_norm,
                                     ggml_tensor *               mul);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
