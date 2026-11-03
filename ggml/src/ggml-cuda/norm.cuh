#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);



bool ggml_cuda_op_dsv4_hc_post_rms_norm_mul_fused(ggml_backend_cuda_context & ctx,
                                                  ggml_tensor *               hc_post,
                                                  ggml_tensor *               scale_inv_hc,
                                                  ggml_tensor *               scale_gate,
                                                  ggml_tensor *               rms_norm,
                                                  ggml_tensor *               mul);




bool ggml_cuda_hc_mix_norm_supported(const ggml_tensor * hc_post, const ggml_tensor * scale_inv_hc,
                                     const ggml_tensor * rms_norm, const ggml_tensor * mul);



void ggml_cuda_op_hc_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post, ggml_tensor * scale_inv_hc,
                              ggml_tensor * scale_gate, ggml_tensor * rms_norm, ggml_tensor * mul, half * mul_f16,
                              const ggml_tensor * inject = nullptr);



void ggml_cuda_op_rms_norm_mul_silu_mul_fused(ggml_backend_cuda_context & ctx,
                                              ggml_tensor *               rms_norm,
                                              ggml_tensor *               mul,
                                              ggml_tensor *               silu,
                                              ggml_tensor *               mul2,
                                              half *                      dst_f16 = nullptr);

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
