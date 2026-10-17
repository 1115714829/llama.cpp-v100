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

// The norm that opens a qwen4exp hyper-connection mixer: [gated DSV4_HC_POST ->] RMS_NORM -> MUL in
// one kernel (hc_post == nullptr: only the norm), bitwise as the unfused kernels; mul_f16, if not
// nullptr, receives the F16 copy of the MUL output (rounded like the F32 -> F16 conversion).
bool ggml_cuda_hc_mix_norm_supported(const ggml_tensor * hc_post, const ggml_tensor * scale_inv_hc,
                                     const ggml_tensor * rms_norm, const ggml_tensor * mul);
// inject, if not null: the MUL_MAT node that produces the post input of the gate (scale_inv_hc->src[0]), an
// F32 inject projection that ggml_cuda_mul_mat runs on the split narrow GEMV; the kernel computes it itself
// with that GEMV's arithmetic instead of reading it (the node need not be computed).
void ggml_cuda_op_hc_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post, ggml_tensor * scale_inv_hc,
                              ggml_tensor * scale_gate, ggml_tensor * rms_norm, ggml_tensor * mul, half * mul_f16,
                              const ggml_tensor * inject = nullptr);

// RMS_NORM -> MUL -> UNARY(SILU or SIGMOID) -> MUL in one kernel. dst_f16, if not null, also
// receives mul2 rounded to F16 (mul2 must be contiguous), bitwise the F32 -> F16 conversion of it.
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
