#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// dst_f16, if not null, also receives the result rounded to F16 (contiguous [n_embd, n_tokens]),
// bitwise the F32 -> F16 conversion of dst
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst, half * dst_f16 = nullptr);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * scale_inv_hc, ggml_tensor * scale_gate);
