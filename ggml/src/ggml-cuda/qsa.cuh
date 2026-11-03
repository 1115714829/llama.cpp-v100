#include "common.cuh"


void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_select_supported(int device, const ggml_tensor * dst);


void ggml_cuda_op_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_attn_supported(int device, const ggml_tensor * dst);


void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_expand_supported(int device, const ggml_tensor * dst);
