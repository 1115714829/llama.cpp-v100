#include "common.cuh"

// QSA selection: indexer scores, top pools per token, cell ids (GGML_OP_QSA_SELECT)
void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_select_supported(int device, const ggml_tensor * dst);

// QSA attention over the selected cells (GGML_OP_QSA_ATTN)
void ggml_cuda_op_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_attn_supported(int device, const ggml_tensor * dst);

// QSA pool rows expanded into cell rows (GGML_OP_QSA_EXPAND)
void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_expand_supported(int device, const ggml_tensor * dst);
