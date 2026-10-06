#pragma once

#include "common.cuh"

// Grouped-expert MUL_MAT_ID kernel for Volta (sm_70).
// The activation is quantized once and staged per expert, then one block serves all tokens
// routed to that expert, so no tile is padded to a fixed J like the MMQ MUL_MAT_ID path.
// The kernel structure follows Strata's s2_expert_grouped.cu (MIT), see moe-grouped.cu.
bool ggml_cuda_moe_grouped_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_moe_grouped_mul_mat_id_supported(const ggml_tensor * dst, const int cc);
