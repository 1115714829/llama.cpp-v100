#pragma once

#include "common.cuh"

// Grouped-expert MUL_MAT_ID kernel for Volta (sm_70).
// The activation is quantized once and staged per expert, then one block serves all tokens
// routed to that expert, so no tile is padded to a fixed J like the MMQ MUL_MAT_ID path.
// The kernel structure follows Strata's s2_expert_grouped.cu (MIT), see moe-grouped.cu.





#define MOE_GROUPED_MAX_TOKENS 8





#define MOE_GROUPED_CT_DECODE  8
#define MOE_GROUPED_CT_PREFILL 32
bool ggml_cuda_moe_grouped_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs = nullptr);
bool ggml_cuda_moe_grouped_mul_mat_id_supported(const ggml_tensor * dst, const int cc);
