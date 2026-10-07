#pragma once

#include "common.cuh"

// Grouped-expert MUL_MAT_ID kernel for Volta (sm_70).
// The activation is quantized once and staged per expert, then one block serves all tokens
// routed to that expert, so no tile is padded to a fixed J like the MMQ MUL_MAT_ID path.
// The kernel structure follows Strata's s2_expert_grouped.cu (MIT), see moe-grouped.cu.

// Decode and verify windows up to this many tokens take the grouped kernel on Volta: it
// stages every entry routed to one expert and reads the weight row once, while the MMVQ MoE
// kernel gives one warp per token and re-reads the expert's weight for every routed token.
// See the dispatch in ggml_cuda_mul_mat_id.
#define MOE_GROUPED_MAX_TOKENS 8
bool ggml_cuda_moe_grouped_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs = nullptr);
bool ggml_cuda_moe_grouped_mul_mat_id_supported(const ggml_tensor * dst, const int cc);
