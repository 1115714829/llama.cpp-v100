// Portions from Strata and its AC922 fork; Copyright (c) 2026 Niko1221 and the Strata contributors (MIT); see licenses/LICENSE-strata.
#pragma once

#include "common.cuh"

#define MOE_GROUPED_MAX_TOKENS 8

#define MOE_GROUPED_CT_DECODE  8
#define MOE_GROUPED_CT_PREFILL 32
bool ggml_cuda_moe_grouped_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs = nullptr);
bool ggml_cuda_moe_grouped_mul_mat_id_supported(const ggml_tensor * dst, const int cc);
