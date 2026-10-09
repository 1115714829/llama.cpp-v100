#pragma once

#include "common.cuh"

// W4A16 fused-expert MUL_MAT_ID for Volta (sm_70).
//
// The MMA tile, the XOR swizzle, the per-step independent accumulators (Volta's tensor cores
// truncate when they accumulate) and the Q4_K/Q5_K/Q5_1/Q8_0 dequantization are ported 1:1
// from Strata-AC922's src/kernels/cuda/fused_expert.cu (MIT, Copyright (c) 2026 Niko1221 and
// the Strata contributors, see licenses/LICENSE-strata). Only the expert addressing and the
// interface are adapted to our grouped tables (expert_bounds / ids_src1 / ids_dst).
//
// Without the SwiGLU epilogue flag (op_params[4]) the output is plain F32 and the graph does the
// SwiGLU. With the flag the 256 kernel pairs the weight rows (2c gate, 2c+1 up) and writes the
// F16 H itself; only this kernel can produce that dst, so the predicate is the single gate.
//
// This is a capability branch: Volta's only tensor core instruction is mma.sync.m8n8k4, so
// int8 tensor cores (sm_80+) are unavailable and the supported quant types are the four the
// ported dequantizers implement. Types outside the set keep their existing path.
bool ggml_cuda_fused_expert_mul_mat_id_supported(const ggml_tensor * dst, const int cc);
bool ggml_cuda_fused_expert_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const char * const * expert_ptrs);
