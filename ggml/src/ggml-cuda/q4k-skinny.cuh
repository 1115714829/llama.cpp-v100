// Portions from 1Cat-vLLM, v100-skinny and haohervchb/sglang-V100; see licenses/LICENSE-1Cat-vLLM and licenses/LICENSE-v100-skinny.

#pragma once

#include "common.cuh"

bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t);

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
