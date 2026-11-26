// Portions from 1Cat-vLLM, v100-skinny and haohervchb/sglang-V100; see licenses/LICENSE-1Cat-vLLM and licenses/LICENSE-v100-skinny.

#pragma once

#include "common.cuh"

bool ggml_cuda_q8_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_q8_skinny_is_repacked(const ggml_tensor * t);

void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);

bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                 const half * input_f16 = nullptr);

bool ggml_cuda_q8_skinny_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1,
                                       const half * input_f16 = nullptr);

bool ggml_cuda_q8_skinny_mul_mat_multi_supported(const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

bool ggml_cuda_q8_skinny_hc_down_up_supported(const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t m);
void ggml_cuda_q8_skinny_hc_down_up(ggml_backend_cuda_context & ctx, const ggml_tensor * w_down, const ggml_tensor * w_up,
                                    const half * input, int64_t m, float scale, float * gate,
                                    const float * pre_xn = nullptr, float * pre_mixed = nullptr,
                                    half * pre_mixed_f16 = nullptr, float pre_scale = 0.0f);
bool ggml_cuda_q8_skinny_hc_up_pre_supported(const ggml_tensor * w_up, int64_t m, int64_t hc);

void ggml_cuda_q8_skinny_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const half * lo, int64_t m,
                                   float * gate, const float * xn, float * mixed, half * mixed_f16, float pre_scale);

bool ggml_cuda_q8_skinny_gated_down(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                    const ggml_tensor * src1, const ggml_tensor * down_w, ggml_tensor * dst,
                                    const ggml_tensor * side_w = nullptr, ggml_tensor * side_dst = nullptr);

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
