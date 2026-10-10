// Q8_0 skinny GEMM for sm_70 (Volta).
//
// Adapted from 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (Apache-2.0). The QPN8
// execution layout is derived from dnv2003/v100-skinny (MIT) and its block-scale adaptation
// in haohervchb/sglang-V100. See LICENSE.v100-skinny in this directory.

#pragma once

#include "common.cuh"

// Weights are repacked in place from the row-major Q8_0 layout (ne0 = K, ne1 = N) into:
//   codes:  [N/32][K/16][32][16] uint8 at data,            N*K bytes
//   scales: [K/32][N/32][32] half at data + N*K,           N*K/16 bytes
// The byte count is the same as Q8_0. A repacked tensor is tagged through tensor->extra;
// reading it with ggml_backend_tensor_get() returns the repacked bytes, there is no
// conversion back to the Q8_0 layout.
bool ggml_cuda_q8_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_q8_skinny_is_repacked(const ggml_tensor * t);

// Repacks all repackable weights of the graph that are not read by another op, then marks
// them. Repacking all of them up front keeps it out of CUDA graph captures.
void ggml_cuda_q8_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
void ggml_cuda_q8_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);

// Returns true if the small-M kernel handled the multiplication. If it returns false the
// caller must expand the repacked weights to dense F16 and use the regular path.
bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Fused gate/up projection with a trailing silu(gate) * up, writing to the GLU node dst.
// Returns false for anything the kernel does not handle, the caller then runs the nodes
// on the regular path.
bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

// Runs 2 to 4 MUL_MAT nodes that share one src1 with a single input conversion and a single
// kernel launch. Each src0 is either repacked or a narrow row-major Q8_0 weight (N is not a
// multiple of 32). Returns false if any node does not fit, the caller then runs them all on
// the regular path.
bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

// The down and up projections of a qwen4exp hyper-connection mixer from the F16 copy of the normed
// input: gate = up(F16(silu(scale * down(input)))), with the arithmetic of the unfused nodes
// (q8 skinny down, SCALE -> UNARY(SILU), F32 -> F16, q8 skinny up), so gate is bitwise the same.
// The down projection must take the split path (a narrow repacked weight, M <= 16).
bool ggml_cuda_q8_skinny_hc_down_up_supported(const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t m);
void ggml_cuda_q8_skinny_hc_down_up(ggml_backend_cuda_context & ctx, const ggml_tensor * w_down, const ggml_tensor * w_up,
                                    const half * input, int64_t m, float scale, float * gate);

// A gated pair followed by its down projection, dst = down(swiglu(gate(src1), up(src1))): the pair
// (split path, a narrow weight, M <= 16) writes silu(gate) * up rounded to F16 and the down
// projection reads it, instead of an F32 GLU output and a conversion; bitwise the same. Returns
// false if the shapes are not covered, the caller then runs the nodes unfused.
bool ggml_cuda_q8_skinny_gated_down(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                    const ggml_tensor * src1, const ggml_tensor * down_w, ggml_tensor * dst);

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
