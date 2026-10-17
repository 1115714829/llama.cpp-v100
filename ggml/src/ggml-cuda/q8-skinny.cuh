// Q8_0 skinny GEMM for sm_70 (Volta).
//
// Adapted from 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (Apache-2.0). The QPN8
// execution layout is derived from dnv2003/v100-skinny (MIT) and its block-scale adaptation
// in haohervchb/sglang-V100. See licenses/LICENSE-v100-skinny in the repository root.

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
// caller must expand the repacked weights to dense F16 and use the regular path. input_f16, if
// not null, is src1 already rounded to F16 (contiguous [k, m]) by its producer; otherwise src1
// is converted here.
bool ggml_cuda_q8_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                 const half * input_f16 = nullptr);
// True if ggml_cuda_q8_skinny_mul_mat handles these nodes (src0 repacked, a covered shape).
bool ggml_cuda_q8_skinny_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Fused gate/up projection with a trailing silu(gate) * up, writing to the GLU node dst.
// Returns false for anything the kernel does not handle, the caller then runs the nodes
// on the regular path.
bool ggml_cuda_q8_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w, const ggml_tensor * src1, ggml_tensor * dst);

// Runs 2 to 4 MUL_MAT nodes that share one src1 with a single input conversion and a single
// kernel launch. Each src0 is either repacked or a narrow row-major Q8_0 weight (N is not a
// multiple of 32). Returns false if any node does not fit, the caller then runs them all on
// the regular path. input_f16 as for ggml_cuda_q8_skinny_mul_mat.
bool ggml_cuda_q8_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1,
                                       const half * input_f16 = nullptr);
// True if ggml_cuda_q8_skinny_mul_mat_multi handles these nodes.
bool ggml_cuda_q8_skinny_mul_mat_multi_supported(const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);

// The down and up projections of a qwen4exp hyper-connection mixer from the F16 copy of the normed
// input: gate = up(F16(silu(scale * down(input)))), with the arithmetic of the unfused nodes
// (q8 skinny down, SCALE -> UNARY(SILU), F32 -> F16, q8 skinny up), so gate is bitwise the same.
// The down projection must take the split path (a narrow repacked weight, M <= 16).
// pre_mixed, if not null: the gated DSV4_HC_PRE that reads the up output right after it, mixed =
// pre_scale * sum_c xn[c] * sigmoid(gate[c]) over the hc = 4 streams of the F32 normed input pre_xn
// ([n_embd, 4, m]), into pre_mixed ([n_embd, m]) and, if not null, its F16 copy pre_mixed_f16: the up
// kernel computes it in its epilogue with the expressions of dsv4_hc_pre_f32, bitwise the same
// (ggml_cuda_q8_skinny_hc_up_pre_supported() must hold).
bool ggml_cuda_q8_skinny_hc_down_up_supported(const ggml_tensor * w_down, const ggml_tensor * w_up, int64_t m);
void ggml_cuda_q8_skinny_hc_down_up(ggml_backend_cuda_context & ctx, const ggml_tensor * w_down, const ggml_tensor * w_up,
                                    const half * input, int64_t m, float scale, float * gate,
                                    const float * pre_xn = nullptr, float * pre_mixed = nullptr,
                                    half * pre_mixed_f16 = nullptr, float pre_scale = 0.0f);
bool ggml_cuda_q8_skinny_hc_up_pre_supported(const ggml_tensor * w_up, int64_t m, int64_t hc);
// The up projection alone with the hc_pre epilogue (lo: the F16 input [K, m]); see above.
void ggml_cuda_q8_skinny_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const half * lo, int64_t m,
                                   float * gate, const float * xn, float * mixed, half * mixed_f16, float pre_scale);

// A gated pair followed by its down projection, dst = down(swiglu(gate(src1), up(src1))): the pair
// (split path, a narrow weight, M <= 16) writes silu(gate) * up rounded to F16 and the down
// projection reads it, instead of an F32 GLU output and a conversion; bitwise the same. Returns
// false if the shapes are not covered, the caller then runs the nodes unfused.
// side_w/side_dst, if not null: a narrow F32 projection of the same src1 (the qwen4exp shared expert
// gate, F32 [K, rows], M <= 4) that ggml_cuda_mul_mat would run on the split narrow GEMV
// (ggml_cuda_mul_mat_vec_f_split_covers()); extra CTAs of the pair launch compute it with that GEMV's
// arithmetic, bitwise the same. The pair's tiles plus its rows must fit GGML_CUDA_MAX_SPLIT_TILES.
bool ggml_cuda_q8_skinny_gated_down(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                    const ggml_tensor * src1, const ggml_tensor * down_w, ggml_tensor * dst,
                                    const ggml_tensor * side_w = nullptr, ggml_tensor * side_dst = nullptr);

void ggml_cuda_q8_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
