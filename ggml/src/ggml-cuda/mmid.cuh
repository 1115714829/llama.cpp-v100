#pragma once

#include "common.cuh"

// lut is the expert map of the launch (ggml_cuda_expert_lut): the ids are routing ids, the
// experts of the compact layout are the local expert numbers the map gives them.
// One device-side grouping pass over the (token, slot) pairs writes expert_bounds (the prefix
// sums of the counts, n_experts + 1 entries) and the two row maps; expert_fill holds the per
// expert placement cursors of that pass (n_experts entries) and is left at expert_bounds[e + 1].
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * expert_fill,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse,
        const ggml_cuda_expert_lut lut, cudaStream_t stream);

// Expert parallelism: zero the output entries whose expert is not local (lut maps the id to -1),
// so that the kernels which skip them (MMQ, grouped) leave no stale data behind. stride_slot and
// stride_token are in elements of the output type, dst_f16 selects the F16 output of the
// fused-expert SwiGLU epilogue over F32 for every other path.
void ggml_cuda_mmid_zero_nonlocal(
        void * dst, const int32_t * ids, const ggml_cuda_expert_lut lut,
        int64_t n_tokens, int n_expert_used, int64_t ne0,
        int64_t ids_stride, int64_t stride_slot, int64_t stride_token, cudaStream_t stream, bool dst_f16 = false);
