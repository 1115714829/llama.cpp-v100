#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse,
        int expert_offset, cudaStream_t stream);

// Expert parallelism: zero the output entries whose expert is not local, so that the kernels
// which skip them (MMQ, grouped) leave no stale data behind.
void ggml_cuda_mmid_zero_nonlocal(
        float * dst, const int32_t * ids, int expert_offset, int n_experts,
        int64_t n_tokens, int n_expert_used, int64_t ne0,
        int64_t ids_stride, int64_t stride_slot, int64_t stride_token, cudaStream_t stream);
