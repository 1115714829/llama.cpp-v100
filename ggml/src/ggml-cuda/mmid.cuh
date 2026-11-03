#pragma once

#include "common.cuh"






void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * expert_fill,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse,
        const ggml_cuda_expert_lut lut, cudaStream_t stream);





void ggml_cuda_mmid_zero_nonlocal(
        void * dst, const int32_t * ids, const ggml_cuda_expert_lut lut,
        int64_t n_tokens, int n_expert_used, int64_t ne0,
        int64_t ids_stride, int64_t stride_slot, int64_t stride_token, cudaStream_t stream, bool dst_f16 = false);
