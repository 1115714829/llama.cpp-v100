














#pragma once

#include "common.cuh"




void ggml_cuda_fused_expert_x16_scaled(
        const float * x, __half * y, int32_t * row_k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream);
