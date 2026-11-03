#pragma once

#include "common.cuh"





#if defined(__CUDA_ARCH_LIST__)
#define GGML_CUDA_SM70_D256_COMPILED ggml_cuda_has_arch(GGML_CUDA_CC_VOLTA)
#else
#define GGML_CUDA_SM70_D256_COMPILED false
#endif

bool   ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst);
size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst);
void   ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);



size_t ggml_cuda_sm70_d256_assist_workspace_size(int q_pad, int heads_q);
bool   ggml_cuda_sm70_d256_assist_run(void * backend, const struct ggml_backend_meta_assist_rank * desc);
bool   ggml_cuda_sm70_d256_assist_scratch_layout(const struct ggml_tensor * dst, struct ggml_backend_meta_assist_scratch * layout);
