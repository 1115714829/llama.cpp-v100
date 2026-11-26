// Portions from 1Cat-vLLM and SGLang-V100; see licenses/LICENSE-1Cat-vLLM (Apache-2.0).
#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

struct ggml_cuda_ar_push;

ggml_cuda_ar_push * ggml_cuda_ar_push_init(const int * devices, size_t n);

void ggml_cuda_ar_push_free(ggml_cuda_ar_push * ar);

bool ggml_cuda_ar_push_allreduce(
    ggml_cuda_ar_push * ar,
    ggml_backend_t    * backends,
    ggml_tensor       ** tensors,
    bool                exact);

bool ggml_cuda_ar_push_can(
    const ggml_cuda_ar_push * ar,
    const ggml_tensor       * t,
    bool                      exact);

bool ggml_cuda_ar_push_allreduce_rank(
    ggml_cuda_ar_push * ar,
    ggml_backend_t      backend,
    size_t              rank,
    ggml_tensor       * t,
    bool                exact);
