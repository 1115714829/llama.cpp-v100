#pragma once

#include "common.cuh"

// SM70 (Volta) chunked prefill for GATED_DELTA_NET (staged f16, TileLang/FlashQLA
// kernels). Device kernels: gdn-chunk-sm70/gdn-chunk-*.cuh. The kernels are
// compiled by a dedicated sm_70-only object target, so the dispatch site guards
// its calls with GGML_CUDA_GDN_CHUNK_SM70_COMPILED.
#if defined(__CUDA_ARCH_LIST__)
#define GGML_CUDA_GDN_CHUNK_SM70_COMPILED ggml_cuda_has_arch(GGML_CUDA_CC_VOLTA)
#else
#define GGML_CUDA_GDN_CHUNK_SM70_COMPILED false
#endif

struct ggml_cuda_gdn_chunk_args {
    const float * q; const float * k;
    int64_t sq1, sq2, sk1, sk2;
    bool    l2norm;
    float   eps_q, scale_q, eps_k, scale_k;
    const float * v; int64_t sv1, sv2;
    const float * g; const float * beta;
    const float * s0;
    int64_t n_tokens;
    int64_t H;
    int64_t H_k;
    float * dst;
    float * h_out;
};


int64_t ggml_cuda_gdn_chunk_sm70_n_tokens(int cc, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3, bool kda, int K);
void    ggml_cuda_gdn_chunk_sm70(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & args);
