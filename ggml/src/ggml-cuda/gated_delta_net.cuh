#include "common.cuh"
#include "ggml.h"

struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;
    int64_t slot_stride;
};

struct ggml_cuda_gdn_prologue {
    const float * q_raw;
    const float * k_raw;
    int64_t sq1, sq2, sq3;
    int64_t sk1, sk2, sk3;
    bool    standalone_rms_order;
    float   eps_q, scale_q;
    float   eps_k, scale_k;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gdn_prologue & prologue,
                                        const ggml_cuda_gated_delta_net_fused_cache * cache);
