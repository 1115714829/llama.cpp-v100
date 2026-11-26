#include "common.cuh"

struct ggml_cuda_ssm_conv_rollback_params {
    const float *   state;
    int64_t         state_stride;
    const int32_t * row;
    const float *   qkv;
    int64_t         qkv_nb0;
    int64_t         qkv_nb1;
    const float *   w;
    int64_t         w_nb1;
    float *         dst;
    int64_t         dst_nb1;
    int64_t         n_tok;
    int64_t         n_channels;
    int64_t         snap_off[16];
    float *         snap_dst[16];
    int             n_snap;
};

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_ssm_conv_rollback_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & params);

void ggml_cuda_op_ssm_conv_rollback_long(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & params, const float * history, float * halo);

void ggml_cuda_op_ple_conv4(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & p,
                           const float * history, const ggml_tensor * weights, const float * hidden, const float * gated);

void ggml_cuda_op_ple_broadcast(ggml_backend_cuda_context & ctx, const ggml_tensor * value,
                               const ggml_tensor * gate, ggml_tensor * dst);

void ggml_cuda_op_ple_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * key,
                          const ggml_tensor * query, float * dst, float scale);
