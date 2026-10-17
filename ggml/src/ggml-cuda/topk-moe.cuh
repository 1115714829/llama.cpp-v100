#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

// The q8_1 quantization of the MoE gate_up activation done in the routing launch (extra blocks after
// the routing rows): x is [ne00] F32 per token with row stride s_row (floats), x_q8 the q8_1 rows of
// ne0 values (ne0: the padded row length, a multiple of TOPK_MOE_ROWS_PER_BLOCK*WARP_SIZE),
// n_blocks = tokens * ne0 / (TOPK_MOE_ROWS_PER_BLOCK*WARP_SIZE). n_blocks = 0: routing only.
struct ggml_cuda_topk_moe_quant {
    const float * x        = nullptr;
    void *        x_q8     = nullptr;
    int64_t       ne00     = 0;
    int64_t       s_row    = 0;
    int64_t       ne0      = 0;
    int           n_blocks = 0;
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args,
                           const ggml_cuda_topk_moe_quant * quant = nullptr);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
