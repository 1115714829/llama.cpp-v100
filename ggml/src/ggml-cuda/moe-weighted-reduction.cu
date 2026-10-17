#include "moe-weighted-reduction.cuh"

#include "ggml-cuda-expert.cuh"

// M3: with expert parallelism the down projection only writes the slots of the experts this device
// holds. The slots of the other devices used to be read from the zeroed rows the MUL_MAT_ID launch
// wrote; they are skipped here instead, from the ids and the expert map of the node, so neither the
// zero rows nor their bytes are needed. The value substituted for such a slot is 0.0f, the value the
// zero kernel stored, so the arithmetic (and with it the sum) is the same term by term.
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const int32_t * __restrict__ ids,
                                                  const int     ids_stride,
                                                  const ggml_cuda_expert_lut lut) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    // the slot is computed on this device when the expert map knows its id; ids == nullptr (no
    // expert parallelism) means every slot is
    const bool first_local = ids == nullptr || ggml_cuda_expert_local(lut, ids[token*ids_stride]) >= 0;
    const float first_val  = first_local ? experts[first_row * n_embd + col] : 0.0f;
    float          sum         = (first_val * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const bool    local = ids == nullptr || ggml_cuda_expert_local(lut, ids[token*ids_stride + expert]) >= 0;
        const float   val   = local ? experts[row * n_embd + col] : 0.0f;
        sum += (val * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          const int32_t * ids,
                                          int             ids_stride,
                                          const ggml_cuda_expert_lut & lut,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used, ids, ids_stride, lut);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         mm_id) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    // the routing ids and the expert map of the node that wrote experts, when it is an expert
    // parallel MUL_MAT_ID: they tell the kernel which slots no device wrote
    const int32_t * ids          = nullptr;
    int             ids_stride   = 0;
    ggml_cuda_expert_lut lut{};
    if (mm_id != nullptr && ggml_mul_mat_id_is_expert_parallel(mm_id)) {
        const ggml_tensor * ids_t = mm_id->src[2];
        GGML_ASSERT(experts->ne[3] == 1 && ids_t->ne[3] == 1 && "the ids are indexed by the token of the batch");
        ids        = (const int32_t *) ids_t->data;
        ids_stride = ids_t->nb[1] / (int64_t) sizeof(int32_t);
        lut        = ggml_cuda_mul_mat_id_lut(ctx, mm_id);
    }

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used,
                                  ids, ids_stride, lut, stream);
    CUDA_CHECK(cudaGetLastError());
}
