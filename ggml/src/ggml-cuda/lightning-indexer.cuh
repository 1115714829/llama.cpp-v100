#include "common.cuh"

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst);

// unmasked indexer scores of a tile of tokens, see QSA_SELECT
void ggml_cuda_lightning_indexer_scores(
        const float * q_d, size_t nbq1, size_t nbq2, int64_t n_batch,
        const ggml_tensor * k, const float * w_d, float * dst_d, size_t nb1, cudaStream_t stream);
