#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

// True when ggml_cuda_mul_mat computes this node with the split narrow GEMV (few long F32/BF16 rows).
bool ggml_cuda_mul_mat_vec_f_split_covers(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                          const ggml_tensor * src1, const ggml_tensor * dst);

// 2..4 MUL_MAT nodes that read one src1, each covered by ggml_cuda_mul_mat_vec_f_split_covers() and all of
// one weight type and K: the rows of all of them in one launch, bitwise the same results. Returns false
// (nothing launched) when a node is not covered or there are more rows than split counters.
bool ggml_cuda_mul_mat_vec_f_narrow_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4],
                                          ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1);
