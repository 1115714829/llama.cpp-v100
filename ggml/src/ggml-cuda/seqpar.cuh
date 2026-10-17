#include "common.cuh"
#include "ggml.h"

// Sequence-parallel token split/gather, single-device (identity/slice/cast) forms.
// The multi-device reduce-scatter/all-gather forms are the comm procs in ggml-cuda.cu.
void ggml_cuda_op_seq_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_seq_gather (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
