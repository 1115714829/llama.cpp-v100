#include "common.cuh"
#include "ggml.h"



void ggml_cuda_op_seq_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_seq_gather (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
