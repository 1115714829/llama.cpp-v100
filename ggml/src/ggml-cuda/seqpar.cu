// Sequence-parallel token split/gather, single-device forms.
//
// Under the meta backend the split/gather across devices is carried by the per-rank
// reduce-scatter/all-gather comm procs in ggml-cuda.cu; these ops then only run on a
// single device, where seq_scatter copies the [t0, t0+n) slice of the token axis and
// seq_gather is the cast/dup of the whole tensor.
#include "common.cuh"
#include "cpy.cuh"
#include "seqpar.cuh"


void ggml_cuda_op_seq_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    const int32_t axis = ggml_get_op_params_i32(dst, 0);
    const int64_t t0   = ggml_get_op_params_i32(dst, 1);
    const int64_t n    = ggml_get_op_params_i32(dst, 2);
    GGML_ASSERT(axis >= 0 && axis < GGML_MAX_DIMS);
    GGML_ASSERT(dst->ne[axis] == n);
    GGML_ASSERT(t0 >= 0 && t0 + n <= src0->ne[axis]);

    int64_t inner = 1;
    for (int i = 0; i < axis; i++) {
        inner *= dst->ne[i];
    }
    int64_t outer = 1;
    for (int i = axis + 1; i < GGML_MAX_DIMS; i++) {
        outer *= dst->ne[i];
    }

    const size_t row_bytes = (size_t) inner * ggml_type_size(dst->type);
    cudaStream_t stream = ctx.stream();

    if (t0 == 0 && n == src0->ne[axis]) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src0->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    for (int64_t o = 0; o < outer; o++) {
        const char * src = (const char *) src0->data + (size_t) (o*src0->ne[axis] + t0)*row_bytes;
        char       * d   = (char *)       dst->data  + (size_t)  o*n*row_bytes;
        CUDA_CHECK(cudaMemcpyAsync(d, src, (size_t) n*row_bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void ggml_cuda_op_seq_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    ggml_cuda_cpy(ctx, src0, dst);
}
