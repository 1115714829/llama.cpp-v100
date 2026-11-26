#pragma once

#include "common.cuh"

#if defined(__CUDA_ARCH_LIST__)
#define GGML_CUDA_FP8_SM70_COMPILED ggml_cuda_has_arch(GGML_CUDA_CC_VOLTA)
#else
#define GGML_CUDA_FP8_SM70_COMPILED false
#endif

bool ggml_cuda_fp8_sm70_mul_mat_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c);
void ggml_cuda_mul_mat_fp8_sm70(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stddef.h>
#include <stdint.h>

static __host__ __device__ inline uint16_t ggml_cuda_fp8_e4m3_half_bits(uint8_t x) {
    const unsigned a = x & 0x7f;
    const unsigned s = (unsigned) (x & 0x80) << 8;
    const unsigned e = a >> 3;
    const unsigned m = a & 7;
    if (a == 0x7f) { return 0x7e00; }
    if (e != 0) { return (uint16_t) (s | ((e + 8) << 10) | (m << 7)); }
    if (m >= 4) { return (uint16_t) (s | (8 << 10) | ((m - 4) << 8)); }
    if (m >= 2) { return (uint16_t) (s | (7 << 10) | ((m - 2) << 9)); }
    return (uint16_t) (s | (m == 1 ? 6 << 10 : 0));
}

enum ggml_cuda_fp8_sm70_status {
    GGML_FP8_SM70_SUCCESS,
    GGML_FP8_SM70_INVALID_ARGUMENT,
    GGML_FP8_SM70_INSUFFICIENT_WORKSPACE,
    GGML_FP8_SM70_UNSUPPORTED_DEVICE,
    GGML_FP8_SM70_CUDA_ERROR,
    GGML_FP8_SM70_CUBLAS_ERROR,
};

struct ggml_cuda_fp8_sm70_problem {
    int64_t m, n, k;
    const uint8_t * a;
    size_t a_bytes, a_row_stride;
    const half * b;
    size_t b_bytes, b_row_stride;
    float * c;
    size_t c_bytes, c_col_stride;
    const float * weight_scale;
    size_t weight_scale_bytes;
};

struct ggml_cuda_fp8_sm70_counters {
    unsigned long long decoded_elements;
    unsigned long long conversion_launches;
    int conversion_arch;
};
struct ggml_cuda_fp8_sm70_report {
    size_t workspace_bytes;
    uint64_t gemm_calls;
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
};

size_t ggml_cuda_fp8_sm70_workspace_size();
const ggml_cuda_fp8_sm70_counters * ggml_cuda_fp8_sm70_device_counters(const void * workspace);

ggml_cuda_fp8_sm70_status ggml_cuda_fp8_sm70_mul_mat(
    const ggml_cuda_fp8_sm70_problem & p, void * workspace, size_t workspace_bytes,
    cublasHandle_t handle, cudaStream_t stream, ggml_cuda_fp8_sm70_report * report);

ggml_cuda_fp8_sm70_status ggml_cuda_fp8_sm70_decode(
    const uint8_t * src, size_t src_bytes, size_t src_stride,
    half * dst, size_t dst_bytes, size_t dst_stride,
    int rows, int cols, cudaStream_t stream);

cudaError_t ggml_cuda_fp8_sm70_code_version(int * binary_version, int * ptx_version);
const char * ggml_cuda_fp8_sm70_status_name(ggml_cuda_fp8_sm70_status status);

#endif
