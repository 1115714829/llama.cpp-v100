#include "../ggml-impl.h"
#include "fp8-sm70.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "convert.cuh"
#include <algorithm>
#include <climits>
#include <cstdio>

namespace {
constexpr int TM = 64, TN = 32, TK = 256;
constexpr size_t A_BYTES = TM * TK * sizeof(half);
constexpr size_t B_BYTES = TN * TK * sizeof(half);
constexpr size_t C_BYTES = TM * TN * sizeof(float);
constexpr size_t COUNTER_OFFSET = A_BYTES + B_BYTES + C_BYTES;
constexpr size_t WORK_BYTES = COUNTER_OFFSET + 256;
static_assert(sizeof(ggml_cuda_fp8_sm70_counters) <= 256, "counter reservation");

bool span(size_t rows, size_t cols, size_t stride, size_t element, size_t bytes, size_t & used) {
    if (rows == 0 || cols == 0 || cols > SIZE_MAX / element) { return false; }
    const size_t row = cols * element;
    if (stride < row || stride % element != 0 || (rows - 1) > (SIZE_MAX - row) / stride) { return false; }
    used = (rows - 1) * stride + row;
    return bytes >= used;
}

bool address_ok(const void * p, size_t bytes) {
    return p && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}

bool overlap(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + bn && y < x + an;
}

ggml_cuda_fp8_sm70_status device_status() {
    int device, major, minor;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return GGML_FP8_SM70_CUDA_ERROR;
    }
    return major == 7 && minor == 0 ? GGML_FP8_SM70_SUCCESS : GGML_FP8_SM70_UNSUPPORTED_DEVICE;
}

__global__ void decode_tile(const uint8_t * src, size_t src_stride, half * dst, size_t dst_stride,
                           int rows, int cols, int padded_rows, int padded_cols,
                           ggml_cuda_fp8_sm70_counters * counters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < padded_rows * padded_cols) {
        const int r = i / padded_cols, c = i % padded_cols;
        half * row = reinterpret_cast<half *>(reinterpret_cast<char *>(dst) + r * dst_stride);
        row[c] = __ushort_as_half(r < rows && c < cols ?
                ggml_cuda_fp8_e4m3_half_bits(src[r * src_stride + c]) : 0);
    }
    if (i == 0 && counters) {
        atomicAdd(&counters->decoded_elements, (unsigned long long) rows * cols);
        atomicAdd(&counters->conversion_launches, 1ULL);
#if defined(__CUDA_ARCH__)
        counters->conversion_arch = __CUDA_ARCH__;
#endif
    }
}

__global__ void pack_rhs(const half * src, size_t src_stride, half * dst,
                         int rows, int cols, int padded_rows, int padded_cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded_rows * padded_cols) { return; }
    const int r = i / padded_cols, c = i % padded_cols;
    half value = __ushort_as_half(0);
    if (r < rows && c < cols) {
        const half * row = reinterpret_cast<const half *>(reinterpret_cast<const char *>(src) + r * src_stride);
        value = row[c];
    }
    dst[i] = value;
}

__global__ void store_result(const float * tile, int padded_m, float * dst, size_t stride,
                             int m, int n, const float * scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) { return; }
    const int col = i / m, row = i % m;
    float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + col * stride);
    const float v = tile[col * padded_m + row];
    out[row] = scale ? v * *scale : v;
}
}

size_t ggml_cuda_fp8_sm70_workspace_size() { return WORK_BYTES; }

const ggml_cuda_fp8_sm70_counters * ggml_cuda_fp8_sm70_device_counters(const void * workspace) {
    return workspace ? reinterpret_cast<const ggml_cuda_fp8_sm70_counters *>(
        static_cast<const char *>(workspace) + COUNTER_OFFSET) : nullptr;
}

cudaError_t ggml_cuda_fp8_sm70_code_version(int * binary_version, int * ptx_version) {
    if (!binary_version || !ptx_version) { return cudaErrorInvalidValue; }
    cudaFuncAttributes attr{};
    const cudaError_t e = cudaFuncGetAttributes(&attr, decode_tile);
    if (e == cudaSuccess) { *binary_version = attr.binaryVersion; *ptx_version = attr.ptxVersion; }
    return e;
}

ggml_cuda_fp8_sm70_status ggml_cuda_fp8_sm70_decode(
    const uint8_t * src, size_t src_bytes, size_t src_stride,
    half * dst, size_t dst_bytes, size_t dst_stride,
    int rows, int cols, cudaStream_t stream) {
    size_t sn, dn;
    if (rows < 1 || rows > TM || cols < 1 || cols > TK ||
        !span(rows, cols, src_stride, 1, src_bytes, sn) ||
        !span(rows, cols, dst_stride, sizeof(half), dst_bytes, dn) ||
        !address_ok(src, sn) || !address_ok(dst, dn) ||
        reinterpret_cast<uintptr_t>(dst) % alignof(half) || overlap(src, sn, dst, dn)) {
        return GGML_FP8_SM70_INVALID_ARGUMENT;
    }
    const auto dev = device_status();
    if (dev != GGML_FP8_SM70_SUCCESS) { return dev; }
    decode_tile<<<(rows * cols + 255) / 256, 256, 0, stream>>>(
        src, src_stride, dst, dst_stride, rows, cols, rows, cols, nullptr);
    return cudaGetLastError() == cudaSuccess ? GGML_FP8_SM70_SUCCESS : GGML_FP8_SM70_CUDA_ERROR;
}

ggml_cuda_fp8_sm70_status ggml_cuda_fp8_sm70_mul_mat(
    const ggml_cuda_fp8_sm70_problem & p, void * workspace, size_t workspace_bytes,
    cublasHandle_t handle, cudaStream_t stream, ggml_cuda_fp8_sm70_report * report) {
    if (report) { *report = {WORK_BYTES, 0, cudaSuccess, CUBLAS_STATUS_SUCCESS}; }
    size_t as, bs, cs;
    if (p.m <= 0 || p.n <= 0 || p.k <= 0 || p.m > INT_MAX || p.n > INT_MAX || p.k > INT_MAX ||
        !span(p.m, p.k, p.a_row_stride, 1, p.a_bytes, as) ||
        !span(p.n, p.k, p.b_row_stride, sizeof(half), p.b_bytes, bs) ||
        !span(p.n, p.m, p.c_col_stride, sizeof(float), p.c_bytes, cs) ||
        !address_ok(p.a, as) || !address_ok(p.b, bs) || !address_ok(p.c, cs) || !handle ||
        reinterpret_cast<uintptr_t>(p.b) % alignof(half) ||
        reinterpret_cast<uintptr_t>(p.c) % alignof(float) ||
        (p.weight_scale && (p.weight_scale_bytes < sizeof(float) ||
            !address_ok(p.weight_scale, sizeof(float)) || reinterpret_cast<uintptr_t>(p.weight_scale) % alignof(float))) ||
        (!p.weight_scale && p.weight_scale_bytes != 0) ||
        overlap(p.a, as, p.c, cs) || overlap(p.b, bs, p.c, cs) ||
        (p.weight_scale && overlap(p.weight_scale, sizeof(float), p.c, cs))) {
        return GGML_FP8_SM70_INVALID_ARGUMENT;
    }
    if (!workspace || workspace_bytes < WORK_BYTES) { return GGML_FP8_SM70_INSUFFICIENT_WORKSPACE; }
    if (!address_ok(workspace, WORK_BYTES) || reinterpret_cast<uintptr_t>(workspace) % 256 ||
        overlap(workspace, WORK_BYTES, p.a, as) || overlap(workspace, WORK_BYTES, p.b, bs) ||
        overlap(workspace, WORK_BYTES, p.c, cs) ||
        (p.weight_scale && overlap(workspace, WORK_BYTES, p.weight_scale, sizeof(float)))) {
        return GGML_FP8_SM70_INVALID_ARGUMENT;
    }
    const auto dev = device_status();
    if (dev != GGML_FP8_SM70_SUCCESS) { return dev; }
    cudaStream_t handle_stream;
    cublasPointerMode_t pointer_mode;
    cublasMath_t math_mode;
    cublasStatus_t cb = cublasGetStream(handle, &handle_stream);
    if (cb == CUBLAS_STATUS_SUCCESS) { cb = cublasGetPointerMode(handle, &pointer_mode); }
    if (cb == CUBLAS_STATUS_SUCCESS) { cb = cublasGetMathMode(handle, &math_mode); }
    if (cb != CUBLAS_STATUS_SUCCESS) {
        if (report) { report->cublas_status = cb; }
        return GGML_FP8_SM70_CUBLAS_ERROR;
    }
    if (handle_stream != stream || pointer_mode != CUBLAS_POINTER_MODE_HOST ||
        (math_mode != CUBLAS_TENSOR_OP_MATH && math_mode != CUBLAS_TF32_TENSOR_OP_MATH)) {
        return GGML_FP8_SM70_INVALID_ARGUMENT;
    }
    char * w = static_cast<char *>(workspace);
    half * at = reinterpret_cast<half *>(w);
    half * bt = reinterpret_cast<half *>(w + A_BYTES);
    float * ct = reinterpret_cast<float *>(w + A_BYTES + B_BYTES);
    auto * counters = reinterpret_cast<ggml_cuda_fp8_sm70_counters *>(w + COUNTER_OFFSET);
    cudaError_t ce = cudaMemsetAsync(counters, 0, sizeof(*counters), stream);
    if (ce != cudaSuccess) {
        if (report) { report->cuda_status = ce; }
        return GGML_FP8_SM70_CUDA_ERROR;
    }
    for (int64_t n0 = 0; n0 < p.n; n0 += TN) {
        const int nt = (int) std::min<int64_t>(TN, p.n - n0), np = (nt + 7) & ~7;
        for (int64_t m0 = 0; m0 < p.m; m0 += TM) {
            const int mt = (int) std::min<int64_t>(TM, p.m - m0), mp = (mt + 7) & ~7;
            for (int64_t k0 = 0; k0 < p.k; k0 += TK) {
                const int kt = (int) std::min<int64_t>(TK, p.k - k0), kp = (kt + 7) & ~7;
                decode_tile<<<(mp * kp + 255) / 256, 256, 0, stream>>>(
                    p.a + m0 * p.a_row_stride + k0, p.a_row_stride, at, kp * sizeof(half), mt, kt, mp, kp, counters);
                ce = cudaGetLastError();
                if (ce == cudaSuccess) {
                    pack_rhs<<<(np * kp + 255) / 256, 256, 0, stream>>>(
                        reinterpret_cast<const half *>(reinterpret_cast<const char *>(p.b) + n0 * p.b_row_stride) + k0,
                        p.b_row_stride, bt, nt, kt, np, kp);
                    ce = cudaGetLastError();
                }
                if (ce != cudaSuccess) {
                    if (report) { report->cuda_status = ce; }
                    return GGML_FP8_SM70_CUDA_ERROR;
                }
                const float alpha = 1.0f, beta = k0 == 0 ? 0.0f : 1.0f;

                cb = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, mp, np, kp,
                    &alpha, at, CUDA_R_16F, kp, bt, CUDA_R_16F, kp,
                    &beta, ct, CUDA_R_32F, mp, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (cb != CUBLAS_STATUS_SUCCESS) {
                    if (report) { report->cublas_status = cb; }
                    return GGML_FP8_SM70_CUBLAS_ERROR;
                }
                if (report) { ++report->gemm_calls; }
            }
            store_result<<<(mt * nt + 255) / 256, 256, 0, stream>>>(ct, mp,
                reinterpret_cast<float *>(reinterpret_cast<char *>(p.c) + n0 * p.c_col_stride) + m0,
                p.c_col_stride, mt, nt, p.weight_scale);
            ce = cudaGetLastError();
            if (ce != cudaSuccess) {
                if (report) { report->cuda_status = ce; }
                return GGML_FP8_SM70_CUDA_ERROR;
            }
        }
    }
    return GGML_FP8_SM70_SUCCESS;
}

const char * ggml_cuda_fp8_sm70_status_name(ggml_cuda_fp8_sm70_status s) {
    switch (s) {
        case GGML_FP8_SM70_SUCCESS: return "success";
        case GGML_FP8_SM70_INVALID_ARGUMENT: return "invalid-argument";
        case GGML_FP8_SM70_INSUFFICIENT_WORKSPACE: return "insufficient-workspace";
        case GGML_FP8_SM70_UNSUPPORTED_DEVICE: return "unsupported-device";
        case GGML_FP8_SM70_CUDA_ERROR: return "cuda-error";
        case GGML_FP8_SM70_CUBLAS_ERROR: return "cublas-error";
    }
    return "unknown";
}

bool ggml_cuda_fp8_sm70_mul_mat_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c) {
    if (a->type != GGML_TYPE_F8_E4M3 || c->type != GGML_TYPE_F32) return false;
    if (b->type != GGML_TYPE_F16 && b->type != GGML_TYPE_F32 && b->type != GGML_TYPE_BF16) return false;
    if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 || c->ne[2] != 1 || c->ne[3] != 1) return false;
    if (a->ne[0] != b->ne[0] || a->ne[1] != c->ne[0] || b->ne[1] != c->ne[1]) return false;
    if (a->nb[0] != 1 || b->nb[0] != ggml_element_size(b) || c->nb[0] != sizeof(float)) return false;
    if (a->nb[1] < a->ne[0] || b->nb[1] < b->ne[0]*ggml_element_size(b) || c->nb[1] < c->ne[0]*sizeof(float)) return false;

    if ((a->ne[1]-1)*a->nb[1] + a->ne[0] > ggml_nbytes(a)) return false;
    if ((b->ne[1]-1)*b->nb[1] + b->ne[0]*ggml_element_size(b) > ggml_nbytes(b)) return false;
    if ((c->ne[1]-1)*c->nb[1] + c->ne[0]*sizeof(float) > ggml_nbytes(c)) return false;
    return true;
}

void ggml_cuda_mul_mat_fp8_sm70(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_fp8_sm70_mul_mat_ok(src0, src1, dst));

    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const int64_t n = src1->ne[1];

    ggml_cuda_fp8_sm70_problem p = {};
    p.m = m;
    p.n = n;
    p.k = k;
    p.a            = (const uint8_t *) src0->data;
    p.a_bytes      = ggml_nbytes(src0);
    p.a_row_stride = src0->nb[1];
    p.c            = (float *) dst->data;
    p.c_bytes      = ggml_nbytes(dst);
    p.c_col_stride = dst->nb[1];
    p.weight_scale       = nullptr;
    p.weight_scale_bytes = 0;

    ggml_cuda_pool_alloc<char> ws(ctx.pool());
    ws.alloc(ggml_cuda_fp8_sm70_workspace_size());
    if (ws.get() == nullptr) {
        GGML_ABORT("%s: fp8-sm70 workspace pool exhausted (%zu bytes)", __func__, (size_t) ggml_cuda_fp8_sm70_workspace_size());
    }

    ggml_cuda_pool_alloc<half> b16(ctx.pool());
    if (src1->type == GGML_TYPE_F16) {
        p.b            = (const half *) src1->data;
        p.b_bytes      = ggml_nbytes(src1);
        p.b_row_stride = src1->nb[1];
    } else {

        b16.alloc(ggml_nelements(src1));
        if (b16.get() == nullptr) {
            GGML_ABORT("%s: fp8-sm70 rhs conversion buffer pool exhausted", __func__);
        }
        if (ggml_is_contiguously_allocated(src1)) {
            ggml_get_to_fp16_cuda(src1->type)(src1->data, b16.get(), ggml_nelements(src1), ctx.stream());
        } else {
            const size_t src1_ts = ggml_type_size(src1->type);
            ggml_get_to_fp16_nc_cuda(src1->type)(src1->data, b16.get(),
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[1] / src1_ts, src1->nb[2] / src1_ts, src1->nb[3] / src1_ts, ctx.stream());
        }
        CUDA_CHECK(cudaGetLastError());
        p.b            = b16.get();
        p.b_bytes      = ggml_nelements(src1)*sizeof(half);
        p.b_row_stride = k*sizeof(half);
    }

    ggml_cuda_fp8_sm70_report rep;
    const ggml_cuda_fp8_sm70_status st = ggml_cuda_fp8_sm70_mul_mat(p, ws.get(), ggml_cuda_fp8_sm70_workspace_size(),
        ctx.cublas_handle(), ctx.stream(), &rep);
    if (st != GGML_FP8_SM70_SUCCESS) {
        GGML_ABORT("%s: fp8-sm70 mul_mat failed: %s (m=%lld n=%lld k=%lld)", __func__,
                   ggml_cuda_fp8_sm70_status_name(st), (long long) m, (long long) n, (long long) k);
    }

    static bool traced = false;
    if (!traced) {
        traced = true;
        GGML_LOG_DEBUG("fp8 mm %lld %lld %lld %s\n",
                (long long) m, (long long) n, (long long) k, ggml_type_name(src1->type));

        const ggml_cuda_fp8_sm70_counters * c = ggml_cuda_fp8_sm70_device_counters(ws.get());
        ggml_cuda_fp8_sm70_counters h = {};
        if (cudaMemcpyAsync(&h, c, sizeof(h), cudaMemcpyDeviceToHost, ctx.stream()) == cudaSuccess &&
            cudaStreamSynchronize(ctx.stream()) == cudaSuccess) {
            GGML_LOG_DEBUG("fp8 run %d %llu %llu\n",
                    h.conversion_arch, h.decoded_elements, h.conversion_launches);
        }
    }
}
#endif
