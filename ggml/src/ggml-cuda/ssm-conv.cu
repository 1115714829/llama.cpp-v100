#include "common.cuh"
#include "convert.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

static constexpr size_t d_conv_fused = 4;

static __global__ void ssm_conv_rollback_fused_f32(const ggml_cuda_ssm_conv_rollback_params p) {
    ggml_cuda_pdl_lc();

    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= p.n_channels) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float * state_c = p.state + (int64_t) p.row[0] * p.state_stride + c * (d_conv_fused - 1);

    float x[d_conv_fused];
#pragma unroll
    for (size_t j = 0; j < d_conv_fused - 1; ++j) {
        x[j] = state_c[j];
    }

    for (int k = 0; k < p.n_snap; ++k) {
        float * snap_c = p.snap_dst[k] + c * (d_conv_fused - 1);
#pragma unroll
        for (size_t j = 0; j < d_conv_fused - 1; ++j) {
            const int64_t t = p.snap_off[k] + j - (d_conv_fused - 1);
            snap_c[j] = t < 0 ? x[p.snap_off[k] + j] : p.qkv[t * p.qkv_nb0 + c * p.qkv_nb1];
        }
    }

    float w[d_conv_fused];
#pragma unroll
    for (size_t j = 0; j < d_conv_fused; ++j) {
        w[j] = p.w[c * p.w_nb1 + j];
    }

    x[d_conv_fused - 1] = p.qkv[c * p.qkv_nb1];

    float * dst_c = p.dst + c;
    for (int64_t i = 0; i < p.n_tok; ++i) {
        if (i > 0) {
            x[(i - 1) % d_conv_fused] = p.qkv[i * p.qkv_nb0 + c * p.qkv_nb1];
        }

        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv_fused; ++j) {
            sumf += x[(i + j) % d_conv_fused] * w[j];
        }
        dst_c[i * p.dst_nb1] = ggml_cuda_op_silu_single(sumf);
    }
}

void ggml_cuda_op_ssm_conv_rollback_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & params) {
    const int    threads = 256;
    const dim3   blocks((params.n_channels + threads - 1) / threads);
    cudaStream_t stream = ctx.stream();

    ssm_conv_rollback_fused_f32<<<blocks, threads, 0, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ssm_conv_rollback_halo_f32(const ggml_cuda_ssm_conv_rollback_params p,
                                                 const float * history, float * halo) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const float * q = p.qkv + c * p.qkv_nb1;
    const int64_t tile_idx = blockIdx.y;
    if (tile_idx != 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            halo[((tile_idx - 1) * 3 + j) * p.n_channels + c] = q[(tile_idx * 32 - 3 + j) * p.qkv_nb0];
        }
    } else {
        for (int k = 0; k < p.n_snap; ++k) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int64_t t = p.snap_off[k] + j - 3;
                p.snap_dst[k][c * 3 + j] = t < 0 ? history[c * 3 + t + 3] : q[t * p.qkv_nb0];
            }
        }
    }
}

static __global__ void ssm_conv_rollback_long_f32(const ggml_cuda_ssm_conv_rollback_params p,
                                                  const float * history, const float * halo) {
    constexpr int tile = 32;
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t begin = (int64_t) blockIdx.y * tile;
    const int64_t remaining = p.n_tok - begin;
    const int count = remaining < tile ? (int) remaining : tile;
    const float * q = p.qkv + c * p.qkv_nb1;
    const float * w = p.w + c * p.w_nb1;
    const int64_t h = ((int64_t) blockIdx.y - 1) * 3 * p.n_channels + c;
    float x0 = begin == 0 ? history[c * 3 + 0] : (halo ? halo[h] : q[(begin - 3) * p.qkv_nb0]);
    float x1 = begin == 0 ? history[c * 3 + 1] : (halo ? halo[h + p.n_channels] : q[(begin - 2) * p.qkv_nb0]);
    float x2 = begin == 0 ? history[c * 3 + 2] : (halo ? halo[h + 2 * p.n_channels] : q[(begin - 1) * p.qkv_nb0]);
    for (int i = 0; i < count; ++i) {
        const float x3 = q[(begin + i) * p.qkv_nb0];
        float sumf = 0.0f;
        sumf += x0 * w[0];
        sumf += x1 * w[1];
        sumf += x2 * w[2];
        sumf += x3 * w[3];
        sumf += 0.0f;
        p.dst[(begin + i) * p.dst_nb1 + c] = ggml_cuda_op_silu_single(sumf);
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    if (halo == nullptr && blockIdx.y == 0) {
        for (int k = 0; k < p.n_snap; ++k) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int64_t t = p.snap_off[k] + j - 3;
                p.snap_dst[k][c * 3 + j] = t < 0 ? history[c * 3 + p.snap_off[k] + j] : q[t * p.qkv_nb0];
            }
        }
    }
}

void ggml_cuda_op_ssm_conv_rollback_long(ggml_backend_cuda_context & ctx,
                                        const ggml_cuda_ssm_conv_rollback_params & params, const float * history, float * halo) {
    constexpr int threads = 128;
    const dim3 blocks(params.n_channels / threads, (params.n_tok + 31) / 32);
    if (halo != nullptr) {
        ssm_conv_rollback_halo_f32<<<blocks, threads, 0, ctx.stream()>>>(params, history, halo);
        CUDA_CHECK(cudaGetLastError());
    }
    ssm_conv_rollback_long_f32<<<blocks, threads, 0, ctx.stream()>>>(params, history, halo);
    CUDA_CHECK(cudaGetLastError());
}

template<typename W>
static __global__ void ple_conv4_f32(const ggml_cuda_ssm_conv_rollback_params p, const float * history, const W * weights, const float * hidden, const float * gated) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const float * q = p.qkv + c;
    const float * h = history + c * 9;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int64_t j = t - 9 + 3 * k;
        const float x = j < 0 ? h[j + 9] : q[j * p.qkv_nb0];
        const float term = __fmul_rn(x, ggml_cuda_cast<float>(weights[c * p.w_nb1 + k]));
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const int64_t idx = t * p.dst_nb1 + c;
    const float activation = ggml_cuda_op_silu_single(sum);
    p.dst[idx] = hidden ? __fadd_rn(hidden[idx], __fadd_rn(gated[idx], activation)) : activation;
    if (t == 0) {
        for (int k = 0; k < p.n_snap; ++k) {
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                const int64_t off = p.snap_off[k] + j;
                p.snap_dst[k][c * 9 + j] = off < 9 ? h[off] : q[(off - 9) * p.qkv_nb0];
            }
        }
    }
}

void ggml_cuda_op_ple_conv4(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_rollback_params & p,
                           const float * history, const ggml_tensor * weights, const float * hidden, const float * gated) {
    const dim3 blocks(p.n_channels / 128, p.n_tok);
    if (weights->type == GGML_TYPE_F16) {
        ple_conv4_f32<<<blocks, 128, 0, ctx.stream()>>>(p, history, (const half *) weights->data, hidden, gated);
    } else if (weights->type == GGML_TYPE_BF16) {
        ple_conv4_f32<<<blocks, 128, 0, ctx.stream()>>>(p, history, (const nv_bfloat16 *) weights->data, hidden, gated);
    } else {
        ple_conv4_f32<<<blocks, 128, 0, ctx.stream()>>>(p, history, (const float *) weights->data, hidden, gated);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ple_gate_f32(const float * key, const float * query, float * dst, int width, float scale) {
    const int64_t row = blockIdx.x;
    key += row * width;
    query += row * width;
    float partial[8] = {};
    for (int col = threadIdx.x; col < width;) {
#pragma unroll
        for (int j = 0; j < 8; ++j, col += blockDim.x) {
            partial[j] += col < width ? __fmul_rn(key[col], query[col]) : 0.0f;
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += partial[j];
    __shared__ float shared[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared);
    if (threadIdx.x == 0) {
        const float x = __fmaf_rn(scale, sum, 0.0f);
        const float mag = sqrtf(fmaxf(fabsf(x), 1e-6f));
        dst[row] = ggml_cuda_op_sigmoid_single(__fmul_rn(float((x > 0.0f) - (x < 0.0f)), mag));
    }
}

void ggml_cuda_op_ple_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * key,
                          const ggml_tensor * query, float * dst, float scale) {
    const int rows = ggml_nrows(key);
    const int threads = rows / ggml_cuda_info().devices[ctx.device].nsm < 2 ? 512 : (key->ne[0] < 1024 ? 32 : 128);
    ple_gate_f32<<<rows, threads, 0, ctx.stream()>>>((const float *) key->data, (const float *) query->data,
                                                   dst, key->ne[0], scale);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ple_broadcast_f32(const float * value, const float * gate, float * dst,
                                         int64_t width, int64_t streams, int64_t count) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t row = i / width;
        dst[i] = __fmul_rn(value[(row / streams) * width + i % width], gate[row]);
    }
}

void ggml_cuda_op_ple_broadcast(ggml_backend_cuda_context & ctx, const ggml_tensor * value,
                               const ggml_tensor * gate, ggml_tensor * dst) {
    const int64_t count = ggml_nelements(dst);
    ple_broadcast_f32<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(
        (const float *) value->data, (const float *) gate->data, (float *) dst->data, dst->ne[0], dst->ne[1], count);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    GGML_ASSERT(!fuse_bias || fuse_silu);

    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = out->ne[1];
    const int64_t n_s = out->ne[2];

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
