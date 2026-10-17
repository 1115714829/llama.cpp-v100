#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmvf-narrow.cuh"
#include "fwht.cuh"
#include "mmf.cuh"
#include "ggml-cuda-expert.cuh"
#include "convert.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const ggml_cuda_expert_lut lut) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row         = blockIdx.x;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    // expert parallelism: the expert map gives the local expert of the routed id, -1 for an expert
    // held by another device - that slot is written as zeros
    const int expert_local = ids != nullptr ? ggml_cuda_expert_local(lut, channel_x) : channel_x;
    if (expert_local < 0) {
        if (tid < ncols_dst) {
            float * dst_e = dst + int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst + row;
            if constexpr (is_multi_token_id) {
                dst_e += token_idx*stride_col_dst;
            }
            dst_e[tid*stride_col_dst] = 0.0f;
        }
        return;
    }

    // src0 only holds the local experts, the weight of the routed expert is its local number
    const int channel_x_local = expert_local;

    // expert store: the weight base comes from the device pointer table, the channel stride
    // within src0 is then not used
    if (ids != nullptr && fusion.expert_ptrs != nullptr) {
        x = (const T *) (fusion.expert_ptrs[channel_x_local]) + int64_t(sample_x) *stride_sample_x + row*stride_row;
    } else {
        x += int64_t(sample_x)  *stride_sample_x   + channel_x_local*stride_channel_x + row*stride_row;
    }
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    float glu_limit = 0.0f;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;
        glu_limit = fusion.glu_limit;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        // expert store: the gate weight of a fused gate/up pair is read through its own table
        if (ids != nullptr && fusion.expert_ptrs_gate != nullptr) {
            gate_x = (const T *) (fusion.expert_ptrs_gate[channel_x_local]) + int64_t(sample_x) *stride_sample_x + row*stride_row;
        } else {
            gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x_local*stride_channel_x   + row*stride_row;
        }
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x_local : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, glu_limit);
                    break;
                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, glu_limit, gate_x, x_bias, gate_bias, sumf_gate);
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride,
        const ggml_cuda_expert_lut lut, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, const ggml_cuda_expert_lut lut, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, lut, stream);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, const ggml_cuda_expert_lut lut, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, lut, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, lut, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, lut, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, const ggml_cuda_expert_lut lut, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, stream);
}

// ---- narrow F32/BF16 GEMV: few rows with a long K, or many rows at M > 1 ----
//
// mul_mat_vec_f runs one block per output row. For a few long rows (the Qwen3.8-Flash-Next
// decode graph has the hyper-connection inject [10240, 4], the GDN beta/alpha slices [2560, 8]
// and the shared expert gate [2560, 1]) that is 1 to 8 SMs, each pulling its row and the whole
// input through one SM at the per-SM load limit. For many rows at M > 1 (the router [2560, 512]
// at M = 4) every block reads the whole input again, 20 MiB of L2 reads for 5 MiB of weights.
// The two kernels below keep the arithmetic of mul_mat_vec_f with its 256-thread block (thread
// t adds columns t, t + 256, ... with the same multiply-adds, then the same warp and block
// reductions), so the results are bitwise the same:
//  - mul_mat_vec_f_split runs each of the 8 warps of a row as its own CTA; a warp stores its
//    warp-reduced sums, the last CTA of the row (one atomic counter per row) reduces the 8 sums
//    with the same 32-lane butterfly (lanes 8..31 add zero, as the zeroed shared buffer does)
//    and resets the counter, so the counters are zero between launches;
//  - mul_mat_vec_f_rows runs Rows rows per 256-thread block and loads each input value once.
// Only taken when mul_mat_vec_f would use a 256-thread block (K a multiple of 512).

// mmvf_narrow_block_size, mul_mat_vec_f_narrow_accumulate() and the split warp: mmvf-narrow.cuh

template <typename T, int ncols_dst>
static __global__ void __launch_bounds__(WARP_SIZE) mul_mat_vec_f_split(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols2, const int64_t stride_row, const int stride_col_y2, const int stride_col_dst,
        float * __restrict__ scratch, unsigned int * __restrict__ counters) {
    constexpr int n_warps = mmvf_narrow_split_warps;
    const int row  = blockIdx.x / n_warps;
    const int warp = blockIdx.x % n_warps;
    mul_mat_vec_f_split_warp<T, ncols_dst>(x + row*stride_row, (const float2 *) y, ncols2, stride_col_y2,
                                           dst + row, stride_col_dst, scratch + (int64_t) row*n_warps*ncols_dst,
                                           counters + row, warp, threadIdx.x);
}

// Several narrow weights that read one input (the GDN ssm_beta and ssm_alpha slices of a qwen4exp
// layer, expanded next to each other): the rows of all of them in one launch, every row on the warps
// of mul_mat_vec_f_split. first_row is the running sum of the row counts of the previous weights;
// row r of the launch uses scratch row r and counter r.
struct mmvf_narrow_multi_seg {
    const void * x;
    float *      dst;
    int64_t      stride_row;
    int          stride_col_dst;
    int          first_row;
};

struct mmvf_narrow_multi_params {
    mmvf_narrow_multi_seg seg[4];
    int                   n_seg;
    const float *         y;
    int                   ncols2;
    int                   stride_col_y2;
    float *               scratch;
    unsigned int *        counters;
};

template <typename T, int ncols_dst>
static __global__ void __launch_bounds__(WARP_SIZE) mul_mat_vec_f_split_multi(const mmvf_narrow_multi_params p) {
    constexpr int n_warps = mmvf_narrow_split_warps;
    const int row_all = blockIdx.x / n_warps;
    const int warp    = blockIdx.x % n_warps;
    int s = 0;
    while (s + 1 < p.n_seg && row_all >= p.seg[s + 1].first_row) {
        ++s;
    }
    const int row = row_all - p.seg[s].first_row;
    mul_mat_vec_f_split_warp<T, ncols_dst>((const T *) p.seg[s].x + row*p.seg[s].stride_row, (const float2 *) p.y,
                                           p.ncols2, p.stride_col_y2, p.seg[s].dst + row, p.seg[s].stride_col_dst,
                                           p.scratch + (int64_t) row_all*n_warps*ncols_dst, p.counters + row_all,
                                           warp, threadIdx.x);
}

template <typename T, int ncols_dst, int rows>
static __global__ void __launch_bounds__(mmvf_narrow_block_size) mul_mat_vec_f_rows(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols2, const int64_t stride_row, const int stride_col_y2, const int stride_col_dst) {
    constexpr int n_warps = mmvf_narrow_block_size / WARP_SIZE;
    const int row0 = blockIdx.x * rows;
    const int tid  = threadIdx.x;
    __shared__ float buf_iw[rows][WARP_SIZE];

    if (tid < WARP_SIZE) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            buf_iw[r][tid] = 0.0f;
        }
    }
    __syncthreads();

    float sumf[rows][ncols_dst];
#pragma unroll
    for (int r = 0; r < rows; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[r][j] = 0.0f;
        }
    }

    const float2 * y2 = (const float2 *) y;
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll 2
        for (int col2 = tid; col2 < ncols2; col2 += mmvf_narrow_block_size) {
            float2 tmpy[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmpy[j] = y2[j*stride_col_y2 + col2];
            }
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                const float2 tmpx = ((const float2 *) (x + (row0 + r)*stride_row))[col2];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    ggml_cuda_mad(sumf[r][j], tmpx.x, tmpy[j].x);
                    ggml_cuda_mad(sumf[r][j], tmpx.y, tmpy[j].y);
                }
            }
        }
    } else {
        static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported type");
#pragma unroll 2
        for (int col2 = tid; col2 < ncols2; col2 += mmvf_narrow_block_size) {
            float2 tmpy[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmpy[j] = y2[j*stride_col_y2 + col2];
            }
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                const nv_bfloat162 tmpx = ((const nv_bfloat162 *) (x + (row0 + r)*stride_row))[col2];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    ggml_cuda_mad(sumf[r][j], tmpx.x, tmpy[j].x);
                    ggml_cuda_mad(sumf[r][j], tmpx.y, tmpy[j].y);
                }
            }
        }
    }

    // the reductions of mul_mat_vec_f, per row
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            sumf[r][j] = warp_reduce_sum<WARP_SIZE>(sumf[r][j]);
            buf_iw[r][tid/WARP_SIZE] = sumf[r][j];
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                sumf[r][j] = warp_reduce_sum<WARP_SIZE>(buf_iw[r][tid]);
            }
        }
        __syncthreads();
    }
    GGML_UNUSED(n_warps);

    if (tid < ncols_dst) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (tid == j) {
#pragma unroll
                for (int r = 0; r < rows; ++r) {
                    dst[j*stride_col_dst + row0 + r] = sumf[r][j];
                }
            }
        }
    }
}

template <typename T, int ncols_dst>
static void mul_mat_vec_f_narrow_launch(ggml_backend_cuda_context & ctx, const T * x, const float * y, float * dst,
                                        int64_t ncols, int64_t nrows, int64_t stride_row, int64_t stride_col_y,
                                        int64_t stride_col_dst) {
    const int ncols2 = (int) (ncols / 2);
    if (nrows <= 16) {
        constexpr int n_warps = mmvf_narrow_split_warps;
        GGML_ASSERT(ctx.tile_counters != nullptr && nrows <= GGML_CUDA_MAX_SPLIT_TILES);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), nrows * n_warps * ncols_dst);
        mul_mat_vec_f_split<T, ncols_dst><<<nrows * n_warps, WARP_SIZE, 0, ctx.stream()>>>(
            x, y, dst, ncols2, stride_row, (int) (stride_col_y / 2), (int) stride_col_dst, scratch.get(),
            ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES);
    } else {
        constexpr int rows = 4;
        mul_mat_vec_f_rows<T, ncols_dst, rows><<<nrows / rows, mmvf_narrow_block_size, 0, ctx.stream()>>>(
            x, y, dst, ncols2, stride_row, (int) (stride_col_y / 2), (int) stride_col_dst);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Returns false if the shape is not one of the two narrow cases; the caller runs mul_mat_vec_f.
static bool ggml_cuda_mul_mat_vec_f_narrow(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ncols     = src0->ne[0];
    const int64_t nrows     = src0->ne[1];
    const int64_t ncols_dst = src1->ne[1];
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16) ||
            ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_VOLTA ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            ncols % (2*mmvf_narrow_block_size) != 0 || ncols_dst < 1 || ncols_dst > MMVF_MAX_BATCH_SIZE) {
        return false;
    }
    const bool few_rows  = nrows <= 16 && ncols / (2*mmvf_narrow_block_size) >= 4;
    const bool many_rows = nrows > 16 && ncols_dst >= 2 && nrows % 4 == 0;
    if (!few_rows && !many_rows) {
        return false;
    }

    const int64_t stride_row     = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t stride_col_y   = src1->nb[1] / sizeof(float);
    const int64_t stride_col_dst = dst->nb[1] / sizeof(float);
    const float * y = (const float *) src1->data;
    float * d = (float *) dst->data;

#define MMVF_NARROW_CASE(N)                                                                                  \
        case N:                                                                                              \
            if (src0->type == GGML_TYPE_F32) {                                                               \
                mul_mat_vec_f_narrow_launch<float, N>(ctx, (const float *) src0->data, y, d, ncols, nrows,   \
                                                      stride_row, stride_col_y, stride_col_dst);             \
            } else {                                                                                         \
                mul_mat_vec_f_narrow_launch<nv_bfloat16, N>(ctx, (const nv_bfloat16 *) src0->data, y, d,     \
                                                            ncols, nrows, stride_row, stride_col_y,          \
                                                            stride_col_dst);                                 \
            }                                                                                                \
            break
    switch (ncols_dst) {
        MMVF_NARROW_CASE(1); MMVF_NARROW_CASE(2); MMVF_NARROW_CASE(3); MMVF_NARROW_CASE(4);
        MMVF_NARROW_CASE(5); MMVF_NARROW_CASE(6); MMVF_NARROW_CASE(7); MMVF_NARROW_CASE(8);
        default: GGML_ABORT("fatal error");
    }
#undef MMVF_NARROW_CASE
    return true;
}

// True when ggml_cuda_mul_mat computes the node with mul_mat_vec_f_split: the route to
// ggml_cuda_mul_mat_vec_f (no Hadamard hint, mul_mat_vec_f chosen for this batch size) and the
// few-rows case of ggml_cuda_mul_mat_vec_f_narrow.
bool ggml_cuda_mul_mat_vec_f_split_covers(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                          const ggml_tensor * src1, const ggml_tensor * dst) {
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16) || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || ggml_cuda_op_mul_mat_use_fwht(dst)) {
        return false;
    }
    const bool bad_padding_clear = src0->buffer != nullptr &&
        ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    if (bad_padding_clear || cc != GGML_CUDA_CC_VOLTA ||
            !ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1])) {
        return false;
    }
    const int64_t ncols     = src0->ne[0];
    const int64_t nrows     = src0->ne[1];
    const int64_t ncols_dst = src1->ne[1];
    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src1->ne[0] == ncols && dst->ne[0] == nrows && dst->ne[1] == ncols_dst &&
           ncols % (2*mmvf_narrow_block_size) == 0 && ncols_dst >= 1 && ncols_dst <= MMVF_MAX_BATCH_SIZE &&
           nrows >= 1 && nrows <= 16 && ncols / (2*mmvf_narrow_block_size) >= 4;
}

template <typename T, int ncols_dst>
static void mul_mat_vec_f_split_multi_launch(const mmvf_narrow_multi_params & p, int rows, cudaStream_t stream) {
    mul_mat_vec_f_split_multi<T, ncols_dst><<<rows * mmvf_narrow_split_warps, WARP_SIZE, 0, stream>>>(p);
}

bool ggml_cuda_mul_mat_vec_f_narrow_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4],
                                          ggml_tensor * const dsts[4], int n_nodes, const ggml_tensor * src1) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const ggml_type type      = src0s[0]->type;
    const int64_t   ncols     = src0s[0]->ne[0];
    const int64_t   ncols_dst = src1->ne[1];
    mmvf_narrow_multi_params p = {};
    int rows = 0;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        const ggml_tensor * d = dsts[i];
        if (w->type != type || w->ne[0] != ncols || !ggml_cuda_mul_mat_vec_f_split_covers(ctx, w, src1, d)) {
            return false;
        }
        mmvf_narrow_multi_seg & seg = p.seg[p.n_seg++];
        seg.x              = w->data;
        seg.dst            = (float *) d->data;
        seg.stride_row     = w->nb[1] / ggml_type_size(w->type);
        seg.stride_col_dst = (int) (d->nb[1] / sizeof(float));
        seg.first_row      = rows;
        rows += (int) w->ne[1];
    }
    if (rows > GGML_CUDA_MAX_SPLIT_TILES) {
        return false;
    }
    GGML_ASSERT(ctx.tile_counters != nullptr);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) rows * mmvf_narrow_split_warps * ncols_dst);
    p.y             = (const float *) src1->data;
    p.ncols2        = (int) (ncols / 2);
    p.stride_col_y2 = (int) (src1->nb[1] / sizeof(float) / 2);
    p.scratch       = scratch.get();
    p.counters      = ctx.tile_counters + ctx.curr_stream_no * GGML_CUDA_MAX_SPLIT_TILES;

#define MMVF_NARROW_MULTI_CASE(N)                                                              \
        case N:                                                                                \
            if (type == GGML_TYPE_F32) {                                                       \
                mul_mat_vec_f_split_multi_launch<float, N>(p, rows, ctx.stream());             \
            } else {                                                                           \
                mul_mat_vec_f_split_multi_launch<nv_bfloat16, N>(p, rows, ctx.stream());       \
            }                                                                                  \
            break
    switch (ncols_dst) {
        MMVF_NARROW_MULTI_CASE(1); MMVF_NARROW_MULTI_CASE(2); MMVF_NARROW_MULTI_CASE(3); MMVF_NARROW_MULTI_CASE(4);
        MMVF_NARROW_MULTI_CASE(5); MMVF_NARROW_MULTI_CASE(6); MMVF_NARROW_MULTI_CASE(7); MMVF_NARROW_MULTI_CASE(8);
        default: GGML_ABORT("fatal error");
    }
#undef MMVF_NARROW_MULTI_CASE
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    if (ids == nullptr && fusion == nullptr && ggml_cuda_mul_mat_vec_f_narrow(ctx, src0, src1, dst)) {
        return;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // expert store: read the weights through the device pointer table of the tensor instead
    // of src0->data + expert*nb02. A fused gate/up pair looks up the gate weight as well.
    ggml_cuda_expert_table ex_table;
    bool is_expert_store = false;
    if (ids != nullptr && ggml_cuda_expert_table_get(src0, &ex_table)) {
        is_expert_store = true;
        fusion_local.expert_ptrs = ex_table.main;
    }
    if (ids != nullptr && fusion != nullptr && fusion->gate != nullptr && ggml_cuda_expert_is_store(fusion->gate)) {
        ggml_cuda_expert_table ex_table_gate;
        if (!ggml_cuda_expert_table_get(fusion->gate, &ex_table_gate)) {
            GGML_ABORT("%s: cannot look up the gate weight table\n", __func__);
        }
        fusion_local.expert_ptrs_gate = ex_table_gate.main;
        if (!is_expert_store) {
            GGML_ABORT("%s: expert store gate without expert store weights\n", __func__);
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    // expert map of the launch: which routed id is which local expert of the weight (identity
    // without expert parallelism). The fused kernels write the GLU/ADD node, which has no expert
    // state, so the MUL_MAT_ID node that carries it is the one asked. An expert store view reads
    // its weights from the view start (the pointer table is shifted there), so the ids stay
    // view-local numbers.
    ggml_cuda_expert_lut lut;
    if (ids != nullptr) {
        lut = ggml_cuda_mul_mat_id_lut(ctx, fusion != nullptr && fusion->mm_id != nullptr ? fusion->mm_id : dst);
        GGML_ASSERT(lut.n >= ne02);
    }

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, lut, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, lut, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, lut, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, ggml_cuda_expert_lut{}, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, ggml_cuda_expert_lut{}, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, ggml_cuda_expert_lut{}, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                // Volta: a 4-token batch (speculative verify) on cuBLAS is a SIMT SGEMM plus a split-K
                // reduction; MMVF beats it on small matrices and is within 10% on large ones up to 4 columns
                if (cc >= GGML_CUDA_CC_VOLTA) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        default:
            return false;
    }
}
