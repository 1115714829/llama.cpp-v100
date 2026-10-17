#include "mmvq.cuh"
#include "ggml-cuda-expert.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MMID_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// On Volta the dedicated MoE kernel tolerates more tokens per block than MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Volta can run the MUL_MAT_ID MMVQ kernel with more tokens per block than the regular MMVQ
// batch limit, but the block is ncols_dst warps wide, so only the types with a cheap vec_dot
// are promoted. The heavier K/IQ types keep the default to avoid register pressure at 512 threads.
static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_volta(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return MMVQ_MMID_MAX_BATCH_VOLTA;
        default:
            return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA) {
            return get_mmvq_mmid_max_batch_volta(type);
        }
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    return get_mmvq_mmid_max_batch_volta(type);
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const ggml_cuda_expert_lut lut) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    // expert parallelism: the expert map gives the local expert of the routed id, -1 for an expert
    // held by another device - that slot is written as zeros
    const int expert_local = shared_expert || ids == nullptr ? (int) channel_x : ggml_cuda_expert_local(lut, (int) channel_x);
    if (expert_local < 0) {
        if (threadIdx.y == 0) {
            float * dst_e = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
            for (int i = threadIdx.x; i < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x); i += warp_size) {
                dst_e[i] = 0.0f;
            }
        }
        return;
    }

    // src0 only holds the local experts, the weight of the routed expert is its local number
    const uint32_t channel_x_local = shared_expert ? 0 : (uint32_t) expert_local;

    // expert store: the weight base comes from the device pointer table, the channel stride
    // within src0 is then not used
    if (!shared_expert && fusion.expert_ptrs != nullptr) {
        vx = fusion.expert_ptrs[channel_x_local];
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }

    // expert store: the gate weight of a fused gate/up pair is read through its own table
    if (!shared_expert && use_gate && fusion.expert_ptrs_gate != nullptr) {
        vgate = fusion.expert_ptrs_gate[channel_x_local];
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x_local : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x_local : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x_local : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + (fusion.expert_ptrs != nullptr ? 0 : channel_x_local*stride_channel_x) + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
// swiglu: MUL_MAT_ID epilogue, the weight rows are interleaved (2r gate, 2r+1 up) and dst is the
// F16 H of silu(gate) * up, one row per pair. c_rows_per_block must be even so that a pair is never
// split across two blocks; the launch builds the grid from the same c_rows_per_block.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false, bool swiglu = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        void * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const ggml_cuda_expert_lut lut) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = (float *) dst_ptr;
    // the SwiGLU epilogue writes the F16 H through the same pointer, all dst strides below are in
    // elements of the output type. Not GGML_CUDA_RESTRICT: it aliases dst on purpose.
    __half * dst_h = (__half *) dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    // expert parallelism: the expert map gives the local expert of the routed id, -1 for an expert
    // held by another device - that slot is written as zeros
    const int expert_local = shared_expert ? 0 : ggml_cuda_expert_local(lut, (int) channel_x);
    if (expert_local < 0) {
        if constexpr (swiglu) {
            // silu(0) * 0 is 0, so the whole H row of the pair is zero; it is half as wide and
            // half as many rows as the weight block
            if (threadIdx.x < c_rows_per_block/2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
                dst_h[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0/2 + threadIdx.x] = __float2half(0.0f);
            }
        } else if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = 0.0f;
        }
        return;
    }

    // src0 only holds the local experts, the weight of the routed expert is its local number
    const uint32_t channel_x_local = (uint32_t) expert_local;

    // expert store: the weight base comes from the device pointer table, the channel stride
    // within src0 is then not used
    if (!shared_expert && fusion.expert_ptrs != nullptr) {
        vx = fusion.expert_ptrs[channel_x_local];
    }

    // expert store: the gate weight of a fused gate/up pair is read through its own table
    if (!shared_expert && use_gate && fusion.expert_ptrs_gate != nullptr) {
        vgate = fusion.expert_ptrs_gate[channel_x_local];
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = (fusion.expert_ptrs != nullptr ? 0 : channel_x_local*stride_channel_x) + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if constexpr (swiglu) {
        // The warp reduced every row of the block, so the gate/up pair of row t is tmp[2t] and
        // tmp[2t+1] of the same thread - no exchange needed to fuse them. Weight row 2t and 2t+1
        // are H row (row0 + t) / 2, written as one F16 element.
        if (threadIdx.x < c_rows_per_block/2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
            const float h = ggml_cuda_op_silu_single(tmp[2*threadIdx.x]) * tmp[2*threadIdx.x + 1];
            dst_h[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0/2 + threadIdx.x] = __float2half(h);
        }
    } else if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x_local*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x_local];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x_local];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
    if constexpr (swiglu) {
        GGML_UNUSED_VARS(dst);
    } else {
        GGML_UNUSED_VARS(dst_h);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, const ggml_cuda_expert_lut lut, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const ggml_cuda_expert_lut lut,
        const bool swiglu, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    // the SwiGLU epilogue pairs weight rows 2r and 2r+1, so a pair has to sit in one block:
    // rows_per_block must be even (the grid below starts every block at a multiple of it) and
    // nrows_x must be even, which ggml_cuda_mul_mat_vec_q checks before it gets here
    static_assert(rows_per_block % 2 == 0);
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (swiglu) {
        // the epilogue consumes the (gate, up) row pair itself and writes the F16 H, so it replaces
        // the fusion args instead of running next to them
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, lut);
    } else if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, lut);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, lut);
    }
}

// ---------------------------------------------------------------------------------------------
// Compacted MUL_MAT_ID kernels for the decode and verify batches (the MMVQ mmid window).
//
// mul_mat_vec_q_moe launches one block per (row pair, slot) with one warp per token. Under expert
// parallelism a warp whose expert lives on another device only writes zeros: with 4..6 devices
// 75..83% of the warps of the 6400 (gate_up, 1280 rows) and 12800 (down, 2560 rows) blocks of a
// verify step do no dot product, so about a quarter of the resident warps stream weights and the
// down projection (K = 640: two loop trips per row) does not reach the DRAM bandwidth. The kernels
// below build the list of the (token, slot) pairs whose expert is local once per block from ids,
// and walk a fixed grid over (local pair, row group) items: every resident warp has rows to stream
// and the block count no longer scales with the slots of other devices.
//
// Every row is still reduced by one warp exactly as in mul_mat_vec_q_moe: lane l accumulates the
// blocks kbx = l/(qi/vdr) + i*blocks_per_iter in order, then warp_reduce_sum. The results are bit
// identical to mul_mat_vec_q_moe at any token count. For a plain (no epilogue) launch with one
// token the other path is mul_mat_vec_q<type, 1>; it is bit identical as well as long as its K loop
// is a single trip (mmvq_moe_single_trip): warp w of that kernel covers exactly the blocks of trip
// w of the one-warp loop and its block reduction adds the warps in order, with plain additions of
// rounded vec_dot results, so that launch sums its trips with __fadd_rn (mmvq_moe_rows, n1).
// ---------------------------------------------------------------------------------------------

// MMVQ_MMID_MAX_BATCH_SIZE tokens times 32 slots
#define MMVQ_MOE_MAX_PAIRS 512

enum mmvq_moe_out {
    MMVQ_MOE_OUT_F32  = 0, // dst[token][slot][row] F32, zero rows for the slots of other devices
    MMVQ_MOE_OUT_H16  = 1, // SwiGLU epilogue: dst[token][slot][row/2] = F16 silu(gate) * up, zero rows likewise
    MMVQ_MOE_OUT_H16L = 2, // the same H rows for the local slots only (a private buffer of the fused MoE FFN)
};

// The (token, slot) pairs of the launch: the local ones first, then (with_nonlocal) the others,
// each group in (token, slot) order. Every block builds the same list; the order only decides
// which block takes which item. s_expert is the local expert number of a local pair (the expert
// map of the launch, as in mul_mat_vec_q_moe). Returns the number of local pairs.
static __device__ __forceinline__ int mmvq_moe_build_pairs(
        const int32_t * __restrict__ ids, const int ids_stride, const int n_tokens, const int n_used,
        const ggml_cuda_expert_lut & lut, uint16_t * s_pair, uint16_t * s_expert,
        int * s_scan, const bool with_nonlocal) {
    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int nwarps   = blockDim.y;
    const int nthreads = nwarps*WARP_SIZE;
    const int n_pairs  = n_tokens*n_used;

    int n_local = 0;
    for (int pass = 0; pass < (with_nonlocal ? 2 : 1); ++pass) {
        int base = n_local;
        for (int p0 = 0; p0 < n_pairs; p0 += nthreads) {
            const int p = p0 + warp*WARP_SIZE + lane;
            bool take = false;
            int  el   = -1;
            if (p < n_pairs) {
                const int t = p / n_used;
                const int s = p - t*n_used;
                el = ggml_cuda_expert_local(lut, ids[s + t*ids_stride]);
                take = pass == 0 ? el >= 0 : el < 0;
            }
            const uint32_t m = __ballot_sync(0xffffffff, take);
            if (lane == 0) {
                s_scan[warp] = __popc(m);
            }
            __syncthreads();
            int off = base;
            int tot = 0;
            for (int w = 0; w < nwarps; ++w) {
                const int c = s_scan[w];
                off += w < warp ? c : 0;
                tot += c;
            }
            if (take) {
                const int idx = off + __popc(m & ((1u << lane) - 1u));
                s_pair[idx]   = (uint16_t) p;
                s_expert[idx] = (uint16_t) (el < 0 ? 0 : el);
            }
            base += tot;
            __syncthreads();
        }
        if (pass == 0) {
            n_local = base;
        }
    }
    return n_local;
}

// R consecutive rows of one (expert, token) pair, reduced by the calling warp in the order of
// mul_mat_vec_q_moe (see above). kbx_row0 is the block index of the first row within vx.
// n1: the rounding of mul_mat_vec_q<type, 1> instead (one token, no epilogue). Its K loop is a single
// trip (the callers check it), so every thread holds one rounded vec_dot result and the results of
// the other warps are added with plain additions; here the trips are summed with __fadd_rn, which is
// never contracted into an FMA with the product that ends some vec_dots (Q8_0: d8_0*d8_1*sumi). With
// tmp += vec_dot the compiler fuses that product into the accumulation, as it does in
// mul_mat_vec_q_moe, which is what the other launches have to reproduce.
template <ggml_type type, int R, bool n1 = false>
static __device__ __forceinline__ void mmvq_moe_rows(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int kbx_row0,
        const int stride_row_x, const int blocks_per_row_x, float (&tmp)[R]) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

#pragma unroll
    for (int i = 0; i < R; ++i) {
        tmp[i] = 0.0f;
    }
    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if constexpr (n1) {
                tmp[i] = __fadd_rn(tmp[i], vec_dot_q_cuda(vx, &y[kby], kbx_row0 + i*stride_row_x + kbx, kqs));
            } else {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_row0 + i*stride_row_x + kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        tmp[i] = warp_reduce_sum<WARP_SIZE>(tmp[i]);
    }
}

// One item = R*W rows (W warps of R rows) of one (token, slot) pair. nrows_x is a multiple of R.
// Four blocks per SM (at most 64 registers; no spills on the types below at R = 2).
// n1: a one-token launch without an epilogue, which replaces mul_mat_vec_q<type, 1> (see mmvq_moe_rows).
template <ggml_type type, int R, int W, mmvq_moe_out out, bool n1>
__launch_bounds__(W*WARP_SIZE, 4)
static __global__ void mul_mat_vec_q_moe_compact(
        const void * __restrict__ vx_base, const char * const * __restrict__ expert_ptrs,
        const void * __restrict__ vy, const int32_t * __restrict__ ids, void * __restrict__ dst_ptr,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_channel_x,
        const int nchannels_y, const int stride_col_y, const int stride_channel_y,
        const int stride_col_dst, const int stride_channel_dst,
        const int n_tokens, const int n_used, const int ids_stride,
        const ggml_cuda_expert_lut lut) {
    static_assert(out == MMVQ_MOE_OUT_F32 || R % 2 == 0, "the SwiGLU epilogue pairs rows 2r and 2r+1 inside one warp");
    static_assert(!n1 || out == MMVQ_MOE_OUT_F32, "a launch with the SwiGLU epilogue replaces mul_mat_vec_q_moe at any width");
    constexpr int  qk            = ggml_cuda_type_traits<type>::qk;
    constexpr bool zero_nonlocal = out != MMVQ_MOE_OUT_H16L;

    __shared__ uint16_t s_pair[MMVQ_MOE_MAX_PAIRS];
    __shared__ uint16_t s_expert[MMVQ_MOE_MAX_PAIRS];
    __shared__ int      s_scan[W];

    const int n_local = mmvq_moe_build_pairs(ids, ids_stride, n_tokens, n_used, lut,
        s_pair, s_expert, s_scan, zero_nonlocal);

    constexpr int rows_per_item = R*W;
    const int n_groups         = (nrows_x + rows_per_item - 1) / rows_per_item;
    const int blocks_per_row_x = ncols_x / qk;
    const int n_items_compute  = n_local*n_groups;
    const int n_items          = zero_nonlocal ? n_tokens*n_used*n_groups : n_items_compute;

    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int pl   = item / n_groups;
        const int g    = item - pl*n_groups;
        const int p    = s_pair[pl];
        const int t    = p / n_used;
        const int s    = p - t*n_used;
        const int row0 = g*rows_per_item + threadIdx.y*R;
        if (row0 >= nrows_x) {
            continue;
        }

        if constexpr (zero_nonlocal) {
            if (item >= n_items_compute) {
                // a slot routed to another device: zero rows, as mul_mat_vec_q_moe writes them
                if constexpr (out == MMVQ_MOE_OUT_F32) {
                    if (threadIdx.x < R) {
                        ((float *) dst_ptr)[s*stride_channel_dst + t*stride_col_dst + row0 + threadIdx.x] = 0.0f;
                    }
                } else {
                    if (threadIdx.x < R/2) {
                        ((half *) dst_ptr)[s*stride_channel_dst + t*stride_col_dst + row0/2 + threadIdx.x] = __float2half(0.0f);
                    }
                }
                continue;
            }
        }

        const int          el       = s_expert[pl];
        const void *       vx       = expert_ptrs != nullptr ? (const void *) expert_ptrs[el] : vx_base;
        const int          kbx_row0 = (expert_ptrs != nullptr ? 0 : el*stride_channel_x) + row0*stride_row_x;
        const block_q8_1 * y        = (const block_q8_1 *) vy + (s % nchannels_y)*stride_channel_y + t*stride_col_y;

        float tmp[R];
        mmvq_moe_rows<type, R, n1>(vx, y, kbx_row0, stride_row_x, blocks_per_row_x, tmp);

        if constexpr (out == MMVQ_MOE_OUT_F32) {
            float * dst = (float *) dst_ptr + s*stride_channel_dst + t*stride_col_dst + row0;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if (threadIdx.x == i) {
                    dst[i] = tmp[i];
                }
            }
        } else {
            half * dst = (half *) dst_ptr + s*stride_channel_dst + t*stride_col_dst + row0/2;
#pragma unroll
            for (int i = 0; i < R/2; ++i) {
                if (threadIdx.x == i) {
                    // the epilogue of mul_mat_vec_q_moe: gate row 2r, up row 2r+1, F16 H
                    dst[i] = __float2half(ggml_cuda_op_silu_single(tmp[2*i]) * tmp[2*i + 1]);
                }
            }
        }
    }
}

struct mmvq_moe_args {
    const void *         vx;
    const char * const * expert_ptrs;
    const void *         vy;
    const int32_t *      ids;
    void *               dst;
    int ncols_x;
    int nrows_x;
    int stride_row_x;
    int stride_channel_x;
    int nchannels_y;
    int stride_col_y;
    int stride_channel_y;
    int stride_col_dst;
    int stride_channel_dst;
    int n_tokens;
    int n_used;
    int ids_stride;
    ggml_cuda_expert_lut lut; // expert map of the launch (identity without expert parallelism)
};

// Two rows per warp, as mul_mat_vec_q_moe, and four 8-warp blocks per SM (__launch_bounds__): 32
// resident warps that all stream weights. Eight rows per warp (110 registers on Q4_K) leave two
// blocks per SM and stop at 41..54% of the DRAM bandwidth (ncu on V100).
#define MMVQ_MOE_COMPACT_R 2
#define MMVQ_MOE_COMPACT_W 8

static int mmvq_moe_blocks_per_sm(const void * kernel, const int threads, const size_t smem) {
    int n = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem));
    return n > 0 ? n : 1;
}

// A fixed grid: as many blocks as fit on the device at once, at most one per item of the largest
// possible launch (every slot local). The occupancy query runs once per kernel and device.
template <ggml_type type, mmvq_moe_out out, bool n1>
static void mul_mat_vec_q_moe_compact_launch_n1(const mmvq_moe_args & a, cudaStream_t stream) {
    constexpr int R = MMVQ_MOE_COMPACT_R;
    constexpr int W = MMVQ_MOE_COMPACT_W;
    static std::atomic<int> blocks_per_sm[GGML_CUDA_MAX_DEVICES];

    const int device = ggml_cuda_get_device();
    int bps = blocks_per_sm[device].load(std::memory_order_relaxed);
    if (bps == 0) {
        bps = mmvq_moe_blocks_per_sm((const void *) mul_mat_vec_q_moe_compact<type, R, W, out, n1>, W*WARP_SIZE, 0);
        blocks_per_sm[device].store(bps, std::memory_order_relaxed);
    }
    const int n_groups    = (a.nrows_x + R*W - 1) / (R*W);
    const int n_items_max = a.n_tokens*a.n_used*n_groups;
    const int nblocks     = std::max(1, std::min(n_items_max, ggml_cuda_info().devices[device].nsm*bps));

    mul_mat_vec_q_moe_compact<type, R, W, out, n1><<<nblocks, dim3(WARP_SIZE, W, 1), 0, stream>>>(
        a.vx, a.expert_ptrs, a.vy, a.ids, a.dst,
        a.ncols_x, a.nrows_x, a.stride_row_x, a.stride_channel_x,
        a.nchannels_y, a.stride_col_y, a.stride_channel_y,
        a.stride_col_dst, a.stride_channel_dst,
        a.n_tokens, a.n_used, a.ids_stride, a.lut);
}

// The F32 rows of a one-token launch replace mul_mat_vec_q<type, 1> and take its rounding; every
// other launch (wider, or with the SwiGLU epilogue at any width) replaces mul_mat_vec_q_moe.
template <ggml_type type, mmvq_moe_out out>
static void mul_mat_vec_q_moe_compact_launch(const mmvq_moe_args & a, cudaStream_t stream) {
    if constexpr (out == MMVQ_MOE_OUT_F32) {
        if (a.n_tokens == 1) {
            mul_mat_vec_q_moe_compact_launch_n1<type, out, true>(a, stream);
            return;
        }
    }
    mul_mat_vec_q_moe_compact_launch_n1<type, out, false>(a, stream);
}

template <mmvq_moe_out out>
static void mul_mat_vec_q_moe_compact_switch_type(const ggml_type type, const mmvq_moe_args & a, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q1_0: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q1_0, out>(a, stream); break;
        case GGML_TYPE_Q2_0: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q2_0, out>(a, stream); break;
        case GGML_TYPE_Q4_0: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q4_0, out>(a, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q4_1, out>(a, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q5_0, out>(a, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q5_1, out>(a, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q8_0, out>(a, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q4_K, out>(a, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q5_K, out>(a, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q_moe_compact_launch<GGML_TYPE_Q6_K, out>(a, stream); break;
        default:
            GGML_ABORT("%s: type %s has no compact MoE kernel", __func__, ggml_type_name(type));
    }
}

// The down projection followed by the weighted sum over the slots (moe_weighted_reduction_f32),
// dst[token][row] F32. Every block works on one token (block b takes token b % n_tokens; the grid is
// a multiple of n_tokens). Once per block it reads the expert map of the slots of the token and
// quantizes the activation rows of its local slots (F16 or F32) into shared memory with the lanes,
// reductions and rounding of quantize_q8_1, so the dot products read the q8_1 blocks
// quantize_row_q8_1_cuda writes for the (converted) activation, as mul_mat_vec_q_moe does: no
// conversion and no quantization launch in front of the down projection. Then it walks its items:
// one item = RP row pairs of the token with all of its local slots. The (local slot, row pair) units
// of an item go round robin over the W warps, and every unit is two rows reduced by one warp exactly
// as in mul_mat_vec_q_moe (mmvq_moe_rows); the slots are summed in order per row afterwards.
// An item has RP = W row pairs, so with the one or two local slots a token has under expert
// parallelism every warp takes one row pair per slot (ggml_cuda_mmvq_moe_sparse).
#define MMVQ_MOE_REDUCE_MAX_USED 16
#define MMVQ_MOE_REDUCE_W 8

template <ggml_type type, int RP, int W, typename act_t, bool n1>
__launch_bounds__(W*WARP_SIZE, 4)
static __global__ void mul_mat_vec_q_moe_reduce(
        const void * __restrict__ vx_base, const char * const * __restrict__ expert_ptrs,
        const act_t * __restrict__ act, const int64_t s_act_slot, const int64_t s_act_tok, const int nchannels_act,
        const int32_t * __restrict__ ids, const float * __restrict__ weights, const float * __restrict__ expert_scale,
        float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_channel_x,
        const int stride_col_dst, const int n_tokens, const int n_used, const int ids_stride,
        const ggml_cuda_expert_lut lut) {
    constexpr int R             = 2; // rows of a unit
    constexpr int rows_per_item = R*RP;
    constexpr int QT            = 4; // quantization tasks of a warp in flight
    static_assert(rows_per_item <= W*WARP_SIZE, "the epilogue gives every row of an item its own thread");
    constexpr int qk = ggml_cuda_type_traits<type>::qk;

    extern __shared__ char mmvq_moe_reduce_smem_raw[];
    block_q8_1 * s_y = (block_q8_1 *) mmvq_moe_reduce_smem_raw; // [n_used][ncols_x/QK8_1]
    __shared__ float    s_res[MMVQ_MOE_REDUCE_MAX_USED][rows_per_item];
    __shared__ int      s_e[MMVQ_MOE_REDUCE_MAX_USED];    // local expert of a slot
    __shared__ int      s_slot[MMVQ_MOE_REDUCE_MAX_USED]; // slot of the k-th local slot
    __shared__ uint32_t s_mask;

    const int blocks_per_row_y = ncols_x / QK8_1;
    const int blocks_per_row_x = ncols_x / qk;
    const int n_groups         = (nrows_x + rows_per_item - 1) / rows_per_item;
    const int t                = blockIdx.x % n_tokens;

    if (threadIdx.y == 0) {
        bool local = false;
        if ((int) threadIdx.x < n_used) {
            // the expert map of the launch, as in mul_mat_vec_q_moe
            const int el = ggml_cuda_expert_local(lut, ids[threadIdx.x + t*ids_stride]);
            local = el >= 0;
            s_e[threadIdx.x] = el;
        }
        const uint32_t m = __ballot_sync(0xffffffff, local);
        if (local) {
            s_slot[__popc(m & ((1u << threadIdx.x) - 1u))] = threadIdx.x;
        }
        if (threadIdx.x == 0) {
            s_mask = m;
        }
    }
    __syncthreads();
    const uint32_t mask    = s_mask;
    const int      n_local = __popc(mask);

    // quantize_q8_1 on the activation rows of the local slots: one q8_1 block per warp task, lane l
    // takes value l of the block (the element order of the 32 lanes of quantize_q8_1); QT tasks per
    // pass, so that their loads and reductions overlap
    const int n_tasks = n_local*blocks_per_row_y;
    for (int task0 = threadIdx.y; task0 < n_tasks; task0 += QT*W) {
        float xi[QT];
        int   yb_idx[QT];
#pragma unroll
        for (int u = 0; u < QT; ++u) {
            const int task = task0 + u*W;
            xi[u]     = 0.0f;
            yb_idx[u] = -1;
            if (task < n_tasks) {
                const int k = task / blocks_per_row_y;
                const int b = task - k*blocks_per_row_y;
                const int s = s_slot[k];
                xi[u]     = (float) act[t*s_act_tok + (s % nchannels_act)*s_act_slot + b*QK8_1 + threadIdx.x];
                yb_idx[u] = s*blocks_per_row_y + b;
            }
        }
#pragma unroll
        for (int u = 0; u < QT; ++u) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(xi[u]));
            const float sum  = warp_reduce_sum<QK8_1>(xi[u]);

            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi[u] / d);

            if (yb_idx[u] >= 0) {
                block_q8_1 * yb = s_y + yb_idx[u];
                yb->qs[threadIdx.x] = q;
                if (threadIdx.x == 0) {
                    yb->ds = make_half2(d, sum);
                }
            }
        }
    }
    __syncthreads();

    const int n_units = n_local*RP;
    for (int g = blockIdx.x / n_tokens; g < n_groups; g += gridDim.x / n_tokens) {
        for (int unit = threadIdx.y; unit < n_units; unit += W) {
            const int k    = unit / RP;
            const int rp   = unit - k*RP;
            const int row0 = g*rows_per_item + rp*R;
            if (row0 < nrows_x) {
                const int    s        = s_slot[k];
                const int    el       = s_e[s];
                const void * vx       = expert_ptrs != nullptr ? (const void *) expert_ptrs[el] : vx_base;
                const int    kbx_row0 = (expert_ptrs != nullptr ? 0 : el*stride_channel_x) + row0*stride_row_x;

                float tmp[R];
                mmvq_moe_rows<type, R, n1>(vx, s_y + s*blocks_per_row_y, kbx_row0, stride_row_x, blocks_per_row_x, tmp);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    if (threadIdx.x == i) {
                        s_res[s][rp*R + i] = tmp[i];
                    }
                }
            }
        }
        __syncthreads();

        // moe_weighted_reduction_f32 on the row: the slots in order, a slot of another device is the
        // zero row mul_mat_vec_q_moe would have written
        const int r   = threadIdx.y*WARP_SIZE + threadIdx.x;
        const int row = g*rows_per_item + r;
        if (r < rows_per_item && row < nrows_x) {
            const uint64_t first_row   = (uint64_t) t * n_used;
            const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
            const float    e0          = (mask & 1u) ? s_res[0][r] : 0.0f;
            float          sum         = (e0 * first_scale) * weights[first_row];

            for (int expert = 1; expert < n_used; ++expert) {
                const uint64_t row_w = first_row + expert;
                const float    scale = expert_scale != nullptr ? expert_scale[row_w] : 1.0f;
                const float    e     = ((mask >> expert) & 1u) ? s_res[expert][r] : 0.0f;
                sum += (e * scale) * weights[row_w];
            }
            dst[t*stride_col_dst + row] = sum;
        }
        __syncthreads();
    }
}

struct mmvq_moe_reduce_args {
    const void *         vx;
    const char * const * expert_ptrs;
    const void *         act;
    int64_t              s_act_slot;
    int64_t              s_act_tok;
    int                  nchannels_act;
    const int32_t *      ids;
    const float *        weights;
    const float *        expert_scale;
    float *              dst;
    int ncols_x;
    int nrows_x;
    int stride_row_x;
    int stride_channel_x;
    int stride_col_dst;
    int n_tokens;
    int n_used;
    int ids_stride;
    ggml_cuda_expert_lut lut; // expert map of the launch (identity without expert parallelism)
};

static size_t mmvq_moe_reduce_smem(const int64_t ncols_x, const int64_t n_used) {
    return (size_t) n_used * (size_t) (ncols_x / QK8_1) * sizeof(block_q8_1);
}

template <ggml_type type, typename act_t, bool n1>
static void mul_mat_vec_q_moe_reduce_launch_n1(const mmvq_moe_reduce_args & a, cudaStream_t stream) {
    constexpr int W  = MMVQ_MOE_REDUCE_W;
    constexpr int RP = MMVQ_MOE_REDUCE_W;
    static std::atomic<int>    blocks_per_sm[GGML_CUDA_MAX_DEVICES];
    static std::atomic<size_t> blocks_per_sm_smem[GGML_CUDA_MAX_DEVICES];

    const int    device = ggml_cuda_get_device();
    const size_t smem   = mmvq_moe_reduce_smem(a.ncols_x, a.n_used);
    int bps = blocks_per_sm[device].load(std::memory_order_relaxed);
    if (bps == 0 || blocks_per_sm_smem[device].load(std::memory_order_relaxed) != smem) {
        bps = mmvq_moe_blocks_per_sm((const void *) mul_mat_vec_q_moe_reduce<type, RP, W, act_t, n1>, W*WARP_SIZE, smem);
        blocks_per_sm_smem[device].store(smem, std::memory_order_relaxed);
        blocks_per_sm[device].store(bps, std::memory_order_relaxed);
    }
    // a fixed grid of at most one block per item, a multiple of the token count: every block keeps
    // one token, so it reads the expert map and quantizes the activation of that token once
    const int n_items = a.n_tokens*((a.nrows_x + 2*RP - 1) / (2*RP));
    int nblocks = std::min(n_items, ggml_cuda_info().devices[device].nsm*bps);
    nblocks = std::max(a.n_tokens, nblocks - nblocks % a.n_tokens);

    mul_mat_vec_q_moe_reduce<type, RP, W, act_t, n1><<<nblocks, dim3(WARP_SIZE, W, 1), smem, stream>>>(
        a.vx, a.expert_ptrs, (const act_t *) a.act, a.s_act_slot, a.s_act_tok, a.nchannels_act,
        a.ids, a.weights, a.expert_scale, a.dst,
        a.ncols_x, a.nrows_x, a.stride_row_x, a.stride_channel_x, a.stride_col_dst,
        a.n_tokens, a.n_used, a.ids_stride, a.lut);
}

// A one-token launch replaces mul_mat_vec_q<type, 1> and takes its rounding (see mmvq_moe_rows).
template <ggml_type type, typename act_t>
static void mul_mat_vec_q_moe_reduce_launch(const mmvq_moe_reduce_args & a, cudaStream_t stream) {
    if (a.n_tokens == 1) {
        mul_mat_vec_q_moe_reduce_launch_n1<type, act_t, true>(a, stream);
    } else {
        mul_mat_vec_q_moe_reduce_launch_n1<type, act_t, false>(a, stream);
    }
}

template <typename act_t>
static void mul_mat_vec_q_moe_reduce_switch_type(const ggml_type type, const mmvq_moe_reduce_args & a, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q1_0: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q1_0, act_t>(a, stream); break;
        case GGML_TYPE_Q2_0: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q2_0, act_t>(a, stream); break;
        case GGML_TYPE_Q4_0: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q4_0, act_t>(a, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q4_1, act_t>(a, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q5_0, act_t>(a, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q5_1, act_t>(a, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q8_0, act_t>(a, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q4_K, act_t>(a, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q5_K, act_t>(a, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q_moe_reduce_launch<GGML_TYPE_Q6_K, act_t>(a, stream); break;
        default:
            GGML_ABORT("%s: type %s has no compact MoE kernel", __func__, ggml_type_name(type));
    }
}

// Whether mul_mat_vec_q<type, 1> (the one-token MUL_MAT_ID launch without an epilogue) runs its K
// loop in a single trip, which is what makes the compact kernel bit identical to it.
template <ggml_type type>
static bool mmvq_moe_single_trip_t(const int64_t ncols_x, const mmvq_parameter_table_id table_id) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    return ncols_x / qk <= (int64_t) calc_nwarps(type, 1, table_id) * (vdr * WARP_SIZE / qi);
}

static bool mmvq_moe_single_trip(const ggml_type type, const int64_t ncols_x, const int cc) {
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    switch (type) {
        case GGML_TYPE_Q1_0: return mmvq_moe_single_trip_t<GGML_TYPE_Q1_0>(ncols_x, table_id);
        case GGML_TYPE_Q2_0: return mmvq_moe_single_trip_t<GGML_TYPE_Q2_0>(ncols_x, table_id);
        case GGML_TYPE_Q4_0: return mmvq_moe_single_trip_t<GGML_TYPE_Q4_0>(ncols_x, table_id);
        case GGML_TYPE_Q4_1: return mmvq_moe_single_trip_t<GGML_TYPE_Q4_1>(ncols_x, table_id);
        case GGML_TYPE_Q5_0: return mmvq_moe_single_trip_t<GGML_TYPE_Q5_0>(ncols_x, table_id);
        case GGML_TYPE_Q5_1: return mmvq_moe_single_trip_t<GGML_TYPE_Q5_1>(ncols_x, table_id);
        case GGML_TYPE_Q8_0: return mmvq_moe_single_trip_t<GGML_TYPE_Q8_0>(ncols_x, table_id);
        case GGML_TYPE_Q4_K: return mmvq_moe_single_trip_t<GGML_TYPE_Q4_K>(ncols_x, table_id);
        case GGML_TYPE_Q5_K: return mmvq_moe_single_trip_t<GGML_TYPE_Q5_K>(ncols_x, table_id);
        case GGML_TYPE_Q6_K: return mmvq_moe_single_trip_t<GGML_TYPE_Q6_K>(ncols_x, table_id);
        default:             return false;
    }
}

// The compact and the fused MoE kernels walk the local (token, slot) pairs. That pays where expert
// parallelism leaves a token one or two local slots of its n_used: the dedicated kernels launch
// blocks for every slot and most of their warps only write zeros. With most of the experts on the
// device (the weight split over one or two devices) nearly every slot is local, the dedicated kernels
// keep their warps busy, and the fused down projection would quantize the activation of every slot
// in every block (measured: single device n=1 +21..37%, two devices n=2..8 +9..13%), so those
// launches stay on the dedicated kernels. Sparse: fewer than four local slots expected per token
// (n_used times the share of the experts the device holds).
static bool ggml_cuda_mmvq_moe_sparse(const int64_t n_used, const int64_t n_experts_local, const int64_t n_expert_total) {
    return n_used*n_experts_local < 4*n_expert_total;
}

// global expert count of a MUL_MAT_ID node (its own count without expert parallelism)
static int64_t mmvq_moe_n_expert_total(const ggml_tensor * node) {
    const int64_t n_total = ggml_mul_mat_id_is_expert_parallel(node) ? ggml_mul_mat_id_get_n_expert_total(node) : 0;
    return n_total > 0 ? n_total : node->src[0]->ne[2];
}

// The compact kernel covers the MUL_MAT_ID launches of the MMVQ mmid window that carry no fusion
// arguments (the plain rows and the SwiGLU epilogue) on NVIDIA, for the types listed in
// mul_mat_vec_q_moe_compact_switch_type, where the routing is sparse on the device
// (ggml_cuda_mmvq_moe_sparse). A plain launch with one token replaces mul_mat_vec_q<type, 1> and is
// only taken where that kernel's K loop is a single trip (the results stay bit identical).
static bool ggml_cuda_mmvq_moe_compact_supported(
        const ggml_type type, const int64_t ncols_x, const int64_t nrows_x, const int64_t n_tokens,
        const int64_t n_used, const int64_t n_experts_local, const int64_t n_expert_total, const bool swiglu,
        const int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ggml_cuda_mmvq_moe_sparse(n_used, n_experts_local, n_expert_total)) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }
    if (n_tokens < 1 || n_tokens > MMVQ_MMID_MAX_BATCH_SIZE || n_used < 1 || n_tokens*n_used > MMVQ_MOE_MAX_PAIRS) {
        return false;
    }
    if (nrows_x % MMVQ_MOE_COMPACT_R != 0) {
        return false;
    }
    return swiglu || n_tokens > 1 || mmvq_moe_single_trip(type, ncols_x, cc);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const ggml_cuda_expert_lut lut, const bool swiglu, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ids ? (ncols_dst <= MMVQ_MMID_MAX_BATCH_SIZE) : (ncols_dst <= MMVQ_MAX_BATCH_SIZE));

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    // The SwiGLU epilogue is a MUL_MAT_ID layout, so its node also belongs to the dedicated MoE
    // kernel. That kernel handles ncols_dst == 1 (the block is one warp wide) and the epilogue
    // needs the row pairs, which only this kernel's even row block gives.
    if (has_ids && (ncols_dst > 1 || swiglu)) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, lut, swiglu, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    lut, stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, lut, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const ggml_cuda_expert_lut lut, const bool swiglu, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, lut, swiglu, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const char * const * expert_ptrs) {
    // MUL_MAT_ID epilogue: the weight rows are interleaved (2r gate, 2r+1 up) and the dst is the F16
    // H of silu(gate) * up. The MoE kernel fuses the pair and writes it, the dst strides below are
    // in elements of the output type, so the F16 H needs no separate path. Only MUL_MAT_ID has an
    // epilogue, a plain MUL_MAT never sets it.
    const bool swiglu = ids != nullptr &&
        ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU;

    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32 || (swiglu && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    // the epilogue has no fusion args of its own, it consumes the gate/up pair itself
    GGML_ASSERT(!swiglu || fusion == nullptr);

    GGML_TENSOR_BINARY_OP_LOCALS;

    // a gate/up pair must be inside one row block of the MoE kernel (its rows_per_block is even),
    // and one pair is one dst row
    GGML_ASSERT(!swiglu || (ne01 % 2 == 0 && ne0 == ne01/2));

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MMID_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

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
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // expert store: read the weights through the device pointer table of the tensor instead
    // of src0->data + expert*nb02. A fully resident tensor is one strided block of device
    // memory, so it is addressed arithmetically instead - reading the table would put a
    // dependent global load in front of every weight access of every block.
    // A fused gate/up pair looks up the gate weight as well (the two lanes are separate store
    // tensors unless the model merges them); that would need a second base, so a fused pair
    // keeps the tables.
    ggml_cuda_expert_table ex_table;
    bool is_expert_store = false;
    const char * vx_weights = (const char *) src0->data;
    int64_t s02_weights    = 0; // 0: take the expert pitch from src0->nb[2]
    if (ids != nullptr && ggml_cuda_expert_table_get(src0, &ex_table)) {
        is_expert_store = true;
        if (ex_table.uniform) {
            vx_weights  = ex_table.plain_base;
            s02_weights = (int64_t) ex_table.plain_pitch / ts_src0;
        } else {
            // expert_ptrs is the staging table of the launch when the decode batch routed to
            // host experts, the main table otherwise (null)
            fusion_local.expert_ptrs = expert_ptrs != nullptr ? expert_ptrs : ex_table.main;
        }
    }
    if (ids != nullptr && fusion != nullptr && fusion->gate != nullptr && ggml_cuda_expert_is_store(fusion->gate)) {
        ggml_cuda_expert_table ex_table_gate;
        if (!ggml_cuda_expert_table_get(fusion->gate, &ex_table_gate)) {
            GGML_ABORT("%s: cannot look up the gate weight table\n", __func__);
        }
        if (is_expert_store && s02_weights > 0) {
            vx_weights            = (const char *) src0->data;
            s02_weights           = 0;
            fusion_local.expert_ptrs = ex_table.main;
        }
        fusion_local.expert_ptrs_gate = ex_table_gate.main;
        if (!is_expert_store) {
            GGML_ABORT("%s: expert store gate without expert store weights\n", __func__);
        }
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = s02_weights > 0 ? s02_weights : src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

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

    // the MUL_MAT_ID launches without fusion arguments (the plain rows and the SwiGLU epilogue) of
    // the mmid window: walk the local (token, slot) pairs instead of one block per slot
    if (ids != nullptr && fusion == nullptr && ne03 == 1 && ne3 == 1) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (ggml_cuda_mmvq_moe_compact_supported(src0->type, ne00, ne01, ncols_dst, nchannels_dst, ne02, lut.n, swiglu, cc)) {
            mmvq_moe_args a = {};
            a.vx                 = vx_weights;
            a.expert_ptrs        = fusion_local.expert_ptrs;
            a.vy                 = src1_q8_1.get();
            a.ids                = ids_d;
            a.dst                = dst->data;
            a.ncols_x            = (int) ne00;
            a.nrows_x            = (int) ne01;
            a.stride_row_x       = (int) s01;
            a.stride_channel_x   = (int) s02;
            a.nchannels_y        = (int) nchannels_y;
            a.stride_col_y       = (int) stride_col_y;
            a.stride_channel_y   = (int) stride_channel_y;
            a.stride_col_dst     = (int) stride_col_dst;
            a.stride_channel_dst = (int) stride_channel_dst;
            a.n_tokens           = (int) ncols_dst;
            a.n_used             = (int) nchannels_dst;
            a.ids_stride         = (int) ids_stride;
            a.lut                = lut;
            if (swiglu) {
                mul_mat_vec_q_moe_compact_switch_type<MMVQ_MOE_OUT_H16>(src0->type, a, stream);
            } else {
                mul_mat_vec_q_moe_compact_switch_type<MMVQ_MOE_OUT_F32>(src0->type, a, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    mul_mat_vec_q_switch_type(
        vx_weights, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, lut, swiglu, stream);
}

// The weight of a MUL_MAT_ID node as the compact kernels read it: the expert store gives either one
// strided block (uniform, addressed arithmetically) or a device pointer table (the staging table of
// the launch when the decode batch routed to host experts), as in ggml_cuda_mul_mat_vec_q.
struct mmvq_moe_src0 {
    const void *         vx;
    int64_t              s02;
    const char * const * expert_ptrs;
};

static mmvq_moe_src0 mmvq_moe_src0_resolve(const ggml_tensor * src0, const char * const * expert_ptrs) {
    const size_t ts_src0 = ggml_type_size(src0->type);
    mmvq_moe_src0 r = { src0->data, (int64_t) (src0->nb[2] / ts_src0), nullptr };
    ggml_cuda_expert_table ex_table;
    if (ggml_cuda_expert_table_get(src0, &ex_table)) {
        if (ex_table.uniform) {
            r.vx  = ex_table.plain_base;
            r.s02 = (int64_t) ex_table.plain_pitch / ts_src0;
        } else {
            r.expert_ptrs = expert_ptrs != nullptr ? expert_ptrs : ex_table.main;
        }
    }
    return r;
}

bool ggml_cuda_mmvq_moe_reduce_supported(const ggml_tensor * down, const ggml_tensor * weights,
        const ggml_tensor * expert_scale, const ggml_tensor * dst) {
    const ggml_tensor * src0 = down->src[0];
    const ggml_tensor * src1 = down->src[1];
    const ggml_tensor * ids  = down->src[2];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (down->op != GGML_OP_MUL_MAT_ID || ggml_mul_mat_id_get_epilogue(down) != GGML_MUL_MAT_ID_EPILOGUE_NONE) {
        return false;
    }
    if (!ggml_is_quantized(src0->type) || down->type != GGML_TYPE_F32 || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if ((src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) || src1->nb[0] != ggml_type_size(src1->type)) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || down->ne[3] != 1 || src0->nb[0] != ggml_type_size(src0->type)) {
        return false;
    }
    const int64_t n_tokens = down->ne[2];
    const int64_t n_used   = down->ne[1];
    if (n_used > MMVQ_MOE_REDUCE_MAX_USED || src0->ne[0] % QK8_1 != 0 ||
            !ggml_cuda_mmvq_moe_compact_supported(src0->type, src0->ne[0], src0->ne[1], n_tokens, n_used,
                src0->ne[2], mmvq_moe_n_expert_total(down), false, cc)) {
        return false;
    }
    // the q8_1 activation of the slots of one token sits in shared memory
    if (mmvq_moe_reduce_smem(src0->ne[0], n_used) > 32*1024) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != down->ne[0] || ggml_nrows(dst) != n_tokens) {
        return false;
    }
    auto per_slot = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n_used*n_tokens;
    };
    return per_slot(weights) && (expert_scale == nullptr || per_slot(expert_scale));
}

bool ggml_cuda_mmvq_moe_gate_up_h_supported(const ggml_tensor * gate_up) {
    const ggml_tensor * src0 = gate_up->src[0];
    const ggml_tensor * src1 = gate_up->src[1];
    const ggml_tensor * ids  = gate_up->src[2];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (gate_up->op != GGML_OP_MUL_MAT_ID || ggml_mul_mat_id_get_epilogue(gate_up) != GGML_MUL_MAT_ID_EPILOGUE_SWIGLU) {
        return false;
    }
    if (!ggml_is_quantized(src0->type) || gate_up->type != GGML_TYPE_F16 || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src0->nb[0] != ggml_type_size(src0->type)) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || gate_up->ne[3] != 1 || gate_up->ne[0] != src0->ne[1]/2) {
        return false;
    }
    return ggml_cuda_mmvq_moe_compact_supported(src0->type, src0->ne[0], src0->ne[1], gate_up->ne[2], gate_up->ne[1],
        src0->ne[2], mmvq_moe_n_expert_total(gate_up), true, cc);
}

void ggml_cuda_mmvq_moe_gate_up_h(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_up,
        half * h, const char * const * expert_ptrs, const void * src1_q8) {
    const ggml_tensor * src0 = gate_up->src[0];
    const ggml_tensor * src1 = gate_up->src[1];
    const ggml_tensor * ids  = gate_up->src[2];
    cudaStream_t stream = ctx.stream();

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const mmvq_moe_src0 w = mmvq_moe_src0_resolve(src0, expert_ptrs);

    // the q8_1 activation exactly as ggml_cuda_mul_mat_vec_q quantizes it (rows of ne10_padded values),
    // unless the caller quantized it already in that layout
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    if (src1_q8 == nullptr) {
        src1_q8_1.alloc(ne13*ne12*ne11*ne10_padded*sizeof(block_q8_1)/QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type, ne10,
            src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            ne10_padded, ne11, ne12, ne13, stream);
        src1_q8 = src1_q8_1.get();
    }
    const int64_t s11 = ne10_padded / QK8_1;

    mmvq_moe_args a = {};
    a.vx                 = w.vx;
    a.expert_ptrs        = w.expert_ptrs;
    a.vy                 = src1_q8;
    a.ids                = (const int32_t *) ids->data;
    a.dst                = h;
    a.ncols_x            = (int) src0->ne[0];
    a.nrows_x            = (int) src0->ne[1];
    a.stride_row_x       = (int) (src0->nb[1] / ggml_type_size(src0->type));
    a.stride_channel_x   = (int) w.s02;
    a.nchannels_y        = (int) ne11;
    a.stride_col_y       = (int) (ne11*s11);
    a.stride_channel_y   = (int) s11;
    a.stride_col_dst     = (int) (gate_up->ne[1]*gate_up->ne[0]);
    a.stride_channel_dst = (int) gate_up->ne[0];
    a.n_tokens           = (int) gate_up->ne[2];
    a.n_used             = (int) gate_up->ne[1];
    a.ids_stride         = (int) (ids->nb[1] / ggml_type_size(ids->type));
    a.lut                = ggml_cuda_mul_mat_id_lut(ctx, gate_up);
    GGML_ASSERT(a.lut.n >= src0->ne[2]);
    mul_mat_vec_q_moe_compact_switch_type<MMVQ_MOE_OUT_H16L>(src0->type, a, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mmvq_moe_down_reduce(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
        const ggml_tensor * weights, const ggml_tensor * expert_scale, ggml_tensor * dst,
        const char * const * expert_ptrs, const half * act_h) {
    const ggml_tensor * src0 = down->src[0];
    const ggml_tensor * src1 = down->src[1];
    const ggml_tensor * ids  = down->src[2];

    const mmvq_moe_src0 w = mmvq_moe_src0_resolve(src0, expert_ptrs);

    mmvq_moe_reduce_args a = {};
    a.vx               = w.vx;
    a.expert_ptrs      = w.expert_ptrs;
    if (act_h != nullptr) {
        // the private H of the fused MoE FFN: [token][slot][ne10] F16
        a.act           = act_h;
        a.s_act_slot    = src1->ne[0];
        a.s_act_tok     = src1->ne[0]*src1->ne[1];
        a.nchannels_act = (int) src1->ne[1];
    } else {
        a.act           = src1->data;
        a.s_act_slot    = src1->nb[1] / ggml_type_size(src1->type);
        a.s_act_tok     = src1->nb[2] / ggml_type_size(src1->type);
        a.nchannels_act = (int) src1->ne[1];
    }
    a.ids              = (const int32_t *) ids->data;
    a.weights          = (const float *) weights->data;
    a.expert_scale     = expert_scale != nullptr ? (const float *) expert_scale->data : nullptr;
    a.dst              = (float *) dst->data;
    a.ncols_x          = (int) src0->ne[0];
    a.nrows_x          = (int) src0->ne[1];
    a.stride_row_x     = (int) (src0->nb[1] / ggml_type_size(src0->type));
    a.stride_channel_x = (int) w.s02;
    a.stride_col_dst   = (int) (dst->nb[1] / sizeof(float));
    a.n_tokens         = (int) down->ne[2];
    a.n_used           = (int) down->ne[1];
    a.ids_stride       = (int) (ids->nb[1] / ggml_type_size(ids->type));
    a.lut              = ggml_cuda_mul_mat_id_lut(ctx, down);
    GGML_ASSERT(a.lut.n >= src0->ne[2]);
    if (act_h != nullptr || src1->type == GGML_TYPE_F16) {
        mul_mat_vec_q_moe_reduce_switch_type<half>(src0->type, a, ctx.stream());
    } else {
        mul_mat_vec_q_moe_reduce_switch_type<float>(src0->type, a, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    // a plain MUL_MAT has no MUL_MAT_ID epilogue
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, ggml_cuda_expert_lut{}, /*swiglu=*/false, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
