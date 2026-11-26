#pragma once

#include "ggml.h"
#include "gguf.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#if defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)

#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ggml_print_backtrace(void);

uint64_t ggml_graph_next_uid(void);

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
    #ifndef static_assert
        #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, msg) _Static_assert(cond, msg)
        #else
            #define static_assert(cond, msg) struct global_scope_noop_trick
        #endif
    #endif
#endif

static inline int ggml_up32(int n) {
    return (n + 31) & ~31;
}

static inline int ggml_up(int n, int m) {

    GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool ggml_op_is_empty(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
}

static inline bool ggml_impl_is_view(const struct ggml_tensor * t) {
    return t->view_src != NULL;
}

static inline float ggml_compute_softplus_f32(float input) {
    return (input > 20.0f) ? input : logf(1 + expf(input));
}

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API void ggml_log_internal        (enum ggml_log_level level, const char * format, ...);
GGML_API void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

static void ggml_set_op_params(struct ggml_tensor * tensor, const void * params, size_t params_size) {
    GGML_ASSERT(tensor != NULL);
    assert(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((const int32_t *)(tensor->op_params))[i];
}

static float ggml_get_op_params_f32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}

static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void ggml_set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}

static inline int ggml_expert_layer_of_name(const char * name) {
    const char * p = name != NULL ? strstr(name, "blk.") : NULL;
    if (p == NULL) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    return layer;
}

GGML_API const int16_t * ggml_expert_store_node_lut(const struct ggml_tensor * node);

GGML_API int32_t ggml_expert_assignment_of_weight(const struct ggml_tensor * weight);

struct ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    int                n_tasks;
    void             * userdata;
};

struct ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    int                 n_tasks;
    void              * userdata;
};

struct ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    int               n_tasks;
    void            * userdata;
};

struct ggml_custom_op_params {
    ggml_custom_op_t fun;
    int              n_tasks;
    void           * userdata;
};

typedef uint32_t ggml_bitset_t;

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1)

static size_t ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

static inline bool ggml_bitset_get(const ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

static inline void ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

static inline void ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;
    struct ggml_tensor ** keys;
};

struct ggml_hash_set ggml_hash_set_new(size_t size);
void                 ggml_hash_set_free(struct ggml_hash_set * hash_set);

size_t ggml_hash_size(size_t min_sz);

void ggml_hash_set_reset(struct ggml_hash_set * hash_set);

static bool ggml_hash_contains(const struct ggml_hash_set * hash_set, struct ggml_tensor * key);

static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key);

static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key);

static size_t ggml_hash_find_or_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key);

static inline size_t ggml_hash(const struct ggml_tensor * p) {

    return (size_t)(uintptr_t)p >> 4;
}

static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    size_t i = h;
    while (ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {

            return GGML_HASHSET_FULL;
        }
    }
    return i;
}

static bool ggml_hash_contains(const struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t i = ggml_hash_find(hash_set, key);
    return i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    GGML_ABORT("fatal error");
}

static size_t ggml_hash_find_or_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return i;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    GGML_ABORT("fatal error");
}

enum ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct ggml_cgraph {
    int size;
    int n_nodes;
    int n_leafs;

    struct ggml_tensor ** nodes;
    struct ggml_tensor ** grads;
    struct ggml_tensor ** grad_accs;
    struct ggml_tensor ** leafs;
    int32_t             * use_counts;

    struct ggml_hash_set visited_hash_set;

    enum ggml_cgraph_eval_order order;

    uint64_t uid;

    uint64_t uid_prev;
};

struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph, int i0, int i1);

GGML_API bool ggml_op_can_inplace(enum ggml_op op);

GGML_API void * ggml_aligned_malloc(size_t size);
GGML_API void ggml_aligned_free(void * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {

        bits = 0x00400000;
    }

    else {

        bits = (uint32_t) x << 23;
    }

    float result;

    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    if (x < 2) {

        bits = 0x00200000 << x;
    }

    else {

        bits = (uint32_t)(x - 1) << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

static inline int ggml_round_to_nearest_even(float x) {
    const int value = (int) floorf(x);
    const float fraction = x - value;
    return fraction > 0.5f || (fraction == 0.5f && (value & 1)) ? value + 1 : value;
}

static inline float ggml_f8_e4m3_to_fp32(uint8_t x) {
    const uint8_t ax = x & 0x7F;
    if (ax == 0x7F) {

        return NAN;
    }

    const int exp = (ax >> 3) & 0xF;
    const int man = ax & 0x7;

    const float value = exp == 0
        ? ldexpf((float) man, -9)
        : ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    return x & 0x80 ? -value : value;
}

static inline uint8_t ggml_fp32_to_f8_e4m3(float x) {
    const uint8_t sign = signbit(x) ? 0x80 : 0;
    x = fabsf(x);

    if (isnan(x)) {

        return 0x7F;
    }
    if (x == 0.0f) {
        return sign;
    }
    if (isinf(x) || x >= 448.0f) {
        return sign | 0x7E;
    }
    if (x < 0.015625f) {

        return sign | (uint8_t) ggml_round_to_nearest_even(x * 512.0f);
    }

    int exp;
    const float mantissa = frexpf(x, &exp) * 2.0f;
    int encoded_exp = exp + 6;
    int encoded_man = ggml_round_to_nearest_even((mantissa - 1.0f) * 8.0f);
    if (encoded_man == 8) {
        encoded_man = 0;
        encoded_exp++;
    }
    if (encoded_exp > 15 || (encoded_exp == 15 && encoded_man > 6)) {
        return sign | 0x7E;
    }
    return sign | (uint8_t) ((encoded_exp << 3) | encoded_man);
}

static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

static inline uint8_t ggml_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {

        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) {
        h.bits = (u.i >> 16) | 64;
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

static inline int32_t ggml_node_get_use_count(const struct ggml_cgraph * cgraph, int node_idx) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    if (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        return 0;
    }
    return cgraph->use_counts[hash_pos];
}

static inline bool ggml_node_has_n_uses(const struct ggml_cgraph * cgraph, int node_idx, int32_t n_uses) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];

    if (ggml_node_get_use_count(cgraph, node_idx) != n_uses) {
        return false;
    }

    if (node->view_src) {
        return false;
    }

    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }

    return true;
}

static inline bool ggml_can_fuse_ext(const struct ggml_cgraph * cgraph, const int * node_idxs, const enum ggml_op * ops, int num_ops) {
    for (int i = 0; i < num_ops; ++i) {
        if (node_idxs[i] >= cgraph->n_nodes) {
            return false;
        }

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];
        if (node->op != ops[i]) {
            return false;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idxs[i], 1)) {
            return false;
        }
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;
            }
            if (!ggml_are_same_shape(node, prev)) {
                return false;
            }
        }
    }
    return true;
}

static inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, const enum ggml_op * ops, int num_ops) {
    assert(num_ops < 32);

    if (node_idx + num_ops > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];
    for (int i = 0; i < num_ops; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_ext(cgraph, idxs, ops, num_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph * cgraph,
                                         const int *                node_idxs,
                                         int                        count,
                                         const enum ggml_op *       ops,
                                         const int *                outputs,
                                         int                        num_outputs);

static inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph * cgraph,
                                          int                        node_idx,
                                          int                        count,
                                          const enum ggml_op *       ops,
                                          const int *                outputs,
                                          int                        num_outputs) {
    GGML_ASSERT(count < 32);
    if (node_idx + count > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];

    for (int i = 0; i < count; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outputs, num_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <array>
#include <initializer_list>
#include <vector>

inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
}

inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *          cgraph,
                                   int                                 start_idx,
                                   std::initializer_list<enum ggml_op> ops,
                                   std::initializer_list<int>          outputs = {}) {
    return ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

inline bool ggml_check_edges(const struct ggml_cgraph *                cgraph,
                             int                                       start_idx,
                             std::initializer_list<std::array<int, 3>> edges) {
    for (const auto & edge : edges) {
        int dst_node = edge[0];
        int src_idx  = edge[1];
        int src_node = edge[2];
        if (cgraph->nodes[start_idx + dst_node]->src[src_idx] != cgraph->nodes[start_idx + src_node]) {
            return false;
        }
    }
    return true;
}

GGML_API size_t gguf_type_size(enum gguf_type type);
GGML_API void gguf_write_to_buf(const struct gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif
