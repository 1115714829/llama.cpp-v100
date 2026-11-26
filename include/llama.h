#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 11

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0,
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
        LLAMA_VOCAB_TYPE_TEST   = 7,
        LLAMA_VOCAB_TYPE_PLAMO3 = 8,
    };

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type {
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,

        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10,
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18,
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19,
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20,
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21,
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22,
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23,
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24,
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25,
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26,
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27,
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28,
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29,
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30,
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31,
        LLAMA_FTYPE_MOSTLY_BF16          = 32,

        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36,
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37,
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38,
        LLAMA_FTYPE_MOSTLY_NVFP4         = 39,
        LLAMA_FTYPE_MOSTLY_Q1_0          = 40,
        LLAMA_FTYPE_MOSTLY_Q2_0          = 41,
        LLAMA_FTYPE_MOSTLY_F8_E4M3       = 42,

        LLAMA_FTYPE_GUESSED = 1024,
    };

    LLAMA_API const char * llama_ftype_name(enum llama_ftype ftype);

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4,
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLAMA_API const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE   = 0,
        LLAMA_SPLIT_MODE_LAYER  = 1,
        LLAMA_SPLIT_MODE_ROW    = 2,
        LLAMA_SPLIT_MODE_TENSOR = 3,
    };

    enum llama_load_mode {
        LLAMA_LOAD_MODE_AUTO       = -1, // auto-detect based on device capabilities
        LLAMA_LOAD_MODE_NONE       =  0,
        LLAMA_LOAD_MODE_MMAP       =  1,
        LLAMA_LOAD_MODE_MLOCK      =  2,
        LLAMA_LOAD_MODE_MMAP_MLOCK =  3,
        LLAMA_LOAD_MODE_DIRECT_IO  =  4,
    };

    LLAMA_API const char * llama_load_mode_name(enum llama_load_mode load_mode);
    LLAMA_API enum llama_load_mode llama_load_mode_from_str(const char * str);

    enum llama_lazy_mode {
        LLAMA_LAZY_MODE_OFF  = 0,
        LLAMA_LAZY_MODE_AUTO = 1,
        LLAMA_LAZY_MODE_ON   = 2,
    };

    enum llama_context_type {
        LLAMA_CONTEXT_TYPE_DEFAULT = 0,
        LLAMA_CONTEXT_TYPE_MTP     = 1,
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id;
        float logit;
        float p;
    } llama_token_data;

    typedef struct llama_token_data_array {

        llama_token_data * data;
        size_t size;
        int64_t selected;
        bool sorted;
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    enum llama_process_type {
        LLAMA_PROCESS_TYPE_ENCODE,
        LLAMA_PROCESS_TYPE_DECODE,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_read_session;

    LLAMA_API struct llama_model_read_session * llama_model_read_session_open(const char * path, enum llama_load_mode mode);
    LLAMA_API struct gguf_context * llama_model_read_qualified_metadata(const char * path);
    LLAMA_API void llama_model_read_session_free(struct llama_model_read_session * session);

    struct llama_model_params {

        ggml_backend_dev_t * devices;

        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers;
        enum llama_split_mode split_mode;
        enum llama_load_mode  load_mode;

        enum llama_lazy_mode lazy_mode;

        int32_t main_gpu;

        const float * tensor_split;

        llama_progress_callback progress_callback;

        void * progress_callback_user_data;

        const struct llama_model_kv_override * kv_overrides;

        const struct llama_model * model_shared;
        struct llama_model_read_session * read_session;

        bool vocab_only;
        bool check_tensors;
        bool use_extra_bufts;
        bool no_host;
        bool no_alloc;
        bool load_mtp;
        bool mtp_host;
        bool no_expert_store;
        bool ple_host_read;
    };

    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t n_ctx;
        uint32_t n_batch;
        uint32_t n_ubatch;
        uint32_t n_seq_max;
        uint32_t n_rs_seq;
        uint32_t n_outputs_max;
        uint32_t n_outputs_max_per_seq;
        int32_t  n_threads;
        int32_t  n_threads_batch;

        enum llama_context_type      ctx_type;
        enum llama_rope_scaling_type rope_scaling_type;
        enum llama_pooling_type      pooling_type;
        enum llama_attention_type    attention_type;
        enum llama_flash_attn_type   flash_attn_type;

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;
        float    rope_freq_scale;
        float    yarn_ext_factor;
        float    yarn_attn_factor;
        float    yarn_beta_fast;
        float    yarn_beta_slow;
        uint32_t yarn_orig_ctx;
        uint32_t n_swa_mtp;
        uint32_t n_h_sink_rows;
        float    defrag_thold;

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k;
        enum ggml_type type_v;

        size_t moe_cache_size;

        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        bool embeddings;
        bool offload_kqv;
        bool no_perf;
        bool op_offload;
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/llama.cpp/pull/14363

        struct llama_sampler_seq_config * samplers;
        size_t                            n_samplers;

        struct llama_context * ctx_other;
    };

    struct llama_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct llama_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    typedef struct llama_model_quantize_params {
        int32_t nthread;
        enum llama_ftype ftype;
        enum ggml_type output_tensor_type;
        enum ggml_type token_embedding_type;
        bool allow_requantize;
        bool quantize_output_tensor;
        bool only_copy;
        bool pure;
        bool keep_split;
        bool dry_run;
        const struct llama_model_imatrix_data * imatrix;
        const struct llama_model_kv_override * kv_overrides;
        const struct llama_model_tensor_override * tt_overrides;
        const int32_t * prune_layers;
        size_t max_buf_size;
    } llama_model_quantize_params;

    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf;
    } llama_sampler_chain_params;

    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    struct llama_adapter_lora;

    LLAMA_API const char * llama_version(void);

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    LLAMA_API void llama_backend_init(void);

    LLAMA_API void llama_backend_free(void);

    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    typedef void (*llama_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    LLAMA_API struct llama_model * llama_model_init_from_user(
                    struct gguf_context * metadata,
          llama_model_set_tensor_data_t   set_tensor_data,
                                   void * set_tensor_data_ud,
              struct llama_model_params   params);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);

    LLAMA_API struct llama_model * llama_model_load_from_file_ptr(
                                   FILE * file,
              struct llama_model_params   params);

    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    DEPRECATED(LLAMA_API void llama_free_model(struct llama_model * model),
            "use llama_model_free instead");

    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);
    LLAMA_API size_t llama_max_parallel_sequences(void);
    LLAMA_API size_t llama_max_tensor_buft_overrides(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);
    LLAMA_API bool llama_supports_rpc        (void);

    // NOTE: After creating a llama_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via llama_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_rs_seq   (const struct llama_context * ctx);

    DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");

    DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");

    LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
    LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    LLAMA_API  enum llama_pooling_type   llama_pooling_type(const struct llama_context * ctx);

    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);

    LLAMA_API int32_t llama_model_n_ctx_train  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd       (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp   (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_out   (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer      (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer_nextn(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head       (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_swa        (const struct llama_model * model);

    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);

    LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);

    LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);

    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    LLAMA_API enum llama_ftype llama_model_ftype(const struct llama_model * model);

    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);

    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);

    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(
            struct llama_model * model,
            FILE * file);

    LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);

    LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);

    LLAMA_API uint64_t            llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
    LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens  (const struct llama_adapter_lora * adapter);

    LLAMA_API int32_t llama_set_adapters_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    LLAMA_API int32_t llama_set_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);

    LLAMA_API bool llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1);

    LLAMA_API void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1);

    LLAMA_API void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id);

    LLAMA_API void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta);

    LLAMA_API void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d);

    LLAMA_API llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id);

    LLAMA_API llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id);

    LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);

    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id);

    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#define LLAMA_STATE_SEQ_FLAGS_NONE 0

#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t llama_state_seq_flags;

    LLAMA_API size_t llama_state_seq_get_size_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_set_data_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    LLAMA_API void llama_batch_free(struct llama_batch batch);

    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    struct llama_batch_ext;

    struct llama_embd {
        const float * data;
        size_t n_rows;
        size_t n_embd;
    };

    LLAMA_API struct llama_batch_ext * llama_batch_ext_init (struct llama_context * ctx);
    LLAMA_API void                     llama_batch_ext_free (struct llama_batch_ext * batch);
    LLAMA_API void                     llama_batch_ext_clear(struct llama_batch_ext * batch);

    LLAMA_API int32_t llama_batch_ext_add      (struct llama_batch_ext * batch, llama_seq_id seq_id);

    LLAMA_API int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, llama_seq_id seq_id, llama_token id);
    LLAMA_API int32_t llama_batch_ext_add_embd (struct llama_batch_ext * batch, llama_seq_id seq_id, struct llama_embd embd);

    LLAMA_API bool llama_batch_ext_add_seq(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                          llama_seq_id   seq_id);

    LLAMA_API bool llama_batch_ext_set_embd_token(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                     struct llama_embd   embd);

    LLAMA_API bool llama_batch_ext_set_embd_state(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                     struct llama_embd   embd);

    LLAMA_API bool llama_batch_ext_set_output_embd(
                                struct llama_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    LLAMA_API bool llama_batch_ext_set_output_logits(
                                struct llama_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    LLAMA_API bool llama_batch_ext_set_pos(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                       const llama_pos * pos);

    LLAMA_API int32_t llama_process(
                                struct llama_context * ctx,
                             enum llama_process_type   type,
                              struct llama_batch_ext * batch);

    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    LLAMA_API bool llama_get_causal_attn(const struct llama_context * ctx);

    DEPRECATED(LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup),
            "user code should do warmup runs manually [TAG_LLAMA_GRAPH_NO_WARMUP]");

    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API void llama_set_logits_topk(struct llama_context * ctx, int32_t k);

    LLAMA_API void llama_reserve(struct llama_context * ctx);

    LLAMA_API int32_t llama_get_logits_topk_ith(
            struct llama_context * ctx,
                         int32_t   i,
               const llama_token ** ids,
                     const float ** logits);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab);

    LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);

    LLAMA_API const llama_token * llama_vocab_get_suppress_tokens(const struct llama_vocab * vocab, int32_t * n_suppress_tokens);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);

    DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
    DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
    DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
    DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
    DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
    DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
    DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
    DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
    DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
    DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
    DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab),
            "use llama_vocab_bos instead");

    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    typedef void * llama_sampler_context_t;

    struct llama_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
        void                   (*reset) (      struct llama_sampler * smpl);
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);
        void                   (*free)  (      struct llama_sampler * smpl);

        bool (*backend_init)(
                struct llama_sampler       * smpl,
                ggml_backend_buffer_type_t   buft,
                uint32_t                     n_outputs_max_per_seq);

        void (*backend_accept)(
                struct llama_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        void (*backend_apply)(
                struct llama_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llama_sampler_data * data);

        void (*backend_set_input)(struct llama_sampler * smpl);

        void (*backend_reset)(struct llama_sampler * smpl);

        void (*copy_state)(const struct llama_sampler * src, struct llama_sampler * dst);
    };

    struct llama_sampler {
        struct llama_sampler_i * iface;

        llama_sampler_context_t ctx;
    };

    LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);

    LLAMA_API struct llama_sampler * llama_sampler_init  (      struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_copy  (const struct llama_sampler * src, struct llama_sampler * dst);

    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);

    LLAMA_API struct llama_sampler * llama_sampler_chain_get(      struct llama_sampler * chain, int32_t i);

    LLAMA_API int32_t                llama_sampler_chain_n  (const struct llama_sampler * chain);

    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);

    LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

    LLAMA_API float llama_sampler_dist_draw_u(struct llama_sampler * smpl);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");

    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   n_vocab,
                             int32_t   penalty_last_n,
                               float   penalty_repeat,
                               float   penalty_freq,
                               float   penalty_present);

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    LLAMA_API int32_t llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    LLAMA_API const char * llama_print_system_info(void);

    LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
    LLAMA_API void llama_log_set(ggml_log_callback   log_callback, void *  user_data);

    struct llama_perf_context_data {

        double t_start_ms;
        double t_load_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_p_eval;
        int32_t n_eval;
        int32_t n_reused;
        int32_t n_reallocs_same;
        int32_t n_reallocs_diff;
    };

    struct llama_perf_sampler_data {
        double t_sample_ms;

        int32_t n_sample;
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train;

        llama_opt_param_filter param_filter;
        void * param_filter_ud;

        ggml_opt_get_optimizer_params get_opt_pars;
        void * get_opt_pars_ud;

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif
