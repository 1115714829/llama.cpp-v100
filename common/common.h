

#pragma once

#include "llama-cpp.h"
#include "fit.h"

#include "ggml-opt.h"
#include "ggml.h"
#include "llama.h"

#include <array>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif

#define COM_DBG(fmt, ...) LOG_DBG("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_TRC(fmt, ...) LOG_TRC("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_INF(fmt, ...) LOG_INF("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_WRN(fmt, ...) LOG_WRN("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_ERR(fmt, ...) LOG_ERR("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

struct common_adapter_lora_info {
    std::string path;
    float scale;

    std::string task_name;
    std::string prompt_prefix;

    struct llama_adapter_lora * ptr;
};

using llama_tokens = std::vector<llama_token>;

struct common_control_vector_load_info;

struct common_cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false};
    bool     mask_valid                  = false;
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;
    bool     strict_cpu                  = false;
    uint32_t poll                        = 50;
};

int32_t common_cpu_get_num_physical_cores();
int32_t common_cpu_get_num_math();

enum llama_example {
    LLAMA_EXAMPLE_BATCHED,
    LLAMA_EXAMPLE_DEBUG,
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_COMPLETION,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_DIFFUSION,
    LLAMA_EXAMPLE_FINETUNE,
    LLAMA_EXAMPLE_FIT_PARAMS,
    LLAMA_EXAMPLE_RESULTS,
    LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,
    LLAMA_EXAMPLE_DOWNLOAD,
    LLAMA_EXAMPLE_TOKENIZE,

    LLAMA_EXAMPLE_COUNT,
};

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,

    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
    COMMON_SAMPLER_TYPE_PENALTIES   = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
    COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12,
};

enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

enum common_params_sampling_config : uint64_t {
    COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS        = 1 << 0,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_K           = 1 << 1,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_P           = 1 << 2,
    COMMON_PARAMS_SAMPLING_CONFIG_MIN_P           = 1 << 3,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY = 1 << 4,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD   = 1 << 5,
    COMMON_PARAMS_SAMPLING_CONFIG_TEMP            = 1 << 6,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N  = 1 << 7,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT  = 1 << 8,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT        = 1 << 9,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU    = 1 << 10,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA    = 1 << 11,
};

enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding based on n-grams
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_COUNT
};

enum common_grammar_type {
    COMMON_GRAMMAR_TYPE_NONE,
    COMMON_GRAMMAR_TYPE_USER,
    COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
    COMMON_GRAMMAR_TYPE_TOOL_CALLS,
};

struct common_grammar {
    common_grammar_type type = COMMON_GRAMMAR_TYPE_NONE;
    std::string grammar;

    common_grammar() = default;

    common_grammar(common_grammar_type t, std::string g) : type(t), grammar(std::move(g)) {
        GGML_ASSERT(type != COMMON_GRAMMAR_TYPE_NONE || !grammar.empty());
    }

    bool empty() const { return type == COMMON_GRAMMAR_TYPE_NONE || grammar.empty(); }
};

inline const std::string & common_grammar_value(const common_grammar & g) {
    return g.grammar;
}

inline bool common_grammar_needs_prefill(const common_grammar & g) {
    return g.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
        || g.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS;
}

struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED;

    int32_t n_prev             = 64;
    int32_t n_probs            = 0;
    int32_t min_keep           = 0;
    int32_t top_k              = 40;
    float   top_p              = 0.95f;
    float   min_p              = 0.05f;
    float   xtc_probability    = 0.00f;
    float   xtc_threshold      = 0.10f;
    float   typ_p              = 1.00f;
    float   temp               = 0.80f;
    float   dynatemp_range     = 0.00f;
    float   dynatemp_exponent  = 1.00f;
    int32_t penalty_last_n     = 64;
    float   penalty_repeat     = 1.00f;
    float   penalty_freq       = 0.00f;
    float   penalty_present    = 0.00f;
    float   dry_multiplier     = 0.0f;
    float   dry_base           = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = 64;
    float   adaptive_target    = -1.0f;
    float   adaptive_decay     = 0.90f;
    int32_t mirostat           = 0;
    float   top_n_sigma        = -1.00f;
    float   mirostat_tau       = 5.00f;
    float   mirostat_eta       = 0.10f;
    bool    ignore_eos         = false;
    bool    no_perf            = false;
    bool    timing_per_token   = false;

    uint64_t user_sampling_config = 0;

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};

    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    common_grammar                      grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::set<llama_token>               preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;
    std::vector<llama_logit_bias> logit_bias_eog;

    std::string generation_prompt;

    // reasoning budget sampler parameters
    // these are populated by the server/CLI based on chat template params
    int32_t                   reasoning_budget_tokens   = -1;
    std::vector<llama_token>  reasoning_budget_start;
    std::vector<llama_tokens> reasoning_budget_end;
    std::vector<llama_token>  reasoning_budget_forced;
    std::string               reasoning_budget_message;
    bool                      reasoning_control = false;

    bool backend_sampling = false;

    std::string print() const;
};

struct common_params_model {
    std::string path        = "";
    std::string url         = "";
    std::string hf_repo     = "";
    std::string hf_file     = "";
    std::string docker_repo = "";

    std::string get_name() const {
        if (!hf_repo.empty()) {
            return hf_repo;
        }
        if (!docker_repo.empty()) {
            return docker_repo;
        }
        return path;
    }

    bool empty() const {
        return get_name().empty();
    }
};

struct common_params_speculative_draft {
    int32_t n_max = 3;
    int32_t n_min = 0;

    float p_split = 0.1f;
    float p_min   = 0.0f;

    bool backend_sampling = true;

    bool probabilistic = false;

    int32_t window = 32768;

    int32_t n_ubatch = 0;

    bool mtp_host = false;

    common_params_model mparams;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    int32_t n_gpu_layers = -1;

    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    std::vector<ggml_backend_dev_t> devices;

    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
};

struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;

    int32_t n_max = 64;
    int32_t n_min = 48;
};

struct common_params_speculative_ngram_map {
    uint16_t size_n   = 12;
    uint16_t size_m   = 48;
    uint16_t min_hits = 1;
};

struct common_params_speculative_ngram_cache {
    std::string lookup_cache_static;
    std::string lookup_cache_dynamic;
};

struct common_params_speculative {
    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };

    static constexpr int32_t lookup_chain_max = 4;
    int32_t lookup_chain = 0;
    int32_t lookup_chain_min_match = 12;

    int32_t mtp_verify_max(int32_t mtp_max) const {
        return lookup_chain > 0 && !draft.probabilistic && !has_synth() && mtp_max > 0 && mtp_max < 7
            ? mtp_max + std::min(lookup_chain, 7 - mtp_max) : mtp_max;
    }

    double synth_len = -1.0;
    std::vector<double> synth_rates;

    common_params_speculative_draft draft;

    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;

    common_params_speculative_ngram_cache ngram_cache;

    bool has_dft() const {
        return !draft.mparams.empty();
    }

    bool has_synth() const {
        return synth_len != -1.0 || !synth_rates.empty();
    }

    uint32_t need_n_rs_seq() const {
        bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

        if (lookup_chain > 0 && std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end()) {
            return mtp_verify_max(draft.n_max);
        }
        return needs_rs_seq ? draft.n_max : 0u;
    }
};

struct common_params_diffusion {
    int32_t steps         = 128;
    bool    visual_mode   = false;

    float   eps           = 0;
    int32_t block_length  = 0;

    int32_t algorithm     = 4;
    float   alg_temp      = 0.0f;

    float   cfg_scale     = 0;
    bool    add_gumbel_noise = false;
};

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    COMMON_REASONING_FORMAT_DEEPSEEK,        // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
    // do not extend this enum unless you absolutely have to
    // in most cases, use COMMON_REASONING_FORMAT_AUTO
    // see: https://github.com/ggml-org/llama.cpp/pull/15408
};

struct lr_opt {
    float    lr0          = 1e-5;
    float    lr_min       = -1;
    float    decay_epochs = -1;
    float    scale_epoch  = 0;
    float    wd           = 0;
    unsigned epochs       = 2;

    unsigned epoch;

    float get_lr(float e) const;
    float get_lr() const { return get_lr(epoch); }

    void init();
};

struct ggml_opt_optimizer_params common_opt_lr_pars(void * userdata);

struct common_params {
    llama_model_read_session * draft_read_session = nullptr;
    int32_t n_predict             =    -1;
    int32_t n_ctx                 =     0;
    int32_t n_batch               =  2048;
    int32_t n_ubatch              =   512;
    int32_t n_keep                =     0;
    int32_t n_chunks              =    -1;
    int32_t n_parallel            =     1;
    int32_t n_sequences           =     1;
    int32_t n_outputs_max         =     0;
    int32_t n_outputs_max_per_seq =     1;
    int32_t grp_attn_n            =     1;
    int32_t grp_attn_w            =   512;
    int32_t n_print               =    -1;
    float   rope_freq_base        =  0.0f;
    float   rope_freq_scale       =  0.0f;
    float   yarn_ext_factor       = -1.0f;
    float   yarn_attn_factor      = -1.0f;
    float   yarn_beta_fast        = -1.0f;
    float   yarn_beta_slow        = -1.0f;
    int32_t yarn_orig_ctx         =     0;

    std::vector<ggml_backend_dev_t> devices;

    int32_t n_gpu_layers       = -1;
    int32_t main_gpu           = 0;
    float   tensor_split[128]  = {0};
    bool    fit_params         = true;
    bool    fit_params_print   = false;
    bool    fit_mmproj         = true;
    int32_t fit_params_min_ctx = 4096;

    std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), COMMON_FIT_TARGET_DEFAULT);
    bool fit_params_target_set = false;

    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
    enum llama_load_mode  load_mode  = LLAMA_LOAD_MODE_AUTO;

    enum llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_AUTO;

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED;
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO;

    struct common_params_sampling    sampling;
    struct common_params_speculative speculative;
    struct common_params_diffusion   diffusion;

    bool    cpu_mtp         = false;
    int32_t cpu_mtp_context = 0;

    struct common_params_model model;

    std::set<std::string> model_alias;     // model aliases                                                 // NOLINT
    std::set<std::string> model_tags;      // model tags (informational, not used for routing)              // NOLINT
    std::string hf_token             = ""; // HF token (aka bearer token)                                   // NOLINT
    std::string prompt               = "";                                                                  // NOLINT
    std::string system_prompt        = "";                                                                  // NOLINT
    std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
    std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
    std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
    std::string logits_file          = ""; // file for saving *all* logits                                  // NOLINT
    std::string path_prompts_log_dir = ""; // directory with logged prompts                                 // NOLINT

    std::string logits_output_dir = "data"; // directory for saving logits output files                     // NOLINT
    bool        save_logits       = false;  // whether to save logits to files                              // NOLINT
    std::vector<std::string> tensor_filter; // filter tensor names for debug output (regex)                 // NOLINT

    std::vector<std::string> in_files;
    std::vector<std::string> antiprompt;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    std::string expert_profile      = "";

    float   expert_host_gib        = -1.0f;
    float   expert_disk_gib        = -1.0f;
    std::string expert_disk_dir   = "";

    std::string dump_expert_routing = "";
    std::string dump_routing_trace  = "";
    int dump_routing_trace_size     = 4;

    bool lora_init_without_apply = false;
    std::vector<common_adapter_lora_info> lora_adapters;

    std::vector<common_control_vector_load_info> control_vectors;

    int32_t verbosity                  = 3;
    int32_t control_vector_layer_start = -1;
    int32_t control_vector_layer_end   = -1;
    bool    offline                    = false;

    int32_t ppl_stride      = 0;
    int32_t ppl_output_type = 0;

    bool   hellaswag        = false;
    size_t hellaswag_tasks  = 400;

    bool   winogrande       = false;
    size_t winogrande_tasks = 0;

    bool   multiple_choice  = false;
    size_t multiple_choice_tasks = 0;

    bool   kl_divergence    = false;

    bool check             = false;

    bool usage             = false;
    bool completion        = false;
    bool use_color         = false;
    bool special           = false;
    bool interactive       = false;
    bool interactive_first = false;
    bool prompt_cache_all  = false;
    bool prompt_cache_ro   = false;

    bool escape            = true;
    bool multiline_input   = false;
    bool simple_io         = false;
    bool cont_batching     = true;
    bool no_perf           = false;
    bool show_timings      = true;
    bool ctx_shift         = false;
    bool swa_full          = false; // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    bool kv_unified        = false;

    bool input_prefix_bos  = false;
    bool verbose_prompt    = false;
    bool display_prompt    = true;
    bool no_kv_offload     = false;
    bool warmup            = true;
    bool check_tensors     = false;
    bool no_op_offload     = false;
    bool no_extra_bufts    = false;
    bool no_host           = false;
    bool load_mtp          = false;

    bool single_turn       = false;

    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;

    size_t moe_cache_size = 0;

    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    struct common_params_model mmproj;
    bool mmproj_use_gpu = true;
    ggml_backend_dev_t mmproj_device = nullptr;
    bool mmproj_device_explicit = false;
    bool no_mmproj = false;
    std::vector<std::string> image;
    int image_min_tokens = -1;
    int image_max_tokens = -1;
    int mtmd_batch_max_tokens = 1024;

    float       video_fps                   = 4.0f;
    int64_t     video_timestamp_interval_ms = 5000;
    std::string video_ffmpeg_bin_dir        = "";

    struct lr_opt lr;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    float val_split = 0.05f;

    bool embedding         = false;
    int32_t embd_normalize = 2;
    std::string embd_out   = "";
    std::string embd_sep   = "\n";
    std::string cls_sep    = "\t";

    int32_t port                = 8080;
    bool    reuse_port          = false;
    int32_t timeout_read        = 3600;
    int32_t timeout_write       = timeout_read;
    int32_t sse_ping_interval   = 30;
    int32_t n_threads_http      = -1;
    int32_t n_cache_reuse       = 0;
    bool    cache_prompt        = true;
    bool    cache_idle_slots    = true;
    int32_t n_ctx_checkpoints    = 32;
    int32_t prefill_pace         = 30;
    int32_t kv_unified_per_slot  = 0;
    int32_t checkpoint_min_step  = 8192;
    int32_t cache_ram_mib        = 8192;
    bool    cache_ram_explicit   = false;

    std::string public_path   = "";                                                                         // NOLINT
    std::string api_prefix    = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    std::vector<std::string> hostnames = {"127.0.0.1"};
    bool use_jinja = true;                                                                                  // NOLINT

    std::string cors_origins = "*";
    std::string cors_methods = "GET, POST, DELETE, OPTIONS";
    std::string cors_headers = "*";
    bool cors_credentials = true;
    bool cors_origins_explicit = false;

    bool enable_chat_template = true;
    bool force_pure_content_parser = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1;
    bool prefill_assistant = true;
    int sleep_idle_seconds = -1;

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    std::map<std::string, std::string> default_template_kwargs;
    bool preserve_reasoning_specified = false;

    std::string server_base;

    bool ui = true;
    bool ui_mcp_proxy = false;
    std::string ui_config_json;

    bool endpoint_slots   = true;
    bool endpoint_props   = false;
    bool endpoint_metrics = false;

    std::vector<std::string> server_tools;
    std::string server_tools_runtime;

    std::string mcp_servers_config;
    std::string mcp_servers_json;

    std::string models_dir    = "";
    std::string models_preset = "";
    int models_max = 4;
    bool models_autoload = true;
    std::string models_preset_hf = "";

    bool log_json = false;

    std::string slot_save_path;
    std::string media_path;

    float slot_prompt_similarity = 0.1f;

    bool is_pp_shared   = false;
    bool is_tg_separate = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    std::vector<std::string> context_files;

    int32_t chunk_size = 64;

    std::string chunk_separator = "\n";

    int32_t n_junk = 250;
    int32_t i_pos  = -1;

    int32_t n_out_freq  = 10;
    int32_t n_save_freq =  0;
    int32_t i_chunk     =  0;
    int8_t  imat_dat    =  0;

    bool process_output         = false;
    bool compute_ppl            = true;
    bool show_statistics        = false;
    bool activation_statistics  = false;
    bool parse_special          = false;

    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    bool spm_infill = false;

    bool batched_bench_output_jsonl = false;

    bool tokenize_ids        = false;
    bool tokenize_stdin      = false;
    bool tokenize_no_bos     = false;
    bool tokenize_show_count = false;

    std::string out_file;

    llama_progress_callback load_progress_callback = NULL;
    void *                  load_progress_callback_user_data = NULL;
    bool no_alloc = false;

    std::string tts_lang = "";
    std::string tts_speaker_file = "";

    bool is_gen_docs = false;
};

void common_init();

void common_params_print_info(const common_params & params, bool print_devices = true);
std::string common_params_get_system_info(const common_params & params);

bool parse_cpu_range(const std::string & range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string string_format(const char * fmt, ...);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string string_lcs(std::string_view a, std::string_view b);

std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

std::string regex_escape(const std::string & s);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        if (!(token_stream >> value)) {
            throw std::invalid_argument("invalid value: \"" + token + "\"");
        }
        values.push_back(value);
    }
    return values;
}

template<>
inline std::vector<std::string> string_split<std::string>(const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char) {
                if (string_ends_with(str, stop.substr(0, len))) {
                    return str.size() - len;
                }
            }
        }
    }
    return std::string::npos;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);
void string_process_escapes(std::string & input);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);

bool glob_match(const std::string & pattern, const std::string & str);

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str);
std::string  wstring_to_utf8(const std::wstring & str);
#endif

std::string fs_path_to_utf8(const std::filesystem::path & path);

std::string common_get_env(const std::string & name);
void        common_set_env(const std::string & name, const std::string & value);

std::filesystem::path common_get_path_from_env(const std::string & name);

bool fs_validate_filename(const std::string & filename, bool allow_subdirs = false);
bool fs_is_directory(const std::string & path);

// some old libstdc++ versions don't follow symlinks here, so adding a trailing "/" fixes it: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101510
inline bool common_create_directories(const std::filesystem::path & path, std::error_code & ec) {
#if defined(__linux__)
    return std::filesystem::create_directories(path / "", ec);
#else
    return std::filesystem::create_directories(path, ec);
#endif
}

std::filesystem::path fs_get_cache_directory();
std::filesystem::path fs_get_cache_file(const std::string & filename);
std::filesystem::path fs_get_config_directory();

void fs_write_atomic(const std::filesystem::path & path, const std::string & data);

// Auto-detect if colors can be enabled based on terminal and environment
bool tty_can_use_colors();
bool tty_enable_ansi();

bool common_is_tty(FILE * file);

struct common_sampler;

enum common_decision_type {
    COMMON_DECISION_TYPE_NONE,
    COMMON_DECISION_TYPE_OPENJEV,
    COMMON_DECISION_TYPE_LEV,
    COMMON_DECISION_TYPE_KEV,
    COMMON_DECISION_TYPE_NIMBLE,
    COMMON_DECISION_TYPE_LAYA,
    COMMON_DECISION_TYPE_CLEF,
    COMMON_DECISION_TYPE_PPLX_DECIDER,
    COMMON_DECISION_TYPE_VLLM_DECISION2,
    COMMON_DECISION_TYPE_VLLM_D3,
    COMMON_DECISION_TYPE_LFM2_D1,
    COMMON_DECISION_TYPE_LFM2_D1_OMNI,
    COMMON_DECISION_TYPE_UNKNOWN,
};

common_decision_type common_get_decision_type(const struct llama_model * model);

struct common_gguf_info {
    common_decision_type decision_type = COMMON_DECISION_TYPE_UNKNOWN;
    uint32_t             n_ctx_train   = 0;
};

common_gguf_info common_get_gguf_info(const std::string & fname);

struct common_init_result {
    common_init_result(common_params & params, bool model_only = false);
    ~common_init_result();

    llama_model * model();
    llama_context * context();

    common_sampler * sampler(llama_seq_id seq_id);
    void reset_samplers();

    std::vector<llama_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr common_init_from_params(common_params & params, bool model_only = false);

struct llama_model_params   common_model_params_to_llama  (      common_params & params);
struct llama_context_params common_context_params_to_llama(const common_params & params);

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

std::string common_get_model_endpoint();

char * common_get_model_or_exit(int, char*[]);

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

struct common_threadpools {
    common_threadpools() = default;
    ~common_threadpools();

    common_threadpools(const common_threadpools &) = delete;
    common_threadpools & operator=(const common_threadpools &) = delete;

    void init(llama_context * ctx, const common_params & params);

private:
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    decltype(ggml_threadpool_free) * free_fn = nullptr;
};

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0,
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1,
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2,
    COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3,
};

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx);

struct common_memory {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    void init(llama_context * ctx_tgt, llama_context * ctx_dft = nullptr);

    void seq_rm (llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const;
    void seq_cp (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const;
};

struct common_batch {
    struct token {
        llama_token  id;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos;
        llama_seq_id seq_id;
        bool         output;
        llama_embd   embd;
        llama_embd   state;
        std::vector<llama_seq_id> seq_ids_extra;
        int32_t      decision_order = 0;
    };

    std::vector<token> tokens;
    llama_batch_ext_ptr batch;

    int32_t n_pos = 1;

    common_batch() = default;
    common_batch(struct llama_context * ctx);

    llama_batch_ext * get() { return get_sub_batch(0, size()); }

    llama_batch_ext * get_sub_batch(int32_t off, int32_t n);

    bool has_token() const { return !tokens.empty() && tokens[0].id != LLAMA_TOKEN_NULL; }
    bool has_embd () const { return !tokens.empty() && tokens[0].embd.data != nullptr; }

    void clear();

    int32_t add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output);

    int32_t add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output);

    bool add_seq(int32_t idx, llama_seq_id seq_id);

    bool set_output(int32_t idx, bool value);

    bool set_embd(int32_t idx, llama_embd embd);

    bool set_embd_state(int32_t idx, llama_embd state);

    int32_t add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output);

    int32_t size() const { return (int32_t) tokens.size(); }
};

common_batch common_batch_get_one(struct llama_context * ctx, const llama_token * tokens, int32_t n_tokens);
common_batch common_batch_get_one(struct llama_context * ctx, const llama_tokens & tokens);

bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos);

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

std::string common_detokenize(
            const struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

struct common_control_vector_data {
    int n_embd;

    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

const char * const LLM_FFN_DENSE_REGEX = "\\.ffn_(up|down|gate)\\.";

inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex, std::vector<llama_model_tensor_buft_override> & overrides) {

    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

struct common_prompt_checkpoint {
    int64_t n_tokens;

    int id_task = -1;

    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    std::vector<uint8_t> data_spec;

    size_t size() const;

    bool empty() const;
    void clear();

    void update_pos(
            int64_t n_tokens,
            llama_pos pos_min,
            llama_pos pos_max);

    void update_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    void update_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    bool load_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    bool load_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    void clear_tgt();
    void clear_dft();
};
