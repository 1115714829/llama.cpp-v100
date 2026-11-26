#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

struct llama_memory_breakdown_data {
    size_t model   = 0;
    size_t context = 0;
    size_t compute = 0;

    size_t compute_prefill = 0;
    size_t compute_decode  = 0;

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);

LLAMA_API int32_t llama_model_n_expert_used(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API struct ggml_backend_meta_split_state llama_model_tensor_split_state(
        const struct llama_model * model, const struct ggml_tensor * tensor);

LLAMA_API std::vector<ggml_backend_dev_t> llama_default_devices();

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

LLAMA_API const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_memory_tensors(const llama_context * ctx);

LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

LLAMA_API void llama_internal_comm_connect(llama_context * ctx);

LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0,
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1,
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4,
};

LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API bool llama_enable_layer_inp_sink(struct llama_context * ctx, const uint32_t * lids, uint32_t n_layers);

LLAMA_API bool llama_context_link_embd(struct llama_context * ctx_tgt, struct llama_context * ctx_dft);

LLAMA_API void llama_set_embd_source(struct llama_context * ctx, bool enable, int32_t token_offset);

LLAMA_API bool llama_layer_inp_sink_get(struct llama_context * ctx, size_t token_offset, size_t n_tokens, float * dst);

LLAMA_API int32_t llama_layer_inp_sink_col(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

LLAMA_API bool llama_enable_h_nextn_sink(struct llama_context * ctx, bool is_target, uint32_t n_h_rows);

LLAMA_API bool llama_context_link_h_nextn(struct llama_context * ctx_tgt, struct llama_context * ctx_dft);

LLAMA_API void llama_set_h_source(struct llama_context * ctx, bool enable, const int32_t * cols, size_t n_cols);

LLAMA_API void llama_set_h_src_group(struct llama_context * ctx, int32_t group);

LLAMA_API int32_t llama_h_nextn_sink_group(struct llama_context * ctx);
LLAMA_API uint64_t llama_mtp_sink_generation(struct llama_context * ctx);
LLAMA_API bool llama_mtp_can_defer_sink(struct llama_context * ctx);
LLAMA_API bool llama_mtp_can_overlap_tail(struct llama_context * ctx, uint32_t tokens, int32_t pending_group);
LLAMA_API void llama_mtp_defer_sink(struct llama_context * ctx, bool enable);
LLAMA_API void llama_mtp_select_sink(struct llama_context * ctx, int32_t group, uint64_t generation);

LLAMA_API void llama_set_h_sink_offset(struct llama_context * ctx, int32_t offset);

LLAMA_API void llama_set_h_sink_write(struct llama_context * ctx, bool write);

LLAMA_API void llama_set_h_sink_first_pos(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos_first);

LLAMA_API bool llama_h_nextn_sink_capture(struct llama_context * ctx, const int32_t * src_cols, size_t n_cols);

LLAMA_API bool llama_h_nextn_sink_get(struct llama_context * ctx, int32_t group, size_t token_offset, size_t n_tokens, float * dst);

LLAMA_API int32_t llama_h_nextn_sink_col(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

LLAMA_API bool llama_synchronize_ubatch(struct llama_context * ctx, int64_t ubatch_id);

LLAMA_API void llama_synchronize_outputs(struct llama_context * ctx);

LLAMA_API void llama_set_logits_topk_only(struct llama_context * ctx, bool only);

LLAMA_API bool llama_draft_ring_enable(struct llama_context * ctx, int32_t n_max, bool with_p);
LLAMA_API void llama_draft_ring_begin(struct llama_context * ctx, const llama_token * ids, size_t n);

LLAMA_API const llama_token * llama_draft_ring_tokens(const struct llama_context * ctx, int32_t step);
LLAMA_API const float       * llama_draft_ring_p     (const struct llama_context * ctx, int32_t step);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);

LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
