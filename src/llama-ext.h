#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
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

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

// per-device slice of a model tensor, as the meta backend splits it (MIRRORED/PARTIAL for
// tensors every device holds whole)
LLAMA_API struct ggml_backend_meta_split_state llama_model_tensor_split_state(
        const struct llama_model * model, const struct ggml_tensor * tensor);

// the devices llama uses when no explicit device list is given: RPC servers first, then
// discrete GPUs, integrated GPUs only when no discrete GPU exists. Same selection as the
// model load applies, so callers can match where a model will run without its own rules.
LLAMA_API std::vector<ggml_backend_dev_t> llama_default_devices();

// all tensors of the loaded model, by name; the fit code reads the expert tensor metadata
// (name and shape) from it to plan the expert store without another trial load. Matches the
// declaration in llama-model.h, the function itself is internal to libllama.
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a joint decision head (clef) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
// Both force a full synchronize() to read the host copy. The MTP draft no longer uses them
// (it reads the device sink), but the other draft implementations (dflash, eagle3, dspark)
// and tools/imatrix still do
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// DFlash2: keep the target layer inputs in a device sink instead of host buffers. The order of
// lids is the row order within the sink and must match the draft's target_layer_ids. Returns
// false if unsupported or the allocation failed - the caller then uses the host path.
LLAMA_API bool llama_enable_layer_inp_sink(struct llama_context * ctx, const uint32_t * lids, uint32_t n_layers);

// DFlash2: link a target and a draft context for the direct device transfer of the layer inputs.
// Creates the events and allocates the draft staging sink. Returns false if the devices are not
// compatible or the allocation failed - the caller then uses the host path.
LLAMA_API bool llama_context_link_embd(struct llama_context * ctx_tgt, struct llama_context * ctx_dft);

// DFlash2: select the source of the next embd batch of a linked draft: true reads the staging
// sink (token_offset selects the first target-sink column to copy, and is ignored for the graph),
// false uses the host input. No-op when the context is not linked.
LLAMA_API void llama_set_embd_source(struct llama_context * ctx, bool enable, int32_t token_offset);

// DFlash2 host fallback: copy the sink rows [token_offset, token_offset + n_tokens) into dst,
// in the same per-token layer order as the injection embd rows. Requires a sink.
LLAMA_API bool llama_layer_inp_sink_get(struct llama_context * ctx, size_t token_offset, size_t n_tokens, float * dst);

// DFlash2: the target sink column of token (seq_id, pos) of the last decoded batch, -1 if not in the sink
LLAMA_API int32_t llama_layer_inp_sink_col(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

// r-mtpdev: keep the nextn hidden state in a device sink. The target writes its t_h_nextn rows
// there; the draft reads and writes the same sink through the link. is_target selects the role
// of the context; n_h_rows is the number of extra sink columns the MTP draft needs on top of
// n_batch (the second batch-column group followed by the pending rows and the draft-step rows).
// Returns false if unsupported or the allocation failed - the caller reports the failure instead
// of silently going back to the host path.
LLAMA_API bool llama_enable_h_nextn_sink(struct llama_context * ctx, bool is_target, uint32_t n_h_rows);

// r-mtpdev: link a target and a draft context for the direct device access to the target h sink.
// Creates the per-group events that order the target writes, the captures and the draft
// reads/writes. Returns false if the devices are not compatible or the events are not available.
LLAMA_API bool llama_context_link_h_nextn(struct llama_context * ctx_tgt, struct llama_context * ctx_dft);

// r-mtpdev: select the h source of the next draft batch: true gathers the h rows in the graph
// from the sink columns [cols, cols + n_cols), false disables the device gather. No-op when the
// context is not linked.
LLAMA_API void llama_set_h_source(struct llama_context * ctx, bool enable, const int32_t * cols, size_t n_cols);

// r-mtpdev draft: the sink group (0 or 1) that the next draft graph reads, -1 when it reads no
// group columns. The catch-up decode reads the group of the last target write; the draft steps
// only touch the pending and step columns and must pass -1.
LLAMA_API void llama_set_h_src_group(struct llama_context * ctx, int32_t group);

// r-mtpdev: the sink group written by the most recent target graph (0 or 1). The driver maps its
// columns through this group instead of counting target submissions on its own.
LLAMA_API int32_t llama_h_nextn_sink_group(struct llama_context * ctx);

// r-mtpdev draft: column offset of the next sink write. The chained-head draft uses it to keep
// one row per draft step; the non-chained draft keeps the first draft-step column. No-op for a
// target context.
LLAMA_API void llama_set_h_sink_offset(struct llama_context * ctx, int32_t offset);

// r-mtpdev draft: enable/disable the sink write of the next draft graph. The catch-up decode
// does not consume its own h rows, so it must not write the sink; the draft steps write their
// output rows for the next step to read
LLAMA_API void llama_set_h_sink_write(struct llama_context * ctx, bool write);

// r-mtpskip: target-side prefill skip. The target skips its sink write and the cross-context
// ordering for a ubatch whose positions all lie below the first position of their sequence. The
// MTP draft can never read those rows (the draft KV ring keeps only the last window + n_ubatch
// positions), so the write is wasted work. A first position of 0 disables the skip for that
// sequence (all positions are written). Set per sequence and cleared when the prompt is done.
LLAMA_API void llama_set_h_sink_first_pos(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos_first);

// r-mtpdev draft: copy the sink rows src_cols into the reserved pending columns
// [2*n_batch, 2*n_batch + n_cols), so that the next batch may overwrite the source columns. The
// copies run on the draft's output-device stream, ordered after the target write and after the
// draft graph, and before the next draft read. Returns false if the context is not a linked draft
// or n_cols exceeds the reserved space.
LLAMA_API bool llama_h_nextn_sink_capture(struct llama_context * ctx, const int32_t * src_cols, size_t n_cols);

// r-mtpdev: copy the target sink rows [group*n_batch + token_offset, ... + n_tokens) into dst.
// Requires a target sink.
LLAMA_API bool llama_h_nextn_sink_get(struct llama_context * ctx, int32_t group, size_t token_offset, size_t n_tokens, float * dst);

// r-mtpdev: the h-sink column of token (seq_id, pos) of the last decoded batch, -1 if not in the sink
LLAMA_API int32_t llama_h_nextn_sink_col(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

// r-mtpdev: wait for one prefill pipeline ubatch without draining the pipeline. A negative id
// waits for the most recent one. Returns false if the context has no scheduler or the ubatch is
// not tracked by the pipeline (decode-sized batches).
LLAMA_API bool llama_synchronize_ubatch(struct llama_context * ctx, int64_t ubatch_id);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
