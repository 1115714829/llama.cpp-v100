#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <unordered_map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// DFlash2: device-side link between a target context (writes the layer-input sink) and a draft
// context (copies from it). The same object is held by both contexts.
struct llama_embd_link {
    ggml_backend_event_t ev_tgt_done  = nullptr; // target finished writing the sink
    ggml_backend_event_t ev_copy_done = nullptr; // draft finished copying from the sink

    ggml_tensor * t_layer_inp_sink = nullptr;    // target sink, [n_embd * n_layers, n_batch]
    ggml_tensor * t_embd_src_sink  = nullptr;    // draft staging sink, [n_embd_inp, n_ubatch]

    // r-mtpdev: device link for the nextn hidden state (see llama_context_link_h_nextn). The
    // batch columns of the h sink are split into two groups; every event that guards them exists
    // once per group and is indexed by the group of the write or read it orders. The pending
    // columns need no event: only the draft stream reads and writes them
    ggml_backend_event_t ev_h_tgt_done[2]  = { nullptr, nullptr }; // target finished writing the group
    ggml_backend_event_t ev_h_read_done[2] = { nullptr, nullptr }; // draft finished reading the group
    ggml_tensor * t_h_sink_tgt = nullptr;          // target h sink, read by the draft graph
    int32_t h_sink_group_last = 0;                 // group written by the most recent target graph

    ~llama_embd_link() {
        ggml_backend_event_free(ev_tgt_done);
        ggml_backend_event_free(ev_copy_done);
        for (int i = 0; i < 2; ++i) {
            ggml_backend_event_free(ev_h_tgt_done[i]);
            ggml_backend_event_free(ev_h_read_done[i]);
        }
    }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    // wait for all in-flight prefill pipeline ubatches without touching the scheduler copy index
    // or the perf counters. callers use it before mutating state the pipeline may still be writing
    void pipeline_drain();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    void    set_logits_topk(int32_t k);
    void    reserve_if_needed();
    int32_t get_logits_topk_ith(int32_t i, const llama_token ** ids, const float ** logits);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // DFlash2: keep the target layer inputs in a device sink instead of host buffers
    bool enable_layer_inp_sink(const std::vector<uint32_t> & lids);
    // DFlash2: link a draft context to this target context (allocates the draft staging sink)
    bool link_embd(llama_context * ctx_dft);
    // DFlash2: switch the next embd batch of a linked draft between the device sink and the host input
    void set_embd_source(bool enable, int32_t token_offset);
    // DFlash2: read back [token_offset, token_offset + n_tokens) rows of the sink into dst
    bool layer_inp_sink_get(size_t token_offset, size_t n_tokens, float * dst);
    // DFlash2: the sink column that holds the features of token (seq_id, pos) of the last decoded
    // batch, or -1 if the token is not in the sink
    int32_t layer_inp_sink_col(llama_seq_id seq_id, llama_pos pos) const;

    // r-mtpdev: keep the nextn hidden state in a device sink. The target writes its t_h_nextn rows
    // there; the draft reads them as its graph input. is_target selects the role of the context.
    // n_h_rows is the number of extra sink columns on top of n_batch: the second batch-column
    // group followed by the pending rows and the draft-step rows; the target allocates them, the
    // draft only reserves the role and is bound to the target sink by link_h_nextn
    bool enable_h_nextn_sink(bool is_target, uint32_t n_h_rows);
    // r-mtpdev: link a draft context to this target context, so that the draft reads and writes
    // the target h sink in place and the two sides are ordered by per-group CUDA events
    bool link_h_nextn(llama_context * ctx_dft);
    // r-mtpdev: select the h source of the next draft batch: true gathers the h in the graph from
    // the sink columns [cols, cols + n_cols), false disables the device gather
    void set_h_source(bool enable, const int32_t * cols, size_t n_cols);
    // r-mtpdev draft: the sink group that the next draft graph reads. The driver sets it to the
    // group of the last target write before a catch-up and to -1 before the draft steps, so that
    // only the catch-up records the group as read
    void set_h_src_group(int32_t group);
    // r-mtpdev: the sink group written by the most recent target graph (0 or 1)
    int32_t h_nextn_sink_group() const;
    // r-mtpdev draft: column offset of the next sink write (the chained-head draft step index
    // times n_seq_max). target writes get the group base from their own rotation
    void set_h_sink_offset(int32_t offset);
    // r-mtpdev draft: enable/disable the sink write of the next draft graph (catch-up decodes
    // keep it off so that they do not overwrite the target batch columns)
    void set_h_sink_write(bool write);
    // r-mtpskip target: skip the sink write and the cross-context ordering of a ubatch whose
    // positions all lie below pos_first (0 = write every position). Set per sequence before a
    // prompt starts and cleared once the prompt is done
    void set_h_sink_first_pos(llama_seq_id seq_id, llama_pos pos_first);
    // r-mtpskip: true when every token of the ubatch lies below the sink boundary of all of its
    // sequences, so the target writes no sink rows for it
    bool h_sink_ubatch_skip(const llama_ubatch & ubatch) const;
    // r-mtpdev draft: copy the sink rows src_cols into the reserved pending columns
    // [2*n_batch, 2*n_batch + n_cols). Runs on the draft's output-device stream, ordered after
    // the target write and after the draft graph, and before the next draft read
    bool h_nextn_sink_capture(const int32_t * src_cols, size_t n_cols);
    // r-mtpdev: read back the sink rows [group*n_batch + token_offset, ... + n_tokens) into dst
    bool h_nextn_sink_get(int32_t group, size_t token_offset, size_t n_tokens, float * dst);
    // r-mtpdev: the h-sink column of token (seq_id, pos) of the last decoded batch, -1 if absent
    int32_t h_nextn_sink_col(llama_seq_id seq_id, llama_pos pos) const;
    // r-mtpdev: wait for one in-flight prefill pipeline ubatch without draining the rest of the
    // pipeline. A negative id waits for the most recent one. Returns false when no prefill
    // pipeline ubatch has been submitted (decode-sized batches are not tracked there)
    bool synchronize_ubatch(int64_t ubatch_id);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    // pipeline: submit the ubatch to the prefill pipeline instead of computing it synchronously.
    // ubatch_id orders the submission and is used later to wait for this ubatch's outputs
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret,
                         size_t     token_offset = 0,
                           bool     pipeline = false,
                        int64_t     ubatch_id = 0);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(ggml_backend_sched_t sched, const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    // n_embd > 0 reserves the graph of an embd ubatch with that row width (e.g. DFlash feature injection)
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr, uint32_t n_embd = 0,
        ggml_backend_sched_t sched_override = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev(int slot);

    // graph result of one pipeline stage, see gf_res_pipe
    llm_graph_result * get_gf_res_pipe(int stage);

    // slot of the previous-graph cache for this ubatch
    int gf_res_slot(const llama_ubatch & ubatch) const;

    // scheduler that holds the graph state of the given slot
    ggml_backend_sched_t sched_for_slot(int slot) const;

    llm_graph_params graph_params(
                     ggml_backend_sched_t sched,
                         llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
                             size_t        token_offset = 0,
                                bool        h_sink_skip = false) const;

    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched) const;

    ggml_status graph_compute_impl(ggml_backend_sched_t sched, ggml_cgraph * gf, bool batched);

    // wait until the submission threads of the schedulers of this context have enqueued the work
    // already posted for the given backend on its stream. no-op for schedulers without submission
    // threads, or when the backend is not part of them
    void wait_submitted(ggml_backend_t backend);

    // set the threadpool and thread counts for the given batch type
    void graph_set_threads(bool batched);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    // DFlash2: device-side sink for the target layer inputs, [n_embd * n_layers, n_batch]
    ggml_context_ptr        ctx_sink; // declared before buf_sink so that the buffer is freed first
    ggml_backend_buffer_ptr buf_sink;
    ggml_tensor *           t_layer_inp_sink = nullptr;
    std::vector<uint32_t>   layer_inp_sink_layers;

    // DFlash2: the (seq_id, pos) of each sink column written by the last batch, in column order
    std::vector<std::pair<llama_seq_id, llama_pos>> sink_cols;
    // DFlash2: per-sequence (pos, column) index into sink_cols; positions grow within a batch
    std::unordered_map<llama_seq_id, std::vector<std::pair<llama_pos, int32_t>>> sink_cols_by_seq;

    // DFlash2: draft-side staging sink for the injected features, [n_embd_inp, n_ubatch]
    ggml_context_ptr        ctx_embd_src;
    ggml_backend_buffer_ptr buf_embd_src;
    ggml_tensor *           t_embd_src_sink = nullptr;

    // r-mtpdev: device-side sink for the nextn hidden state, [n_embd_out, 2*n_batch + n_h_rows].
    // The target writes its t_h_nextn rows and the MTP bookkeeping rows; a linked draft context
    // shares this tensor (t_h_sink points at the target sink) and reads/writes its columns
    ggml_context_ptr        ctx_h_sink; // declared before buf_h_sink so that the buffer is freed first
    ggml_backend_buffer_ptr buf_h_sink;
    ggml_tensor *           t_h_sink = nullptr;
    bool                    h_sink_is_target = false; // true: this sink holds the target h rows
    // non-owning: this context's own backend, its stream queues the h sink event waits/records and
    // the pending capture copy. The target backend owns the sink buffer; the draft queues on its
    // own stream so the cross-context events order the two streams
    ggml_backend_t          backend_h_sink = nullptr;
    uint32_t                h_sink_extra = 0;         // target: extra columns after n_batch (second group + bookkeeping)

    // r-mtpdev: the (seq_id, pos) of each h-sink column written by the last batch, in column order
    std::vector<std::pair<llama_seq_id, llama_pos>> h_sink_cols;
    // r-mtpdev: per-sequence (pos, column) index into h_sink_cols; positions grow within a batch
    std::unordered_map<llama_seq_id, std::vector<std::pair<llama_pos, int32_t>>> h_sink_cols_by_seq;

    // r-mtpdev draft: the next draft graph gathers its h from the sink at these columns.
    // The vector only grows, so the graph input pointer stays valid across batches
    bool                 h_src_enable = false;
    std::vector<int32_t> h_src_cols;

    // r-mtpdev draft: sink column offset of the next write (see set_h_sink_offset). The target
    // stores the group base [0, n_batch) or [n_batch, 2*n_batch) of its next sink write here
    int32_t h_sink_offset = 0;
    // r-mtpdev: write the nextn rows of the next graph into the sink (draft catch-up turns it off)
    bool    h_sink_write = true;
    // r-mtpdev target: the group that the next sink-writing graph uses. Only the pipelined
    // prefill rotates between the groups; every other target graph writes group 0
    int32_t h_sink_group = 0;
    // r-mtpdev draft: the group that the next draft graph reads, -1 when it reads no group
    // columns (draft steps only touch the pending and step columns)
    int32_t h_src_group = -1;

    // r-mtpskip target: per-sequence first position of the MTP draft KV window. A prefill ubatch
    // whose positions all lie below this value writes no sink rows and does no cross-context
    // ordering; 0 means the sequence writes every position (no skip)
    std::vector<llama_pos> h_sink_first_pos;

    // r-mtpdev: the id of the most recent ubatch submitted through the prefill pipeline, -1 if none
    int64_t pipeline_ubatch_last = -1;

    // producer/consumer link, shared with the linked context
    std::shared_ptr<llama_embd_link> embd_link;
    bool    embd_src_enable = false; // draft: the next embd batch reads the staging sink
    int32_t embd_src_offset = 0;     // draft: first target-sink column to copy

    // top-k logits output (per row: n_shards blocks of k candidates), see llama_set_logits_topk;
    // the graph ids are local to their vocab shard; topk_ids_row holds the global ids of the last
    // row read, rebased with the shard prefix sums (cparams.logits_topk_shard_off)
    buffer_view<float>       topk_logits = {nullptr, 0};
    buffer_view<int32_t>     topk_ids    = {nullptr, 0};
    std::vector<llama_token> topk_ids_row;

    // false when the last batch only copied its top-k logits
    bool logits_full = true;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // DFlash drafts keep the embd injection graph on a separate scheduler so that
    // the alternating inject/block passes do not invalidate each other's graph state
    ggml_backend_sched_ptr sched_slot0;

    // decode graphs (verification/single-token batches) get their own scheduler so that
    // prefill allocations do not invalidate them across requests, see gf_res_slot
    ggml_backend_sched_ptr sched_dec;

    // max tokens of a batch that sched_dec accepts and is reserved for, set once in the constructor
    uint32_t n_dec_max = 0;

    // monotonic id of the prefill ubatches submitted to the pipeline. it spans decode() calls so
    // the pipeline can stay in flight across the n_batch-sized chunks of one prompt
    int64_t pipeline_ubatch_id = 0;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with different graph layouts distinct CUDA graph cache keys:
    // by outputs for regular contexts, by input kind (embd/token) for DFlash drafts,
    // slot 2 for decode batches when sched_dec is enabled
    std::array<llm_graph_result_ptr, 3> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // one graph result per pipeline stage: a ubatch's graph result must stay alive while its
    // submission workers still read its nodes, so it is reset only when the stage is reused.
    // indexed by the stage returned by ggml_backend_sched_pipeline_next_stage
    std::vector<llm_graph_result_ptr> gf_res_pipe;

    std::array<llm_graph_result *, 3> gf_res_prev_active = { nullptr, nullptr, nullptr };

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
