#pragma once

#include "llama-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus an indexer cache for block-sparse attention (glm5-next k-pool DSA, qwen4exp QSA)
// two layouts:
//  - per cell (glm5-next): a third kv cache with one indexer row per token, a side buffer over the attention cells: same
//    size, padding, streams and slots, so cell j is one token in both
//  - compact (qwen4exp, indexer_kpool_compact): one pooled key per complete pool, and the raw keys of the cells whose
//    pool may still need them in a small ring, see kpool_store

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                 uint32_t   n_ubatch,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout and kpool_store are incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    // the indexer cache tracks the attention cache cell for cell, so it takes the same n_kv rule
    void set_kq_range_allowed(bool allowed) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void collect_tensors(std::vector<std::pair<std::string, ggml_tensor *>> & out) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer, or keeps it compact

    // Whether the model keeps an indexer cache, in either layout.
    bool has_idx() const;

    // The compact layout's storage, nullptr in the per-cell layout.
    struct kpool_store;
    const kpool_store * get_kstore() const { return kstore.get(); }

    // The cache whose cells the pool layout follows: the indexer cache, or in the compact layout the attention cache.
    const llama_kv_cache * get_idx_cells() const;

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. A sequence edit drops it from the position
    // it staled on, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The graph's pool count for a layout of n_pool_real pools over the cells in use now: padded like the n_kv of a
    // ubatch that does not view the whole cache, so a decode step changes it only where its n_kv changes. Always
    // above n_pool_real: the last padded pool is unused. With full, the count is over the whole cache, matching the
    // n_kv of a 2+ token range-mask ubatch, so the prompt blocks share one shape (see kpool_build_state).
    uint32_t kpool_n_pool(uint32_t n_pool_real, bool full = false) const;

    // For tests: kpool_n_pool for the current layout.
    uint32_t kpool_n_pool_layout() const;

    // For tests: the first way the current layout's pool count breaks its bounds (no padded pool after the real ones,
    // more pools than the reserved graph holds), empty if none. Call it where kpool_layout_diff is called.
    std::string kpool_n_pool_check() const;

    // For tests: the first difference between the layout as the last update left it and one rebuilt from scratch
    // over the cache as it is now, empty if none. Call it after a decode, before the next sequence edit.
    std::string kpool_layout_diff() const;

    // For tests: picks the cells of padded new-pool entries over the current layout as set_input_kpool does, for
    // counts up to every cell that is no rep, and returns the first pick that is out of range, a pool's rep, taken
    // twice or short of the count, empty if none. Call it where kpool_layout_diff is called.
    // The compact layout pads into rows of its own and has nothing to check here.
    std::string kpool_pad_check() const;

    // For tests: the first inconsistency of the compact layout's pooled slots (a pool without one, reference counts,
    // the registry of identical pools) and raw-key ring (a cell whose pool may still need its raw key without one),
    // empty if none. Call it where kpool_layout_diff is called.
    std::string kpool_store_check() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_store is incomplete here
    std::unique_ptr<kpool_store> kstore;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // whether the current layout has cells shared between sequences (kpool_layout is incomplete here, so out of line)
    bool kpool_layout_shared() const;

    // brings lay up to date with the cache, keeping each sequence's cells before its stale position
    // dropped: if set, gets the pooled slots of the pools that no longer stand, with the global cell each ended at
    // (compact layout)
    void kpool_layout_build(kpool_layout & lay, const stale_pos_t & stale, std::vector<std::pair<int32_t, int64_t>> * dropped) const;

    // compact layout, see kpool_store:
    // brings the layout up to date after the cells changed, the sequences edited from their stale positions: a pool that
    // now stands takes the slot of an identical one, or a new slot that the next ubatch pools into (fresh); the slots
    // of the pools that no longer stand are released after that, so a pool rebuilt as it was keeps its key
    // false if the table has no slot left for a new pool (its pools keep slot -1, the caller fails the batch)
    bool kstore_sync(const stale_pos_t & stale, std::vector<std::pair<llama_seq_id, uint32_t>> & fresh);
    // the same after an edit, which never needs a pool computed: no fresh pool is allowed
    void kstore_sync_edit(const stale_pos_t & stale, const char * what);
    // whether seq_rm(seq_id, p0, p1) leaves a layout the compact cache can keep: a tail cut whose new incomplete pool has
    // its raw keys in the ring, a sequence removed whole, or nothing removed
    bool kstore_seq_rm_ok(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    // the ubatch at hand, after the layout took its cells: pools to compute (fresh), raw-key rows for its cells, and
    // the rows of every cell no pool can need any more released. false if a fresh pool misses a raw key
    bool kstore_apply(const llama_kv_cache::slot_info & sinfo, std::vector<std::pair<llama_seq_id, uint32_t>> & fresh);
    void kstore_clear(bool data);
    // whether seq_id (every sequence if < 0) has cells at positions in [p0, p1)
    bool kstore_has_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    // forgets sequence s: the slots of its pools are released and its layout emptied (its cells are about to change)
    void kstore_drop_seq(llama_seq_id s);
    // the same for every sequence if seq_id < 0
    void kstore_forget(llama_seq_id seq_id);
    // a copy across streams: dst takes src's layout in its own stream, with src's slots and raw-key rows
    void kstore_copy_stream(llama_seq_id src, llama_seq_id dst);
    void kstore_state_write(llama_io_write_i & io, llama_seq_id seq_id, bool partial) const;
    void kstore_state_read (llama_io_read_i  & io, llama_seq_id seq_id, bool partial);

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();

    // applies a ubatch to the compact layout, see kstore_apply
    friend class llama_memory_hybrid_idx_context;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        // the per-token rows of the pools' members: cache rows of the cells (per-cell layout), ring rows (compact)
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;
        // the pooled keys in place, [n_embd, 1, n_rows]: the key table of ggml_qsa_select, rows picked by pool_cells
        ggml_tensor * pooled_table() const { return pooled_rows; }
        // compact layout: stores the raw keys [n_embd, n_tokens] of the ubatch in their ring rows, see set_input_kring
        ggml_tensor * store_raw(ggml_tensor * values, ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        // per-cell layout: k holds the per-token rows and the pooled key side by side
        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);
        // compact layout: the raw-key ring and the pooled table
        kpool_access(ggml_context * ctx, ggml_tensor * ring, ggml_tensor * pooled, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
        ggml_tensor  * pooled_rows;
        bool           compact = false;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer, and in the compact layout, which has no per-cell indexer cache
    const llama_kv_cache_context * get_idx() const;

    // whether the model keeps an indexer cache, in either layout
    bool has_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused, see kpool_n_pool.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    // pool_rng (I32 [2, n_tokens]): the pools [lo, hi) each token sees, for ggml_qsa_select; with it pool_mask may be
    // null and the padded member and tail cells are -1 instead of the n_kv sentinel
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr, ggml_tensor * pool_rng = nullptr) const;
    // compact layout: the ring row of each ubatch token's raw key, I64 [n_tokens], token order as the slot info's
    void set_input_kring(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
