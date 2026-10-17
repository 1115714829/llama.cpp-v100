#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

//
// llama_memory_hybrid_idx::kpool_store
//

// The compact indexer cache (qwen4exp QSA, the layout of the official implementation and of SGLang). The per-cell layout
// keeps a raw key | pooled key row for every cell; this keeps one pooled key per complete pool of kpool cells, in a table
// of pooled slots, and the raw keys only of the cells whose pool may still have to be pooled, in a ring:
//  - a pool is pooled once, from its members' raw keys, by the ubatch that completes it, into a slot of its own
//  - a raw key stays while its cell is in the incomplete tail of a sequence, could become part of one again after a
//    rollback of up to n_rs_seq tokens (the recurrent cache refuses longer ones), or belongs to the ubatch at hand
//  - a checkpoint (partial state) carries the raw keys of its incomplete tail, so the cut that follows its restore finds
//    them again; a full state carries the pooled keys and the same tail
// Edits that would regroup pools whose raw keys are gone (a cut in the middle or at the front, a copy into a sequence
// that has cells or from past the source's first position) are refused, see kstore_seq_rm_ok and seq_cp.
//
// Slots are addressed through pool_cells, so the pools of a sequence can sit anywhere in the table; they are handed out
// lowest first, so a single sequence keeps pool i in slot i. A slot is shared by every layout entry of the same pool (the
// same member cells): a registry maps a pool's last member to its slot, which is how a copy and a rebuilt layout keep
// their pooled keys without the raw keys. A copy across streams shares the slot from other cells, so a slot counts the
// entries that hold its own member cells apart (owners): the registry entry goes with the last owner, even while a copy
// keeps the slot, and a live registry entry never names a rewritten cell: a cell is rewritten only once no sequence
// holds it, and every pool it was in went with that.
struct llama_memory_hybrid_idx::kpool_store {
    struct layer {
        uint32_t il;

        ggml_tensor * pooled; // [n_embd, n_slot + n_trash]: the pooled keys, then one row per padded new-pool entry
        ggml_tensor * ring;   // [n_embd, n_ring]: raw keys
    };

    const ggml_type type;
    const int64_t   n_embd;
    const uint32_t  kpool;
    const uint32_t  n_slot;   // slots for pools
    const uint32_t  n_trash;  // rows after them for the padded new-pool entries, the most a graph pads
    const uint32_t  n_ring;
    const uint32_t  n_rs_seq; // the longest cut the recurrent cache rolls back without a checkpoint
    const int64_t   n_cells;  // global cells of the attention cache: size * streams
    const bool      no_alloc; // a dry run (fit): the tensors have dummy buffers

    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // pooled slots
    std::vector<uint32_t> slot_refs;    // [n_slot] layout entries in the slot
    std::vector<uint32_t> slot_owns;    // [n_slot] the entries among them made of the slot's member cells
    std::vector<int64_t>  slot_members; // [n_slot*kpool] member cells of the slot's pool, -1 for a free slot
    std::vector<int32_t>  rep_slot;     // [n_cells] the registered slot of the pool ending at the cell, -1 if none
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> slot_free;

    // raw-key ring
    std::vector<int32_t>  ring_of;      // [n_cells] the ring row with the cell's raw key, -1 if none
    std::vector<uint32_t> ring_refs;    // [n_ring] cells reading the row: a copy across streams shares rows
    std::vector<int64_t>  ring_cells;   // the cells with a row, in no order
    std::vector<uint32_t> ring_free;    // free rows, a stack
    std::vector<uint32_t> stamp;        // [n_cells] the cells a release pass keeps
    uint32_t              stamp_cur = 0;

    kpool_store(const llama_model & model, ggml_type type, bool offload, const llama_memory_i::layer_filter_cb & filter,
            int64_t n_embd, uint32_t kpool, uint32_t n_slot, uint32_t n_trash, uint32_t n_ring, uint32_t n_rs_seq,
            int64_t n_cells) :
        type(type), n_embd(n_embd), kpool(kpool), n_slot(n_slot), n_trash(n_trash), n_ring(n_ring), n_rs_seq(n_rs_seq),
        n_cells(n_cells), no_alloc(model.hparams.no_alloc) {
        const auto & hparams = model.hparams;

        struct buft_cmp {
            bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
                return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
            }
        };
        std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_cmp> ctx_map;

        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (!hparams.has_kv(il) || (filter && !filter(il))) {
                continue;
            }

            // the layer's own device, as llama_kv_cache places its K/V; the meta device mirrors cache_idx_* tensors
            ggml_backend_buffer_type_t buft = offload ? ggml_backend_dev_buffer_type(model.dev_layer(il)) : ggml_backend_cpu_buffer_type();

            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = {
                    /*.mem_size   =*/ size_t(2u*hparams.n_layer_all*ggml_tensor_overhead()),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context * ctx = ggml_init(params);
                if (!ctx) {
                    throw std::runtime_error("failed to create ggml context for the compact indexer cache");
                }
                it = ctx_map.emplace(buft, ctx).first;
            }

            ggml_tensor * pooled = ggml_new_tensor_2d(it->second.get(), type, n_embd, (int64_t) n_slot + n_trash);
            ggml_tensor * ring   = ggml_new_tensor_2d(it->second.get(), type, n_embd, n_ring);
            ggml_format_name(pooled, "cache_idx_k_l%d", il);
            ggml_format_name(ring,   "cache_idx_r_l%d", il);

            map_layer_ids[il] = (int32_t) layers.size();
            layers.push_back({ il, pooled, ring });
        }

        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_t buf;
            if (no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer, as llama_kv_cache does
                for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                    t->buffer = buf;
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
            }
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for the compact indexer cache");
            }
            // padded pools are scored from the first spare row, so every row starts finite
            ggml_backend_buffer_clear(buf, 0);
            LLAMA_LOG_INFO("%s: %10s compact indexer buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf),
                    ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            ctxs_bufs.emplace_back(std::move(ctx), buf);
        }

        reset();
    }

    void reset() {
        slot_refs.assign(n_slot, 0);
        slot_owns.assign(n_slot, 0);
        slot_members.assign((size_t) n_slot*kpool, -1);
        rep_slot.assign((size_t) n_cells, -1);
        slot_free = decltype(slot_free)();
        for (uint32_t r = 0; r < n_slot; ++r) {
            slot_free.push(r);
        }

        ring_of.assign((size_t) n_cells, -1);
        ring_refs.assign(n_ring, 0);
        ring_cells.clear();
        ring_free.clear();
        for (uint32_t r = n_ring; r > 0; --r) {
            ring_free.push_back(r - 1);
        }
        stamp.assign((size_t) n_cells, 0);
        stamp_cur = 0;
    }

    size_t row_size() const {
        return ggml_row_size(type, n_embd);
    }

    const layer & get_layer(int32_t il) const {
        return layers.at(map_layer_ids.at(il));
    }

    // the registered slot of the pool with these members, -1 if none; a match is one more owner
    int32_t slot_find(const int64_t * m) {
        const int32_t r = rep_slot[m[kpool - 1]];
        if (r < 0 || !std::equal(m, m + kpool, slot_members.begin() + (size_t) r*kpool)) {
            return -1;
        }
        ++slot_refs[r];
        ++slot_owns[r];
        return r;
    }

    // a new slot for the pool with these members, -1 if the table is full
    int32_t slot_alloc(const int64_t * m) {
        if (slot_free.empty()) {
            return -1;
        }
        const int32_t r = (int32_t) slot_free.top();
        slot_free.pop();
        std::copy(m, m + kpool, slot_members.begin() + (size_t) r*kpool);
        slot_refs[r] = 1;
        slot_owns[r] = 1;
        // two pools differing only before their last member keep one registered, the other is pooled on its own
        if (rep_slot[m[kpool - 1]] < 0) {
            rep_slot[m[kpool - 1]] = r;
        }
        return r;
    }

    // one more entry reading the slot from other cells (a copy across streams)
    void slot_share(int32_t r) {
        GGML_ASSERT(r >= 0 && (uint32_t) r < n_slot && slot_refs[r] > 0);
        ++slot_refs[r];
    }

    // an entry whose pool ends at cell rep drops the slot
    void slot_unref(int32_t r, int64_t rep) {
        GGML_ASSERT(r >= 0 && (uint32_t) r < n_slot && slot_refs[r] > 0);
        const int64_t rep_r = slot_members[(size_t) r*kpool + kpool - 1];
        if (rep == rep_r) {
            GGML_ASSERT(slot_owns[r] > 0);
            // without an owner the members can be rewritten, so nothing may find the slot by them any more
            if (--slot_owns[r] == 0 && rep_slot[rep_r] == r) {
                rep_slot[rep_r] = -1;
            }
        }
        if (--slot_refs[r] > 0) {
            return;
        }
        GGML_ASSERT(slot_owns[r] == 0 && rep_slot[rep_r] != r);
        std::fill(slot_members.begin() + (size_t) r*kpool, slot_members.begin() + (size_t) (r + 1)*kpool, -1);
        slot_free.push((uint32_t) r);
    }

    // drops the cell's ring row; the caller keeps ring_cells in step
    void ring_unref(int64_t g) {
        const int32_t row = ring_of[g];
        GGML_ASSERT(row >= 0 && ring_refs[row] > 0);
        ring_of[g] = -1;
        if (--ring_refs[row] == 0) {
            ring_free.push_back((uint32_t) row);
        }
    }

    // a row of the cell's own to write its raw key into: kept if the cell has one alone, else a new one
    bool ring_take(int64_t g) {
        const int32_t row = ring_of[g];
        if (row >= 0 && ring_refs[row] == 1) {
            return true;
        }
        if (ring_free.empty()) {
            return false;
        }
        if (row >= 0) {
            ring_unref(g); // shared with a copy: copy on write
        } else {
            ring_cells.push_back(g);
        }
        ring_of[g] = (int32_t) ring_free.back();
        ring_free.pop_back();
        ring_refs[ring_of[g]] = 1;
        return true;
    }

    // the cell reads the row of another one, which holds the same raw key (a copy across streams)
    void ring_share(int64_t g_dst, int64_t g_src) {
        const int32_t row = ring_of[g_src];
        GGML_ASSERT(row >= 0);
        if (ring_of[g_dst] >= 0) {
            ring_unref(g_dst);
        } else {
            ring_cells.push_back(g_dst);
        }
        ring_of[g_dst] = row;
        ++ring_refs[row];
    }

    void stamp_begin() {
        if (++stamp_cur == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            stamp_cur = 1;
        }
    }

    // releases the rows of the cells the pass did not stamp
    void ring_release_unstamped() {
        size_t n = 0;
        for (const int64_t g : ring_cells) {
            if (ring_of[g] < 0) {
                continue;
            }
            if (stamp[g] != stamp_cur) {
                ring_unref(g);
                continue;
            }
            ring_cells[n++] = g;
        }
        ring_cells.resize(n);
    }

    // drops the rows of these cells, whose raw keys no longer hold
    template <typename F>
    void ring_forget(F && for_each_cell) {
        for_each_cell([&](int64_t g) {
            if (ring_of[g] >= 0) {
                ring_unref(g);
            }
        });
        size_t n = 0;
        for (const int64_t g : ring_cells) {
            if (ring_of[g] >= 0) {
                ring_cells[n++] = g;
            }
        }
        ring_cells.resize(n);
    }

    size_t total_size() const {
        size_t size = 0;
        for (const auto & [ctx, buf] : ctxs_bufs) {
            size += ggml_backend_buffer_get_size(buf.get());
        }
        return size;
    }
};

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;
        // The pooled slot of each pool, in step with pools (compact layout, see kpool_store); -1 until one is taken.
        std::vector<int32_t>  slots;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;

        // Whether any cell also carries another sequence, which rules out caching this sequence's pooled keys.
        // (per-cell layout: the compact one keeps a slot per pool and shares it only between identical pools)
        bool shared = false;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
    bool cache_safe      = true;
};

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr || model.hparams.indexer_kpool_compact ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    if (filter_idx == nullptr || !model.hparams.indexer_kpool_compact) {
        return;
    }

    const uint32_t kpool = model.hparams.indexer_kpool;
    GGML_ASSERT(kpool > 1 && model.hparams.indexer_kpool_by_order && "the compact indexer cache groups cells in sequence order");

    const llama_kv_cache * attn = get_mem_attn();
    const int64_t  n_cells = (int64_t) attn->get_size()*attn->get_n_stream();
    const uint32_t n_ub    = std::max<uint32_t>(1, std::min<uint32_t>(n_ubatch, (uint32_t) n_cells));

    // every pool of every sequence when none shares a cell, plus one pool per sequence that straddles the end of a
    // shared prefix (the copies of seq_cp, tokens of several sequences)
    const uint32_t n_slot  = (uint32_t) (n_cells/kpool) + n_seq_max;
    // a ubatch pads its new-pool list to at most a pool per kpool tokens plus one per sequence, see kpool_build_state
    const uint32_t n_trash = n_ub/kpool + n_seq_max;
    // the raw keys of a whole ubatch, plus per sequence its incomplete tail and the cells a rollback can make one
    const uint32_t n_ring  = n_ub + n_seq_max*(n_rs_seq + 2*kpool);

    kstore = std::make_unique<kpool_store>(model, type_k, offload, filter_idx, (int64_t) model.hparams.indexer_head_size,
            kpool, n_slot, n_trash, n_ring, n_rs_seq, n_cells);

    LLAMA_LOG_INFO("%s: compact indexer cache: %u pooled slots + %u pad rows, %u raw-key ring rows, %zu layers, %.2f MiB\n",
            __func__, n_slot, n_trash, n_ring, kstore->layers.size(), kstore->total_size()/1024.0/1024.0);

    // the layout is kept in step with every edit, so it exists from the start
    kpool_lay = std::make_unique<kpool_layout>();
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (has_idx()) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::set_kq_range_allowed(bool allowed) {
    llama_memory_hybrid::set_kq_range_allowed(allowed);
    if (mem_idx) {
        mem_idx->set_kq_range_allowed(allowed);
    }
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }

    if (kstore) {
        kstore_clear(data);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (kstore) {
        // the compact cache keeps no raw key a regrouping would need, so it refuses before any cache is touched
        if (!kstore_seq_rm_ok(seq_id, p0, p1)) {
            return false;
        }

        if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
            return false;
        }

        const bool res = get_mem_attn()->seq_rm(seq_id, p0, p1);

        // a tail cut keeps the cells before p0 and the pools made of them, a sequence removed whole loses them all
        const llama_pos p_stale = std::max<llama_pos>(p0, 0);
        stale_pos_t stale = stale_pos_clean();
        if (seq_id < 0) {
            stale.fill(p_stale);
        } else {
            stale[seq_id] = p_stale;
        }
        kstore_sync_edit(stale, "seq_rm");

        return res;
    }

    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);

        // removing a sequence can free cells another sequence shared, but only this one is marked stale, so the
        // survivor would keep shared = true and pin cache_safe off forever; stale every sequence to re-derive it
        if (kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (kstore && seq_id_src != seq_id_dst) {
        GGML_ASSERT(seq_id_src >= 0 && seq_id_src < (llama_seq_id) LLAMA_MAX_SEQ);
        GGML_ASSERT(seq_id_dst >= 0 && seq_id_dst < (llama_seq_id) LLAMA_MAX_SEQ);

        const llama_kv_cache * cells = get_idx_cells();

        if (cells->get_stream(seq_id_src) == cells->get_stream(seq_id_dst)) {
            // the cells become shared. The destination keeps the pooled keys of the source's pools only as the same
            // pools, so its cells must be the source's from its first one on: with cells of its own, or starting later,
            // the copied cells would regroup, and their raw keys are gone. Every caller in the tree copies a whole
            // prompt into an emptied sequence (server: copy_prompt_to, copy_state_to, the cross-slot prefix reuse)
            const auto & src = kpool_lay->seqs[seq_id_src];
            GGML_ASSERT(kpool_lay->seqs[seq_id_dst].cells.empty() &&
                    "compact QSA indexer cache: seq_cp needs an empty destination sequence (seq_rm it first)");
            GGML_ASSERT((src.cells.empty() || p0 <= src.cells.front().first) &&
                    "compact QSA indexer cache: seq_cp must start at the source's first position");

            llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

            // the destination's pools are the source's: they find their slots in the registry
            stale_pos_t stale = stale_pos_clean();
            stale[seq_id_dst] = 0;
            kstore_sync_edit(stale, "seq_cp");
        } else {
            // across streams llama_kv_cache copies the whole stream (it asserts the full range): the destination's pools
            // are the source's in the destination stream, they read the same slots and raw-key rows
            kstore_drop_seq(seq_id_dst);
            llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
            kstore_copy_stream(seq_id_src, seq_id_dst);
        }

        return;
    }

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // the copy shares cells, which cannot hold two groupings, so both sides drop their cached keys
        mem_idx_stale_set(seq_id_src, 0);
        mem_idx_stale_set(seq_id_dst, 0);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (kstore) {
        // the other sequences lose their cells and with them their pools; the kept one stays as it is
        stale_pos_t stale;
        stale.fill(0);
        if (seq_id >= 0 && seq_id < (llama_seq_id) LLAMA_MAX_SEQ) {
            stale[seq_id] = POS_CLEAN;
        }
        kstore_sync_edit(stale, "seq_keep");

        return;
    }

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // cells shared with the dropped sequences become exclusive again, their keys were never cached
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (kstore) {
        // the pooled keys are rotated to their first member's position when pooled, and the raw keys to rotate them
        // again are gone; qwen4exp ropes with M-RoPE, so the attention cache cannot shift either (get_can_shift)
        GGML_ASSERT((shift == 0 || !kstore_has_cells(seq_id, p0, p1)) &&
                "compact QSA indexer cache: positions cannot be shifted (llama_memory_can_shift is false for this model)");
    }

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (kstore) {
        // see seq_add
        GGML_ASSERT((d == 1 || !kstore_has_cells(seq_id, p0, p1)) &&
                "compact QSA indexer cache: positions cannot be divided (llama_memory_can_shift is false for this model)");
    }

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

void llama_memory_hybrid_idx::collect_tensors(std::vector<std::pair<std::string, ggml_tensor *>> & out) const {
    llama_memory_hybrid::collect_tensors(out);

    if (mem_idx) {
        mem_idx->collect_tensors(out);
    }

    if (kstore) {
        for (const auto & layer : kstore->layers) {
            out.push_back({ layer.pooled->name, layer.pooled });
            out.push_back({ layer.ring->name,   layer.ring   });
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (kstore) {
        for (const auto & [ctx, buf] : kstore->ctxs_bufs) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
            // a dry run holds dummy buffers, so the size comes from the tensors (as llama_kv_cache does)
            mb[buft] += kstore->no_alloc ?
                ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) : ggml_backend_buffer_get_size(buf.get());
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    if (kstore) {
        // [TAG_HYBRID_IDX_STATE] last, as below. Written in both modes: a checkpoint (PARTIAL_ONLY) carries the raw keys of
        // its incomplete pool, which the cut back to it after its restore needs
        kstore_state_write(io, seq_id, (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0);

        return;
    }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    const bool partial = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;

    try {
        if (kstore && !partial) {
            // the restored cells replace the sequence's: its pools go first, so that no restored pool can take the slot
            // of one that held other keys in the same cells
            kstore_forget(seq_id);
        }

        if (!partial) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        if (kstore) {
            // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
            kstore_state_read(io, seq_id, partial);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
                // it can also change which cells are shared; re-derive sharing for every sequence, as seq_rm does
                if (kpool_layout_shared()) {
                    mem_idx_stale_set(-1, 0);
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (kstore) {
        // the sequence has no cell left, so its pools go
        stale_pos_t stale = stale_pos_clean();
        stale[seq_id] = 0;
        kstore_sync_edit(stale, "state_drop");

        return;
    }

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
        // clearing this sequence can end a sharing the survivor would otherwise keep flagged (see seq_rm)
        if (kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

bool llama_memory_hybrid_idx::has_idx() const {
    return mem_idx != nullptr || kstore != nullptr;
}

const llama_kv_cache * llama_memory_hybrid_idx::get_idx_cells() const {
    return mem_idx ? mem_idx.get() : get_mem_attn();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    uint32_t generation = 0;

    // compact layout: the pools this ubatch pools, as (sequence, pool index); the slots they took are new
    std::vector<std::pair<llama_seq_id, uint32_t>> fresh;

    uint32_t n_pool_real = 0;
    uint32_t n_pool_pad  = 1; // graph size of the pool list, see llama_memory_hybrid_idx::kpool_n_pool
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
    bool     cache_safe  = true;
};

namespace {

// The graph's pool count for n_pool_real pools over n_cells padded cells, see llama_memory_hybrid_idx::kpool_n_pool.
// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool_real, uint64_t n_cells, uint32_t kpool) {
    const uint64_t n_pool = std::max<uint64_t>(n_cells/kpool, n_pool_real) + 1;
    GGML_ASSERT(n_pool <= UINT32_MAX);

    return (uint32_t) n_pool;
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

// Brings one sequence's pools up to date with its cells in the cache.
// An edit staled from p leaves every cell before p as it was: same position, same cell, so the same rank, which is
// all the pools and both scans below read. So the cells before p stay, with the pools wholly made of them, and the
// rest is dropped and appended again like new cells. A clean sequence (POS_CLEAN) keeps every cell, one whose first
// position moved keeps none, as pools count from it; keeping none is the full rebuild, on the same path.
// The pools that go hand their slots (compact layout) to dropped, with the global cell their pool ends at (cells of the
// stream are kv_size apart); the new ones get none yet (-1). The compact layout keeps a slot per pool whoever else holds
// its cells, so it does not track sharing (track_shared).
void kpool_seq_update(llama_memory_hybrid_idx::kpool_layout::seq & sq, const llama_kv_cells & cells, llama_seq_id s,
        llama_pos stale, uint32_t kpool, bool by_order, bool unified, bool track_shared, int64_t kv_size,
        std::vector<std::pair<int32_t, int64_t>> * dropped) {
    const auto & sp = cells.seq_pos_get(s);

    const llama_pos p_keep = !sp.empty() && sp.begin()->first == sq.pos_min ? stale : 0;

    size_t n_keep = (size_t) (std::partition_point(sq.cells.begin(), sq.cells.end(),
            [&](const std::pair<llama_pos, uint32_t> & c) { return c.first < p_keep; }) - sq.cells.begin());

    // the kept cells must still open the sequence: an embedding repeating the last position can land on a lower cell
    // and rank among them without an edit, which the count shows, and then none is known to stand
    if (n_keep > 0 && n_keep + (size_t) std::distance(sp.upper_bound(sq.cells[n_keep - 1]), sp.end()) != sp.size()) {
        n_keep = 0;
    }

    // where the sharing scan resumes: it went through the kept cells, but a set flag cannot tell which cell set it
    size_t j_shared = n_keep;

    if (n_keep < sq.cells.size()) {
        // the pools reaching past the kept cells go; after the last pool left the scan stepped through the cells one by
        // one, so it resumes right after that pool, at a start it took then and a full rebuild takes now
        const size_t n_pool_keep = (size_t) (std::partition_point(sq.pools.begin(), sq.pools.end(),
                [&](uint32_t j) { return j + kpool <= n_keep; }) - sq.pools.begin());
        if (dropped != nullptr) {
            for (size_t i = n_pool_keep; i < sq.slots.size(); ++i) {
                if (sq.slots[i] >= 0) {
                    dropped->emplace_back(sq.slots[i], (int64_t) sq.strm*kv_size + sq.cells[sq.pools[i] + kpool - 1].second);
                }
            }
        }
        sq.pools.resize(n_pool_keep);
        sq.slots.resize(std::min(sq.slots.size(), n_pool_keep));
        sq.j_next = sq.pools.empty() ? 0 : sq.pools.back() + kpool;
        sq.cells.resize(n_keep);

        if (sq.shared) {
            sq.shared = false;
            j_shared  = 0;
        }
    }

    // unchanged while any cell is kept
    sq.pos_min = sp.empty() ? 0 : sp.begin()->first;

    for (auto it = sq.cells.empty() ? sp.begin() : sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
        sq.cells.push_back(*it);
    }

    // sharing starts with a seq_cp and ends only with a cut above: a seq_rm/state_drop/state_read that can free shared
    // cells stales every sequence from 0, so once set it holds until a cut and the rescan can be skipped
    if (track_shared && unified && !sq.shared) {
        for (size_t j = j_shared; j < sq.cells.size(); ++j) {
            if (cells.seq_count(sq.cells[j].second) > 1) {
                sq.shared = true;
                break;
            }
        }
    }

    // Pools start at the first valid token
    size_t j = sq.j_next;
    if (by_order) {
        // consecutive cells in sequence order, whatever their positions
        for (; j + kpool <= sq.cells.size(); j += kpool) {
            sq.pools.push_back((uint32_t) j);
        }
    } else {
        while (j + kpool <= sq.cells.size()) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
            } else {
                ++j;
            }
        }
    }
    sq.j_next = j;
    sq.slots.resize(sq.pools.size(), -1);
}

// The cells for the n_pad padded entries of the new pool list, handed to take() one at a time.
// A padded entry pools kpool copies of one cell's key | gate into that cell's pooled slot, so the cell must not be a rep:
// pool_cells reads the reps' slots, the ones the real entries write this ubatch and the cached ones alike. The entries
// must not share a cell either, the scatter would write it from several threads. Beyond that the pick has no effect on
// the result: pool_cells reads no other slot but the first ubatch cell's, for the padded pools, which lie outside every
// token's visible pools (pool_rng) or under its -inf mask (pool_mask), where a finite key scores -inf all the same; and
// a cell that comes to end a pool later is re-pooled by the ubatch that forms that pool, before anything reads its slot
// (see kpool_build_state).
// So no global order is needed. Pools are disjoint runs of kpool cells, so a sequence's cells taken in order, skipping
// each pool's last one, are no rep. Sequences never share a cell here: a layout that shares cells re-pools every pool
// and pads only while it has none, when every cell qualifies. Fewer such cells than pads takes a nearly empty cache
// (many sequences at their first tokens), and then the cells no sequence holds follow. O(n_pad) whatever the context.
template <typename F>
void kpool_pick_pads(const llama_memory_hybrid_idx::kpool_layout & lay, const llama_kv_cache & idx, uint32_t kpool,
        uint32_t n_pad, F && take) {
    if (n_pad == 0) {
        return;
    }

    const uint32_t kv_size  = idx.get_size();
    const uint32_t n_stream = idx.get_n_stream();

    if (lay.n_pool_real == 0) {
        GGML_ASSERT((uint64_t) n_pad <= (uint64_t) kv_size*n_stream);
        for (uint32_t c = 0; c < n_pad; ++c) {
            take((int64_t) c);
        }
        return;
    }

    GGML_ASSERT(lay.cache_safe && "a layout sharing cells pads only while it has no pool");

    uint32_t n_left = n_pad;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ && n_left > 0; ++s) {
        const auto & sq = lay.seqs[s];

        size_t pi = 0; // the next pool to end at or after rank r
        for (size_t r = 0; r < sq.cells.size() && n_left > 0; ++r) {
            if (pi < sq.pools.size() && sq.pools[pi] + kpool - 1 == r) {
                ++pi;
                continue;
            }
            take((int64_t) sq.strm*kv_size + sq.cells[r].second);
            --n_left;
        }
    }

    // a non unified cache keeps one stream per sequence, see kpool_layout_build
    for (uint32_t s = 0; s < n_stream && n_left > 0; ++s) {
        const auto & cells = idx.get_cells((llama_seq_id) s);
        const int64_t base = (int64_t) lay.seqs[s].strm*kv_size;
        for (uint32_t c = 0; c < cells.size() && n_left > 0; ++c) {
            if (cells.is_empty(c)) {
                take(base + c);
                --n_left;
            }
        }
    }

    GGML_ASSERT(n_left == 0 && "more padded entries than cells that are no rep");
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

bool llama_memory_hybrid_idx::kpool_layout_shared() const {
    return kpool_lay && !kpool_lay->cache_safe;
}

// A new pool count rebuilds the graph, and with tensor parallelism re-records its CUDA graphs (about 0.3 s for the
// decode graph of a big MoE). A count padded to 64 pools moved every 256 tokens of decode, so the count is padded like
// the n_kv of a ubatch that does not view the whole cache instead: the pools the used cells of each stream can form,
// with the cells padded to the n_kv grain (2048 cells, doubling beyond 64K), so a decode step moves it only where its
// own n_kv moves. The padded pools lie outside every token's visible range, see set_input_kpool.
// Without shared cells a stream's pools are at most its cells / kpool. A layout sharing cells can have more; it
// re-pools every pool each ubatch, so its graph follows the pool count anyway (n_new_g), and it takes the count itself.
uint32_t llama_memory_hybrid_idx::kpool_n_pool(uint32_t n_pool_real, bool full) const {
    GGML_ASSERT(has_idx());

    const llama_kv_cache * idx = get_idx_cells();

    uint64_t n_cells = 0;
    for (uint32_t s = 0; s < idx->get_n_stream(); ++s) {
        // a non unified cache keeps one stream per sequence, see kpool_layout_build
        const auto & cells = idx->get_cells((llama_seq_id) s);
        if (cells.used_max_p1() > 0) {
            // full: the whole cache, as the attention n_kv under a range mask (see llama_kv_cache_context::apply);
            // the padded pools stay out of every token's visible range, see set_input_kpool
            n_cells += full ? cells.size() : idx->get_n_kv_used(cells);
        }
    }

    return kpool_pad(n_pool_real, n_cells, get_kpool());
}

uint32_t llama_memory_hybrid_idx::kpool_n_pool_layout() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return kpool_n_pool(kpool_lay->n_pool_real);
}

std::string llama_memory_hybrid_idx::kpool_n_pool_check() const {
    GGML_ASSERT(has_idx());

    const llama_kv_cache * idx = get_idx_cells();

    if (!kpool_lay) {
        return "no layout yet";
    }

    const uint32_t n_pool_real = kpool_lay->n_pool_real;
    const uint32_t n_pool      = kpool_n_pool(n_pool_real);

    // set_input_kpool lays the real pools out first and needs one padded pool after them
    if (n_pool <= n_pool_real) {
        return format("%u pools in the graph for %u real ones", n_pool, n_pool_real);
    }

    // the reserved graph holds every count a decode can take, see the full-cache context
    const uint64_t n_pool_max = uint64_t(idx->get_size()/get_kpool())*idx->get_n_seq_max();
    const uint32_t n_pool_rsv = kpool_pad((uint32_t) std::max<uint64_t>(n_pool_max, n_pool_real),
            uint64_t(idx->get_size())*idx->get_n_stream(), get_kpool());
    if (n_pool > n_pool_rsv) {
        return format("%u pools in the graph, %u reserved", n_pool, n_pool_rsv);
    }

    return "";
}

void llama_memory_hybrid_idx::kpool_layout_build(kpool_layout & lay, const stale_pos_t & stale,
        std::vector<std::pair<int32_t, int64_t>> * dropped) const {
    const llama_kv_cache * idx = get_idx_cells();

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;
    lay.cache_safe  = true;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        sq.strm = unified ? 0 : idx->get_stream(s);

        kpool_seq_update(sq, idx->get_cells(unified ? 0 : s), s, stale[s], kpool, get_kpool_by_order(), unified,
                kstore == nullptr, (int64_t) idx->get_size(), dropped);

        lay.n_pool_real += (uint32_t) sq.pools.size();
        lay.cache_safe   = lay.cache_safe && !sq.shared;
    }
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them from the position it staled on, which mem_idx_stale records.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    // the compact layout is brought up to date by every edit and every ubatch, see kstore_sync
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    kpool_layout_build(*kpool_lay, mem_idx_stale, nullptr);

    return *kpool_lay;
}

std::string llama_memory_hybrid_idx::kpool_layout_diff() const {
    GGML_ASSERT(has_idx());

    if (!kpool_lay) {
        return "no layout yet";
    }

    // a fresh layout keeps no cell whatever the stale positions say, so this is the full rebuild
    auto ref = std::make_unique<kpool_layout>();
    stale_pos_t stale;
    stale.fill(0);
    kpool_layout_build(*ref, stale, nullptr);

    const auto & lay = *kpool_lay;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & a = lay.seqs[s];
        const auto & b = ref->seqs[s];

        if (a.cells != b.cells) {
            size_t i = 0;
            while (i < a.cells.size() && i < b.cells.size() && a.cells[i] == b.cells[i]) {
                ++i;
            }
            return format("seq %d: %zu cells, rebuilt %zu, first difference at %zu", s, a.cells.size(), b.cells.size(), i);
        }
        if (a.pools != b.pools) {
            size_t i = 0;
            while (i < a.pools.size() && i < b.pools.size() && a.pools[i] == b.pools[i]) {
                ++i;
            }
            return format("seq %d: %zu pools, rebuilt %zu, first difference at %zu", s, a.pools.size(), b.pools.size(), i);
        }
        if (a.pos_min != b.pos_min || a.strm != b.strm || a.j_next != b.j_next || a.shared != b.shared) {
            return format("seq %d: pos_min %d strm %u j_next %zu shared %d, rebuilt %d %u %zu %d", s,
                    a.pos_min, a.strm, a.j_next, (int) a.shared, b.pos_min, b.strm, b.j_next, (int) b.shared);
        }
    }

    if (lay.n_pool_real != ref->n_pool_real || lay.cache_safe != ref->cache_safe) {
        return format("n_pool_real %u cache_safe %d, rebuilt %u %d",
                lay.n_pool_real, (int) lay.cache_safe, ref->n_pool_real, (int) ref->cache_safe);
    }

    return "";
}

std::string llama_memory_hybrid_idx::kpool_pad_check() const {
    GGML_ASSERT(has_idx());

    if (!kpool_lay) {
        return "no layout yet";
    }

    // the compact layout pads into rows of its own, see set_input_kpool
    if (kstore) {
        return "";
    }

    const auto & lay = *kpool_lay;

    // set_input_kpool pads a layout that shares cells only while it has no pool, see kpool_build_state
    if (!lay.cache_safe && lay.n_pool_real > 0) {
        return "";
    }

    const uint32_t kpool   = get_kpool();
    const uint32_t kv_size = mem_idx->get_size();
    const int64_t  n_cells = (int64_t) kv_size*mem_idx->get_n_stream();

    std::vector<uint8_t> is_rep((size_t) n_cells, 0);
    int64_t n_free = n_cells; // the cells that are no rep
    int64_t n_lay  = 0;       // the layout's cells that are no rep, taken before any cell no sequence holds
    for (const auto & sq : lay.seqs) {
        for (const uint32_t j : sq.pools) {
            const int64_t c = (int64_t) sq.strm*kv_size + sq.cells[j + kpool - 1].second;
            n_free -= is_rep[c] ? 0 : 1;
            is_rep[c] = 1;
        }
        n_lay += (int64_t) (sq.cells.size() - sq.pools.size());
    }

    // a few counts inside the layout, the ones around running out of it, and every cell that is no rep
    const int64_t counts[] = { 1, 2, 3, kpool + 1, n_lay - 1, n_lay, n_lay + 1, n_free };
    for (const int64_t n_pad : counts) {
        if (n_pad < 1 || n_pad > n_free) {
            continue;
        }

        std::vector<uint8_t> taken((size_t) n_cells, 0);
        int64_t     n_taken = 0;
        std::string err;
        kpool_pick_pads(lay, *mem_idx, kpool, (uint32_t) n_pad, [&](int64_t c) {
            if (err.empty()) {
                if (c < 0 || c >= n_cells) {
                    err = format("%" PRId64 " pads: pad %" PRId64 " takes cell %" PRId64 " of %" PRId64, n_pad, n_taken, c, n_cells);
                } else if (is_rep[c]) {
                    err = format("%" PRId64 " pads: pad %" PRId64 " takes cell %" PRId64 ", a pool's rep", n_pad, n_taken, c);
                } else if (taken[c]) {
                    err = format("%" PRId64 " pads: pad %" PRId64 " takes cell %" PRId64 " again", n_pad, n_taken, c);
                } else {
                    taken[c] = 1;
                }
            }
            ++n_taken;
        });
        if (!err.empty()) {
            return err;
        }
        if (n_taken != n_pad) {
            return format("%" PRId64 " pads: %" PRId64 " taken", n_pad, n_taken);
        }
    }

    return "";
}

//
// llama_memory_hybrid_idx: the compact layout, see kpool_store
//

bool llama_memory_hybrid_idx::kstore_sync(const stale_pos_t & stale, std::vector<std::pair<llama_seq_id, uint32_t>> & fresh) {
    GGML_ASSERT(kstore != nullptr && kpool_lay != nullptr);

    auto & ks  = *kstore;
    auto & lay = *kpool_lay;

    std::vector<std::pair<int32_t, int64_t>> dropped;
    kpool_layout_build(lay, stale, &dropped);

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();

    bool ok = true;

    std::vector<int64_t> m(kpool);
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ && ok; ++s) {
        auto & sq = lay.seqs[s];

        // the update appends the pools it forms, so the ones without a slot are the last ones
        size_t pi = sq.slots.size();
        while (pi > 0 && sq.slots[pi - 1] < 0) {
            --pi;
        }

        for (; pi < sq.pools.size(); ++pi) {
            const uint32_t j = sq.pools[pi];
            for (uint32_t k = 0; k < kpool; ++k) {
                m[k] = (int64_t) sq.strm*kv_size + sq.cells[j + k].second;
            }

            int32_t r = ks.slot_find(m.data());
            if (r < 0) {
                r = ks.slot_alloc(m.data());
                if (r < 0) {
                    LLAMA_LOG_ERROR("%s: the compact indexer cache has no pooled slot left (%u slots)\n", __func__, ks.n_slot);
                    ok = false;
                    break;
                }
                fresh.emplace_back(s, (uint32_t) pi);
            }
            sq.slots[pi] = r;
        }
    }

    // after the matching above, so that a pool rebuilt as it was keeps its slot and its key
    for (const auto & [r, rep] : dropped) {
        ks.slot_unref(r, rep);
    }

    return ok;
}

void llama_memory_hybrid_idx::kstore_sync_edit(const stale_pos_t & stale, const char * what) {
    std::vector<std::pair<llama_seq_id, uint32_t>> fresh;

    const bool ok = kstore_sync(stale, fresh);
    if (!fresh.empty()) {
        const auto & sq = kpool_lay->seqs[fresh[0].first];
        LLAMA_LOG_ERROR("%s: %s left %zu pools without a pooled key, the first one of seq %d at position %d\n", __func__,
                what, fresh.size(), fresh[0].first, sq.cells[sq.pools[fresh[0].second]].first);
    }

    // an edit removes cells or copies pools, it never completes one: every pool that stands has its key already
    GGML_ASSERT(ok && fresh.empty() && "compact QSA indexer cache: an edit left a pool without its pooled key");
}

bool llama_memory_hybrid_idx::kstore_seq_rm_ok(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    const auto & ks  = *kstore;
    const auto & lay = *kpool_lay;

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();

    const llama_pos p_end = std::numeric_limits<llama_pos>::max();

    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? p_end : p1;

    auto ok_seq = [&](llama_seq_id s) {
        const auto & sq = lay.seqs[s];

        // the cells at positions [p0, p1) go: the cells are sorted by position, so they are the ranks [i0, i1)
        const size_t i0 = (size_t) (std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(p0, 0u)) - sq.cells.begin());
        const size_t i1 = p1 == p_end ? sq.cells.size() :
            (size_t) (std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(p1, 0u)) - sq.cells.begin());

        if (i0 >= i1) {
            return true; // nothing goes
        }

        if (i1 < sq.cells.size()) {
            // the cells after the removed ones would rank lower and regroup, from raw keys the cache no longer keeps
            LLAMA_LOG_WARN("%s: seq %d: the compact QSA indexer cache cannot remove positions [%d, %d) from inside the sequence\n",
                    __func__, s, p0, p1);
            return false;
        }

        // a tail cut: the pools before it stay, the incomplete one it leaves is pooled later from the raw keys
        for (size_t r = i0/kpool*kpool; r < i0; ++r) {
            if (ks.ring_of[(int64_t) sq.strm*kv_size + sq.cells[r].second] < 0) {
                LLAMA_LOG_WARN("%s: seq %d: the compact QSA indexer cache has no raw key of position %d left, cannot cut at %d\n",
                        __func__, s, sq.cells[r].first, p0);
                return false;
            }
        }

        return true;
    };

    if (seq_id < 0) {
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (!ok_seq(s)) {
                return false;
            }
        }

        return true;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    return ok_seq(seq_id);
}

bool llama_memory_hybrid_idx::kstore_apply(const llama_kv_cache::slot_info & sinfo, std::vector<std::pair<llama_seq_id, uint32_t>> & fresh) {
    auto & ks = *kstore;

    const auto & lay = *kpool_lay;

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();

    ks.stamp_begin();

    // the ubatch writes the raw keys of its cells
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            ks.stamp[(int64_t) sinfo.strm[s]*kv_size + sinfo.idxs[s][i]] = ks.stamp_cur;
        }
    }

    // the pools it completes are pooled from their members' raw keys: this ubatch's, or the ring's
    for (const auto & [s, pi] : fresh) {
        const auto &   sq = lay.seqs[s];
        const uint32_t j  = sq.pools[pi];
        for (uint32_t k = 0; k < kpool; ++k) {
            const int64_t g = (int64_t) sq.strm*kv_size + sq.cells[j + k].second;
            if (ks.stamp[g] != ks.stamp_cur && ks.ring_of[g] < 0) {
                LLAMA_LOG_ERROR("%s: seq %d: the pool at position %d needs the raw key of position %d, which the compact "
                        "QSA indexer cache no longer keeps\n", __func__, s, sq.cells[j].first, sq.cells[j + k].first);
                return false;
            }
            ks.stamp[g] = ks.stamp_cur;
        }
    }

    // every sequence keeps the raw keys of its incomplete pool and of the cells a rollback can turn into one
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];

        const size_t n  = sq.cells.size();
        const size_t r0 = (n > ks.n_rs_seq ? n - ks.n_rs_seq : 0)/kpool*kpool;
        for (size_t r = r0; r < n; ++r) {
            ks.stamp[(int64_t) sq.strm*kv_size + sq.cells[r].second] = ks.stamp_cur;
        }
    }

    ks.ring_release_unstamped();

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            if (!ks.ring_take((int64_t) sinfo.strm[s]*kv_size + sinfo.idxs[s][i])) {
                LLAMA_LOG_ERROR("%s: the compact QSA indexer cache's raw-key ring is full (%u rows)\n", __func__, ks.n_ring);
                return false;
            }
        }
    }

    return true;
}

void llama_memory_hybrid_idx::kstore_clear(bool data) {
    kstore->reset();
    kpool_lay = std::make_unique<kpool_layout>();

    if (data) {
        for (auto & [ctx, buf] : kstore->ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_memory_hybrid_idx::kstore_has_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    p0 = std::max<llama_pos>(p0, 0);
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    auto has = [&](llama_seq_id s) {
        const auto & cells = kpool_lay->seqs[s].cells;
        auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(p0, 0u));
        return it != cells.end() && it->first < p1;
    };

    if (seq_id >= 0) {
        GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);
        return has(seq_id);
    }

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (has(s)) {
            return true;
        }
    }

    return false;
}

void llama_memory_hybrid_idx::kstore_drop_seq(llama_seq_id s) {
    GGML_ASSERT(s >= 0 && s < (llama_seq_id) LLAMA_MAX_SEQ);

    auto & lay = *kpool_lay;
    auto & sq  = lay.seqs[s];

    const int64_t kv_size = get_idx_cells()->get_size();
    for (size_t pi = 0; pi < sq.slots.size(); ++pi) {
        if (sq.slots[pi] >= 0) {
            kstore->slot_unref(sq.slots[pi], (int64_t) sq.strm*kv_size + sq.cells[sq.pools[pi] + get_kpool() - 1].second);
        }
    }

    // the raw-key rows of its cells go with the next release pass, or when a restore or a copy reuses the cells
    lay.n_pool_real -= (uint32_t) sq.pools.size();
    sq = kpool_layout::seq();
}

void llama_memory_hybrid_idx::kstore_forget(llama_seq_id seq_id) {
    if (seq_id < 0) {
        kstore->reset();
        kpool_lay = std::make_unique<kpool_layout>();

        return;
    }

    kstore_drop_seq(seq_id);
}

void llama_memory_hybrid_idx::kstore_copy_stream(llama_seq_id src, llama_seq_id dst) {
    auto & ks  = *kstore;
    auto & lay = *kpool_lay;

    const llama_kv_cache * idx = get_idx_cells();
    const int64_t kv_size = idx->get_size();

    const auto & a = lay.seqs[src];
    auto       & b = lay.seqs[dst];
    GGML_ASSERT(b.cells.empty() && b.pools.empty());

    // the destination stream holds the source's cells at the same indices
    b = a;
    b.strm = idx->get_stream(dst);
    for (const int32_t r : b.slots) {
        ks.slot_share(r);
    }
    lay.n_pool_real += (uint32_t) b.pools.size();

    // the raw keys are the source's; a row the destination cell had before belongs to a token it no longer holds
    std::vector<int64_t> stale_rows;
    for (const auto & c : b.cells) {
        const int64_t g_src = (int64_t) a.strm*kv_size + c.second;
        const int64_t g_dst = (int64_t) b.strm*kv_size + c.second;
        if (ks.ring_of[g_src] >= 0) {
            ks.ring_share(g_dst, g_src);
        } else if (ks.ring_of[g_dst] >= 0) {
            stale_rows.push_back(g_dst);
        }
    }
    ks.ring_forget([&](auto && f) {
        for (const int64_t g : stale_rows) {
            f(g);
        }
    });
}

static constexpr uint32_t LLAMA_KSTORE_STATE_MAGIC = 0x31415351; // "QSA1"

// state: per sequence, the pooled keys of its pools in pool order (full state only) and the raw keys of the cells of its
// incomplete pool, by rank. A full restore rebuilds the pools from the restored cells, which come back in the same order;
// a checkpoint (partial state) is restored over the longer sequence it was taken from and cut back to it after that, so
// the ranks of its incomplete pool still name the same tokens.
void llama_memory_hybrid_idx::kstore_state_write(llama_io_write_i & io, llama_seq_id seq_id, bool partial) const {
    const auto & ks  = *kstore;
    const auto & lay = *kpool_lay;

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();
    const size_t   row     = ks.row_size();

    const uint32_t magic   = LLAMA_KSTORE_STATE_MAGIC;
    const uint32_t n_layer = (uint32_t) ks.layers.size();
    const int32_t  type    = (int32_t) ks.type;
    const uint32_t n_embd  = (uint32_t) ks.n_embd;
    const uint32_t mode    = partial ? 1 : 0;

    io.write(&magic,   sizeof(magic));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&type,    sizeof(type));
    io.write(&n_embd,  sizeof(n_embd));
    io.write(&kpool,   sizeof(kpool));
    io.write(&mode,    sizeof(mode));

    std::vector<llama_seq_id> seqs;
    if (seq_id >= 0) {
        GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);
        seqs.push_back(seq_id);
    } else {
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (!lay.seqs[s].cells.empty()) {
                seqs.push_back(s);
            }
        }
    }

    const uint32_t n_seq = (uint32_t) seqs.size();
    io.write(&n_seq, sizeof(n_seq));

    for (const llama_seq_id s : seqs) {
        const auto & sq = lay.seqs[s];

        io.write(&s, sizeof(s));

        const uint32_t n_pool = partial ? 0 : (uint32_t) sq.pools.size();
        io.write(&n_pool, sizeof(n_pool));

        // runs of consecutive slots, the whole table in one go for a single sequence
        for (const auto & layer : ks.layers) {
            for (uint32_t pi = 0; pi < n_pool; ) {
                const int32_t r0 = sq.slots[pi];
                GGML_ASSERT(r0 >= 0);
                uint32_t n = 1;
                while (pi + n < n_pool && sq.slots[pi + n] == r0 + (int32_t) n) {
                    ++n;
                }
                io.write_tensor(layer.pooled, (size_t) r0*row, (size_t) n*row);
                pi += n;
            }
        }

        const uint32_t n_cell = (uint32_t) sq.cells.size();
        const uint32_t t0     = n_cell/kpool*kpool;
        const uint32_t n_tail = n_cell - t0;
        io.write(&n_tail, sizeof(n_tail));

        std::vector<int32_t> rows(n_tail, -1);
        for (uint32_t t = 0; t < n_tail; ++t) {
            const int32_t   rank = (int32_t) (t0 + t);
            const llama_pos pos  = sq.cells[t0 + t].first;
            rows[t] = ks.ring_of[(int64_t) sq.strm*kv_size + sq.cells[t0 + t].second];
            const uint8_t have = rows[t] >= 0;
            io.write(&rank, sizeof(rank));
            io.write(&pos,  sizeof(pos));
            io.write(&have, sizeof(have));
        }

        for (const auto & layer : ks.layers) {
            for (uint32_t t = 0; t < n_tail; ++t) {
                if (rows[t] >= 0) {
                    io.write_tensor(layer.ring, (size_t) rows[t]*row, row);
                }
            }
        }
    }
}

void llama_memory_hybrid_idx::kstore_state_read(llama_io_read_i & io, llama_seq_id seq_id, bool partial) {
    auto & ks = *kstore;

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();
    const size_t   row     = ks.row_size();

    uint32_t magic   = 0;
    uint32_t n_layer = 0;
    int32_t  type    = 0;
    uint32_t n_embd  = 0;
    uint32_t kpool_r = 0;
    uint32_t mode    = 0;

    io.read(&magic,   sizeof(magic));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&type,    sizeof(type));
    io.read(&n_embd,  sizeof(n_embd));
    io.read(&kpool_r, sizeof(kpool_r));
    io.read(&mode,    sizeof(mode));

    if (magic != LLAMA_KSTORE_STATE_MAGIC) {
        throw std::runtime_error("compact indexer cache state: magic mismatch");
    }
    if (n_layer != ks.layers.size() || type != (int32_t) ks.type || n_embd != (uint32_t) ks.n_embd || kpool_r != kpool) {
        throw std::runtime_error(format("compact indexer cache state: %u layers of %u type %d pooled by %u, the cache has %zu of %u type %d by %u",
                n_layer, n_embd, type, kpool_r, ks.layers.size(), (uint32_t) ks.n_embd, (int) ks.type, kpool));
    }
    if (mode != (partial ? 1u : 0u)) {
        throw std::runtime_error("compact indexer cache state: partial flag mismatch");
    }

    if (!partial) {
        // the restored cells make new pools; kstore_forget released the old ones, so every pool takes a new slot (or the
        // slot of an identical pool of another restored sequence), filled from the state below
        stale_pos_t stale = stale_pos_clean();
        if (seq_id < 0) {
            stale.fill(0);
        } else {
            stale[seq_id] = 0;
        }

        std::vector<std::pair<llama_seq_id, uint32_t>> fresh;
        if (!kstore_sync(stale, fresh)) {
            throw std::runtime_error("compact indexer cache state: no pooled slot left for the restored pools");
        }

        // a raw-key row a restored cell had before belongs to a token it no longer holds
        ks.ring_forget([&](auto && f) {
            for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (seq_id >= 0 && s != seq_id) {
                    continue;
                }
                const auto & sq = kpool_lay->seqs[s];
                for (const auto & c : sq.cells) {
                    f((int64_t) sq.strm*kv_size + c.second);
                }
            }
        });
    }

    uint32_t n_seq = 0;
    io.read(&n_seq, sizeof(n_seq));

    std::vector<uint8_t> scratch(row);

    for (uint32_t i = 0; i < n_seq; ++i) {
        llama_seq_id s_src = -1;
        io.read(&s_src, sizeof(s_src));

        const llama_seq_id s = seq_id >= 0 ? seq_id : s_src;
        if (s < 0 || s >= (llama_seq_id) LLAMA_MAX_SEQ) {
            throw std::runtime_error(format("compact indexer cache state: invalid sequence %d", s));
        }

        const auto & sq = kpool_lay->seqs[s];

        uint32_t n_pool = 0;
        io.read(&n_pool, sizeof(n_pool));

        if (n_pool != (partial ? 0 : sq.pools.size())) {
            throw std::runtime_error(format("compact indexer cache state: seq %d has %u pools, the restored cells make %zu",
                    s, n_pool, partial ? (size_t) 0 : sq.pools.size()));
        }

        for (const auto & layer : ks.layers) {
            for (uint32_t pi = 0; pi < n_pool; ) {
                const int32_t r0 = sq.slots[pi];
                GGML_ASSERT(r0 >= 0);
                uint32_t n = 1;
                while (pi + n < n_pool && sq.slots[pi + n] == r0 + (int32_t) n) {
                    ++n;
                }
                io.read_tensor(layer.pooled, (size_t) r0*row, (size_t) n*row);
                pi += n;
            }
        }

        uint32_t n_tail = 0;
        io.read(&n_tail, sizeof(n_tail));
        if (n_tail >= kpool) {
            throw std::runtime_error(format("compact indexer cache state: seq %d has an incomplete pool of %u cells", s, n_tail));
        }

        // the ring row of each saved raw key, -1 when the sequence has no such cell now (the key is skipped)
        std::vector<int32_t> rows(n_tail, -1);
        std::vector<uint8_t> have(n_tail, 0);
        for (uint32_t t = 0; t < n_tail; ++t) {
            int32_t   rank = -1;
            llama_pos pos  = -1;
            io.read(&rank,    sizeof(rank));
            io.read(&pos,     sizeof(pos));
            io.read(&have[t], sizeof(have[t]));

            if (!have[t] || rank < 0 || (size_t) rank >= sq.cells.size() || sq.cells[rank].first != pos) {
                continue;
            }

            const int64_t g = (int64_t) sq.strm*kv_size + sq.cells[rank].second;
            if (!ks.ring_take(g)) {
                throw std::runtime_error("compact indexer cache state: the raw-key ring is full");
            }
            rows[t] = ks.ring_of[g];
        }

        for (const auto & layer : ks.layers) {
            for (uint32_t t = 0; t < n_tail; ++t) {
                if (!have[t]) {
                    continue;
                }
                if (rows[t] >= 0) {
                    io.read_tensor(layer.ring, (size_t) rows[t]*row, row);
                } else {
                    io.read(scratch.data(), row);
                }
            }
        }
    }
}

std::string llama_memory_hybrid_idx::kpool_store_check() const {
    if (!kstore) {
        return "";
    }

    if (!kpool_lay) {
        return "no layout yet";
    }

    const auto & ks  = *kstore;
    const auto & lay = *kpool_lay;

    const uint32_t kpool   = get_kpool();
    const int64_t  kv_size = get_idx_cells()->get_size();

    // every pool has a slot; an entry made of the slot's member cells owns it, a copy in another stream reads it from
    // the same tokens in other cells, at the same ranks
    std::vector<uint32_t> refs(ks.n_slot, 0);
    std::vector<uint32_t> owns(ks.n_slot, 0);
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        if (sq.slots.size() != sq.pools.size()) {
            return format("seq %d: %zu slots for %zu pools", s, sq.slots.size(), sq.pools.size());
        }
        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const int32_t r = sq.slots[pi];
            if (r < 0 || (uint32_t) r >= ks.n_slot) {
                return format("seq %d pool %zu: slot %d", s, pi, r);
            }
            ++refs[r];
            bool own = true;
            for (uint32_t k = 0; k < kpool; ++k) {
                const int64_t g = (int64_t) sq.strm*kv_size + sq.cells[sq.pools[pi] + k].second;
                const int64_t m = ks.slot_members[(size_t) r*kpool + k];
                own = own && m == g;
                if (m < 0 || m % kv_size != g % kv_size) {
                    return format("seq %d pool %zu: slot %d holds the pool of other cells", s, pi, r);
                }
            }
            owns[r] += own;
        }
    }

    uint32_t n_live = 0;
    for (uint32_t r = 0; r < ks.n_slot; ++r) {
        if (refs[r] != ks.slot_refs[r] || owns[r] != ks.slot_owns[r]) {
            return format("slot %u: %u references (%u owners), the layout has %u (%u)", r, ks.slot_refs[r], ks.slot_owns[r],
                    refs[r], owns[r]);
        }
        n_live += refs[r] > 0;
    }
    if (ks.slot_free.size() != ks.n_slot - n_live) {
        return format("%zu free slots, %u not in use", ks.slot_free.size(), ks.n_slot - n_live);
    }
    for (int64_t g = 0; g < ks.n_cells; ++g) {
        const int32_t r = ks.rep_slot[g];
        if (r >= 0 && (owns[r] == 0 || ks.slot_members[(size_t) r*kpool + kpool - 1] != g)) {
            return format("cell %" PRId64 ": registered to slot %d, which no pool of these cells owns", g, r);
        }
    }

    std::vector<uint32_t> rrefs(ks.n_ring, 0);
    size_t n_with = 0;
    for (int64_t g = 0; g < ks.n_cells; ++g) {
        if (ks.ring_of[g] >= 0) {
            ++rrefs[ks.ring_of[g]];
            ++n_with;
        }
    }
    if (n_with != ks.ring_cells.size()) {
        return format("%zu cells with a raw-key row, %zu listed", n_with, ks.ring_cells.size());
    }
    uint32_t n_rows = 0;
    for (uint32_t row = 0; row < ks.n_ring; ++row) {
        if (rrefs[row] != ks.ring_refs[row]) {
            return format("raw-key row %u: %u references, %u cells read it", row, ks.ring_refs[row], rrefs[row]);
        }
        n_rows += rrefs[row] > 0;
    }
    if (ks.ring_free.size() != ks.n_ring - n_rows) {
        return format("%zu free raw-key rows, %u not in use", ks.ring_free.size(), ks.n_ring - n_rows);
    }

    // the incomplete pool of every sequence is pooled later, from the raw keys of its cells
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        for (size_t r = sq.cells.size()/kpool*kpool; r < sq.cells.size(); ++r) {
            if (ks.ring_of[(int64_t) sq.strm*kv_size + sq.cells[r].second] < 0) {
                return format("seq %d: no raw key for position %d of its incomplete pool", s, sq.cells[r].first);
            }
        }
    }

    return "";
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(!mem->has_idx() ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_idx_cells()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        // the compact layout is in step with the cells at all times, see kstore_sync
        if (mem->get_kstore() == nullptr) {
            mem->kpool_layout_update();
        }
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_idx_cells();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        // the most pools a graph can take: every pool the cache can hold, over every cell of every stream
        st.n_pool_pad = kpool_pad(st.n_pool_real, uint64_t(idx->get_size())*idx->get_n_stream(), mem->get_kpool());
        if (mem->get_kstore() != nullptr) {
            // the compact layout never pools a pool again: a ubatch pools at most one per kpool tokens plus one per
            // sequence, which is how many padded rows the cache keeps (see kpool_build_state)
            st.n_new   = 0;
            st.n_new_g = std::max(1u, std::min<uint32_t>(mem->get_kstore()->n_trash, (uint32_t) n_pool_max));
        } else {
            st.n_new   = st.n_pool_real;
            st.n_new_g = std::max(st.n_new, 1u);
        }
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->has_idx() && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        if (mem->get_kstore() != nullptr) {
            // compact layout: the pools the ubatch completes take new slots, its cells rows for their raw keys
            kpool_st->fresh.clear();
            res = mem->kstore_sync(llama_memory_hybrid_idx::stale_pos_clean(), kpool_st->fresh) &&
                  mem->kstore_apply(sinfos_kpool[i_cur], kpool_st->fresh);
        } else {
            mem->kpool_layout_update();
        }
        if (res) {
            kpool_build_state(get_ubatch());
            i_kpool  = i_cur;
        }
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->has_idx() && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

bool llama_memory_hybrid_idx_context::has_idx() const {
    return mem != nullptr && mem->has_idx();
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate    = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled      = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
    pooled_rows = ggml_view_3d(ctx, k, n_embd, 1, n_cells, ggml_row_size(k->type, n_embd), k->nb[1],
                               ggml_row_size(k->type, n_tok));
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * ring, ggml_tensor * table,
        int64_t n_embd) : ctx(ctx), compact(true) {
    GGML_ASSERT(ring->ne[0] == n_embd && table->ne[0] == n_embd);

    // the pools take the rows of their slots, see kpool_store
    key_gate    = ring;
    pooled      = table;
    pooled_rows = ggml_reshape_3d(ctx, table, n_embd, 1, table->ne[1]);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::store_raw(ggml_tensor * values, ggml_tensor * idxs) const {
    GGML_ASSERT(compact && "the raw keys have a ring of their own only in the compact layout");

    return ggml_set_rows(ctx, key_gate, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->has_idx());

    if (const auto * ks = mem->get_kstore()) {
        const auto & layer = ks->get_layer(il);
        return kpool_access(ctx, layer.ring, layer.pooled, n_embd);
    }

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

ggml_tensor * llama_memory_hybrid_idx_context::gather_mla_rows(
        ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    ggml_tensor * k = mem->get_mem_attn()->get_k_storage(il);
    GGML_ASSERT(k->ne[0] == n_embd);

    ggml_tensor * rows = ggml_view_2d(ctx, k, k->ne[0], k->ne[1]*k->ne[2], k->nb[1], 0);
    return ggml_get_rows(ctx, rows, ggml_reshape_1d(ctx, idxs, n_rows));
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are read through pool_cells, fresh ones just written, older ones from whatever batch last wrote them
//    (qwen4exp: ggml_qsa_select reads the pooled slots in place; glm5-next: one get_rows).
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    st.n_pool_real = lay.n_pool_real;
    // Only prompt batches take the whole-cache pool count: a small verify batch (an MTP draft step) keeps the
    // used grain, so its set_input does not fill the whole cache's pool list every step. 32 is the QSA prompt
    // threshold (see QSA_SEL_POOL_MIN_TOKENS): below it the selection is the per-token cell path anyway.
    const bool full = ubatch.n_tokens >= 32 &&
        (get_attn()->get_kq_range() || mem->get_mem_attn()->get_view_full());
    st.n_pool_pad  = mem->kpool_n_pool(lay.n_pool_real, full);
    st.cache_safe  = lay.cache_safe;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(), st.is_new.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);

    auto mark = [&](uint32_t ip) {
        if (st.is_new[ip] != st.generation) {
            st.is_new[ip] = st.generation;
            ++st.n_new;
        }
    };

    const uint32_t kpool = mem->get_kpool();
    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;
    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        if (!st.cache_safe) {
            continue;
        }

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const auto *   idx        = mem->get_idx_cells();
    const uint32_t n_pool_max = idx->get_size() / kpool * idx->get_n_seq_max();
    const uint32_t bound      = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;

    if (mem->get_kstore() != nullptr) {
        // compact layout: the pools that took a new slot, the ones this ubatch completes; an edit never stales a pool
        // there (a cut drops the pools it reaches) and a shared cell does not either (each pool has its slot)
        for (const auto & [s, pi] : st.fresh) {
            mark(pool_start[s] + pi);
        }
        st.n_new_g = std::max({st.n_new, 1u, std::min({bound, st.n_pool_pad - 1, n_pool_max})});
        return;
    }

    if (!st.cache_safe) {
        std::fill(st.is_new.begin(), st.is_new.end(), st.generation);
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        return;
    }

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(pool_start[s] + (uint32_t) (r / kpool));
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
            }
        }
    }

    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, st.n_pool_pad - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_cur().n_pool_pad;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos, ggml_tensor * pool_rng) const {
    GGML_ASSERT(mem != nullptr && mem->has_idx());
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(pool_mask != nullptr || pool_rng != nullptr);
    GGML_ASSERT(pool_mask == nullptr || ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(pool_rng  == nullptr || ggml_backend_buffer_is_host(pool_rng->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    // the indexer cache tracks the attention cache cell for cell, the compact layout reads the attention cells
    const uint32_t n_kv  = get_idx() != nullptr ? get_idx()->get_n_kv() : get_attn()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    // compact layout: pools read their slots, new pools their members' raw-key rows, see kpool_store
    const llama_memory_hybrid_idx::kpool_store * ks = mem->get_kstore();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == st.n_pool_pad && n_pool > st.n_pool_real);
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask == nullptr || (pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens));
    GGML_ASSERT(pool_rng  == nullptr || (pool_rng->type == GGML_TYPE_I32 && pool_rng->ne[0] == 2 && pool_rng->ne[1] == (int64_t) n_tokens));
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_idx_cells()->get_size();
    const uint32_t n_stream_kv = mem->get_idx_cells()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // Use the first ubatch cell for padded gathers.
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // Gather maps padding to a real cell and masks it separately; ggml_qsa_select takes -1 for a missing cell.
    const int32_t sentinel = gather ? (int32_t) dummy_cell : (pool_rng != nullptr ? -1 : (int32_t) n_kv);

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (gather_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(gather_mask->buffer));
        GGML_ASSERT(gather_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(gather_mask->ne[3] == (int64_t) n_tokens && gather_mask->ne[1] == 1 && gather_mask->ne[2] == 1);
        n_sel = (uint32_t) gather_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) gather_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = !gather && n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row (compact: in the pool's slot).
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = ks ? sq.slots[pi] : (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel :
                    (int32_t) (gather ? gcell(sq, sq.cells[j + k].second) : (int64_t) sq.cells[j + k].second);
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    const int64_t g = gcell(sq, sq.cells[j + k].second);
                    // kstore_apply gave every member of a new pool its raw-key row
                    nidx[(size_t) i_new*kpool + k] = ks ? ks->ring_of[g] : (int32_t) g;
                    GGML_ASSERT(nidx[(size_t) i_new*kpool + k] >= 0);
                }
                nrep[i_new] = ks ? (int64_t) sq.slots[pi] : gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_idx_cells()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    if (ks) {
        // compact layout: the padded entries pool raw-key row 0 into rows of their own after the slots, one each, which
        // nothing reads but the padded pools below, outside every token's visible range
        GGML_ASSERT(n_new_g - n_new <= ks->n_trash);
        for (uint32_t i = n_new; i < n_new_g; ++i) {
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = 0;
            }
            nrep[i] = (int64_t) ks->n_slot + (i - n_new);
        }
    } else {
        // Padded entries re-pool cells whose pooled slot is never read, each its own cell, see kpool_pick_pads.
        uint32_t i = n_new;
        kpool_pick_pads(lay, *mem->get_mem_idx(), kpool, n_new_g - n_new, [&](int64_t cell) {
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) cell;
            }
            nrep[i++] = cell;
        });
        GGML_ASSERT(i == n_new_g);
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        // pool_cells always addresses the key table (compact: the first padded row)
        pcell[ip] = ks ? (int32_t) ks->n_slot : (int32_t) dummy_cell;
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it: the pools of a sequence are
    // laid out in order, so the visible ones are the range [p0, p0 + nv)
    auto visible = [&](uint32_t i, uint32_t & p0, uint32_t & nv) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];

        p0 = seq_pool_start[s];
        const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
        nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
            (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
    };

    if (pool_rng != nullptr) {
        int32_t * rng = (int32_t *) pool_rng->data;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            uint32_t p0 = 0;
            uint32_t nv = 0;
            visible(i, p0, nv);
            rng[2*i + 0] = (int32_t) p0;
            rng[2*i + 1] = (int32_t) (p0 + nv);
        }
    }

    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            uint32_t p0 = 0;
            uint32_t nv = 0;
            visible(i, p0, nv);
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask != nullptr && pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else if (pool_mask != nullptr) {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) (gather ? gcell(sq, c) : (int64_t) c);
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) (gather ? gcell(sq, it->second) : (int64_t) it->second);
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}

void llama_memory_hybrid_idx_context::set_input_kring(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(mem != nullptr && mem->get_kstore() != nullptr);
    GGML_ASSERT(kpool_track() && i_kpool == i_cur && "raw-key rows read before apply()");
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    const auto * ks = mem->get_kstore();

    const auto &  sinfo   = sinfos_kpool[i_cur];
    const int64_t kv_size = mem->get_idx_cells()->get_size();

    GGML_ASSERT(dst->ne[0] == (int64_t) ubatch->n_tokens && (size_t) ubatch->n_tokens == sinfo.size()*sinfo.n_stream());

    // token s*n + i of the ubatch sits in cell idxs[s][i] of stream strm[s], as set_input_k_idxs lays the cells out
    int64_t * data = (int64_t *) dst->data;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const int32_t row = ks->ring_of[(int64_t) sinfo.strm[s]*kv_size + sinfo.idxs[s][i]];
            GGML_ASSERT(row >= 0 && "kstore_apply gives every cell of the ubatch a raw-key row");
            data[(size_t) s*sinfo.size() + i] = row;
        }
    }
}
