#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // when tok = LLAMA_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
    // use case: n-gram embeddings hash
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        *this = llama_kv_cell_ext{};
    }
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    // raw access to the cell arrays, for performance-critical loops
    const llama_pos * pos_data() const {
        return pos.data();
    }

    const seq_set_t * seq_data() const {
        return seq.data();
    }

    // number of cells that carry sequence seq_id
    uint32_t seq_n_cells(llama_seq_id seq_id) const {
        assert(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);

        return seq_pos[seq_id].size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : *used.begin();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : *used.rbegin() + 1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, i);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, i);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // the full set of sequences this cell is visible to
    const seq_set_t & seq_get_all(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // the (pos, cell) pairs of sequence seq_id, ordered by position
    const std::set<std::pair<llama_pos, uint32_t>> & seq_pos_get(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_pos[seq_id];
    }

    // the token of the cell of sequence seq_id at the largest position <= p
    // when several cells share that position, the one with the highest index wins
    // return LLAMA_TOKEN_NULL if the sequence has no cell at or before p
    // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & sp = seq_pos[seq_id];

        auto it = sp.upper_bound({ p, std::numeric_limits<uint32_t>::max() });
        if (it == sp.begin()) {
            return LLAMA_TOKEN_NULL;
        }

        return ext[(--it)->second].tok;
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, i);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].begin()->first;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].rbegin()->first;
    }

    // the indices of all cells of sequence seq_id with position in [p0, p1)
    // note: the result is a snapshot, the caller must collect it before modifying the cells
    std::vector<uint32_t> seq_pos_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        std::vector<uint32_t> res;

        const auto & sp = seq_pos[seq_id];

        for (auto it = sp.lower_bound({ p0, 0 }); it != sp.end() && it->first < p1; ++it) {
            res.push_back(it->second);
        }

        return res;
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // the set seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
    //
    // the cell index is part of the key because a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::set<std::pair<llama_pos, uint32_t>> seq_pos[LLAMA_MAX_SEQ];

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, uint32_t i) {
        const auto n = seq_pos[s].erase({ pos[i], i });
        assert(n == 1);
        GGML_UNUSED(n);
    }

    void seq_pos_inc(llama_seq_id s, uint32_t i) {
        seq_pos[s].insert({ pos[i], i });
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, i);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, i);
            }
        }
    }
};

// A single-sequence SWA cache is written as a ring: the cells take the positions in order, from cell 0 on and
// wrapping around the cell array, the write head pointing at the next cell. Removing the newest positions (the
// rejected tail of a speculative round) frees the cells right before the head and moves the head back to the
// first of them, so the empty cells, if any, are one run in ring order starting at the head,
// [head, head + size - used), and the used cells are one run from the oldest cell, right after the empty run,
// to the newest, right before the head. A ring that has not wrapped yet is the case of the empty run
// [used, size) with the oldest cell 0; a full ring has no empty run and its oldest cell is the head.
//
// Returns the oldest cell when the cells are in that layout and the positions strictly increase by one along
// the used run, -1 otherwise (and for an empty cache). The visible set of a token is then a run of cells
// ending at its own cell, see llama_kv_cells_ring_window.
// true if pos[j] == pos[j - 1] + 1 for every j in [j0, j1): a branch-free scan that vectorizes
static inline bool llama_kv_cells_pos_run_ok(const llama_pos * pos, uint32_t j0, uint32_t j1) {
    int32_t broken = 0;
    for (uint32_t j = j0; j < j1; ++j) {
        broken |= pos[j] != pos[j - 1] + 1;
    }
    return !broken;
}

// true if every cell in [j0, j1) is empty: a branch-free scan that vectorizes
static inline bool llama_kv_cells_pos_empty(const llama_pos * pos, uint32_t j0, uint32_t j1) {
    int32_t used = 0;
    for (uint32_t j = j0; j < j1; ++j) {
        used |= pos[j] != -1;
    }
    return !used;
}

// This runs for every ubatch of the cache (the MTP draft's 32K window: four times a decode round), so both runs
// are walked as at most two contiguous array segments each, without a modulo per cell.
static inline int64_t llama_kv_cells_ring_oldest(const llama_kv_cells & cells, uint32_t head) {
    const uint32_t size = cells.size();
    const uint32_t used = cells.get_used();

    if (size == 0 || used == 0) {
        return -1;
    }

    const llama_pos * pos = cells.pos_data();

    head %= size; // the head is size after a write that ended at the last cell

    // the empty run [head, head + size - used)
    const uint32_t e_end = head + (size - used);
    if (e_end <= size) {
        if (!llama_kv_cells_pos_empty(pos, head, e_end)) {
            return -1;
        }
    } else if (!llama_kv_cells_pos_empty(pos, head, size) || !llama_kv_cells_pos_empty(pos, 0, e_end - size)) {
        return -1;
    }

    const uint32_t oldest = e_end % size;

    // the used run: `used` cells from the oldest one, with the seam pair (size - 1, 0) when it wraps. An empty
    // cell inside the run breaks the chain at itself (pos -1 never follows a used position); only an empty
    // first cell has to be checked on its own
    if (pos[oldest] < 0) {
        return -1;
    }
    const uint32_t u_end = oldest + used;
    if (u_end <= size) {
        if (!llama_kv_cells_pos_run_ok(pos, oldest + 1, u_end)) {
            return -1;
        }
    } else if (!llama_kv_cells_pos_run_ok(pos, oldest + 1, size) || pos[0] != pos[size - 1] + 1 ||
               !llama_kv_cells_pos_run_ok(pos, 1, u_end - size)) {
        return -1;
    }

    return oldest;
}

// The cells a token at cell `end` of such a ring attends with a standard window of n_swa positions: the
// min(n_swa, avail) cells ending at its own cell, where avail counts the used cells from the oldest one up to
// and including its own (the newer cells of its batch and the empty run lie after it). The run wraps the cell
// array at most once: row = [lo0, hi0, lo1, hi1], the second segment empty (hi1 <= lo1) when it does not.
static inline void llama_kv_cells_ring_window(uint32_t size, uint32_t oldest, uint32_t end, uint32_t n_swa, int32_t * row) {
    const uint32_t avail = (end + size - oldest) % size + 1;
    const uint32_t span  = n_swa < avail ? n_swa : avail;

    if (span == 0) {
        row[0] = 0;
        row[1] = 0;
        row[2] = 0;
        row[3] = 0;
        return;
    }

    const uint32_t start = (end + size - (span - 1)) % size;

    if (start <= end) {
        row[0] = (int32_t) start;
        row[1] = (int32_t) end + 1;
        row[2] = 0;
        row[3] = 0;
    } else {
        row[0] = (int32_t) start;
        row[1] = (int32_t) size;
        row[2] = 0;
        row[3] = (int32_t) end + 1;
    }
}

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
