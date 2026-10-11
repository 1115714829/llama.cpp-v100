// Range masks on a single-sequence SWA ring (llama_kv_cells_ring_oldest / llama_kv_cells_ring_window, used by
// llama_kv_cache::get_kq_range_ok and set_input_kq_range).
//
// The cells are driven the way llama_kv_cache drives a ring: find_slot takes empty cells and cells outside the
// window from the head on, apply_ubatch writes them and moves the head behind the last one, seq_rm frees the
// cells of a position range and moves the head back to the first freed cell. After every ubatch that the cache
// would give a range mask, the [lo0, hi0) + [lo1, hi1) cells of every token must be exactly the cells the dense
// SWA mask shows it: used, position in [p - n_swa + 1, p].
//
// Scenarios:
//  - spec: the MTP draft context of a speculative server. The prompt tail is fed in chunks, then every round
//    runs three one-token draft steps, removes them (the server after drafting), feeds the four-token
//    verify batch (the catch-up) and removes the rejected tail (the server after the target accepted 0..3).
//    Once the ring is full, the rejected tail leaves empty cells right at the write head; every draft step
//    must still take the range mask (it used to fall back to the dense mask there, a different graph that was
//    rebuilt in every round after 0 or 1 accepted tokens and rebuilt back two steps later).
//  - random: chunks of random size, random tails removed, removals in the middle (holes elsewhere: no range
//    mask) and clears.
// Every state the previous rule accepted (used cells = prefix [0, used), or a full ring) must still be
// accepted, with the same windows.

#include "../src/llama-kv-cells.h"
#include "../src/llama-hparams.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

struct ring {
    llama_kv_cells cells;
    uint32_t       head = 0;
    uint32_t       n_swa;

    ring(uint32_t size, uint32_t n_swa) : n_swa(n_swa) {
        cells.resize(size);
    }

    // llama_kv_cache::find_slot, non-contiguous
    std::vector<uint32_t> find_slot(uint32_t n_tokens) const {
        const uint32_t size = cells.size();
        uint32_t head_cur = head;
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }
        std::vector<uint32_t> idxs;
        uint32_t n_tested = 0;
        while (idxs.size() < n_tokens && n_tested < size) {
            if (head_cur >= size) {
                head_cur = 0;
            }
            const uint32_t idx = head_cur++;
            n_tested++;
            bool can_use = cells.is_empty(idx);
            if (!can_use && cells.seq_count(idx) == 1) {
                const llama_seq_id s = cells.seq_get(idx);
                can_use = llama_hparams::is_masked_swa(n_swa, LLAMA_SWA_TYPE_STANDARD, cells.pos_get(idx), cells.seq_pos_max(s) + 1);
            }
            if (can_use) {
                idxs.push_back(idx);
            }
        }
        if (idxs.size() < n_tokens) {
            idxs.clear();
        }
        return idxs;
    }

    // llama_kv_cache::seq_rm for the one sequence of a ring: removing the newest positions (not all of them)
    // moves the head to the cell of the first removed position
    void seq_rm(llama_pos p0, llama_pos p1) {
        const bool ring_tail = cells.seq_n_cells(0) > 0 && p0 > cells.seq_pos_min(0) && p1 > cells.seq_pos_max(0);
        uint32_t new_head  = cells.size();
        uint32_t ring_head = cells.size();
        for (const uint32_t i : cells.seq_pos_range(0, p0, p1)) {
            if (cells.seq_rm(i, 0)) {
                new_head = std::min(new_head, i);
                if (ring_head == cells.size()) {
                    ring_head = i;
                }
            }
        }
        if (ring_tail && ring_head != cells.size()) {
            head = ring_head;
        } else if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    }

    // llama_kv_cache::apply_ubatch
    void apply(const std::vector<uint32_t> & idxs, llama_pos p0) {
        llama_pos pos_max_rm = -1;
        for (size_t k = 0; k < idxs.size(); ++k) {
            const uint32_t idx = idxs[k];
            if (!cells.is_empty(idx)) {
                pos_max_rm = std::max(pos_max_rm, cells.pos_get(idx));
                cells.rm(idx);
            }
            cells.pos_set(idx, p0 + (llama_pos) k);
            cells.seq_add(idx, 0);
        }
        if (pos_max_rm >= 0 && cells.seq_pos_min(0) <= pos_max_rm) {
            seq_rm(cells.seq_pos_min(0), pos_max_rm + 1);
        }
        head = idxs.back() + 1;
    }

    void clear() {
        cells.reset();
        head = 0;
    }
};

struct stats {
    int n_ubatch     = 0;
    int n_range      = 0;
    int n_range_old  = 0;
    int n_gap        = 0; // range mask with empty cells right at the head of a ring that wrapped
    int n_fail       = 0;
};

// the rule before the ring layout: used cells = prefix [0, used) in array order, or a full ring from the head
static bool range_ok_old(const ring & r) {
    const llama_kv_cells & c = r.cells;
    const uint32_t n = c.get_used();
    const uint32_t size = c.size();
    if (n == 0 || n != c.used_max_p1()) {
        return false;
    }
    const llama_pos * pos = c.pos_data();
    if (n < size) {
        for (uint32_t j = 1; j < n; ++j) {
            if (pos[j] != pos[j - 1] + 1) {
                return false;
            }
        }
        return true;
    }
    const uint32_t oldest = r.head % size;
    for (uint32_t k = 1; k < size; ++k) {
        if (pos[(oldest + k) % size] != pos[(oldest + k - 1) % size] + 1) {
            return false;
        }
    }
    return true;
}

// feed n tokens from position p0 and check the range mask the cache would build for them
static bool feed(ring & r, llama_pos p0, uint32_t n, stats & st, const char * what) {
    const std::vector<uint32_t> idxs = r.find_slot(n);
    if (idxs.empty()) {
        fprintf(stderr, "%s: no slot for %u tokens at pos %d\n", what, n, p0);
        st.n_fail++;
        return false;
    }
    r.apply(idxs, p0);
    st.n_ubatch++;

    const llama_kv_cells & c = r.cells;
    const uint32_t size = c.size();

    // the conditions of llama_kv_cache::get_kq_range_ok for a single-sequence SWA cache
    const bool base = c.get_used() > 0 && c.seq_n_cells(0) == c.get_used() && r.n_swa + n <= size;
    const int64_t oldest = base ? llama_kv_cells_ring_oldest(c, r.head) : -1;
    const bool range = oldest >= 0;
    const bool range_old = base && range_ok_old(r);

    if (range_old && !range) {
        fprintf(stderr, "%s: pos %d: the previous rule took the range mask, the ring rule does not\n", what, p0);
        st.n_fail++;
    }
    if (!range) {
        return true;
    }
    st.n_range++;
    st.n_range_old += range_old ? 1 : 0;
    if (!range_old && c.get_used() < size) {
        st.n_gap++;
    }

    for (uint32_t i = 0; i < n; ++i) {
        const llama_pos p = p0 + (llama_pos) i;
        int32_t row[4];
        llama_kv_cells_ring_window(size, (uint32_t) oldest, idxs[i], r.n_swa, row);

        // the cells of the range, against the cells the dense SWA mask shows the token
        size_t n_got = 0;
        size_t n_want = 0;
        bool   same = true;
        for (uint32_t j = 0; j < size; ++j) {
            const bool in_range = ((int32_t) j >= row[0] && (int32_t) j < row[1]) || ((int32_t) j >= row[2] && (int32_t) j < row[3]);
            bool visible = false;
            if (!c.is_empty(j)) {
                const llama_pos pj = c.pos_get(j);
                visible = pj <= p && !llama_hparams::is_masked_swa(r.n_swa, LLAMA_SWA_TYPE_STANDARD, pj, p);
            }
            n_got  += in_range ? 1 : 0;
            n_want += visible  ? 1 : 0;
            same = same && in_range == visible;
        }
        if (!same) {
            fprintf(stderr, "%s: pos %d (token %u of %u, cell %u, oldest %lld, head %u, used %u/%u): range [%d,%d)+[%d,%d) has %zu cells, the dense mask %zu\n",
                    what, p, i, n, idxs[i], (long long) oldest, r.head, c.get_used(), size, row[0], row[1], row[2], row[3], n_got, n_want);
            st.n_fail++;
            return false;
        }
    }
    return true;
}

static int test_spec(uint32_t n_swa, uint32_t n_ub, uint32_t n_rounds, uint32_t seed) {
    const uint32_t size = n_swa + n_ub;
    ring r(size, n_swa);
    stats st;
    std::mt19937 rng(seed);

    // the prompt tail the draft keeps (window + slack), in chunks of n_ub
    llama_pos pos = 1000;
    const uint32_t n_prompt = n_swa + n_ub/4;
    for (uint32_t done = 0; done < n_prompt; ) {
        const uint32_t n = std::min(n_ub, n_prompt - done);
        feed(r, pos, n, st, "spec/prompt");
        pos += n;
        done += n;
    }

    int n_step_full = 0;      // draft steps on a ring that has been full
    int n_step_full_range = 0;
    bool was_full = false;
    for (uint32_t round = 0; round < n_rounds && st.n_fail == 0; ++round) {
        // three draft steps from the position of the last accepted token
        for (uint32_t k = 0; k < 3; ++k) {
            const int n_range = st.n_range;
            feed(r, pos + (llama_pos) k, 1, st, "spec/draft");
            if (was_full) {
                n_step_full++;
                n_step_full_range += st.n_range - n_range;
                if (st.n_range == n_range && n_step_full - n_step_full_range == 1) {
                    const llama_kv_cells & c = r.cells;
                    fprintf(stderr, "spec: first draft step without the range mask: round %u step %u pos %d head %u used %u/%u used_max_p1 %u\n",
                            round, k, pos + (llama_pos) k, r.head, c.get_used(), c.size(), c.used_max_p1());
                    for (uint32_t j = 0; j < c.size(); ++j) {
                        fprintf(stderr, "%s%d", j ? " " : "  cells:", c.is_empty(j) ? -1 : c.pos_get(j));
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
        // the server removes the draft steps before the target verifies
        r.seq_rm(pos, std::numeric_limits<llama_pos>::max());
        // the catch-up: the four tokens of the verify batch
        feed(r, pos, 4, st, "spec/catch-up");
        was_full = was_full || r.cells.get_used() == size;
        // the target accepted 0..3 drafts: the rejected tail is removed
        const uint32_t n_acc = rng() % 4;
        pos += (llama_pos) n_acc + 1;
        r.seq_rm(pos, std::numeric_limits<llama_pos>::max());
    }

    printf("spec  n_swa=%u n_ub=%u: ubatches %d, range %d (previous rule %d, with empty cells at the head %d), draft steps on the full ring %d with range %d, failures %d\n",
           n_swa, n_ub, st.n_ubatch, st.n_range, st.n_range_old, st.n_gap, n_step_full, n_step_full_range, st.n_fail);

    if (n_step_full == 0 || n_step_full_range != n_step_full) {
        fprintf(stderr, "spec: every draft step on the full ring must take the range mask\n");
        return 1;
    }
    return st.n_fail == 0 ? 0 : 1;
}

static int test_random(uint32_t n_swa, uint32_t n_ub, uint32_t n_ops, uint32_t seed) {
    const uint32_t size = n_swa + n_ub;
    ring r(size, n_swa);
    stats st;
    std::mt19937 rng(seed);

    llama_pos pos = 0;
    for (uint32_t op = 0; op < n_ops && st.n_fail == 0; ++op) {
        const uint32_t what = rng() % 16;
        if (what < 10 || r.cells.get_used() == 0) {
            const uint32_t n = 1 + rng() % n_ub;
            feed(r, pos, n, st, "random/feed");
            pos += n;
        } else if (what < 14) {
            // remove a random tail
            const uint32_t k = 1 + rng() % 8;
            pos = std::max(r.cells.seq_pos_min(0), pos - (llama_pos) k);
            r.seq_rm(pos, std::numeric_limits<llama_pos>::max());
        } else if (what < 15) {
            // remove a range in the middle: a hole away from the head
            const llama_pos lo = r.cells.seq_pos_min(0);
            const llama_pos hi = r.cells.seq_pos_max(0);
            if (hi - lo > 4) {
                const llama_pos p0 = lo + 1 + (llama_pos) (rng() % (uint32_t) (hi - lo - 2));
                r.seq_rm(p0, p0 + 1);
                // keep the positions consecutive for the next feeds: drop everything after the hole too,
                // or keep the hole and let the next ubatches see it
                if (rng() % 2) {
                    r.seq_rm(p0, std::numeric_limits<llama_pos>::max());
                    pos = p0;
                }
            }
        } else {
            r.clear();
            pos = 0;
        }
    }

    printf("random n_swa=%u n_ub=%u: ubatches %d, range %d (previous rule %d, with empty cells at the head %d), failures %d\n",
           n_swa, n_ub, st.n_ubatch, st.n_range, st.n_range_old, st.n_gap, st.n_fail);
    return st.n_fail == 0 ? 0 : 1;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int rc = 0;
    rc |= test_spec(64, 16, 400, 1);
    rc |= test_spec(32768, 256, 300, 2); // the MTP draft ring of the first-version configuration
    rc |= test_spec(17, 5, 400, 3);
    rc |= test_random(64, 16, 4000, 4);
    rc |= test_random(13, 7, 4000, 5);
    printf("%s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
}
