// tests for the segmented NUMA placement of the expert host arenas (ggml-cuda-numa-plan.h)

#include "ggml-cuda/ggml-cuda-numa-plan.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_cuda_numa_plan::segment_entry;

constexpr int64_t MiB = 1024ll*1024ll;
constexpr int64_t GiB = 1024ll*MiB;

std::string to_string(const std::vector<std::vector<segment_entry>> & seg) {
    std::string s;
    for (size_t d = 0; d < seg.size(); d++) {
        s += (d ? " | " : "") + std::to_string(d) + ":";
        for (const segment_entry & e : seg[d]) {
            s += " node " + std::to_string(e.node) + " " + std::to_string(e.bytes/MiB) + "MiB";
        }
    }
    return s;
}

// the AC922 2 card case: the mapped weights table is one block and the pinned host buffers
// follow it, so no assignment of one node per card holds the arenas and the two fixed blocks
// together (a card arena plus the table is more than the node may fill). The table is solved
// arena by arena, so walking it the way the load does must place every arena.
bool test_two_cards() {
    const size_t  n_dev  = 2;
    const int64_t cap[2] = { 46810*MiB, 46895*MiB }; // MemTotal minus max(2 GiB, 4%)
    const int64_t w  = 27466*MiB;
    const int64_t p  =  3451*MiB;
    const size_t  arena = 355*MiB + 8192;             // one expert arena, page aligned
    const size_t  n_arena = 81;                       // ~28.2 GiB per card
    const int     local[2] = { 0, 0 };                // both cards in the PCI slots of node 0
    const int64_t slack = arena;

    std::vector<int>    dev;
    std::vector<size_t> size;
    for (int d = 0; d < 2; d++) {
        for (size_t k = 0; k < n_arena; k++) {
            dev.push_back(d);
            size.push_back(arena);
        }
    }
    const size_t bytes[2] = { n_arena*arena, n_arena*arena };

    std::vector<std::vector<segment_entry>> seg;
    if (ggml_cuda_numa_plan::segment(n_dev, bytes, dev.data(), size.data(), dev.size(), local,
            2, cap, w, p, slack, &seg) != 1) {
        fprintf(stderr, "%s: the two card case did not segment\n", __func__);
        return false;
    }

    // walk the table the way the load does: every arena takes the segment it fits in, and the
    // next one when it does not. No arena may be left without a segment.
    std::vector<std::vector<int64_t>> left(n_dev);
    for (size_t d = 0; d < n_dev; d++) {
        for (const segment_entry & e : seg[d]) {
            left[d].push_back(e.bytes);
        }
    }
    for (size_t i = 0; i < dev.size(); i++) {
        const size_t d = (size_t) dev[i];
        size_t k = 0;
        while (k < seg[d].size() && (int64_t) size[i] > left[d][k]) {
            if (k + 1 < seg[d].size()) {
                left[d][k + 1] += left[d][k];   // the same carry over the load does
            }
            left[d][k] = 0;
            k++;
        }
        if (k >= seg[d].size()) {
            fprintf(stderr, "%s: arena %zu of device %zu has no segment (%s)\n",
                    __func__, i, d, to_string(seg).c_str());
            return false;
        }
        left[d][k] -= (int64_t) size[i];
    }

    // the table may not promise more than the nodes may hold
    for (size_t n = 0; n < 2; n++) {
        int64_t used = slack;
        used += (n == 1 ? w : 0);
        used += (n == 0 ? p : 0);
        for (const auto & tbl : seg) {
            for (const segment_entry & e : tbl) {
                if ((size_t) e.node == n) {
                    used += e.bytes;
                }
            }
        }
        if (used > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld MiB\n",
                    __func__, n, (long long) (used/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }

    // the first card keeps its own node as far as that node reaches
    if (seg[0].empty() || seg[0][0].node != 0) {
        fprintf(stderr, "%s: card 0 does not start on its own node: %s\n", __func__, to_string(seg).c_str());
        return false;
    }
    return true;
}

// with room for everything the solver is not called at all; this case only checks the table of
// a machine whose cards all fit on the node of their PCI slot (the no-trace path of the solver)
bool test_six_cards_stay_local() {
    const size_t  n_dev    = 6;
    const int64_t cap[2]   = { 46810*MiB, 46895*MiB };
    const size_t  bytes[6] = { (size_t) (980*MiB), (size_t) (980*MiB), (size_t) (980*MiB), (size_t) (970*MiB), (size_t) (970*MiB), (size_t) (970*MiB) };
    const int     local[6] = { 0, 0, 0, 1, 1, 1 };
    const int64_t w  = 27466*MiB;
    const int64_t p  =  4096*MiB;

    std::vector<std::vector<segment_entry>> seg;
    if (ggml_cuda_numa_plan::segment(n_dev, bytes, nullptr, nullptr, 0, local,
            2, cap, w, p, 0, &seg) != 1) {
        fprintf(stderr, "%s: the six card case did not segment\n", __func__);
        return false;
    }
    for (size_t d = 0; d < n_dev; d++) {
        if (seg[d].size() != 1 || seg[d][0].node != local[d] || seg[d][0].bytes != (int64_t) bytes[d]) {
            fprintf(stderr, "%s: device %zu: got %s\n", __func__, d, to_string(seg).c_str());
            return false;
        }
    }
    return true;
}

// The two cards of the cross socket pair on a machine like the ac922-2: card 0 in the PCI slot
// of node 0, card 3 in the slot of node 8. Every card keeps its arena whole on the node of its
// own slot, the table and the pinned host buffers take the room of the big node.
bool test_two_cards_cross_socket() {
    const size_t  n_dev  = 2;
    const int64_t cap[2] = { 94208*MiB, 30720*MiB }; // the physical capacity of the two nodes
    const int64_t w  = 27466*MiB;
    const int64_t p  =  3452*MiB;
    const size_t  arena = 29434*MiB;               // the host complement of one card
    const size_t  n_arena = 81;
    const int     local[2] = { 0, 1 };              // card 0 on node 0, card 3 on node 8
    const int64_t slack = 364*MiB;

    std::vector<int>    dev;
    std::vector<size_t> size;
    for (int d = 0; d < 2; d++) {
        // the arenas share the complement of the card: the last one takes the rest, so that the
        // sum is the complement itself and not the complement minus the remainder of the division
        size_t left = arena;
        for (size_t k = 0; k < n_arena; k++) {
            dev.push_back(d);
            const size_t chunk = (k + 1 == n_arena) ? left : arena / n_arena;
            size.push_back(chunk);
            left -= chunk;
        }
    }
    const size_t bytes[2] = { arena, arena };

    std::vector<std::vector<segment_entry>> seg;
    if (ggml_cuda_numa_plan::segment(n_dev, bytes, dev.data(), size.data(), dev.size(), local,
            2, cap, w, p, slack, &seg) != 1) {
        fprintf(stderr, "%s: the cross socket pair did not segment\n", __func__);
        return false;
    }
    // every card keeps its arena whole on the node of its own slot
    for (size_t d = 0; d < n_dev; d++) {
        int64_t total = 0;
        for (const segment_entry & s : seg[d]) {
            total += s.bytes;
        }
        if (seg[d].size() != 1 || seg[d][0].node != local[d] || total != (int64_t) bytes[d]) {
            fprintf(stderr, "%s: device %zu: got %s\n", __func__, d, to_string(seg).c_str());
            return false;
        }
    }
    return true;
}

// The same pair on a machine whose small node can not hold the arena of its card once the
// mapped weights table took the room: the arena is then split, the node of the card first.
bool test_two_cards_split_when_the_node_is_too_small() {
    const size_t  n_dev  = 2;
    const int64_t cap[2] = { 46810*MiB, 46895*MiB };
    const int64_t w  = 27466*MiB;
    const int64_t p  =  3451*MiB;
    const size_t  arena = 355*MiB + 8192;
    const size_t  n_arena = 81;
    const int     local[2] = { 0, 1 };
    const int64_t slack = arena;

    std::vector<int>    dev;
    std::vector<size_t> size;
    for (int d = 0; d < 2; d++) {
        for (size_t k = 0; k < n_arena; k++) {
            dev.push_back(d);
            size.push_back(arena);
        }
    }
    const size_t bytes[2] = { n_arena*arena, n_arena*arena };

    std::vector<std::vector<segment_entry>> seg;
    if (ggml_cuda_numa_plan::segment(n_dev, bytes, dev.data(), size.data(), dev.size(), local,
            2, cap, w, p, slack, &seg) != 1) {
        fprintf(stderr, "%s: the pair did not segment\n", __func__);
        return false;
    }
    // the first card keeps its arena on the node of its slot
    if (seg[0].size() != 1 || seg[0][0].node != 0) {
        fprintf(stderr, "%s: device 0: got %s\n", __func__, to_string(seg).c_str());
        return false;
    }
    // the second card starts on the node of its slot and spills onto the node with the room
    int64_t total = 0;
    for (const segment_entry & s : seg[1]) {
        total += s.bytes;
    }
    if (seg[1].empty() || seg[1][0].node != 1 || seg[1].size() < 2 ||
            total != (int64_t) bytes[1]) {
        fprintf(stderr, "%s: device 1: got %s\n", __func__, to_string(seg).c_str());
        return false;
    }
    return true;
}

// The check of the fixed pinned host allocations: on the ac922-1 (two nodes of 48 GiB) the two
// card cases can not hold the arenas and the two fixed blocks together, so the placement has to
// be segmented. The six card case holds them: it keeps the plan of the machine and changes
// nothing.
bool test_trigger() {
    const int64_t cap_ac9221[2] = { 44762*MiB, 44847*MiB }; // MemTotal minus the 4 GiB margin
    const int64_t w = 27466*MiB;
    const int64_t p =  3451*MiB;

    // two cards in the slots of node 0: the arenas and the two fixed blocks do not fit
    {
        const size_t bytes[2] = { (size_t) (28912*MiB), (size_t) (28913*MiB) };
        const int     local[2] = { 0, 0 };
        int node_w = -1, node_p = -1;
        if (ggml_cuda_numa_plan::fixed_fits(2, bytes, local, 2, cap_ac9221, w, p, &node_w, &node_p, nullptr)) {
            fprintf(stderr, "%s: the two cards in one node hold the fixed blocks\n", __func__);
            return false;
        }
        if (node_w < 0 || node_p < 0) {
            fprintf(stderr, "%s: the two cards did not get a node for the fixed blocks\n", __func__);
            return false;
        }
    }

    // two cards in the slots of node 0 and node 8: the card of node 0 and the table do not fit
    {
        const size_t bytes[2] = { (size_t) (28912*MiB), (size_t) (28913*MiB) };
        const int     local[2] = { 0, 1 };
        if (ggml_cuda_numa_plan::fixed_fits(2, bytes, local, 2, cap_ac9221, w, p, nullptr, nullptr, nullptr)) {
            fprintf(stderr, "%s: the cross socket pair holds the fixed blocks\n", __func__);
            return false;
        }
    }

    // six cards: the arenas and the two fixed blocks fit, the plan of the machine stays
    {
        const size_t bytes[6] = { (size_t) (980*MiB), (size_t) (980*MiB), (size_t) (980*MiB),
                                  (size_t) (970*MiB), (size_t) (970*MiB), (size_t) (970*MiB) };
        const int     local[6] = { 0, 0, 0, 1, 1, 1 };
        int node_w = -1, node_p = -1;
        if (!ggml_cuda_numa_plan::fixed_fits(6, bytes, local, 2, cap_ac9221, w, p, &node_w, &node_p, nullptr)) {
            fprintf(stderr, "%s: the six cards do not hold the fixed blocks\n", __func__);
            return false;
        }
        if (node_w != 1 || node_p != 0) {
            fprintf(stderr, "%s: the six cards put the fixed blocks on node %d and %d\n",
                    __func__, node_w, node_p);
            return false;
        }
    }
    return true;
}

// a device without a PCI topology spills from the start, and the nodes that can not hold the
// bytes fail instead of placing them anywhere
bool test_no_topology_and_failure() {    const size_t  n_dev    = 2;
    const int64_t cap[2]   = { 10*GiB, 10*GiB };
    const size_t  bytes[2] = { (size_t) (8*GiB), (size_t) (4*GiB) };
    const int     local[2] = { -1, -1 };

    std::vector<std::vector<segment_entry>> seg;
    if (ggml_cuda_numa_plan::segment(n_dev, bytes, nullptr, nullptr, 0, local,
            2, cap, 0, 0, 0, &seg) != 1) {
        fprintf(stderr, "%s: the spill case did not segment\n", __func__);
        return false;
    }
    // the largest device takes node 0 (the lowest index on a tie), then the second device takes
    // node 1, which still holds its 10 GiB
    for (size_t d = 0; d < n_dev; d++) {
        if (seg[d].size() != 1) {
            fprintf(stderr, "%s: device %zu got %s\n", __func__, d, to_string(seg).c_str());
            return false;
        }
    }
    if (seg[0][0].node != 0 || seg[0][0].bytes != (int64_t) bytes[0] ||
        seg[1][0].node != 1 || seg[1][0].bytes != (int64_t) bytes[1]) {
        fprintf(stderr, "%s: got %s\n", __func__, to_string(seg).c_str());
        return false;
    }

    const size_t  big[2] = { (size_t) (12*GiB), (size_t) (12*GiB) };
    std::vector<std::vector<segment_entry>> none;
    if (ggml_cuda_numa_plan::segment(n_dev, big, nullptr, nullptr, 0, local,
            2, cap, 0, 0, 0, &none) == 1) {
        fprintf(stderr, "%s: 24 GiB on 20 GiB of nodes were placed\n", __func__);
        return false;
    }
    if (!none.empty()) {
        fprintf(stderr, "%s: a failed placement left segments behind\n", __func__);
        return false;
    }
    return true;
}

} // namespace

int main(void) {
    if (!test_two_cards() ||
        !test_two_cards_cross_socket() ||
        !test_two_cards_split_when_the_node_is_too_small() ||
        !test_six_cards_stay_local() ||
        !test_trigger() ||
        !test_no_topology_and_failure()) {
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
