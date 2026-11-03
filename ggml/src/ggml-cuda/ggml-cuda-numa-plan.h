#pragma once



















#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_cuda_numa_plan {





static constexpr double TIE = 1.02;



static constexpr uint64_t EXHAUSTIVE = 256;

struct result {
    std::vector<int>     node;
    std::vector<int64_t> node_bytes;
    double               cost = 0.0;
    double               cost_fast = 0.0;
    double               cost_local = 0.0;
    bool                 have_local = false;
};


static double cost_of(const int * a, size_t n_dev, size_t n_nodes, const size_t * dev_bytes, const double * t_ms) {
    double worst = 0.0;
    for (size_t d = 0; d < n_dev; d++) {
        const double td = t_ms[d*n_nodes + a[d]];
        if (td < 0.0) {
            return -1.0;
        }
        worst = std::max(worst, (double) dev_bytes[d]*td);
    }
    return worst;
}


static bool fits_of(const int * a, size_t n_dev, size_t n_nodes, const size_t * dev_bytes,
                    const int64_t * node_cap, std::vector<int64_t> & used) {
    used.assign(n_nodes, 0);
    for (size_t d = 0; d < n_dev; d++) {
        if (a[d] < 0 || (size_t) a[d] >= n_nodes) {
            return false;
        }
        used[a[d]] += (int64_t) dev_bytes[d];
    }
    for (size_t n = 0; n < n_nodes; n++) {
        if (used[n] > node_cap[n]) {
            return false;
        }
    }
    return true;
}

static bool lex_less(const std::vector<int> & a, const std::vector<int> & b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}





static int nodes_used(const std::vector<int> & a, const std::vector<size_t> & part, size_t n_nodes,
        std::vector<char> & seen) {
    seen.assign(n_nodes, 0);
    int n = 0;
    for (size_t d : part) {
        const int node = a[d];
        if (node < 0 || (size_t) node >= n_nodes || seen[(size_t) node]) {
            continue;
        }
        seen[(size_t) node] = 1;
        n++;
    }
    return n;
}



static int fastest_of(size_t device, size_t n_nodes, const double * t_ms) {
    int    best = -1;
    double best_t = 0.0;
    for (size_t n = 0; n < n_nodes; n++) {
        const double td = t_ms[device*n_nodes + n];
        if (td < 0.0) {
            continue;
        }
        if (best < 0 || td < best_t) {
            best   = (int) n;
            best_t = td;
        }
    }
    return best;
}



















static int plan(size_t n_dev, size_t n_nodes,
                const size_t * dev_bytes, const int64_t * node_cap, const double * t_ms,
                const int * topo_node, result * out) {
    out->node.clear();
    out->node_bytes.clear();
    out->cost = out->cost_fast = out->cost_local = 0.0;
    out->have_local = false;

    if (n_dev == 0 || n_nodes == 0) {
        return 0;
    }



    std::vector<int> base(n_dev, -1);
    for (size_t d = 0; d < n_dev; d++) {
        base[d] = fastest_of(d, n_nodes, t_ms);
        if (base[d] < 0) {
            return -1;
        }
    }


    std::vector<size_t> part;
    for (size_t d = 0; d < n_dev; d++) {
        if (dev_bytes[d] > 0) {
            part.push_back(d);
        }
    }



    out->cost_fast = cost_of(base.data(), n_dev, n_nodes, dev_bytes, t_ms);
    {
        std::vector<int> local(n_dev, -1);
        bool             all_local = true;
        for (size_t d = 0; d < n_dev; d++) {
            local[d] = fastest_of(d, n_nodes, t_ms);
            if (dev_bytes[d] == 0) {
                continue;
            }
            const int t = topo_node != nullptr ? topo_node[d] : -1;
            if (t >= 0 && (size_t) t < n_nodes && t_ms[d*n_nodes + t] >= 0.0) {
                local[d] = (int) t;
            } else {
                all_local = false;
            }
        }
        out->cost_local = cost_of(local.data(), n_dev, n_nodes, dev_bytes, t_ms);
        out->have_local = all_local;
    }


    uint64_t count = 1;
    bool     exhaustive = true;
    for (size_t i = 0; i < part.size(); i++) {
        if (count > EXHAUSTIVE) {
            exhaustive = false;
            break;
        }
        count *= n_nodes;
    }
    if (exhaustive && count > EXHAUSTIVE) {
        exhaustive = false;
    }

    std::vector<int>       best;
    std::vector<int64_t>   used;
    double                 best_cost = -1.0;

    if (exhaustive) {

        std::vector<int> a(n_dev);
        for (uint64_t idx = 0; idx < count; idx++) {
            a = base;
            uint64_t v = idx;
            for (size_t i = 0; i < part.size(); i++) {
                a[part[i]] = (int) (v % n_nodes);
                v /= n_nodes;
            }
            if (!fits_of(a.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
                continue;
            }
            const double c = cost_of(a.data(), n_dev, n_nodes, dev_bytes, t_ms);
            if (c >= 0.0 && (best_cost < 0.0 || c < best_cost)) {
                best_cost = c;
                best = a;
            }
        }







        if (best_cost >= 0.0) {
            const double limit = best_cost*TIE;
            std::vector<char> seen;
            int best_nodes = -1;
            for (uint64_t idx = 0; idx < count; idx++) {
                std::vector<int> c(n_dev);
                uint64_t v = idx;
                for (size_t i = 0; i < part.size(); i++) {
                    c[part[i]] = (int) (v % n_nodes);
                    v /= n_nodes;
                }
                if (!fits_of(c.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
                    continue;
                }
                const double cc = cost_of(c.data(), n_dev, n_nodes, dev_bytes, t_ms);
                if (cc < 0.0 || cc > limit) {
                    continue;
                }
                const int nodes = nodes_used(c, part, n_nodes, seen);
                if (nodes > best_nodes || (nodes == best_nodes && lex_less(c, best))) {
                    best_nodes = nodes;
                    best = c;
                }
            }
        }
    } else {


        std::vector<int> a = base;
        if (!fits_of(a.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {



            std::vector<size_t> by_bytes;
            for (;;) {
                size_t over = n_nodes;
                for (size_t n = 0; n < n_nodes; n++) {
                    if (used[n] > node_cap[n]) {
                        over = n;
                        break;
                    }
                }
                if (over == n_nodes) {
                    break;
                }
                by_bytes.clear();
                for (size_t d = 0; d < n_dev; d++) {
                    if (dev_bytes[d] > 0 && (size_t) a[d] == over) {
                        by_bytes.push_back(d);
                    }
                }
                std::sort(by_bytes.begin(), by_bytes.end(), [&](size_t x, size_t y) {
                    if (dev_bytes[x] != dev_bytes[y]) {
                        return dev_bytes[x] > dev_bytes[y];
                    }
                    return x < y;
                });
                bool moved = false;
                for (size_t d : by_bytes) {
                    int    best_n = -1;
                    double best_t = 0.0;
                    for (size_t n = 0; n < n_nodes; n++) {
                        if (n == over || used[n] + (int64_t) dev_bytes[d] > node_cap[n]) {
                            continue;
                        }
                        const double td = t_ms[d*n_nodes + n];
                        if (td < 0.0 || (best_n >= 0 && td >= best_t)) {
                            continue;
                        }
                        best_n = (int) n;
                        best_t = td;
                    }
                    if (best_n >= 0) {
                        a[d]    = best_n;
                        used[over] -= (int64_t) dev_bytes[d];
                        used[best_n] += (int64_t) dev_bytes[d];
                        moved = true;
                    }
                }
                if (!moved) {
                    break;
                }
            }
        }

        if (!fits_of(a.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
            return -1;
        }
        const double cur = cost_of(a.data(), n_dev, n_nodes, dev_bytes, t_ms);
        if (cur < 0.0) {
            return -1;
        }
        for (size_t round = 0; round <= part.size(); round++) {

            std::vector<size_t> crit;
            double worst = 0.0;
            for (size_t d : part) {
                const double v = (double) dev_bytes[d]*t_ms[d*n_nodes + a[d]];
                if (v > worst) {
                    worst = v;
                    crit.clear();
                    crit.push_back(d);
                } else if (v == worst) {
                    crit.push_back(d);
                }
            }

            std::vector<int> best_a;
            double           best_c = cur;
            for (size_t d : crit) {
                for (size_t n = 0; n < n_nodes; n++) {
                    if ((int) n == a[d]) {
                        continue;
                    }
                    std::vector<int> trial = a;
                    trial[d] = (int) n;
                    if (!fits_of(trial.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
                        continue;
                    }
                    const double c = cost_of(trial.data(), n_dev, n_nodes, dev_bytes, t_ms);
                    if (c < 0.0) {
                        continue;
                    }


                    if (c < best_c) {
                        best_c = c;
                        best_a = trial;
                    } else if (c == best_c && !best_a.empty() && lex_less(trial, best_a)) {
                        best_a = trial;
                    }
                }
            }
            if (best_a.empty()) {
                break;
            }
            a = best_a;
        }

        best_cost = cost_of(a.data(), n_dev, n_nodes, dev_bytes, t_ms);
        best = a;
    }

    if (best.empty()) {
        return -1;
    }

    if (!exhaustive) {




        std::vector<int> lex = best;
        if (fits_of(lex.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
            for (size_t d : part) {
                for (int n = 0; n < lex[d]; n++) {
                    std::vector<int> trial = lex;
                    trial[d] = n;
                    if (!fits_of(trial.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
                        continue;
                    }
                    const double c = cost_of(trial.data(), n_dev, n_nodes, dev_bytes, t_ms);
                    if (c >= 0.0 && c <= best_cost*TIE) {
                        lex = trial;
                        fits_of(lex.data(), n_dev, n_nodes, dev_bytes, node_cap, used);
                        break;
                    }
                }
            }
        }
        best = lex;
    }

    out->node = best;
    fits_of(out->node.data(), n_dev, n_nodes, dev_bytes, node_cap, out->node_bytes);
    out->cost = cost_of(out->node.data(), n_dev, n_nodes, dev_bytes, t_ms);
    return 1;
}

// The node of the two fixed pinned host allocations of a load (the mapped weights table and the
// pinned host buffers of the context) next to the arenas of the devices, and whether they hold
// together.
//
// The check is done on the placement every device keeps on the node of its PCI slot, not on the
// measured one: the read times are a measurement, and a measurement that can not tell the
// placements apart must not turn a machine that holds the arenas and the two fixed blocks into
// one that does not. A machine that holds them needs no segmented placement.
//
//   n_dev       number of devices
//   dev_bytes   [device] host bytes the device reads
//   local_node  [device] node index of the PCI slot of the device, negative when the platform
//               has no topology: the device then takes the node with the most room
//   n_nodes     number of host NUMA nodes
//   node_cap    [node] bytes the node may hold
//   fixed_w     bytes of the mapped weights table, one block on one node
//   fixed_p     bytes of the pinned host buffers of the context, one block on one node
//   node_w      receives the node of the mapped weights table
//   node_p      receives the node of the pinned host buffers
//   dev_node    receives the node of every device
//   used_out    receives the bytes the placement puts on every node (may be null)
//
// The mapped weights table goes on the node with the most room and the pinned host buffers on
// the node with the most room left, the lowest index wins a tie.
// the bytes the arenas of the devices carry on every node, when every device keeps its arena on the
// node of its PCI slot (the placement of one node per device)
static std::vector<int64_t> slot_occupancy(size_t n_dev, const size_t * dev_bytes,
        const int * local_node, size_t n_nodes, const int64_t * node_cap) {
    std::vector<int64_t> used(n_nodes, 0);
    for (size_t d = 0; d < n_dev; d++) {
        int n = local_node != nullptr ? local_node[d] : -1;
        if (n < 0 || (size_t) n >= n_nodes) {
            n = 0;
            for (size_t m = 1; m < n_nodes; m++) {
                if (node_cap[m] > node_cap[n]) {
                    n = (int) m;
                }
            }
        }
        used[(size_t) n] += (int64_t) dev_bytes[d];
    }
    return used;
}



















static int table_node_of(const std::vector<int64_t> & node_bytes, size_t n_nodes,
        const int64_t * node_cap, int64_t fixed_w) {
    if (n_nodes == 0) {
        return -1;
    }



    int best = -1;
    for (size_t n = 0; n < n_nodes && n < node_bytes.size(); n++) {
        if (node_bytes[n] != 0 || node_cap[n] < fixed_w) {
            continue;
        }
        if (best < 0 || node_cap[n] > node_cap[(size_t) best]) {
            best = (int) n;
        }
    }
    if (best >= 0) {
        return best;
    }


    best = 0;
    for (size_t n = 1; n < n_nodes; n++) {
        const int64_t used_n = n < node_bytes.size() ? node_bytes[n] : 0;
        const int64_t used_b = (size_t) best < node_bytes.size() ? node_bytes[(size_t) best] : 0;
        if (node_cap[n] - used_n > node_cap[(size_t) best] - used_b) {
            best = (int) n;
        }
    }
    return best;
}




static int pinned_node_of(const std::vector<int64_t> & node_bytes, size_t n_nodes,
        const int64_t * node_cap, int64_t fixed_w) {
    if (n_nodes == 0) {
        return -1;
    }
    const int w = table_node_of(node_bytes, n_nodes, node_cap, fixed_w);
    if (w < 0) {
        return -1;
    }

    std::vector<int64_t> used(node_bytes);
    used.resize(n_nodes, 0);
    used[(size_t) w] += fixed_w;

    int best = 0;
    for (size_t n = 1; n < n_nodes; n++) {
        if (node_cap[n] - used[n] > node_cap[(size_t) best] - used[(size_t) best]) {
            best = (int) n;
        }
    }
    return best;
}




static int table_node(size_t n_dev, const size_t * dev_bytes, const int * local_node,
        size_t n_nodes, const int64_t * node_cap, int64_t fixed_w) {
    if (n_nodes == 0) {
        return -1;
    }
    std::vector<int64_t> node_bytes = slot_occupancy(n_dev, dev_bytes, local_node, n_nodes, node_cap);
    return table_node_of(node_bytes, n_nodes, node_cap, fixed_w);
}



static int pinned_node(size_t n_dev, const size_t * dev_bytes, const int * local_node,
        size_t n_nodes, const int64_t * node_cap, int64_t fixed_w) {
    if (n_nodes == 0) {
        return -1;
    }
    std::vector<int64_t> node_bytes = slot_occupancy(n_dev, dev_bytes, local_node, n_nodes, node_cap);
    return pinned_node_of(node_bytes, n_nodes, node_cap, fixed_w);
}

static bool fixed_fits(size_t n_dev, const size_t * dev_bytes, const int * local_node,
        size_t n_nodes, const int64_t * node_cap, int64_t fixed_w, int64_t fixed_p,
        int * node_w, int * node_p, int * dev_node, std::vector<int64_t> * used_out = nullptr) {
    if (n_nodes == 0) {
        return false;
    }


    std::vector<int64_t> used(n_nodes, 0);
    for (size_t d = 0; d < n_dev; d++) {
        int n = local_node != nullptr ? local_node[d] : -1;
        if (n < 0 || (size_t) n >= n_nodes) {
            n = 0;
            for (size_t m = 1; m < n_nodes; m++) {
                if (node_cap[m] > node_cap[n]) {
                    n = (int) m;
                }
            }
        }
        if (dev_node != nullptr) {
            dev_node[d] = n;
        }
        used[n] += (int64_t) dev_bytes[d];
    }

    int w = table_node(n_dev, dev_bytes, local_node, n_nodes, node_cap, fixed_w);
    if (w < 0) {
        return false;
    }
    used[w] += fixed_w;
    int p = pinned_node(n_dev, dev_bytes, local_node, n_nodes, node_cap, fixed_w);
    if (p < 0) {
        return false;
    }
    used[p] += fixed_p;

    if (node_w != nullptr) {
        *node_w = w;
    }
    if (node_p != nullptr) {
        *node_p = p;
    }

    if (used_out != nullptr) {
        *used_out = used;
    }

    for (size_t n = 0; n < n_nodes; n++) {
        if (used[n] > node_cap[n]) {
            return false;
        }
    }
    return true;
}







struct segment_entry {
    int     node;
    int64_t bytes;

    bool operator==(const segment_entry & other) const {
        return node == other.node && bytes == other.bytes;
    }
};



























static int segment(size_t n_dev, const size_t * dev_bytes, const int * alloc_dev,
        const size_t * alloc_size, size_t n_alloc, const int * local_node,
        size_t n_nodes, const int64_t * node_cap, int64_t fixed_w, int64_t fixed_p,
        int64_t slack, std::vector<std::vector<segment_entry>> * out) {
    out->assign(n_dev, std::vector<segment_entry>());
    if (n_dev == 0 || n_nodes == 0) {
        return 0;
    }




    std::vector<int64_t> room(n_nodes);
    for (size_t n = 0; n < n_nodes; n++) {
        room[n] = node_cap[n] - slack;
    }
    int node_w = 0;
    for (size_t n = 1; n < n_nodes; n++) {
        if (room[n] > room[node_w]) {
            node_w = (int) n;
        }
    }
    room[node_w] -= fixed_w;
    int node_p = 0;
    for (size_t n = 1; n < n_nodes; n++) {
        if (room[n] > room[node_p]) {
            node_p = (int) n;
        }
    }
    room[node_p] -= fixed_p;


    std::vector<std::vector<int>> quota_node(n_dev);
    std::vector<std::vector<int64_t>> quota_left(n_dev);
    std::vector<size_t> order(n_dev);
    for (size_t d = 0; d < n_dev; d++) {
        order[d] = d;
    }
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        if (dev_bytes[x] != dev_bytes[y]) {
            return dev_bytes[x] > dev_bytes[y];
        }
        return x < y;
    });

    for (size_t d : order) {
        int64_t rest = (int64_t) dev_bytes[d];
        if (rest <= 0) {
            continue;
        }
        const int local = local_node != nullptr ? local_node[d] : -1;
        if (local >= 0 && (size_t) local < n_nodes && room[local] > 0) {
            const int64_t take = std::min(rest, room[local]);
            quota_node[d].push_back(local);
            quota_left[d].push_back(take);
            room[local] -= take;
            rest         -= take;
        }
        while (rest > 0) {
            int best = -1;
            for (size_t n = 0; n < n_nodes; n++) {
                if (room[n] > 0 && (best < 0 || room[n] > room[best])) {
                    best = (int) n;
                }
            }
            if (best < 0) {
                out->clear();
                return 0;
            }
            const int64_t take = std::min(rest, room[best]);
            quota_node[d].push_back(best);
            quota_left[d].push_back(take);
            room[best] -= take;
            rest        -= take;
        }
    }



    std::vector<size_t> at(n_dev, 0);
    for (size_t i = 0; i < n_alloc; i++) {
        const int     d    = alloc_dev != nullptr ? alloc_dev[i] : 0;
        const int64_t size = (int64_t) alloc_size[i];
        if (d < 0 || (size_t) d >= n_dev || size <= 0) {
            continue;
        }
        size_t k = at[d];
        while (k < quota_node[d].size() && size > quota_left[d][k]) {
            if (k + 1 < quota_node[d].size()) {
                quota_left[d][k + 1] += quota_left[d][k];
            }
            quota_left[d][k] = 0;
            k++;
        }
        if (k >= quota_node[d].size()) {
            out->clear();
            return 0;
        }
        quota_left[d][k] -= size;
        at[d] = k;

        std::vector<segment_entry> & tbl = (*out)[d];
        if (tbl.empty() || tbl.back().node != quota_node[d][k]) {
            tbl.push_back({ quota_node[d][k], size });
        } else {
            tbl.back().bytes += size;
        }
    }


    if (n_alloc == 0) {
        for (size_t d : order) {
            if (quota_node[d].empty()) {
                continue;
            }
            size_t k = 0;
            int64_t rest = (int64_t) dev_bytes[d];
            while (rest > 0) {
                if (k >= quota_node[d].size()) {
                    out->clear();
                    return 0;
                }
                const int64_t take = std::min(rest, quota_left[d][k]);
                (*out)[d].push_back({ quota_node[d][k], take });
                quota_left[d][k] -= take;
                rest             -= take;
                k++;
            }
        }
    }

    return 1;
}













static int64_t pinned_room(int64_t cap, int64_t zone_high, int64_t margin_proc) {
    return cap - margin_proc - (zone_high > 0 ? zone_high : 0);
}



























static int balanced(size_t n_dev, const size_t * dev_bytes, const int * alloc_dev,
        const size_t * alloc_size, size_t n_alloc, const int * local_node,
        size_t n_nodes, const int64_t * node_cap, int64_t fixed_w, int64_t fixed_p,
        int64_t slack, int node_w, int node_p,
        std::vector<std::vector<segment_entry>> * out) {
    out->assign(n_dev, std::vector<segment_entry>());
    if (n_dev == 0 || n_nodes == 0) {
        return 0;
    }


    std::vector<int64_t> room(n_nodes);
    for (size_t n = 0; n < n_nodes; n++) {
        room[n] = node_cap[n] - slack;
    }
    if (node_w < 0 || (size_t) node_w >= n_nodes || node_p < 0 || (size_t) node_p >= n_nodes) {
        return 0;
    }
    room[(size_t) node_w] -= fixed_w;
    room[(size_t) node_p] -= fixed_p;


    std::vector<std::vector<int>>    quota_node(n_dev);
    std::vector<std::vector<int64_t>> quota_left(n_dev);
    std::vector<size_t> order(n_dev);
    for (size_t d = 0; d < n_dev; d++) {
        order[d] = d;
    }
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        if (dev_bytes[x] != dev_bytes[y]) {
            return dev_bytes[x] > dev_bytes[y];
        }
        return x < y;
    });








    const auto fill_quota = [&](bool balance) {
        std::vector<int64_t> left = room;
        std::vector<int64_t> take_up(n_nodes, 0);
        take_up[(size_t) node_w] += fixed_w;
        take_up[(size_t) node_p] += fixed_p;


        int64_t total = fixed_w + fixed_p;
        for (size_t d = 0; d < n_dev; d++) {
            total += (int64_t) dev_bytes[d];
        }
        const int64_t target = total / (int64_t) n_nodes;

        for (size_t d : order) {
            int64_t rest = (int64_t) dev_bytes[d];
            if (rest <= 0) {
                continue;
            }
            const int local = local_node != nullptr ? local_node[d] : -1;
            if (local >= 0 && (size_t) local < n_nodes) {
                int64_t take = std::min(rest, left[(size_t) local]);
                if (balance) {


                    const int64_t share = target > take_up[(size_t) local] ? target - take_up[(size_t) local] : 0;
                    take = std::min(rest, std::min(share, left[(size_t) local]));
                }
                if (take > 0) {
                    quota_node[d].push_back(local);
                    quota_left[d].push_back(take);
                    take_up[(size_t) local] += take;
                    left[(size_t) local] -= take;
                    rest -= take;
                }
            }
            while (rest > 0) {
                int best = -1;
                for (size_t n = 0; n < n_nodes; n++) {
                    if (left[n] <= 0) {
                        continue;
                    }
                    if (best < 0) {
                        best = (int) n;
                        continue;
                    }
                    if (balance) {

                        if (take_up[n] < take_up[(size_t) best]) {
                            best = (int) n;
                        }
                    } else if (left[n] > left[(size_t) best]) {
                        best = (int) n;
                    }
                }
                if (best < 0) {
                    return false;
                }
                const int64_t take = std::min(rest, left[(size_t) best]);
                quota_node[d].push_back(best);
                quota_left[d].push_back(take);
                take_up[(size_t) best] += take;
                left[(size_t) best] -= take;
                rest -= take;
            }
        }

        return true;
    };

    if (!fill_quota(true)) {
        out->clear();
        return 0;
    }



    std::vector<size_t> at(n_dev, 0);
    for (size_t i = 0; i < n_alloc; i++) {
        const int     d    = alloc_dev != nullptr ? alloc_dev[i] : 0;
        const int64_t size = (int64_t) alloc_size[i];
        if (d < 0 || (size_t) d >= n_dev || size <= 0) {
            continue;
        }
        size_t k = at[d];
        while (k < quota_node[d].size() && size > quota_left[d][k]) {
            if (k + 1 < quota_node[d].size()) {
                quota_left[d][k + 1] += quota_left[d][k];
            }
            quota_left[d][k] = 0;
            k++;
        }
        if (k >= quota_node[d].size()) {
            out->clear();
            return 0;
        }
        quota_left[d][k] -= size;
        at[d] = k;

        std::vector<segment_entry> & tbl = (*out)[d];
        if (tbl.empty() || tbl.back().node != quota_node[d][k]) {
            tbl.push_back({ quota_node[d][k], size });
        } else {
            tbl.back().bytes += size;
        }
    }


    if (n_alloc == 0) {
        for (size_t d : order) {
            if (quota_node[d].empty()) {
                continue;
            }
            size_t k = 0;
            int64_t rest = (int64_t) dev_bytes[d];
            while (rest > 0) {
                if (k >= quota_node[d].size()) {
                    out->clear();
                    return 0;
                }
                const int64_t take = std::min(rest, quota_left[d][k]);
                (*out)[d].push_back({ quota_node[d][k], take });
                quota_left[d][k] -= take;
                rest -= take;
                k++;
            }
        }
    }

    return 1;
}





static constexpr size_t BYTES_TOLERANCE = 1ull << 20;

struct committed {
    bool                 valid    = false;
    bool                 from_fit = false;
    std::vector<int>     node;
    std::vector<size_t> bytes;
};


static void commit_plan(committed * c, const std::vector<int> & node, const std::vector<size_t> & bytes, bool from_fit) {
    c->valid    = true;
    c->from_fit = from_fit;
    c->node     = node;
    c->bytes    = bytes;
}




static int committed_node(const committed & c, size_t device, size_t have_bytes, bool * bytes_match) {
    if (!c.valid || device >= c.node.size() || c.node[device] < 0) {
        return -1;
    }
    if (bytes_match != nullptr) {
        const size_t want = device < c.bytes.size() ? c.bytes[device] : 0;
        const size_t diff = want > have_bytes ? want - have_bytes : have_bytes - want;
        *bytes_match = diff <= BYTES_TOLERANCE;
    }
    return c.node[device];
}

}