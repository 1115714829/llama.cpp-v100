#pragma once

// NUMA placement solver for the host tier of the expert store (used from ggml-cuda.cu).
//
// The node of every device-readable host arena is planned once for the whole machine: the
// per-device read time under concurrent load decides how fast a device reads a node, and the
// capacity of the node is the constraint. The solver is a pure function of those inputs and
// has no CUDA dependency, so it is unit tested on the CPU (tests/test-numa-plan.cpp).
//
// The estimated time of an assignment a is T(a) = max over the devices of
// (bytes_d * read_ms[d][a[d]]): the devices read at the same time, so the slowest one sets
// the step time. Units are byte*ms; the caller divides them by the probe size to get a step
// time in ms.
//
// The plan is decided once per machine and then frozen: fit computes it from the host bytes of
// every device and commits it, the load reads the committed nodes. A device whose bytes differ
// from the ones the plan was made for keeps the planned node, because the load only knows the
// bytes of the devices that already allocated something and planning from those would place the
// machine from partial information.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_cuda_numa_plan {

// an assignment within this factor of the best one counts as tied, and the lexicographically
// smallest tied assignment wins: measurement noise must not decide the placement
static constexpr double TIE = 1.02;

// number of assignments per device: up to this many assignments are enumerated, above it the
// solver starts at the per-device fastest node and moves one critical device per round
static constexpr uint64_t EXHAUSTIVE = 256;

struct result {
    std::vector<int>     node;        // [device] node of every device
    std::vector<int64_t> node_bytes;  // [node] bytes the placement puts on every node
    double               cost = 0.0;  // estimated time of the placement, in byte*ms
    double               cost_fast = 0.0;  // every device on its own fastest node
    double               cost_local = 0.0; // every device on the node of its PCI slot
    bool                 have_local = false; // cost_local is a slot-local placement
};

// estimated time of an assignment, -1 when the assignment uses an unusable (device, node) pair
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

// does the assignment fit into the node capacities? used receives the bytes per node
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

// node with the lowest read time of one device, lowest index on a tie, -1 when the device
// cannot read any node
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

// Plan the node of every device.
//
//   n_dev      number of devices
//   n_nodes    number of host NUMA nodes
//   dev_bytes  [device] host bytes the device reads
//   node_cap   [node] bytes the node may hold
//   t_ms       [device][node] time the device needs to read a probe buffer on the node while
//              every other placed device reads at the same time, negative when unusable
//   topo_node  [device] node of the PCI slot of the device, -1 when the platform has no
//              PCI to NUMA topology; only used for the cost_local comparison
//
// Returns 1 and fills `out` on success, 0 when there is no device or no node at all, and -1
// when no assignment fits into the node capacities (out is then left empty and the caller
// reports the total need against the total capacity).
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

    // every device takes its fastest node; a device without host bytes keeps the lowest
    // usable node index, which is the lexicographically smallest assignment for it
    std::vector<int> base(n_dev, -1);
    for (size_t d = 0; d < n_dev; d++) {
        base[d] = fastest_of(d, n_nodes, t_ms);
        if (base[d] < 0) {
            return -1; // the device cannot read any node
        }
    }

    // the devices that read something take part in the decision, the others only need a node
    std::vector<size_t> part;
    for (size_t d = 0; d < n_dev; d++) {
        if (dev_bytes[d] > 0) {
            part.push_back(d);
        }
    }

    // comparison assignments for the log: every device on its own fastest node, every device
    // on the node of its PCI slot
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

    // number of assignments of the reading devices, saturating at the enumeration limit
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
        // enumerate every assignment in lexicographic order
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
        // then take the first assignment in lexicographic order that is within the tie band
        if (best_cost >= 0.0) {
            const double limit = best_cost*TIE;
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
                if (cc >= 0.0 && cc <= limit) {
                    best = c;
                    break;
                }
            }
        }
    } else {
        // too many assignments to enumerate: start at the per-device fastest node and move
        // the device that sets the step time to the node that lowers it the most
        std::vector<int> a = base;
        if (!fits_of(a.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
            // the fastest nodes are too full: move the devices of an overfull node, largest
            // bytes first and lowest device index first, to the node they read fastest among
            // the nodes that still have room
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
                    break; // every device is on a node that fits it
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
                    break; // nothing can be moved, the capacity repair fails below
                }
            }
        }

        if (!fits_of(a.data(), n_dev, n_nodes, dev_bytes, node_cap, used)) {
            return -1; // the capacity repair could not make it fit
        }
        const double cur = cost_of(a.data(), n_dev, n_nodes, dev_bytes, t_ms);
        if (cur < 0.0) {
            return -1;
        }
        for (size_t round = 0; round <= part.size(); round++) {
            // devices that set the step time; every one of them may be moved
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
                    // the move that lowers the step time the most wins, the lexicographically
                    // smallest one when several lower it by the same amount
                    if (c < best_c) {
                        best_c = c;
                        best_a = trial;
                    } else if (c == best_c && !best_a.empty() && lex_less(trial, best_a)) {
                        best_a = trial;
                    }
                }
            }
            if (best_a.empty()) {
                break; // no move lowers the step time
            }
            a = best_a;
        }

        best_cost = cost_of(a.data(), n_dev, n_nodes, dev_bytes, t_ms);
        best = a;
    }

    if (best.empty()) {
        return -1; // nothing fits into the node capacities
    }

    // ties are resolved by taking the lexicographically smallest assignment within the band of
    // the best cost, so that measurement noise cannot decide the placement
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

    out->node = lex;
    fits_of(out->node.data(), n_dev, n_nodes, dev_bytes, node_cap, out->node_bytes);
    out->cost = cost_of(out->node.data(), n_dev, n_nodes, dev_bytes, t_ms);
    return 1;
}

// the plan the load uses, once it is committed

// the load may read a few MiB more or less than the plan was made for: the expert store rounds
// every arena up to its expert pitch and every arena is a whole number of pages, and the fit
// figure and the allocation are computed by the same plan function
static constexpr size_t BYTES_TOLERANCE = 1ull << 20; // 1 MiB

struct committed {
    bool                 valid    = false; // a machine plan exists
    bool                 from_fit = false; // it is the plan fit computed and committed for the load
    std::vector<int>     node;     // [device] node index of every device
    std::vector<size_t> bytes;    // [device] host bytes the plan was made for
};

// freeze a plan of the solver as the one the load uses
static void commit_plan(committed * c, const std::vector<int> & node, const std::vector<size_t> & bytes, bool from_fit) {
    c->valid    = true;
    c->from_fit = from_fit;
    c->node     = node;
    c->bytes    = bytes;
}

// node index of one device of the committed plan, -1 when there is no plan or no node for the
// device. bytes_match reports whether have_bytes is the byte count the plan was made for; a
// mismatch does not change the node.
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

} // namespace ggml_cuda_numa_plan