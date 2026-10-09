// tests for the NUMA placement solver of the expert host arena (ggml-cuda-numa-plan.h)

#include "ggml-cuda/ggml-cuda-numa-plan.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_cuda_numa_plan::result;

constexpr size_t GiB = 1024ull*1024*1024;

std::string to_string(const std::vector<int> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

bool expect(const char * what, const std::vector<int> & got, const std::vector<int> & want) {
    if (got != want) {
        fprintf(stderr, "%s: got %s, want %s\n", what, to_string(got).c_str(), to_string(want).c_str());
        return false;
    }
    return true;
}

// read time matrix of the 6 card AC922 measurement (t123), one GiB per device, 128 MiB probe
// buffer, in ms. The measured step times were: 0,0,8,0,0,8 -> 90 ms (best), page interleaved
// -> 119, all on node 0 -> 128, 8,8,8,0,0,0 -> 134, slot local 0,0,0,8,8,8 -> 146, all on node 8
// -> 233. The times are per (device, node) under concurrent load, which is why the placement
// that keeps four devices on node 0 beats the slot local one: it uses the total bandwidth of
// the node instead of the local path of every device.
const double T6[6*2] = {
     88.0, 150.0,   // device 0
     85.0, 155.0,   // device 1
    145.0,  90.0,   // device 2
     82.0, 200.0,   // device 3
     80.0, 195.0,   // device 4
    150.0,  87.0,   // device 5
};

const int TOPO6[6] = { 0, 0, 0, 1, 1, 1 };

// every assignment of the six devices within the two node capacities, and the capacity holds
bool test_exhaustive_best(void) {
    const size_t            n_dev   = 6;
    const size_t            n_nodes = 2;
    const std::vector<size_t> bytes  = { GiB, GiB, GiB, GiB, GiB, GiB };
    const int64_t           cap[2]   = { (int64_t) 28*GiB, (int64_t) 28*GiB };

    result r;
    const int ret = ggml_cuda_numa_plan::plan(n_dev, n_nodes, bytes.data(), cap, T6, TOPO6, &r);
    if (ret != 1) {
        fprintf(stderr, "%s: the plan failed (%d)\n", __func__, ret);
        return false;
    }
    // 64 assignments is below the enumeration limit, so the best one is the measured one
    if (!expect(__func__, r.node, { 0, 0, 1, 0, 0, 1 })) {
        return false;
    }
    if (r.node_bytes[0] != (int64_t) 4*GiB || r.node_bytes[1] != (int64_t) 2*GiB) {
        fprintf(stderr, "%s: node bytes %lld/%lld\n", __func__,
            (long long) r.node_bytes[0], (long long) r.node_bytes[1]);
        return false;
    }
    // the estimated step time is the slowest device, and the slot local placement of the same
    // table is worse, which is what the measurement showed
    if (r.cost >= r.cost_local || !r.have_local) {
        fprintf(stderr, "%s: cost %f is not better than the slot local %f\n", __func__, r.cost, r.cost_local);
        return false;
    }
    // every device on its own fastest node is never worse than the chosen plan
    if (r.cost > r.cost_fast) {
        fprintf(stderr, "%s: cost %f is worse than every device on its fastest node %f\n",
            __func__, r.cost, r.cost_fast);
        return false;
    }
    return true;
}

// a node that cannot hold all the devices of the plan forces a different one, and the result
// still respects the capacity
bool test_capacity(void) {
    const size_t              n_dev   = 6;
    const std::vector<size_t> bytes   = { GiB, GiB, GiB, GiB, GiB, GiB };
    // node 0 takes at most two devices
    const int64_t cap[2] = { (int64_t) (2*GiB + GiB/2), (int64_t) 28*GiB };

    result r;
    const int ret = ggml_cuda_numa_plan::plan(n_dev, 2, bytes.data(), cap, T6, TOPO6, &r);
    if (ret != 1) {
        fprintf(stderr, "%s: the plan failed (%d)\n", __func__, ret);
        return false;
    }
    if (!expect(__func__, r.node, { 1, 1, 1, 0, 0, 1 })) {
        return false;
    }
    for (size_t n = 0; n < 2; n++) {
        if (r.node_bytes[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld bytes of %lld\n",
                __func__, n, (long long) r.node_bytes[n], (long long) cap[n]);
            return false;
        }
    }
    return true;
}

// times that differ by one percent make every assignment a tie: the result must be the
// lexicographically smallest one and the same on every call
bool test_tie_is_deterministic(void) {
    const std::vector<size_t> bytes = { GiB, GiB };
    const double  t[2*2] = { 1.00, 0.99, 1.00, 0.99 };
    const int     topo[2] = { 0, 1 };
    const int64_t cap[2]  = { (int64_t) 8*GiB, (int64_t) 8*GiB };

    result first;
    result again;
    for (int i = 0; i < 8; i++) {
        result r;
        if (ggml_cuda_numa_plan::plan(2, 2, bytes.data(), cap, t, topo, &r) != 1) {
            fprintf(stderr, "%s: the plan failed\n", __func__);
            return false;
        }
        if (i == 0) {
            first = r;
        } else if (r.node != first.node) {
            fprintf(stderr, "%s: call %d gave %s, the first call gave %s\n",
                __func__, i, to_string(r.node).c_str(), to_string(first.node).c_str());
            return false;
        }
    }
    if (!expect(__func__, first.node, { 0, 0 })) {
        return false;
    }
    return true;
}

// more bytes than the nodes hold together: the solver reports the failure instead of placing
// the arena anyway
bool test_does_not_fit(void) {
    const std::vector<size_t> bytes = { GiB, GiB };
    const double  t[2*2] = { 1.0, 1.0, 1.0, 1.0 };
    const int     topo[2] = { 0, 0 };
    const int64_t cap[2]  = { (int64_t) GiB/2, (int64_t) GiB/2 };

    result r;
    const int ret = ggml_cuda_numa_plan::plan(2, 2, bytes.data(), cap, t, topo, &r);
    if (ret != -1) {
        fprintf(stderr, "%s: the plan returned %d instead of a failure\n", __func__, ret);
        return false;
    }
    if (!r.node.empty()) {
        fprintf(stderr, "%s: a failed plan returned %s\n", __func__, to_string(r.node).c_str());
        return false;
    }
    return true;
}

// more assignments than the enumeration limit: the solver descends from the per device fastest
// node instead. The result must be legal and the same on every call.
bool test_descent(void) {
    const size_t              n_dev   = 3;
    const size_t              n_nodes = 7; // 7^3 = 343 assignments, above the enumeration limit
    const size_t              limit  = (size_t) ggml_cuda_numa_plan::EXHAUSTIVE;
    const std::vector<size_t> bytes   = { 4*GiB, 2*GiB, 1*GiB };
    std::vector<double>       t(n_dev*n_nodes);
    for (size_t d = 0; d < n_dev; d++) {
        for (size_t n = 0; n < n_nodes; n++) {
            t[d*n_nodes + n] = 50.0 + 10.0*(double) n + (double) d;
        }
    }
    std::vector<int64_t> cap(n_nodes, (int64_t) 8*GiB);
    const int            topo[3] = { 0, 0, 1 };

    result first;
    if (n_nodes*n_nodes*n_nodes <= limit) {
        fprintf(stderr, "%s: %zu assignments are still enumerated\n", __func__, n_nodes*n_nodes*n_nodes);
        return false;
    }
    for (int i = 0; i < 4; i++) {
        result r;
        if (ggml_cuda_numa_plan::plan(n_dev, n_nodes, bytes.data(), cap.data(), t.data(), topo, &r) != 1) {
            fprintf(stderr, "%s: the plan failed\n", __func__);
            return false;
        }
        for (size_t n = 0; n < n_nodes; n++) {
            if (r.node_bytes[n] > cap[n]) {
                fprintf(stderr, "%s: node %zu holds %lld bytes of %lld\n",
                    __func__, n, (long long) r.node_bytes[n], (long long) cap[n]);
                return false;
            }
        }
        if (i == 0) {
            first = r;
            // every device reads node 0 the fastest, and nothing forces it elsewhere
            if (r.node != std::vector<int>({ 0, 0, 0 })) {
                fprintf(stderr, "%s: got %s, want 0,0,0\n", __func__, to_string(r.node).c_str());
                return false;
            }
        } else if (r.node != first.node) {
            fprintf(stderr, "%s: call %d gave %s, the first call gave %s\n",
                __func__, i, to_string(r.node).c_str(), to_string(first.node).c_str());
            return false;
        }
    }
    return true;
}

} // namespace

// the committed plan is what the load reads: a device gets the node of the plan no matter how
// many bytes it has when its arena is allocated
bool test_committed_plan(void) {
    // the placement of the 6 card machine: 0,0,8,0,0,0 is what the machines measure
    ggml_cuda_numa_plan::committed c;
    const std::vector<int>     node  = { 1, 1, 1, 0, 0, 0 };
    const std::vector<size_t>  bytes = { 1514*1024*1024, 1500*1024*1024, 1494*1024*1024,
                                          1501*1024*1024, 1502*1024*1024, 1490*1024*1024 };
    if (ggml_cuda_numa_plan::committed_node(c, 0, bytes[0], nullptr) != -1) {
        fprintf(stderr, "%s: an empty plan returned a node\n", __func__);
        return false;
    }

    ggml_cuda_numa_plan::commit_plan(&c, node, bytes, true);

    // every device gets the node of the plan, with the bytes it reads in the plan
    for (size_t d = 0; d < node.size(); d++) {
        bool     match = false;
        const int got = ggml_cuda_numa_plan::committed_node(c, d, bytes[d], &match);
        if (got != node[d]) {
            fprintf(stderr, "%s: device %zu got node %d, want %d\n", __func__, d, got, node[d]);
            return false;
        }
        if (!match) {
            fprintf(stderr, "%s: device %zu with the planned bytes reports a mismatch\n", __func__, d);
            return false;
        }
    }

    // other bytes than the plan was made for: the node stays, only the mismatch is reported.
    // A load that only knows the bytes of one device must not place the machine again.
    for (size_t d = 0; d < node.size(); d++) {
        bool     match = true;
        const int got = ggml_cuda_numa_plan::committed_node(c, d, bytes[d] + 512*1024*1024, &match);
        if (got != node[d]) {
            fprintf(stderr, "%s: device %zu with other bytes got node %d, want %d\n", __func__, d, got, node[d]);
            return false;
        }
        if (match) {
            fprintf(stderr, "%s: device %zu reads 512 MiB more than planned, no mismatch reported\n", __func__, d);
            return false;
        }
    }

    // a difference inside the tolerance is not a mismatch: the expert store rounds every arena
    // to its expert pitch and every arena is a whole number of pages
    {
        bool match = false;
        const size_t tol = ggml_cuda_numa_plan::BYTES_TOLERANCE;
        if (ggml_cuda_numa_plan::committed_node(c, 0, bytes[0] + tol, &match) != node[0] || !match) {
            fprintf(stderr, "%s: a difference of exactly the tolerance is a mismatch\n", __func__);
            return false;
        }
        if (ggml_cuda_numa_plan::committed_node(c, 0, bytes[0] + tol + 1, &match) != node[0] || match) {
            fprintf(stderr, "%s: a difference past the tolerance is not a mismatch\n", __func__);
            return false;
        }
    }

    // a device the plan does not cover has no node
    if (ggml_cuda_numa_plan::committed_node(c, node.size(), 0, nullptr) != -1) {
        fprintf(stderr, "%s: a device outside the plan returned a node\n", __func__);
        return false;
    }

    return true;
}

int main(void) {
    if (!test_exhaustive_best() ||
        !test_capacity()      ||
        !test_tie_is_deterministic() ||
        !test_does_not_fit()  ||
        !test_descent()       ||
        !test_committed_plan()) {
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}