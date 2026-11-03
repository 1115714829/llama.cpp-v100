

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







const double T6[6*2] = {
     88.0, 150.0,
     85.0, 155.0,
    145.0,  90.0,
     82.0, 200.0,
     80.0, 195.0,
    150.0,  87.0,
};

const int TOPO6[6] = { 0, 0, 0, 1, 1, 1 };


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

    if (!expect(__func__, r.node, { 0, 0, 1, 0, 0, 1 })) {
        return false;
    }
    if (r.node_bytes[0] != (int64_t) 4*GiB || r.node_bytes[1] != (int64_t) 2*GiB) {
        fprintf(stderr, "%s: node bytes %lld/%lld\n", __func__,
            (long long) r.node_bytes[0], (long long) r.node_bytes[1]);
        return false;
    }


    if (r.cost >= r.cost_local || !r.have_local) {
        fprintf(stderr, "%s: cost %f is not better than the slot local %f\n", __func__, r.cost, r.cost_local);
        return false;
    }

    if (r.cost > r.cost_fast) {
        fprintf(stderr, "%s: cost %f is worse than every device on its fastest node %f\n",
            __func__, r.cost, r.cost_fast);
        return false;
    }
    return true;
}



bool test_capacity(void) {
    const size_t              n_dev   = 6;
    const std::vector<size_t> bytes   = { GiB, GiB, GiB, GiB, GiB, GiB };

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

    if (!expect(__func__, first.node, { 0, 1 })) {
        return false;
    }
    return true;
}





bool test_tie_spreads_the_devices(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[2] = { (size_t) (30249*MiB), (size_t) (30243*MiB) };

    const double  t[2*2] = { 4.30, 4.32, 3.30, 3.31 };

    const int     topo[2] = { 0, 0 };

    const int64_t cap[2]  = { (int64_t) 96*1024*MiB, (int64_t) 96*1024*MiB };

    result r;
    if (ggml_cuda_numa_plan::plan(2, 2, bytes, cap, t, topo, &r) != 1) {
        fprintf(stderr, "%s: the plan failed\n", __func__);
        return false;
    }
    if (!expect(__func__, r.node, { 0, 1 })) {
        return false;
    }
    if (r.node[0] == r.node[1]) {
        fprintf(stderr, "%s: both arenas are on node %d\n", __func__, r.node[0]);
        return false;
    }


    const int piled[2] = { 0, 0 };
    const double c_piled = ggml_cuda_numa_plan::cost_of(piled, 2, 2, bytes, t);
    if (c_piled < 0.0 || r.cost > c_piled*ggml_cuda_numa_plan::TIE) {
        fprintf(stderr, "%s: the spread costs %f, the piled placement %f\n", __func__, r.cost, c_piled);
        return false;
    }
    return true;
}





bool test_six_cards_ac9221(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  n_dev = 6;
    const size_t  bytes[6] = { (size_t) (1897*MiB), (size_t) (1832*MiB), (size_t) (1994*MiB),
                                (size_t) (1890*MiB), (size_t) (1841*MiB), (size_t) (2007*MiB) };
    const double  t[6*2] = { 8.3, 13.6,   8.1, 13.5,   7.7, 15.6,
                             13.4, 8.0,  13.1, 8.3,   15.2, 8.6 };
    const int     topo[6] = { 0, 0, 0, 1, 1, 1 };
    const int64_t cap[2]  = { (int64_t) (44762*MiB), (int64_t) (44847*MiB) };

    result r;
    if (ggml_cuda_numa_plan::plan(n_dev, 2, bytes, cap, t, topo, &r) != 1) {
        fprintf(stderr, "%s: the plan failed\n", __func__);
        return false;
    }
    if (!expect(__func__, r.node, { 0, 0, 0, 1, 1, 1 })) {
        return false;
    }
    if (r.node_bytes[0] != (int64_t) (5723*MiB) || r.node_bytes[1] != (int64_t) (5738*MiB)) {
        fprintf(stderr, "%s: node bytes %lld/%lld MiB\n", __func__,
            (long long) (r.node_bytes[0]/MiB), (long long) (r.node_bytes[1]/MiB));
        return false;
    }
    return true;
}



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



bool test_descent(void) {
    const size_t              n_dev   = 3;
    const size_t              n_nodes = 7;
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





static bool test_pinned_room(void) {
    const int64_t MiB = 1024ll*1024ll;
    const int64_t margin_proc = 4*1024*MiB;

    const int64_t cap8 = 44847*MiB;
    const int64_t cap0 = 44762*MiB;
    const int64_t high  = 1024*MiB;



    if (!(43835*MiB > ggml_cuda_numa_plan::pinned_room(cap8, high, margin_proc))) {
        fprintf(stderr, "%s: the 3 card tier does not fill node 8\n", __func__);
        return false;
    }

    if (34079*MiB > ggml_cuda_numa_plan::pinned_room(cap0, high, margin_proc)) {
        fprintf(stderr, "%s: the 6 card tier fills node 0\n", __func__);
        return false;
    }
    return true;
}






static bool test_balanced_one_card_ac9221(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[1] = { (size_t) (71907*MiB) };
    const int     local[1] = { 0 };
    const int64_t cap[2] = { (int64_t) (40744*MiB), (int64_t) (40990*MiB) };
    const int64_t fixed_w = 644*MiB;
    const int64_t fixed_p = 163*MiB;

    const int     node_w  = 1;
    const int     node_p  = 0;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(1, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, node_w, node_p, &seg) != 1) {
        fprintf(stderr, "%s: the balanced placement failed\n", __func__);
        return false;
    }

    int64_t used[2] = { fixed_p, fixed_w };
    for (const ggml_cuda_numa_plan::segment_entry & e : seg[0]) {
        used[e.node] += e.bytes;
    }


    if (seg[0].size() < 2) {
        fprintf(stderr, "%s: the complement of the card is not split over the nodes\n", __func__);
        return false;
    }


    const int64_t diff = used[0] > used[1] ? used[0] - used[1] : used[1] - used[0];
    if (diff > used[0]/100) {
        fprintf(stderr, "%s: the nodes carry %lld and %lld MiB\n", __func__,
            (long long) (used[0]/MiB), (long long) (used[1]/MiB));
        return false;
    }
    for (size_t n = 0; n < 2; n++) {
        if (used[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld\n", __func__, n,
                (long long) (used[n]/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }
    return true;
}




static bool test_table_node_without_arena(void) {
    const int64_t MiB = 1024ll*1024ll;

    const size_t bytes[2] = { (size_t) (29945*MiB), (size_t) (29945*MiB) };
    const int    local[2] = { 0, 0 };
    const int64_t cap[2] = { (int64_t) (61132*MiB), (int64_t) (61132*MiB) };
    const int64_t table  = 27466*MiB;


    if (ggml_cuda_numa_plan::table_node(2, bytes, local, 2, cap, table) != 1) {
        fprintf(stderr, "%s: the table did not take the node without an arena\n", __func__);
        return false;
    }



    const int local_both[2] = { 0, 1 };
    const int got = ggml_cuda_numa_plan::table_node(2, bytes, local_both, 2, cap, table);
    if (got != 0 && got != 1) {
        fprintf(stderr, "%s: the table took no node when every node carries an arena\n", __func__);
        return false;
    }
    return true;
}



static bool test_table_node_shares_with_arena(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t bytes[3] = { (size_t) (15503*MiB), (size_t) (15644*MiB), (size_t) (15644*MiB) };
    const int    local[3] = { 0, 0, 1 };
    const int64_t cap[2] = { (int64_t) (44762*MiB), (int64_t) (44847*MiB) };
    const int64_t table  = 27466*MiB;




    const int got = ggml_cuda_numa_plan::table_node(3, bytes, local, 2, cap, table);
    if (got != 1) {
        fprintf(stderr, "%s: the table went to node %d instead of node 1\n", __func__, got);
        return false;
    }
    return true;
}






static bool test_table_node_of_uses_the_placement(void) {
    const int64_t MiB = 1024ll*1024ll;
    const int64_t cap[2] = { (int64_t) (61132*MiB), (int64_t) (61132*MiB) };


    std::vector<int64_t> placed(2, 0);
    placed[0] = 29945*MiB;
    placed[1] = 29945*MiB;
    const int w = ggml_cuda_numa_plan::table_node_of(placed, 2, cap, 27466*MiB);
    if (w < 0 || w > 1) {
        fprintf(stderr, "%s: the table took node %d of two\n", __func__, w);
        return false;
    }


    std::vector<int64_t> slot(2, 0);
    slot[0] = 59890*MiB;
    if (ggml_cuda_numa_plan::table_node_of(slot, 2, cap, 27466*MiB) != 1) {
        fprintf(stderr, "%s: the table did not take the node without an arena\n", __func__);
        return false;
    }
    return true;
}

} // namespace











static bool test_balanced_three_cards(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  n_dev = 3;
    const size_t  bytes[3] = { (size_t) (15503*MiB), (size_t) (15644*MiB), (size_t) (15644*MiB) };

    const int     local[3] = { 0, 0, 1 };

    const int64_t cap[2] = { (int64_t) (46810*MiB), (int64_t) (46895*MiB) };

    const int64_t fixed_w = 27464*MiB;
    const int64_t fixed_p = 83*MiB;
    const int     node_w  = 1;
    const int     node_p  = 0;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(n_dev, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, node_w, node_p, &seg) != 1) {
        fprintf(stderr, "%s: the balanced placement failed\n", __func__);
        return false;
    }

    int64_t used[2] = { 0, 0 };
    used[node_w] += fixed_w;
    used[node_p] += fixed_p;
    for (size_t d = 0; d < n_dev; d++) {
        for (const ggml_cuda_numa_plan::segment_entry & e : seg[d]) {
            used[e.node] += e.bytes;
        }
    }


    const int64_t diff = used[0] > used[1] ? used[0] - used[1] : used[1] - used[0];
    if (diff > used[0]/100) {
        fprintf(stderr, "%s: the nodes carry %lld and %lld MiB\n", __func__,
            (long long) (used[0]/MiB), (long long) (used[1]/MiB));
        return false;
    }


    if (seg[2].size() < 2) {
        fprintf(stderr, "%s: the third card is not split over the nodes\n", __func__);
        return false;
    }


    for (size_t n = 0; n < 2; n++) {
        if (used[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld\n", __func__, n,
                (long long) (used[n]/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }
    return true;
}



static bool test_balanced_one_card(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[1] = { (size_t) (60000*MiB) };
    const int     local[1] = { 0 };
    const int64_t cap[2] = { (int64_t) (46810*MiB), (int64_t) (46895*MiB) };
    const int64_t fixed_w = 27464*MiB;
    const int64_t fixed_p = 83*MiB;

    const int     node_w  = 1;
    const int     node_p  = 0;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(1, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, node_w, node_p, &seg) != 1) {
        fprintf(stderr, "%s: the balanced placement failed\n", __func__);
        return false;
    }

    int64_t used[2] = { fixed_p, fixed_w };
    for (const ggml_cuda_numa_plan::segment_entry & e : seg[0]) {
        used[e.node] += e.bytes;
    }


    if (seg[0].size() < 2) {
        fprintf(stderr, "%s: the complement of the card is not split over the nodes\n", __func__);
        return false;
    }


    const int64_t diff = used[0] > used[1] ? used[0] - used[1] : used[1] - used[0];
    if (diff > used[0]/100) {
        fprintf(stderr, "%s: the nodes carry %lld and %lld MiB\n", __func__,
            (long long) (used[0]/MiB), (long long) (used[1]/MiB));
        return false;
    }

    for (size_t n = 0; n < 2; n++) {
        if (used[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld\n", __func__, n,
                (long long) (used[n]/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }
    return true;
}



static bool test_balanced_six_cards(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[6] = { (size_t) (1897*MiB), (size_t) (1832*MiB), (size_t) (1994*MiB),
                                (size_t) (1890*MiB), (size_t) (1841*MiB), (size_t) (2007*MiB) };
    const int     local[6] = { 0, 0, 0, 1, 1, 1 };
    const int64_t cap[2] = { (int64_t) (46810*MiB), (int64_t) (46895*MiB) };
    const int64_t fixed_w = 27464*MiB;
    const int64_t fixed_p = 644*MiB;
    const int     node_w  = 0;
    const int     node_p  = 1;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(6, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, node_w, node_p, &seg) != 1) {
        fprintf(stderr, "%s: the balanced placement failed\n", __func__);
        return false;
    }

    int64_t used[2] = { fixed_w, fixed_p };
    for (size_t d = 0; d < 6; d++) {
        for (const ggml_cuda_numa_plan::segment_entry & e : seg[d]) {
            used[e.node] += e.bytes;
        }
    }
    for (size_t n = 0; n < 2; n++) {
        if (used[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld\n", __func__, n,
                (long long) (used[n]/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }
    return true;
}


static bool test_balanced_does_not_fit(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[1] = { (size_t) (90000*MiB) };
    const int     local[1] = { 0 };
    const int64_t cap[2] = { (int64_t) (46810*MiB), (int64_t) (46895*MiB) };
    const int64_t fixed_w = 27464*MiB;
    const int64_t fixed_p = 83*MiB;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(1, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, 1, 0, &seg) != 0) {
        fprintf(stderr, "%s: the balanced placement did not report the failure\n", __func__);
        return false;
    }
    return true;
}




static bool test_balanced_respects_the_room_of_the_node(void) {
    const int64_t MiB = 1024ll*1024ll;
    const size_t  bytes[1] = { (size_t) (1000*MiB) };
    const int     local[1] = { 0 };
    const int64_t cap[2] = { (int64_t) (100*MiB), (int64_t) (1900*MiB) };
    const int64_t fixed_w = 0;
    const int64_t fixed_p = 0;

    std::vector<std::vector<ggml_cuda_numa_plan::segment_entry>> seg;
    if (ggml_cuda_numa_plan::balanced(1, bytes, nullptr, nullptr, 0, local, 2, cap,
            fixed_w, fixed_p, 0, 1, 0, &seg) != 1) {
        fprintf(stderr, "%s: the placement failed although the bytes fit the nodes\n", __func__);
        return false;
    }

    int64_t used[2] = { 0, 0 };
    for (const ggml_cuda_numa_plan::segment_entry & e : seg[0]) {
        used[e.node] += e.bytes;
    }
    for (size_t n = 0; n < 2; n++) {
        if (used[n] > cap[n]) {
            fprintf(stderr, "%s: node %zu holds %lld MiB of %lld\n", __func__, n,
                (long long) (used[n]/MiB), (long long) (cap[n]/MiB));
            return false;
        }
    }
    return true;
}



static bool test_committed_plan(void) {

    ggml_cuda_numa_plan::committed c;
    const std::vector<int>     node  = { 1, 1, 1, 0, 0, 0 };
    const std::vector<size_t>  bytes = { 1514*1024*1024, 1500*1024*1024, 1494*1024*1024,
                                          1501*1024*1024, 1502*1024*1024, 1490*1024*1024 };
    if (ggml_cuda_numa_plan::committed_node(c, 0, bytes[0], nullptr) != -1) {
        fprintf(stderr, "%s: an empty plan returned a node\n", __func__);
        return false;
    }

    ggml_cuda_numa_plan::commit_plan(&c, node, bytes, true);


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
        !test_tie_spreads_the_devices() ||
        !test_six_cards_ac9221() ||
        !test_does_not_fit()  ||
        !test_descent()       ||
        !test_committed_plan()||
        !test_balanced_three_cards() ||
        !test_balanced_one_card()    ||
        !test_balanced_one_card_ac9221() ||
        !test_balanced_six_cards()   ||
        !test_balanced_does_not_fit() ||
        !test_balanced_respects_the_room_of_the_node() ||
        !test_pinned_room()) {
        return 1;
    }

    if (!test_table_node_without_arena() || !test_table_node_shares_with_arena() ||
        !test_table_node_of_uses_the_placement()) {
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}