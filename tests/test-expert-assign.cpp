// Expert-to-device assignment (ggml_expert_store_owner and friends): the invariants every
// consumer of the table relies on. No GPU and no model: the table is computed by the same
// ggml_expert_store_plan() the backend calls, with a small artificial profile.

#include "ggml-expert-store.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

static void check_eq(long long got, long long want, const char * what) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %lld, want %lld\n", what, got, want);
        g_fail++;
    }
}

static void write_profile(const std::string & path, const std::vector<std::vector<int64_t>> & counts) {
    FILE * f = fopen(path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "FAIL: cannot write %s\n", path.c_str());
        g_fail++;
        return;
    }
    for (size_t l = 0; l < counts.size(); ++l) {
        for (size_t e = 0; e < counts[l].size(); ++e) {
            fprintf(f, "%zu %zu %lld\n", l, e, (long long) counts[l][e]);
        }
    }
    fclose(f);
}

// one device of a 32-expert model over 4 devices: 8 experts per device
static ggml_expert_store_plan_tensor device_tensor(int layer) {
    ggml_expert_store_plan_tensor t;
    t.layer      = layer;
    t.n_expert   = 8;
    t.expert_base = 0;
    t.blob_bytes = 1024;
    t.blob_pad   = 0;
    t.type_size  = 32;
    return t;
}

static void plan_of(int n_dev, const char * profile_path, int layer) {
    ggml_expert_store_plan_tensor t = device_tensor(layer);
    ggml_expert_store_placement plan;
    ggml_expert_store_plan(&t, 1, n_dev, -1, profile_path, &plan);
}

// every expert is on exactly one device, every device holds cap experts, the local numbers are
// unique inside a device and count ascending with the global number, and the reverse lookup
// answers the same expert
static void check_invariants(int n_dev, int n_layer, int n_expert, const char * tag) {
    ggml_expert_store_assignment_info info;
    check(ggml_expert_store_assignment_info_get(&info), "assignment info");
    check_eq(info.n_dev,    n_dev,   "n_dev");
    check_eq(info.n_layer,  n_layer, "n_layer");
    check_eq(info.n_expert, n_expert,"n_expert");

    const int cap_max = info.cap_max;
    for (int l = 0; l < n_layer; ++l) {
        for (int j = 0; j < n_dev; ++j) {
            std::vector<int> seen((size_t) cap_max, 0);
            int last = -1;
            for (int e = 0; e < n_expert; ++e) {
                const int16_t * row = ggml_expert_store_local_of(j, l);
                check(row != nullptr, "local_of row");
                if (row == nullptr) {
                    continue;
                }
                const int owner = ggml_expert_store_owner(l, e);
                check_eq(owner >= 0 && owner < n_dev ? 1 : 0, 1, "owner in range");
                check_eq(row[e] >= 0 ? 1 : 0, owner == j ? 1 : 0, "local_of agrees with owner");
                if (row[e] >= 0) {
                    check_eq(row[e], last + 1, "local numbers count up in expert order");
                    last = row[e];
                    check_eq(seen[(size_t) row[e]], 0, "local number used once");
                    seen[(size_t) row[e]] = 1;
                    check_eq(ggml_expert_store_global_of(j, l, row[e]), e, "global_of(local_of) == expert");
                }
            }
            int held = 0;
            for (int k = 0; k < cap_max; ++k) {
                held += seen[(size_t) k];
            }
            check(held > 0, "device holds experts");
            check_eq(ggml_expert_store_global_of(j, l, cap_max), -1, "global_of past the quota");
            check_eq(ggml_expert_store_global_of(j, l, -1), -1, "global_of of a negative slot");
        }
    }
    (void) tag;
}

static std::vector<std::vector<int16_t>> dump_table(int n_dev, int n_layer, int n_expert) {
    std::vector<std::vector<int16_t>> all;
    for (int j = 0; j < n_dev; ++j) {
        for (int l = 0; l < n_layer; ++l) {
            const int16_t * row = ggml_expert_store_local_of(j, l);
            if (row == nullptr) {
                all.push_back({});
                continue;
            }
            all.emplace_back(row, row + n_expert);
        }
    }
    return all;
}

int main() {
    // ---- profile driven assignment: 4 devices, 32 experts, 4 layers, 8 hot experts per layer
    const int n_dev    = 4;
    const int n_expert = 32;
    const int n_layer  = 4;
    const int cap      = 8;

    std::vector<std::vector<int64_t>> counts((size_t) n_layer, std::vector<int64_t>((size_t) n_expert, 0));
    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < 8; ++e) {
            counts[(size_t) l][(size_t) e] = 1000 - 10 * e + l;
        }
    }
    write_profile("test-expert-assign-profile.txt", counts);
    plan_of(n_dev, "test-expert-assign-profile.txt", 0);
    check_invariants(n_dev, n_layer, n_expert, "profile");

    ggml_expert_store_assignment_info info;
    ggml_expert_store_assignment_info_get(&info);
    check(info.has_profile, "assignment used the profile");

    // the hottest expert goes to the first device, the eight hot experts are spread over the
    // devices and no device holds more than its quota
    check_eq(ggml_expert_store_owner(0, 0), 0, "hottest expert on device 0");
    check_eq(ggml_expert_store_owner(0, 1), 1, "second hottest on device 1");
    check_eq(ggml_expert_store_owner(0, 3), 3, "fourth hottest on device 3");
    for (int j = 0; j < n_dev; ++j) {
        const int16_t * row = ggml_expert_store_local_of(j, 0);
        int n_local = 0;
        for (int e = 0; e < n_expert; ++e) {
            if (row[e] >= 0) {
                n_local++;
            }
        }
        check_eq(n_local, cap, "experts per device");
    }
    // the hot experts must not all sit on one device
    int hot_on_dev0 = 0;
    for (int e = 0; e < 8; ++e) {
        hot_on_dev0 += ggml_expert_store_owner(0, e) == 0;
    }
    check(hot_on_dev0 < 8, "the hot experts are spread over the devices");

    // ---- same counts, different file name: the table is computed again and must be identical
    write_profile("test-expert-assign-profile-2.txt", counts);
    const std::vector<std::vector<int16_t>> first = dump_table(n_dev, n_layer, n_expert);
    plan_of(n_dev, "test-expert-assign-profile-2.txt", 0);
    const std::vector<std::vector<int16_t>> second = dump_table(n_dev, n_layer, n_expert);
    check(first.size() == second.size(), "table size");
    bool same = first.size() == second.size();
    for (size_t i = 0; same && i < first.size(); ++i) {
        same = first[i] == second[i];
    }
    check(same, "the same counts give the same table");
    // and the unchanged input keeps the table of the first computation
    plan_of(n_dev, "test-expert-assign-profile-2.txt", 0);
    check(dump_table(n_dev, n_layer, n_expert) == second, "the cached table is reused");

    // ---- no profile: the contiguous split, the only placement that can be right without counts
    {
        ggml_expert_store_plan_tensor t = device_tensor(0);
        t.n_expert = n_expert; // one unsplit tensor, so the global expert count is known
        ggml_expert_store_placement plan;
        ggml_expert_store_plan(&t, 1, n_dev, -1, nullptr, &plan);
    }
    ggml_expert_store_assignment_info_get(&info);
    check(!info.has_profile, "no profile without --expert-profile");
    check_invariants(n_dev, 1, n_expert, "contiguous");
    int base = 0;
    for (int j = 0; j < n_dev; ++j) {
        const int16_t * row = ggml_expert_store_local_of(j, 0);
        for (int e = 0; e < n_expert; ++e) {
            const bool local = e >= base && e < base + cap;
            check_eq(ggml_expert_store_owner(0, e) == j ? 1 : 0, local ? 1 : 0, "contiguous owner");
            if (row != nullptr) {
                check_eq(row[e], local ? e - base : -1, "contiguous local number");
            }
        }
        base += cap;
    }

    // ---- extremes: all counts on one expert, and a layer without any count
    std::vector<std::vector<int64_t>> extreme(2, std::vector<int64_t>((size_t) n_expert, 0));
    extreme[0][5] = 1000000;
    write_profile("test-expert-assign-profile-extreme.txt", extreme);
    plan_of(n_dev, "test-expert-assign-profile-extreme.txt", 1);
    check_invariants(n_dev, 2, n_expert, "extreme");
    check_eq(ggml_expert_store_owner(0, 5), 0, "the only busy expert goes to device 0");
    const int16_t * row0 = ggml_expert_store_local_of(0, 0);
    check(row0 != nullptr, "the busy layer has a table");
    if (row0 != nullptr) {
        check_eq(row0[5], 0, "the busy expert is the first local expert of device 0");
    }
    const int16_t * row1 = ggml_expert_store_local_of(1, 1);
    check(row1 != nullptr, "the empty layer has a table");
    if (row1 != nullptr) {
        int n_local = 0;
        for (int e = 0; e < n_expert; ++e) {
            n_local += row1[e] >= 0;
        }
        check_eq(n_local, cap, "the empty layer still fills every device");
    }

    // ---- no assignment of a layer or device that does not exist
    check_eq(ggml_expert_store_owner(99, 0),      -1, "owner of an unknown layer");
    check_eq(ggml_expert_store_owner(0, 999),     -1, "owner of an unknown expert");
    check(ggml_expert_store_local_of(n_dev, 0) == nullptr, "local_of of an unknown device");

    // ---- the active table: the contiguous split until the loader activates the proposal, the
    // proposal after that, the contiguous split again after a reset. The expert maps of the nodes
    // (ggml_expert_store_node_lut) follow it.
    plan_of(n_dev, "test-expert-assign-profile.txt", 0); // proposal: the profile LPT, 4 layers x 32 experts
    {
        check(!ggml_expert_store_active_is_permuted(), "contiguous until activated");
        const uint64_t v0 = ggml_expert_store_assignment_version();
        check(v0 > 0, "the active table has a version");
        int a_dev = 0;
        int a_layer = 0;
        int a_expert = 0;
        check(ggml_expert_store_active_dims(&a_dev, &a_layer, &a_expert), "active dims");
        check_eq(a_dev, n_dev, "active n_dev");
        check_eq(a_layer, n_layer, "active n_layer");
        check_eq(a_expert, n_expert, "active n_expert");

        // contiguous rows: device j holds [j*cap, (j+1)*cap)
        for (int j = 0; j < n_dev; ++j) {
            for (int l = 0; l < n_layer; ++l) {
                const int16_t * row = ggml_expert_store_active_local_of(j, l);
                check(row != nullptr, "active row");
                for (int e = 0; e < n_expert && row != nullptr; ++e) {
                    const bool local = e >= j * cap && e < (j + 1) * cap;
                    check_eq(row[e], local ? e - j * cap : -1, "active contiguous local number");
                    check_eq(ggml_expert_store_active_owner(l, e) == j ? 1 : 0, local ? 1 : 0, "active contiguous owner");
                }
                for (int kk = 0; kk < cap; ++kk) {
                    check_eq(ggml_expert_store_active_global_of(j, l, kk), j * cap + kk, "active contiguous global_of");
                }
            }
        }

        // the map of a node while contiguous: its segment, the same row on every call; the map
        // of a node without expert parallelism is the identity
        const int16_t * seg = ggml_expert_store_node_lut(true, 1, 0, cap, cap, n_expert);
        check(seg != nullptr, "segment row");
        for (int e = 0; e < n_expert && seg != nullptr; ++e) {
            check_eq(seg[e], e >= cap && e < 2 * cap ? e - cap : -1, "segment row entry");
        }
        check(ggml_expert_store_node_lut(true, 1, 0, cap, cap, n_expert) == seg, "the segment row is kept");
        const int16_t * idn = ggml_expert_store_node_lut(false, 0, 0, 0, n_expert, n_expert);
        check(idn != nullptr, "identity row");
        for (int e = 0; e < n_expert && idn != nullptr; ++e) {
            check_eq(idn[e], e, "identity row entry");
        }

        // activate: the active rows are the proposal, the maps of the expert parallel nodes are them
        ggml_expert_store_activate_assignment();
        check(ggml_expert_store_active_is_permuted(), "permuted after activation");
        check(ggml_expert_store_assignment_version() > v0, "activation bumps the version");
        bool differs = false;
        for (int j = 0; j < n_dev; ++j) {
            for (int l = 0; l < n_layer; ++l) {
                const int16_t * act = ggml_expert_store_active_local_of(j, l);
                const int16_t * pro = ggml_expert_store_local_of(j, l);
                check(act != nullptr && pro != nullptr, "active and proposal rows");
                for (int e = 0; e < n_expert && act != nullptr && pro != nullptr; ++e) {
                    check_eq(act[e], pro[e], "active row == proposal row");
                    check_eq(ggml_expert_store_active_owner(l, e), ggml_expert_store_owner(l, e), "active owner == proposal owner");
                    differs = differs || act[e] != (e >= j * cap && e < (j + 1) * cap ? e - j * cap : -1);
                }
                for (int kk = 0; kk < cap; ++kk) {
                    check_eq(ggml_expert_store_active_global_of(j, l, kk), ggml_expert_store_global_of(j, l, kk), "active global_of == proposal");
                }
                check(ggml_expert_store_node_lut(true, j, l, j * cap, cap, n_expert) == act, "node map is the active row");
            }
        }
        check(differs, "the profile permutes the experts");
        // a node without expert parallelism keeps the identity whatever the active table
        check(ggml_expert_store_node_lut(false, 0, 0, 0, n_expert, n_expert) == idn, "identity row unchanged");

        // a plan without a profile (another device allocating) leaves the active table alone
        {
            ggml_expert_store_plan_tensor t = device_tensor(0);
            ggml_expert_store_placement plan;
            ggml_expert_store_plan(&t, 1, n_dev, -1, nullptr, &plan);
        }
        check(ggml_expert_store_active_is_permuted(), "a later plan keeps the activated table");
        check(ggml_expert_store_active_dims(&a_dev, &a_layer, &a_expert) && a_layer == n_layer, "a later plan keeps the dims");

        // reset: back to the contiguous split, the segment rows again
        const uint64_t v1 = ggml_expert_store_assignment_version();
        ggml_expert_store_reset_assignment();
        check(!ggml_expert_store_active_is_permuted(), "contiguous after reset");
        check(ggml_expert_store_assignment_version() > v1, "reset bumps the version");
        for (int j = 0; j < n_dev; ++j) {
            const int16_t * row = ggml_expert_store_active_local_of(j, 0);
            for (int e = 0; e < n_expert && row != nullptr; ++e) {
                const bool local = e >= j * cap && e < (j + 1) * cap;
                check_eq(row[e], local ? e - j * cap : -1, "contiguous again after reset");
            }
        }
        check(ggml_expert_store_node_lut(true, 1, 0, cap, cap, n_expert) == seg, "the segment row again after reset");
    }

    if (g_fail == 0) {
        printf("expert assignment tests passed\n");
        return 0;
    }
    fprintf(stderr, "%d checks failed\n", g_fail);
    return 1;
}