// Expert-to-device assignment (ggml_expert_store_owner and friends): the invariants every
// consumer of the table relies on. No GPU and no model: the table is computed by the same
// ggml_expert_store_plan() the backend calls, with a small artificial profile.

#include "ggml-expert-store.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static int g_fail = 0;

// memory operations of a store that lives in host memory: the placement check only needs the
// bytes to arrive where the store puts them
static void * fake_alloc_device(void * user, size_t size) {
    (void) user;
    return malloc(size);
}

static void fake_free_device(void * user, void * ptr) {
    (void) user;
    free(ptr);
}

static void * fake_alloc_host(void * user, size_t size, void ** dev_ptr) {
    (void) user;
    void * p = malloc(size);
    *dev_ptr = p;
    return p;
}

static void fake_free_host(void * user, void * ptr) {
    (void) user;
    free(ptr);
}

static void fake_copy(void * user, void * dst, const void * src, size_t size) {
    (void) user;
    memcpy(dst, src, size);
}

static void fake_table_upload(void * user, void * dst, const char * const * src, size_t n) {
    (void) user;
    memcpy(dst, src, n * sizeof(char *));
}

// the bytes of global expert g: a pattern that differs between experts in every window
static void fill_expert(std::vector<uint8_t> & buf, int g) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t) (g * 37 + i * 11 + (i >> 8));
    }
}

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

    // ---- load positions: the device segments follow each other in device order, segment j
    // holds the experts of device j in local order, every position is used once
    {
        int n_dev_q = 0;
        int n_layer_q = 0;
        int n_expert_q = 0;
        check(ggml_expert_store_assignment_dims(&n_dev_q, &n_layer_q, &n_expert_q), "assignment dims");
        check_eq(n_dev_q, n_dev, "dims: devices");
        check_eq(n_layer_q, n_layer, "dims: layers");
        check_eq(n_expert_q, n_expert, "dims: experts");
        for (int j = 0; j < n_dev; ++j) {
            check_eq(ggml_expert_store_assignment_cap(j), cap, "cap of every device");
        }
        check_eq(ggml_expert_store_assignment_cap(n_dev), -1, "cap of an unknown device");
        for (int l = 0; l < n_layer; ++l) {
            std::vector<int> seen((size_t) n_expert, 0);
            for (int e = 0; e < n_expert; ++e) {
                const int64_t pos   = ggml_expert_store_load_position(l, e);
                const int     owner = ggml_expert_store_owner(l, e);
                const int16_t * row = ggml_expert_store_local_of(owner, l);
                check(pos >= 0 && pos < n_expert, "load position in range");
                if (pos < 0 || pos >= n_expert || row == nullptr) {
                    continue;
                }
                check_eq(pos, (long long) owner * cap + row[e], "load position = segment start + local number");
                check_eq(seen[(size_t) pos], 0, "load position used once");
                seen[(size_t) pos] = 1;
            }
        }
        check_eq(ggml_expert_store_load_position(n_layer, 0), -1, "load position of an unknown layer");
    }

    // ---- the device of a slice of the expert dim
    check_eq(ggml_expert_store_slice_device(0, 8),   0, "slice of device 0");
    check_eq(ggml_expert_store_slice_device(8, 8),   1, "slice of device 1");
    check_eq(ggml_expert_store_slice_device(24, 8),  3, "slice of device 3");
    check_eq(ggml_expert_store_slice_device(0, 32), -1, "the whole expert dim");
    check_eq(ggml_expert_store_slice_device(1, 8),  -2, "a shifted slice is no segment");
    check_eq(ggml_expert_store_slice_device(0, 9),  -2, "a longer slice is no segment");
    check_eq(ggml_expert_store_slice_device(32, 8), -2, "a slice past the end is no segment");

    // ---- the active table before any activation is the contiguous split, whatever the proposal
    // says: that is the layout of the memory until a loader placed the experts by the proposal
    check_eq(ggml_expert_store_assignment_version(), 0, "no activation yet");
    {
        bool proposal_differs = false;
        for (int l = 0; l < n_layer; ++l) {
            for (int e = 0; e < n_expert; ++e) {
                check_eq(ggml_expert_store_active_owner(l, e), e / cap, "active owner is contiguous before activation");
                proposal_differs = proposal_differs || ggml_expert_store_owner(l, e) != e / cap;
            }
            for (int j = 0; j < n_dev; ++j) {
                const int16_t * row = ggml_expert_store_active_local_of(j, l);
                check(row != nullptr, "active local_of row");
                for (int e = 0; row != nullptr && e < n_expert; ++e) {
                    check_eq(row[e], e / cap == j ? e % cap : -1, "active local number is contiguous before activation");
                }
                for (int k = 0; k < cap; ++k) {
                    check_eq(ggml_expert_store_active_global_of(j, l, k), j * cap + k, "active global_of is contiguous before activation");
                }
            }
        }
        check(proposal_differs, "the proposal of the profile differs from the contiguous split");
    }

    // ---- activation: the proposal becomes the active table
    ggml_expert_store_activate_assignment();
    check_eq(ggml_expert_store_assignment_version(), 1, "first activation");
    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            check_eq(ggml_expert_store_active_owner(l, e), ggml_expert_store_owner(l, e), "active owner == proposal after activation");
        }
        for (int j = 0; j < n_dev; ++j) {
            const int16_t * a = ggml_expert_store_active_local_of(j, l);
            const int16_t * p = ggml_expert_store_local_of(j, l);
            check(a != nullptr && p != nullptr && memcmp(a, p, (size_t) n_expert * sizeof(int16_t)) == 0,
                "active local_of == proposal after activation");
            for (int k = 0; k < cap; ++k) {
                check_eq(ggml_expert_store_active_global_of(j, l, k), ggml_expert_store_global_of(j, l, k), "active global_of == proposal after activation");
                check_eq(ggml_expert_store_active_global_of_slice(j * cap, cap, l, k), ggml_expert_store_active_global_of(j, l, k),
                    "global_of_slice names the device by its segment");
            }
        }
    }
    check_eq(ggml_expert_store_active_global_of_slice(0, n_expert, 0, 5), 5, "the whole dim maps local to itself");
    check_eq(ggml_expert_store_active_global_of_slice(1, 8, 0, 0),       -1, "a slice that is no segment has no global expert");
    check_eq(ggml_expert_store_active_global_of_slice(8, 8, 0, 8),       -1, "a local number past the segment has no global expert");
    ggml_expert_store_activate_assignment();
    check_eq(ggml_expert_store_assignment_version(), 2, "every activation counts");

    // ---- placement check: a store of device 1 receives the segment the loader lays out for it
    // (the experts of device 1 in local order); the loader registers the hash of every global
    // expert, and the check maps local -> global through the active table
    {
        ggml_expert_store_backend be;
        be.user         = nullptr;
        be.alloc_device = fake_alloc_device;
        be.free_device  = fake_free_device;
        be.alloc_host   = fake_alloc_host;
        be.free_host    = fake_free_host;
        be.copy_h2d     = fake_copy;
        be.copy_d2h     = fake_copy;
        be.table_upload = fake_table_upload;

        ggml_expert_store * store = ggml_expert_store_init(&be, /*device =*/ 1, nullptr);
        const int64_t blob = 1024;
        std::vector<int64_t> slots((size_t) cap);
        std::iota(slots.begin(), slots.end(), 0);
        const int id = ggml_expert_store_add_tensor(store, /*layer =*/ 0, /*kind =*/ 7, /*expert_base =*/ 8, cap,
            (size_t) blob, 0, 32, slots.data(), cap, -1);
        check(id >= 0, "store tensor of device 1");

        std::vector<uint8_t> buf((size_t) blob);
        std::vector<uint8_t> seg((size_t) cap * blob);
        for (int k = 0; k < cap; ++k) {
            fill_expert(buf, ggml_expert_store_active_global_of(1, 0, k));
            memcpy(seg.data() + (size_t) k * blob, buf.data(), (size_t) blob);
        }
        ggml_expert_store_write(store, id, seg.data(), 0, seg.size());

        const char * name = "blk.0.ffn_down_exps.weight";
        for (int g = 0; g < n_expert; ++g) {
            fill_expert(buf, g);
            ggml_expert_store_set_source_hash(name, g, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        }

        int64_t n_checked = 0;
        int64_t n_skipped = 0;
        check(ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped), "the placement check passes");
        check_eq(n_checked, cap, "every local expert checked");
        check_eq(n_skipped, 0, "nothing skipped");

        check(ggml_expert_store_verify_placement(store, id, "blk.0.ffn_up_exps.weight", &n_checked, &n_skipped),
            "a tensor nobody registered is not a failure");
        check_eq(n_checked, 0, "nothing checked without source hashes");
        check_eq(n_skipped, cap, "every expert skipped without source hashes");

        // the file's expert at one local number is another one than the active table says
        const int g_wrong = ggml_expert_store_active_global_of(1, 0, 3);
        fill_expert(buf, g_wrong + 100);
        ggml_expert_store_set_source_hash(name, g_wrong, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        check(!ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped), "a misplaced expert fails the placement check");

        ggml_expert_store_clear_source_hashes();
        check(ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped) && n_skipped == cap,
            "cleared hashes: every expert skipped");

        ggml_expert_store_free(store);
    }

    // ---- the expert hash: head, middle and tail windows of a large expert, all bytes of a small one
    {
        std::vector<uint8_t> a((size_t) 100 * 1024);
        fill_expert(a, 3);
        std::vector<uint8_t> b = a;
        const uint64_t h = ggml_expert_store_hash_expert(a.data(), a.size());
        check(h == ggml_expert_store_hash_expert(b.data(), b.size()), "the hash is deterministic");
        b[0] ^= 1;
        check(h != ggml_expert_store_hash_expert(b.data(), b.size()), "the head window counts");
        b = a;
        b[b.size() - 1] ^= 1;
        check(h != ggml_expert_store_hash_expert(b.data(), b.size()), "the tail window counts");
        b = a;
        b[b.size() / 2] ^= 1;
        check(h != ggml_expert_store_hash_expert(b.data(), b.size()), "the middle window counts");
        b = a;
        b[(size_t) 40 * 1024] ^= 1;
        check(h == ggml_expert_store_hash_expert(b.data(), b.size()), "bytes between the windows are not hashed (windowed check)");

        std::vector<uint8_t> s((size_t) 1000);
        fill_expert(s, 5);
        const uint64_t hs = ggml_expert_store_hash_expert(s.data(), s.size());
        s[700] ^= 1;
        check(hs != ggml_expert_store_hash_expert(s.data(), s.size()), "every byte of a small expert counts");
        const uint8_t zeros[16] = { 0 };
        check(ggml_expert_store_hash_expert(zeros, 8) != ggml_expert_store_hash_expert(zeros, 16), "the length is part of the hash");
    }

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

    if (g_fail == 0) {
        printf("expert assignment tests passed\n");
        return 0;
    }
    fprintf(stderr, "%d checks failed\n", g_fail);
    return 1;
}