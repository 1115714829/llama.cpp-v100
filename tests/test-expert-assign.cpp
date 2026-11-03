




#include "ggml.h"
#include "ggml-expert-store.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static int g_fail = 0;



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

    int hot_on_dev0 = 0;
    for (int e = 0; e < 8; ++e) {
        hot_on_dev0 += ggml_expert_store_owner(0, e) == 0;
    }
    check(hot_on_dev0 < 8, "the hot experts are spread over the devices");




    const uint32_t tab_a = ggml_expert_assignment_from_proposal(n_dev, n_layer, n_expert);
    check(tab_a > 0, "a table from the proposal");
    check(ggml_expert_assignment_from_proposal(n_dev + 1, n_layer, n_expert) == 0, "no table from a proposal of other devices");
    check(ggml_expert_assignment_from_proposal(n_dev, n_layer + 1, n_expert) == 0, "no table for more layers than the proposal has");
    {
        int n_dev_q = 0;
        int n_layer_q = 0;
        int n_expert_q = 0;
        check(ggml_expert_assignment_dims(tab_a, &n_dev_q, &n_layer_q, &n_expert_q), "table dims");
        check_eq(n_dev_q, n_dev, "dims: devices");
        check_eq(n_layer_q, n_layer, "dims: layers");
        check_eq(n_expert_q, n_expert, "dims: experts");
        for (int j = 0; j < n_dev; ++j) {
            check_eq(ggml_expert_assignment_cap(tab_a, j), cap, "cap of every device");
        }
        check_eq(ggml_expert_assignment_cap(tab_a, n_dev), -1, "cap of an unknown device");
        for (int l = 0; l < n_layer; ++l) {
            std::vector<int> seen((size_t) n_expert, 0);
            for (int e = 0; e < n_expert; ++e) {
                const int64_t pos   = ggml_expert_assignment_load_position(tab_a, l, e);
                const int     owner = ggml_expert_store_owner(l, e);
                const int16_t * row = ggml_expert_store_local_of(owner, l);
                check_eq(ggml_expert_assignment_owner(tab_a, l, e), owner, "the table owner is the proposal owner");
                check(pos >= 0 && pos < n_expert, "load position in range");
                if (pos < 0 || pos >= n_expert || row == nullptr) {
                    continue;
                }
                check_eq(pos, (long long) owner * cap + row[e], "load position = segment start + local number");
                check_eq(seen[(size_t) pos], 0, "load position used once");
                seen[(size_t) pos] = 1;
            }
            for (int j = 0; j < n_dev; ++j) {
                const int16_t * t_row = ggml_expert_assignment_local_of(tab_a, j, l);
                const int16_t * p_row = ggml_expert_store_local_of(j, l);
                check(t_row != nullptr && p_row != nullptr && memcmp(t_row, p_row, (size_t) n_expert * sizeof(int16_t)) == 0,
                    "the table rows are the proposal rows");
                for (int k = 0; k < cap; ++k) {
                    check_eq(ggml_expert_assignment_global_of(tab_a, j, l, k), ggml_expert_store_global_of(j, l, k), "table global_of == proposal");
                }
            }
        }
        check_eq(ggml_expert_assignment_load_position(tab_a, n_layer, 0), -1, "load position of an unknown layer");
        check_eq(ggml_expert_assignment_load_position(tab_a + 1000, 0, 0), -1, "load position of an unknown table");
        check(!ggml_expert_assignment_is_active(tab_a), "a table is not active before its loader activates it");
        check(ggml_expert_assignment_fingerprint(tab_a) != 0, "the table has a fingerprint");
    }


    check_eq(ggml_expert_store_slice_device(0, 8),   0, "slice of device 0");
    check_eq(ggml_expert_store_slice_device(8, 8),   1, "slice of device 1");
    check_eq(ggml_expert_store_slice_device(24, 8),  3, "slice of device 3");
    check_eq(ggml_expert_store_slice_device(0, 32), -1, "the whole expert dim");
    check_eq(ggml_expert_store_slice_device(1, 8),  -2, "a shifted slice is no segment");
    check_eq(ggml_expert_store_slice_device(0, 9),  -2, "a longer slice is no segment");
    check_eq(ggml_expert_store_slice_device(32, 8), -2, "a slice past the end is no segment");





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
        std::vector<int64_t> slots((size_t) n_expert);
        std::iota(slots.begin(), slots.end(), 0);
        const int id = ggml_expert_store_add_tensor(store, 0, 7, 8, cap,
            (size_t) blob, 0, 32, slots.data(), cap, -1);
        check(id >= 0, "store tensor of device 1");

        std::vector<uint8_t> buf((size_t) blob);
        std::vector<uint8_t> seg((size_t) cap * blob);
        for (int k = 0; k < cap; ++k) {
            fill_expert(buf, ggml_expert_assignment_global_of(tab_a, 1, 0, k));
            memcpy(seg.data() + (size_t) k * blob, buf.data(), (size_t) blob);
        }
        ggml_expert_store_write(store, id, seg.data(), 0, seg.size());

        const char * name = "blk.0.ffn_down_exps.weight";
        for (int g = 0; g < n_expert; ++g) {
            fill_expert(buf, g);
            ggml_expert_store_set_source_hash(name, g, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        }


        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, n_expert);
        ggml_set_name(w, name);
        const ggml_tensor * bind[1] = { w };
        ggml_expert_assignment_activate(tab_a, bind, 1);
        check(ggml_expert_assignment_is_active(tab_a), "active after the activation");
        check_eq(ggml_expert_assignment_of_weight(w), tab_a, "the weight is bound to its table");

        int64_t n_checked = 0;
        int64_t n_skipped = 0;
        check(ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped), "the placement check passes");
        check_eq(n_checked, cap, "every local expert checked");
        check_eq(n_skipped, 0, "nothing skipped");
        const ggml_expert_store_tensor * st = ggml_expert_store_tensor_at(store, id);
        check(st != nullptr && st->assignment == tab_a && (int64_t) st->global_of.size() == cap, "the check records the table of the tensor");
        for (int k = 0; st != nullptr && k < cap && k < (int) st->global_of.size(); ++k) {
            check_eq(st->global_of[(size_t) k], ggml_expert_assignment_global_of(tab_a, 1, 0, k), "the recorded global numbers are the table's");
        }

        check(ggml_expert_store_verify_placement(store, id, "blk.0.ffn_up_exps.weight", &n_checked, &n_skipped),
            "a tensor nobody registered is not a failure");
        check_eq(n_checked, 0, "nothing checked without source hashes");
        check_eq(n_skipped, cap, "every expert skipped without source hashes");


        const int g_wrong = ggml_expert_assignment_global_of(tab_a, 1, 0, 3);
        fill_expert(buf, g_wrong + 100);
        ggml_expert_store_set_source_hash(name, g_wrong, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        check(!ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped), "a misplaced expert fails the placement check");


        ggml_expert_store_clear_source_hashes();
        for (int g = 0; g < n_expert; ++g) {
            fill_expert(buf, g);
            ggml_expert_store_set_source_hash(name, g, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        }
        check(!ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped), "a split slice without a table fails the check");

        ggml_expert_store_clear_source_hashes();
        check(ggml_expert_store_verify_placement(store, id, name, &n_checked, &n_skipped) && n_skipped == cap,
            "cleared hashes: every expert skipped");


        std::vector<int64_t> all_slots((size_t) n_expert);
        std::iota(all_slots.begin(), all_slots.end(), 0);
        const int id_all = ggml_expert_store_add_tensor(store, 1, 7, 0, n_expert,
            (size_t) blob, 0, 32, all_slots.data(), n_expert, -1);
        std::vector<uint8_t> whole((size_t) n_expert * blob);
        const char * name_all = "blk.1.ffn_down_exps.weight";
        for (int g = 0; g < n_expert; ++g) {
            fill_expert(buf, g);
            memcpy(whole.data() + (size_t) g * blob, buf.data(), (size_t) blob);
            ggml_expert_store_set_source_hash(name_all, g, ggml_expert_store_hash_expert(buf.data(), (size_t) blob));
        }
        ggml_expert_store_write(store, id_all, whole.data(), 0, whole.size());
        check(ggml_expert_store_verify_placement(store, id_all, name_all, &n_checked, &n_skipped) && n_checked == n_expert,
            "the whole expert dim in file order passes with the identity");
        const ggml_expert_store_tensor * st_all = ggml_expert_store_tensor_at(store, id_all);
        check(st_all != nullptr && st_all->assignment == 0 && (int64_t) st_all->global_of.size() == n_expert &&
            st_all->global_of[5] == 5, "the identity is recorded for the whole expert dim");
        ggml_expert_store_clear_source_hashes();

        ggml_expert_store_free(store);
        ggml_free(ctx);
    }


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

    plan_of(n_dev, "test-expert-assign-profile-2.txt", 0);
    check(dump_table(n_dev, n_layer, n_expert) == second, "the cached table is reused");


    {
        ggml_expert_store_plan_tensor t = device_tensor(0);
        t.n_expert = n_expert;
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


    check_eq(ggml_expert_store_owner(99, 0),      -1, "owner of an unknown layer");
    check_eq(ggml_expert_store_owner(0, 999),     -1, "owner of an unknown expert");
    check(ggml_expert_store_local_of(n_dev, 0) == nullptr, "local_of of an unknown device");






    {
        const int64_t caps_b[4] = { 7, 9, 8, 8 };
        const uint32_t tab_b = ggml_expert_assignment_contiguous(n_dev, n_layer + 1, n_expert, caps_b);
        check(tab_b > tab_a, "table ids count up");
        check(ggml_expert_assignment_contiguous(n_dev, n_layer, n_expert + 1, caps_b) == 0, "the caps must cover the experts");
        check_eq(ggml_expert_assignment_cap(tab_b, 1), 9, "the caps of a contiguous table are the slices");
        check_eq(ggml_expert_assignment_slice_device(tab_b, 7, 9), 1, "the segment of device 1 of B");
        check_eq(ggml_expert_assignment_slice_device(tab_b, 8, 8), -2, "the even segment is no segment of B");
        check_eq(ggml_expert_assignment_slice_device(tab_b, 0, n_expert), -1, "the whole dim");
        for (int l = 0; l <= n_layer; ++l) {
            int base = 0;
            for (int j = 0; j < n_dev; ++j) {
                const int16_t * row = ggml_expert_assignment_local_of(tab_b, j, l);
                check(row != nullptr, "contiguous row");
                for (int e = 0; e < n_expert && row != nullptr; ++e) {
                    const bool local = e >= base && e < base + (int) caps_b[j];
                    check_eq(row[e], local ? e - base : -1, "contiguous local number");
                    check_eq(ggml_expert_assignment_load_position(tab_b, l, e), e, "contiguous load position is the expert");
                }
                base += (int) caps_b[j];
            }
        }

        ggml_init_params ip = { 512 * ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);

        const auto make_node = [&](int n_local, bool ep, int offset, int layer, int dev, int32_t table) {
            ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 16, n_local);
            ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
            ggml_tensor * mm  = ggml_mul_mat_id(ctx, as, b, ids);
            if (ep) {
                ggml_mul_mat_id_set_expert_parallel(mm, true);
                ggml_mul_mat_id_set_expert_offset(mm, offset);
                ggml_mul_mat_id_set_n_expert_total(mm, n_expert);
                ggml_mul_mat_id_set_layer(mm, layer);
                ggml_mul_mat_id_set_device(mm, dev);
                ggml_mul_mat_id_set_assignment(mm, table);
            }
            return mm;
        };

        ggml_expert_assignment_activate(tab_b, nullptr, 0);


        const int16_t * idn = ggml_expert_store_node_lut(make_node(n_expert, false, 0, 0, 0, 0));
        for (int e = 0; e < n_expert && idn != nullptr; ++e) {
            check_eq(idn[e], e, "identity row entry");
        }
        const int16_t * seg = ggml_expert_store_node_lut(make_node(cap, true, cap, 0, 1, 0));
        for (int e = 0; e < n_expert && seg != nullptr; ++e) {
            check_eq(seg[e], e >= cap && e < 2 * cap ? e - cap : -1, "segment row entry");
        }
        check(ggml_expert_store_node_lut(make_node(cap, true, cap, 2, 3, 0)) == seg, "the segment row is shared");

        bool rows_differ = false;
        for (int j = 0; j < n_dev; ++j) {
            for (int l = 0; l < n_layer; ++l) {
                const int16_t * ra = ggml_expert_store_node_lut(make_node(cap, true, j * cap, l, j, (int32_t) tab_a));
                check(ra == ggml_expert_assignment_local_of(tab_a, j, l), "a node of the target reads its row of A");
                const int boff = (int) (j == 0 ? 0 : caps_b[0] + (j > 1 ? caps_b[1] : 0) + (j > 2 ? caps_b[2] : 0));
                const int16_t * rb = ggml_expert_store_node_lut(make_node((int) caps_b[j], true, boff, l, j, (int32_t) tab_b));
                check(rb == ggml_expert_assignment_local_of(tab_b, j, l), "a node of the draft of the same layer reads its row of B");
                rows_differ = rows_differ || memcmp(ra, rb, (size_t) n_expert * sizeof(int16_t)) != 0;
            }

            const int boff = (int) (j == 0 ? 0 : caps_b[0] + (j > 1 ? caps_b[1] : 0) + (j > 2 ? caps_b[2] : 0));
            const int16_t * rb = ggml_expert_store_node_lut(make_node((int) caps_b[j], true, boff, n_layer, j, (int32_t) tab_b));
            check(rb != nullptr && rb == ggml_expert_assignment_local_of(tab_b, j, n_layer), "the draft's own layer has a row in its table");
        }
        check(rows_differ, "the two tables give the nodes of one layer different rows");


        ggml_tensor * w_a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 16, n_expert);
        ggml_tensor * w_b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 16, n_expert);
        ggml_tensor * w_off = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 16, n_expert);
        ggml_tensor * v_a   = ggml_view_3d(ctx, w_a, 32, 16, n_expert / 2, w_a->nb[1], w_a->nb[2], 0);
        ggml_set_name(w_a, "blk.0.ffn_gate_exps.weight");
        ggml_set_name(w_b, "blk.4.ffn_gate_exps.weight");
        const uint32_t tab_c = ggml_expert_assignment_contiguous(n_dev, 1, n_expert, caps_b);
        const ggml_tensor * bind_c[1] = { w_b };
        ggml_expert_assignment_activate(tab_c, bind_c, 1);
        check_eq(ggml_expert_assignment_of_weight(w_b), tab_c, "the draft weight is bound to its table");
        check_eq(ggml_expert_assignment_of_weight(w_a), -1, "a weight bound to nothing yet");

        plan_of(n_dev, "test-expert-assign-profile.txt", 0);
        const uint32_t tab_d = ggml_expert_assignment_from_proposal(n_dev, n_layer, n_expert);
        check(tab_d > tab_c, "the target's table");
        const ggml_tensor * bind_d[1] = { w_a };
        ggml_expert_assignment_activate(tab_d, bind_d, 1);
        check_eq(ggml_expert_assignment_of_weight(w_a), tab_d, "the target weight is bound to its table");
        check_eq(ggml_expert_assignment_of_weight(v_a), tab_d, "a view answers for the tensor it views");
        check_eq(ggml_expert_assignment_of_weight(w_off), -1, "an unbound weight");


        const size_t n_tables = ggml_expert_assignment_count();
        ggml_expert_assignment_release(tab_d);
        check_eq((long long) ggml_expert_assignment_count(), (long long) n_tables - 1, "one table less");
        check_eq(ggml_expert_assignment_of_weight(w_a), -1, "the released weight is unbound");
        check_eq(ggml_expert_assignment_of_weight(w_b), tab_c, "the other model keeps its binding");
        check(!ggml_expert_assignment_is_active(tab_d) && ggml_expert_assignment_local_of(tab_d, 0, 0) == nullptr, "the released table is gone");
        const uint32_t tab_e = ggml_expert_assignment_contiguous(n_dev, 1, n_expert, caps_b);
        check(tab_e > tab_d, "a released id is not reused");
        ggml_expert_assignment_release(tab_e);
        ggml_expert_assignment_release(tab_c);
        ggml_expert_assignment_release(tab_b);
        ggml_expert_assignment_release(tab_a);
        ggml_expert_assignment_release(0);
        check_eq((long long) ggml_expert_assignment_count(), 0, "every table released");

        ggml_free(ctx);
    }



    {
        const int nl = 2;
        std::vector<std::vector<int64_t>> cnt((size_t) nl, std::vector<int64_t>((size_t) n_expert, 0));
        std::vector<std::vector<int64_t>> lau((size_t) nl, std::vector<int64_t>((size_t) n_expert, 0));
        for (int l = 0; l < nl; ++l) {
            for (int e = 0; e < n_expert; ++e) {



                cnt[(size_t) l][(size_t) e] = 4000 - 97 * e + l;
                lau[(size_t) l][(size_t) e] = (e / 2) * 3;
            }
        }
        write_profile("test-expert-assign-profile-cnt.txt", cnt);
        {
            FILE * f = fopen("test-expert-assign-profile-lau.txt", "w");
            check(f != nullptr, "write the launches profile");
            if (f != nullptr) {
                fprintf(f, "# layer expert count launches\n");
                for (int l = 0; l < nl; ++l) {
                    for (int e = 0; e < n_expert; ++e) {
                        fprintf(f, "%d %d %lld %lld\n", l, e, (long long) cnt[(size_t) l][(size_t) e], (long long) lau[(size_t) l][(size_t) e]);
                    }
                }
                fclose(f);
            }
        }


        const auto slots_of = [&](const char * path, std::vector<std::vector<int16_t>> & table) {
            ggml_expert_store_plan_tensor t = device_tensor(1);
            ggml_expert_store_placement plan;
            ggml_expert_store_plan(&t, 1, n_dev, 3 * 1024, path, &plan);
            table = dump_table(n_dev, nl, n_expert);
            return plan.slot_experts[0];
        };
        std::vector<std::vector<int16_t>> tab_cnt, tab_lau;
        const std::vector<int64_t> s_cnt = slots_of("test-expert-assign-profile-cnt.txt", tab_cnt);
        const std::vector<int64_t> s_lau = slots_of("test-expert-assign-profile-lau.txt", tab_lau);
        check(tab_cnt == tab_lau, "the launches field does not change the expert assignment");


        std::vector<int> loc;
        const int16_t * row = ggml_expert_store_local_of(0, 1);
        check(row != nullptr, "local_of row of device 0");
        for (int e = 0; row != nullptr && e < n_expert; ++e) {
            if (row[e] >= 0) {
                loc.push_back(e);
            }
        }
        std::vector<int> by_cnt = loc, by_lau = loc;
        std::stable_sort(by_cnt.begin(), by_cnt.end(), [&](int a, int b) { return cnt[1][(size_t) a] > cnt[1][(size_t) b]; });
        std::stable_sort(by_lau.begin(), by_lau.end(), [&](int a, int b) {
            if (lau[1][(size_t) a] != lau[1][(size_t) b]) {
                return lau[1][(size_t) a] > lau[1][(size_t) b];
            }
            return cnt[1][(size_t) a] > cnt[1][(size_t) b];
        });
        std::vector<int64_t> want_cnt, want_lau;
        for (int k = 0; k < 3 && k < (int) loc.size(); ++k) {
            want_cnt.push_back(row[by_cnt[(size_t) k]]);
            want_lau.push_back(row[by_lau[(size_t) k]]);
        }
        std::sort(want_cnt.begin(), want_cnt.end());
        std::sort(want_lau.begin(), want_lau.end());
        check(s_cnt == want_cnt, "without the launches field the slots go to the highest counts");
        check(s_lau == want_lau, "with the launches field the slots go to the most launches");
        check(s_cnt != s_lau, "the two rankings differ in this profile");
    }

    if (g_fail == 0) {
        printf("expert assignment tests passed\n");
        return 0;
    }
    fprintf(stderr, "%d checks failed\n", g_fail);
    return 1;
}