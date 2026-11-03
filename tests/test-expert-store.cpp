

#include "ggml-expert-store.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>




#if defined(__SIZEOF_INT128__)
#define TEST_HAVE_REF 1
__extension__ typedef unsigned __int128 test_u128;

static uint64_t test_mul_div_floor_ref(uint64_t a, uint64_t b, uint64_t c) {
    return (uint64_t) ((test_u128) a * (test_u128) b / (test_u128) c);
}
#else
#define TEST_HAVE_REF 0
#endif

struct mul_div_case {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t expect;
};

static const mul_div_case g_cases[] = {

    { 7,          5,           3,        11 },
    { 1000003,    1000033,     997,      1003045135 },
    { 1,          1,           1,        1 },

    { 3,          4,           1,        12 },
    { 0xFFFFFFFFFFF, 1,        1,        0xFFFFFFFFFFF },

    { 0xFFFFFFFFFFF, 0xFFFFFFFFF, 0x123456789AB, 966367641586ULL },

    { 0xFFFFFFFFFFF, 0xFFFFFFFFF, (uint64_t) 1 << 48, 4294967295ULL },


    { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFULL },

    { 0xFFFFFFFFFFFFFFFF, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000ULL },
};


static bool test_case(uint64_t a, uint64_t b, uint64_t c, uint64_t expect) {
#if TEST_HAVE_REF
    if (test_mul_div_floor_ref(a, b, c) != expect) {
        fprintf(stderr, "%s: the reference does not give %llu for a = %llu b = %llu c = %llu\n",
            __func__, (unsigned long long) expect, (unsigned long long) a,
            (unsigned long long) b, (unsigned long long) c);
        return false;
    }
#endif
    if (ggml_expert_store_mul_div_floor(a, b, c) != expect) {
        fprintf(stderr, "%s: a = %llu b = %llu c = %llu does not give %llu\n",
            __func__, (unsigned long long) a, (unsigned long long) b,
            (unsigned long long) c, (unsigned long long) expect);
        return false;
    }
    return true;
}



struct placement_snapshot {
    std::vector<int32_t> residency;
    std::vector<int64_t> host_offset;
    std::vector<char *>  table;
    int64_t n_slot     = 0;
    int64_t n_host     = 0;
    int64_t host_used  = 0;
    int64_t swap_count = 0;
    int64_t swap_bytes = 0;

    static placement_snapshot take(const ggml_expert_store_tensor * st) {
        placement_snapshot s;
        s.residency   = st->residency;
        s.host_offset = st->host_offset;
        s.table.assign(st->table_host, st->table_host + st->n_expert);
        s.n_slot      = st->n_slot;
        s.n_host      = st->n_host;
        s.host_used   = st->host_used;
        s.swap_count  = st->swap_count;
        s.swap_bytes  = st->swap_bytes;
        return s;
    }

    bool equal(const placement_snapshot & o) const {
        return residency == o.residency && host_offset == o.host_offset && table == o.table &&
               n_slot == o.n_slot && n_host == o.n_host && host_used == o.host_used &&
               swap_count == o.swap_count && swap_bytes == o.swap_bytes;
    }
};



static void * test_alloc(void *, size_t size) {
    return malloc(size);
}

static void test_free(void *, void * p) {
    free(p);
}

static void * test_alloc_host(void *, size_t size, void ** dev_ptr) {
    void * p = malloc(size);
    *dev_ptr = p;
    return p;
}

static void test_copy_h2d(void *, void * dst, const void * src, size_t size) {
    memcpy(dst, src, size);
}

static void test_copy_d2h(void *, void * dst, const void * src, size_t size) {
    memcpy(dst, src, size);
}

static const ggml_expert_store_backend g_test_backend = {
    nullptr, test_alloc, test_free, test_alloc_host, test_free, test_copy_h2d, test_copy_d2h, nullptr,
};


static uint64_t g_rng = 0x0123456789ABCDEFULL;

static int64_t test_rand(int64_t n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (int64_t) (g_rng % (uint64_t) n);
}


static bool test_swap(void) {
    const int64_t n_expert  = 16;
    const size_t  blob      = 1024;
    const size_t  pad       = 512;
    const size_t  type_size = 32;
    const size_t  stride    = ggml_expert_store_stride(blob, pad, type_size);


    const int64_t slot_experts[] = { 2, 5, 9, 11, 13 };
    const int64_t n_slot = (int64_t) (sizeof(slot_experts) / sizeof(slot_experts[0]));

    ggml_expert_store * store = ggml_expert_store_init(&g_test_backend, 0, nullptr);
    if (store == nullptr) {
        fprintf(stderr, "%s: cannot create the store\n", __func__);
        return false;
    }

    const int tensor_id = ggml_expert_store_add_tensor(store, 0, 0, 0, n_expert, blob, pad, type_size,
        slot_experts, n_slot, -1);
    if (tensor_id < 0) {
        fprintf(stderr, "%s: cannot add the tensor\n", __func__);
        ggml_expert_store_free(store);
        return false;
    }

    ggml_expert_store_tensor * st = &store->tensors[(size_t) tensor_id];

    if (!ggml_expert_store_tensor_check(st)) {
        fprintf(stderr, "%s: the placement of a freshly added tensor is not consistent\n", __func__);
        ggml_expert_store_free(store);
        return false;
    }
    if (ggml_expert_store_tensor_n_slot(st) != n_slot || ggml_expert_store_tensor_n_host(st) != n_expert - n_slot ||
            st->host_used != (n_expert - n_slot) * (int64_t) stride) {
        fprintf(stderr, "%s: %" PRId64 " slots and %" PRId64 " host experts, expected %" PRId64 " and %" PRId64 "\n",
            __func__, ggml_expert_store_tensor_n_slot(st), ggml_expert_store_tensor_n_host(st),
            n_slot, n_expert - n_slot);
        ggml_expert_store_free(store);
        return false;
    }
    for (int64_t e = 0; e < n_expert; ++e) {
        const bool in_slot = e == 2 || e == 5 || e == 9 || e == 11 || e == 13;
        const int32_t slot = ggml_expert_store_tensor_residency(st, e);
        const int64_t off  = ggml_expert_store_tensor_host_offset(st, e);
        if (in_slot ? (slot < 0 || off != GGML_EXPERT_STORE_INVALID) : (slot != GGML_EXPERT_STORE_INVALID || off < 0)) {
            fprintf(stderr, "%s: expert %" PRId64 " starts in the wrong tier\n", __func__, e);
            ggml_expert_store_free(store);
            return false;
        }
    }


    for (int64_t i = 0; i < 200; ++i) {
        int64_t host = -1;
        while (host < 0) {
            const int64_t e = test_rand(n_expert);
            host = ggml_expert_store_tensor_residency(st, e) < 0 ? e : -1;
        }
        int64_t slot_expert = -1;
        while (slot_expert < 0) {
            const int64_t e = test_rand(n_expert);
            slot_expert = ggml_expert_store_tensor_residency(st, e) >= 0 ? e : -1;
        }

        const int32_t slot   = ggml_expert_store_tensor_residency(st, slot_expert);
        const int64_t offset = ggml_expert_store_tensor_host_offset(st, host);

        if (!ggml_expert_store_tensor_swap(st, host, slot_expert)) {
            fprintf(stderr, "%s: swap %" PRId64 " of expert %" PRId64 " and slot expert %" PRId64 " was refused\n",
                __func__, i, host, slot_expert);
            ggml_expert_store_free(store);
            return false;
        }

        if (!ggml_expert_store_tensor_check(st)) {
            fprintf(stderr, "%s: swap %" PRId64 " broke the placement bookkeeping\n", __func__, i);
            ggml_expert_store_free(store);
            return false;
        }
        if (ggml_expert_store_tensor_residency(st, host) != slot ||
                ggml_expert_store_tensor_host_offset(st, host) != GGML_EXPERT_STORE_INVALID ||
                ggml_expert_store_tensor_residency(st, slot_expert) != GGML_EXPERT_STORE_INVALID ||
                ggml_expert_store_tensor_host_offset(st, slot_expert) != offset) {
            fprintf(stderr, "%s: swap %" PRId64 " did not exchange the tiers of %" PRId64 " and %" PRId64 "\n",
                __func__, i, host, slot_expert);
            ggml_expert_store_free(store);
            return false;
        }
        if (st->table_host[(size_t) host] != st->slots_dev + (size_t) slot * stride ||
                st->table_host[(size_t) slot_expert] != st->host_dev + offset) {
            fprintf(stderr, "%s: swap %" PRId64 " did not move the table entries\n", __func__, i);
            ggml_expert_store_free(store);
            return false;
        }
        if (ggml_expert_store_tensor_n_slot(st) != n_slot || ggml_expert_store_tensor_n_host(st) != n_expert - n_slot ||
                st->host_used != (n_expert - n_slot) * (int64_t) stride) {
            fprintf(stderr, "%s: swap %" PRId64 " changed the slot or host count\n", __func__, i);
            ggml_expert_store_free(store);
            return false;
        }
        if (st->swap_count != i + 1 || st->swap_bytes != (i + 1) * 2 * (int64_t) stride) {
            fprintf(stderr, "%s: swap %" PRId64 " counted %" PRId64 " swaps and %" PRId64 " bytes\n",
                __func__, i, st->swap_count, st->swap_bytes);
            ggml_expert_store_free(store);
            return false;
        }
    }


    int64_t host_a = -1;
    int64_t host_b = -1;
    int64_t slot_a = -1;
    int64_t slot_b = -1;
    for (int64_t e = 0; e < n_expert; ++e) {
        if (ggml_expert_store_tensor_residency(st, e) < 0) {
            if (host_a < 0) {
                host_a = e;
            } else {
                host_b = e;
            }
        } else {
            if (slot_a < 0) {
                slot_a = e;
            } else {
                slot_b = e;
            }
        }
    }

    struct bad_case {
        const char * name;
        int64_t in;
        int64_t out;
    };
    const bad_case bad[] = {
        { "both in slots",     slot_a, slot_b },
        { "both on the host",  host_a, host_b },
        { "in with itself",    host_a, host_a },
        { "out with itself",   slot_a, slot_a },
        { "in out of range",   n_expert, slot_a },
        { "in negative",      -1, slot_a },
        { "out out of range",  host_a, n_expert },
        { "out negative",      host_a, -1 },
        { "both out of range", n_expert + 1, -7 },
    };

    for (const bad_case & c : bad) {
        const placement_snapshot before = placement_snapshot::take(st);
        if (ggml_expert_store_tensor_swap(st, c.in, c.out)) {
            fprintf(stderr, "%s: %s was accepted\n", __func__, c.name);
            ggml_expert_store_free(store);
            return false;
        }
        if (!before.equal(placement_snapshot::take(st))) {
            fprintf(stderr, "%s: %s changed the placement bookkeeping\n", __func__, c.name);
            ggml_expert_store_free(store);
            return false;
        }
        if (!ggml_expert_store_tensor_check(st)) {
            fprintf(stderr, "%s: %s broke the placement bookkeeping\n", __func__, c.name);
            ggml_expert_store_free(store);
            return false;
        }
    }

    if (st->swap_count != 200 || st->swap_bytes != 200 * 2 * (int64_t) stride) {
        fprintf(stderr, "%s: %" PRId64 " swaps and %" PRId64 " bytes after the rejected ones\n",
            __func__, st->swap_count, st->swap_bytes);
        ggml_expert_store_free(store);
        return false;
    }

    ggml_expert_store_free(store);
    return true;
}







static bool test_swap_to(void) {
    const int64_t n_expert  = 16;
    const size_t  blob      = 1024;
    const size_t  pad       = 512;
    const size_t  type_size = 32;
    const size_t  stride    = ggml_expert_store_stride(blob, pad, type_size);
    const int     n_spare   = 6;

    const int64_t slot_experts[] = { 2, 5, 9, 11, 13 };
    const int64_t n_slot = (int64_t) (sizeof(slot_experts) / sizeof(slot_experts[0]));

    ggml_expert_store * store = ggml_expert_store_init(&g_test_backend, 0, nullptr);
    if (store == nullptr) {
        fprintf(stderr, "%s: cannot create the store\n", __func__);
        return false;
    }


    const auto pattern = [&](int t, int64_t e, size_t i) {
        return (uint8_t) (t * 97 + e * 31 + (int64_t) i * 7 + (int64_t) (i >> 8));
    };

    int tid[2] = { -1, -1 };
    for (int t = 0; t < 2; ++t) {
        tid[t] = ggml_expert_store_add_tensor(store, t, 0, 0, n_expert, blob, pad, type_size, slot_experts, n_slot, -1);
        if (tid[t] < 0) {
            fprintf(stderr, "%s: cannot add tensor %d\n", __func__, t);
            ggml_expert_store_free(store);
            return false;
        }
        std::vector<uint8_t> data((size_t) n_expert * blob);
        for (int64_t e = 0; e < n_expert; ++e) {
            for (size_t i = 0; i < blob; ++i) {
                data[(size_t) e * blob + i] = pattern(t, e, i);
            }
        }
        ggml_expert_store_write(store, tid[t], data.data(), 0, data.size());
    }


    std::vector<std::vector<char>> spare_mem((size_t) n_spare, std::vector<char>(stride, 0));
    std::vector<char *> pool;
    for (auto & m : spare_mem) {
        pool.push_back(m.data());
    }

    bool ok = true;
    for (int64_t i = 0; i < 400 && ok; ++i) {
        const int t = (int) test_rand(2);
        ggml_expert_store_tensor * st = &store->tensors[(size_t) tid[t]];

        int64_t e_in = -1;
        while (e_in < 0) {
            const int64_t e = test_rand(n_expert);
            e_in = ggml_expert_store_tensor_residency(st, e) < 0 ? e : -1;
        }
        int64_t e_out = -1;
        while (e_out < 0) {
            const int64_t e = test_rand(n_expert);
            e_out = ggml_expert_store_tensor_residency(st, e) >= 0 ? e : -1;
        }

        const int32_t slot   = ggml_expert_store_tensor_residency(st, e_out);
        const int64_t off_in = ggml_expert_store_tensor_host_offset(st, e_in);
        char * const  before = st->slot_addr[(size_t) slot];
        char * const  fresh  = pool.back();
        pool.pop_back();


        memcpy(fresh, st->host_host + off_in, stride);
        memcpy(st->host_host + off_in, st->table_host[(size_t) e_out], stride);

        char * old = nullptr;
        if (!ggml_expert_store_tensor_swap_to(st, e_in, e_out, fresh, &old)) {
            fprintf(stderr, "%s: exchange %" PRId64 " of %" PRId64 " and %" PRId64 " was refused\n", __func__, i, e_in, e_out);
            ok = false;
            break;
        }
        pool.push_back(old);

        if (old != before || st->slot_addr[(size_t) slot] != fresh || st->table_host[(size_t) e_in] != fresh ||
                ggml_expert_store_tensor_residency(st, e_in) != slot ||
                ggml_expert_store_tensor_host_offset(st, e_out) != off_in ||
                st->table_host[(size_t) e_out] != st->host_dev + off_in) {
            fprintf(stderr, "%s: exchange %" PRId64 " left the wrong addresses\n", __func__, i);
            ok = false;
        }
        if (!ggml_expert_store_tensor_check(st)) {
            fprintf(stderr, "%s: exchange %" PRId64 " broke the placement bookkeeping\n", __func__, i);
            ok = false;
        }
    }


    for (int t = 0; t < 2 && ok; ++t) {
        std::vector<uint8_t> got(blob);
        for (int64_t e = 0; e < n_expert && ok; ++e) {
            ggml_expert_store_read(store, tid[t], got.data(), (size_t) e * blob, blob);
            for (size_t i = 0; i < blob; ++i) {
                if (got[i] != pattern(t, e, i)) {
                    fprintf(stderr, "%s: tensor %d expert %" PRId64 " reads back wrong bytes at %zu\n", __func__, t, e, i);
                    ok = false;
                    break;
                }
            }
        }
    }


    if (ok) {
        ggml_expert_store_tensor * st = &store->tensors[(size_t) tid[0]];
        int64_t host = -1;
        int64_t slot_e = -1;
        for (int64_t e = 0; e < n_expert; ++e) {
            if (ggml_expert_store_tensor_residency(st, e) < 0) {
                host = host < 0 ? e : host;
            } else {
                slot_e = slot_e < 0 ? e : slot_e;
            }
        }
        const placement_snapshot snap = placement_snapshot::take(st);
        if (ggml_expert_store_tensor_swap_to(st, host, slot_e, st->host_dev, nullptr) ||
                !snap.equal(placement_snapshot::take(st))) {
            fprintf(stderr, "%s: a slot inside the host arena was accepted\n", __func__);
            ok = false;
        }
    }

    ggml_expert_store_free(store);
    return ok;
}

int main(void) {
    for (const mul_div_case & t : g_cases) {
        if (!test_case(t.a, t.b, t.c, t.expect)) {
            return 1;
        }
    }




    for (const mul_div_case & t : g_cases) {
        if (t.c > UINT64_MAX / 2) {
            continue;
        }
        const uint64_t quota = ggml_expert_store_mul_div_floor(t.a, t.b, t.c);
        if (!test_case(t.a, t.b, t.c * 2, quota / 2)) {
            return 1;
        }
    }


    for (uint64_t c = 1; c <= 64; ++c) {
        for (uint64_t b = 1; b <= 64; ++b) {
            const uint64_t a = 0xFFFFFFFFFFF * (b + c);
            if (!test_case(a, b, c, ggml_expert_store_mul_div_floor(a, b, c))) {
                return 1;
            }
        }
    }

    if (!test_swap()) {
        return 1;
    }
    if (!test_swap_to()) {
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
