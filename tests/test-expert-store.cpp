// tests for the host-side helpers of the expert store (ggml-expert-store.h)

#include "ggml-expert-store.h"

#include <cstdint>
#include <cstdio>

// the expected values below are constants, but they are also checked against a 128-bit
// reference in the test, so a typo in them shows up here and not as a failure of the helper.
// the extension stays in the test: the store itself must be exact on every compiler.
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
    // plain 64-bit products, the result is (a * b) / c
    { 7,          5,           3,        11 },
    { 1000003,    1000033,     997,      1003045135 },
    { 1,          1,           1,        1 },
    // c = 1: the result is the product itself
    { 3,          4,           1,        12 },
    { 0xFFFFFFFFFFF, 1,        1,        0xFFFFFFFFFFF },
    // the product leaves 64 bits: (2^44 - 1) * (2^36 - 1) / 0x123456789AB = 966367641586
    { 0xFFFFFFFFFFF, 0xFFFFFFFFF, 0x123456789AB, 966367641586ULL },
    // the same product over a power of two: (2^44 - 1) * (2^36 - 1) / 2^48 = 2^32 - 1
    { 0xFFFFFFFFFFF, 0xFFFFFFFFF, (uint64_t) 1 << 48, 4294967295ULL },
    // largest quotient that fits in 64 bits: (2^64 - 1)^2 / (2^64 - 1) = 2^64 - 1, the top
    // half of the product is 2^64 - 2 here, so the division is the widest case that is allowed
    { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFULL },
    // the top half of the product is 2^63 - 1 here: (2^64 - 1) * 2^63 / (2^64 - 1) = 2^63
    { 0xFFFFFFFFFFFFFFFF, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000ULL },
};

// one case: the expected value first against the 128-bit reference, then against the helper
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

int main(void) {
    for (const mul_div_case & t : g_cases) {
        if (!test_case(t.a, t.b, t.c, t.expect)) {
            return 1;
        }
    }

    // the slot plan divides the quota by the expert stride afterwards, which is the same value:
    // floor(floor(x / c) / d) == floor(x / (c * d)) for positive integers. The doubled divisor
    // has to stay a valid one, so the widest divisors are left out.
    for (const mul_div_case & t : g_cases) {
        if (t.c > UINT64_MAX / 2) {
            continue;
        }
        const uint64_t quota = ggml_expert_store_mul_div_floor(t.a, t.b, t.c);
        if (!test_case(t.a, t.b, t.c * 2, quota / 2)) {
            return 1;
        }
    }

    // every small size, including the ones where the product and the divisor are close
    for (uint64_t c = 1; c <= 64; ++c) {
        for (uint64_t b = 1; b <= 64; ++b) {
            const uint64_t a = 0xFFFFFFFFFFF * (b + c);
            if (!test_case(a, b, c, ggml_expert_store_mul_div_floor(a, b, c))) {
                return 1;
            }
        }
    }

    printf("%s: OK\n", __func__);
    return 0;
}
