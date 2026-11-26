#pragma once
#include <cstdint>

namespace admission_pin_math {

enum class error { ok, uncovered, unproven, overflow, invalid_input };

struct settle_input {
    uint64_t pin_bytes = 0;
    uint64_t pending = 0;
    uint64_t owned_pinned = 0;
    uint64_t rss_file = 0;
    uint64_t vm_locked = 0;
};
struct settle_result {
    error status = error::invalid_input;
    bool retire = false;
    uint64_t retire_bytes = 0;
    uint64_t registered_after = 0;
    uint64_t overhead = 0;

    bool vmlck_moved = false;

};
settle_result settle(const settle_input & in) noexcept;

struct release_input {
    uint64_t pin_bytes = 0;
    uint64_t owned_pinned = 0;
};
struct release_result {
    error status = error::invalid_input;
    uint64_t release_bytes = 0;
};
release_result release(const release_input & in) noexcept;

struct form_input {
    uint64_t pin_bytes = 0;
    uint64_t delta_usable = 0;
    uint64_t delta_rss_file = 0;
    uint64_t delta_vm_locked = 0;
    uint64_t pending_after = 0;
};
enum class form { gup = 0, mlock_migration = 1, observed = 2 };
struct form_result {
    form matched = form::observed;
    bool ledger_exact = false;
    bool bound_held = false;
    uint64_t structural = 0;
    uint64_t residual = 0;
    uint64_t naive_miss = 0;

    bool formula_double = false;

};
form_result classify_form(const form_input & in) noexcept;

}
