#pragma once
#include <cstddef>
#include <cstdint>

namespace admission_owned_math {
enum class error { ok, invalid_owner_observation, overflow, needs_reclaim, invalid_input };
struct held_result {
    error status;
    uint64_t unclassified_locked;
    uint64_t held_file_credit;
    uint64_t legacy_resident_cap;
};
held_result normalize_held(uint64_t rss_file, uint64_t vm_locked,
                           uint64_t owned_locked_anon) noexcept;

struct ancestor {
    uint64_t limit;
    uint64_t current;
    uint64_t headroom;
    bool finite;
    bool hard_usage;
};
struct charge_result {
    error status;
    uint64_t demand;
    uint64_t room;
    size_t limiting_index;
    bool constrained;
};

charge_result charge_gate(const ancestor * levels, size_t count,
                          uint64_t pending, uint64_t additional) noexcept;

struct observation {
    uint64_t rss_file = 0;
    uint64_t vm_locked = 0;
    uint64_t owned_locked_anon = 0;
    uint64_t registry_before = 0;
    uint64_t registry_after = 0;
};

enum class observation_error { ok, unstable, invalid_owner_observation, invalid_input };

struct observed_result {
    observation_error status = observation_error::unstable;
    held_result held{error::ok, 0, 0, 0};
};

observed_result normalize_observed(const observation & observed) noexcept;

enum class ledger_error { ok, unstable, invalid_owner_observation, budget_short, overflow, invalid_input };

struct ledger_result {
    ledger_error status = ledger_error::unstable;
    held_result held{error::ok, 0, 0, 0};
    charge_result charge{error::ok, 0, 0, 0, false};
};

ledger_result evaluate(const observation & observed, const ancestor * levels, size_t count,
                       uint64_t pending, uint64_t additional) noexcept;
}
