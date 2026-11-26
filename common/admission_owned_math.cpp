#include "admission_owned_math.h"
#include <algorithm>
#include <limits>

namespace admission_owned_math {
held_result normalize_held(uint64_t rss, uint64_t locked, uint64_t owned) noexcept {
    if (owned > locked) return {error::invalid_owner_observation, 0, 0, 0};
    const uint64_t remaining = locked - owned;
    const uint64_t held = std::max(rss, remaining);

    return {error::ok, remaining, held, held};
}
charge_result charge_gate(const ancestor * levels, size_t count,
                          uint64_t pending, uint64_t additional) noexcept {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    charge_result result{error::ok, 0, max, count, false};
    if (count && !levels) { result.status = error::invalid_input; return result; }
    if (additional > max - pending) { result.status = error::overflow; return result; }
    result.demand = pending + additional;
    for (size_t i = 0; i != count; ++i) {
        if (!levels[i].finite) continue;
        if (!levels[i].hard_usage) { result.status = error::invalid_input; return result; }
        const uint64_t available = levels[i].current < levels[i].limit
            ? levels[i].limit - levels[i].current : 0;
        const uint64_t room = available > levels[i].headroom ? available - levels[i].headroom : 0;
        if (!result.constrained || room < result.room) {
            result.room = room;
            result.limiting_index = i;
        }
        result.constrained = true;
    }
    if (result.constrained && result.demand > result.room) result.status = error::needs_reclaim;
    return result;
}
observed_result normalize_observed(const observation & observed) noexcept {
    observed_result result;

    if (observed.registry_before != observed.registry_after) {
        result.status = observation_error::unstable;
        return result;
    }
    const held_result held = normalize_held(observed.rss_file, observed.vm_locked,
                                            observed.owned_locked_anon);
    if (held.status == error::invalid_owner_observation) {

        result.status = observation_error::invalid_owner_observation;
        return result;
    }
    if (held.status != error::ok) {
        result.status = observation_error::invalid_input;
        return result;
    }
    result.status = observation_error::ok;
    result.held    = held;
    return result;
}
ledger_result evaluate(const observation & observed, const ancestor * levels, size_t count,
                       uint64_t pending, uint64_t additional) noexcept {
    ledger_result result;
    const observed_result window = normalize_observed(observed);
    if (window.status == observation_error::unstable) {
        result.status = ledger_error::unstable;
        return result;
    }
    if (window.status == observation_error::invalid_owner_observation) {
        result.status = ledger_error::invalid_owner_observation;
        return result;
    }
    if (window.status != observation_error::ok) {
        result.status = ledger_error::invalid_input;
        return result;
    }
    result.charge = charge_gate(levels, count, pending, additional);
    switch (result.charge.status) {
        case error::ok:            result.status = ledger_error::ok; break;
        case error::needs_reclaim: result.status = ledger_error::budget_short; break;
        case error::overflow:      result.status = ledger_error::overflow; break;
        default:                   result.status = ledger_error::invalid_input; break;
    }
    if (result.status == ledger_error::ok) result.held = window.held;
    return result;
}
}
