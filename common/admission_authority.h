#pragma once

#include "admission_gate.h"

#include <cstdint>

namespace common_admission_authority {

enum outcome : uint8_t {
    outcome_published = 0,
    outcome_cancelled = 1,
    outcome_failed    = 2,
    outcome_external  = 3,
    outcome_read      = 4,
};

struct grant_observation {
    common_admission_error error = common_admission_error::unstable;
    uint64_t pending_grants = 0;
    uint64_t released_bytes = 0;
    uint64_t published_grants = 0;
    uint64_t cancelled_grants = 0;
    uint64_t failed_grants = 0;
    uint64_t external_grants = 0;
    uint64_t read_grants = 0;
    uint64_t settled_pins = 0;
    unsigned attempts = 0;
    bool repeated = false;
};
grant_observation observe_grants() noexcept;

bool release(uint64_t token, uint64_t bytes, outcome how) noexcept;

bool settle_grant(uint64_t token, uint64_t bytes) noexcept;

bool release_pin_grant(uint64_t token, uint64_t bytes) noexcept;

bool force_link() noexcept;

}
