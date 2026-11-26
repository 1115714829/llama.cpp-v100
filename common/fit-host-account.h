#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct common_fit_host_account {
    int64_t     mapped = 0;
    int64_t     pinned = 0;
    int64_t     ple    = 0;
    int64_t     embd   = 0;
    uint64_t    id     = 0;
    std::string placement;
    bool        valid  = false;
};

struct common_fit_layer_measurement {
    std::vector<int64_t>    mem;
    common_fit_host_account account;
};

static inline int64_t common_fit_host_demand(int64_t arena_total, int64_t mapped, int64_t pinned) {
    return arena_total + mapped + pinned;
}
