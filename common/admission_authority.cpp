#include "log.h"

#include "admission_authority.h"

#include "../ggml/src/ggml-safe-io-authority.h"

#include <cstdio>
#include <mutex>
#include <utility>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace common_admission_authority {
namespace {

struct grant_state {
    std::mutex mutex;
    uint64_t generation = 0;
    uint64_t pending_bytes = 0;
    uint64_t released_bytes = 0;
    uint64_t published = 0;
    uint64_t cancelled = 0;
    uint64_t failed = 0;
    uint64_t external = 0;
    uint64_t reads = 0;
    uint64_t settled_pins = 0;
    uint64_t tokens[64] = {};
    uint64_t bytes[64] = {};
    int64_t pids[64] = {};
    bool live[64] = {};
    bool batch[64] = {};
    bool claimed[64] = {};
    ggml_safe_io::authority::ops::batch_use uses[64] = {};
    bool used[64] = {};

    common_admission_lease leases[64];
};
grant_state & grants() {
    static grant_state state;
    return state;
}

int64_t authority_pid() noexcept {
#if defined(__linux__)
    return (int64_t) getpid();
#else
    return 0;
#endif
}

uint64_t sign_acquire(uint64_t bytes, uint64_t * usable) noexcept {

    if (!usable) { return 0; }
    *usable = 0;
    const int64_t pid = authority_pid();
    if (!pid) { return 0; }
    try {

        common_admission_scope scope;
        common_admission_lease lease;
        uint64_t usable_at_issue = 0;
        if (bytes) {
            const common_admission_status entered = scope.enter(common_admission_scope_kind::transaction);
            if (entered.error != common_admission_error::ok) { return 0; }
            const common_admission_status grown = lease.grow(bytes);
            if (grown.error != common_admission_error::ok) { return 0; }
            usable_at_issue = grown.usable;
        }
        std::lock_guard<std::mutex> guard(grants().mutex);
        int index = -1;
        for (int j = 0; j < 64; ++j) {
            if (!grants().live[j]) { index = j; break; }
        }
        if (index < 0) { return 0; }

        grants().used[index] = false;
        grants().batch[index] = false;
        grants().claimed[index] = false;
        grants().live[index] = true;
        grants().tokens[index] = ++grants().generation;
        grants().bytes[index] = bytes;
        grants().pids[index] = pid;

        grants().leases[index] = std::move(lease);
        grants().pending_bytes += bytes;
        *usable = usable_at_issue;
        return grants().tokens[index];
    } catch (...) {
        return 0;
    }
}

using batch_use = ggml_safe_io::authority::ops::batch_use;

bool sign_acquire_batch(const uint64_t * sizes, const batch_use * uses, unsigned count, uint64_t * tokens) noexcept {
    if (!sizes || !uses || !tokens || !count || count > 64) { return false; }
    uint64_t total = 0;
    for (unsigned i = 0; i < count; ++i) {
        if ((uses[i] != batch_use::pack_pin && uses[i] != batch_use::pack_write && uses[i] != batch_use::pack_manifest && uses[i] != batch_use::manifest_read) ||
                !sizes[i] || sizes[i] > UINT64_MAX - total) { return false; }
        total += sizes[i];
    }
    const auto pid = authority_pid();
    if (!pid) { return false; }
    try {
        common_admission_scope scope;
        common_admission_lease aggregate;
        if (scope.enter(common_admission_scope_kind::transaction).error != common_admission_error::ok ) {
            return false;
        }
        const auto admission = aggregate.grow(total);
        if (admission.error != common_admission_error::ok) {
            std::fprintf(stderr, "pack batch rejected rc=%d total=%llu pending=%llu usable=%llu\n", int(admission.error),
                (unsigned long long) total, (unsigned long long) admission.pending, (unsigned long long) admission.usable);
            return false;
        }
        std::lock_guard<std::mutex> guard(grants().mutex);
        int slots[64];
        unsigned found = 0;
        for (int i = 0; i < 64 && found < count; ++i) {
            if (!grants().live[i]) { slots[found++] = i; }
        }
        if (found != count || total > UINT64_MAX - grants().pending_bytes ||
                count > UINT64_MAX - grants().generation) { return false; }
        common_admission_lease children[64];
        for (unsigned i = 0; i < count; ++i) {
            if (!aggregate.split_pending(sizes[i], children[i])) { return false; }
        }
        for (unsigned i = 0; i < count; ++i) {
            const int slot = slots[i];
            grants().leases[slot] = std::move(children[i]);
            grants().used[slot] = false;
            grants().batch[slot] = true;
            grants().claimed[slot] = false;
            grants().uses[slot] = uses[i];
            grants().live[slot] = true;
            grants().tokens[slot] = ++grants().generation;
            grants().bytes[slot] = sizes[i];
            grants().pids[slot] = pid;
            tokens[i] = grants().tokens[slot];
            LOG_DBG("alloc %llu %llu %d\n",
                (unsigned long long) tokens[i], (unsigned long long) sizes[i], int(uses[i]));
        }
        grants().pending_bytes += total;
        return true;
    } catch (...) {
        return false;
    }
}

bool sign_claim_batch(uint64_t token, uint64_t bytes, batch_use use) noexcept {
    const auto pid = authority_pid();
    if (!token || !pid) { return false; }
    try {
        auto & state = grants();
        std::lock_guard<std::mutex> guard(state.mutex);
        for (int i = 0; i < 64; ++i) {
            if (state.live[i] && state.batch[i] && !state.claimed[i] && state.tokens[i] == token &&
                    state.pids[i] == pid && state.bytes[i] == bytes && state.uses[i] == use) {
                state.claimed[i] = true;
                LOG_DBG("claim %llu %llu %d\n",
                    (unsigned long long) token, (unsigned long long) bytes, int(use));
                return true;
            }
        }
    } catch (...) {}
    return false;
}

bool sign_release(uint64_t token, uint64_t bytes, int how) noexcept {

    if (!token) { return false; }

    if (how < static_cast<int>(outcome_published) || how > static_cast<int>(outcome_read)) {
        return false;
    }
    const int64_t pid = authority_pid();
    if (!pid) { return false; }
    try {
        std::lock_guard<std::mutex> guard(grants().mutex);
        int index = -1;
        for (int j = 0; j < 64; ++j) {
            if (grants().live[j] && grants().tokens[j] == token &&
                grants().bytes[j] == bytes && grants().pids[j] == pid) {
                index = j;
                break;
            }
        }
        if (index < 0 || grants().used[index] || grants().leases[index].settled_pinned_bytes()) {
            return false;
        }

        if (grants().batch[index]) {
            LOG_DBG("free %llu %llu %d\n",
                (unsigned long long) token, (unsigned long long) bytes, how);
        }
        grants().used[index] = true;
        grants().live[index] = false;
        grants().pending_bytes -= bytes;
        grants().released_bytes += bytes;
        switch (how) {
            case static_cast<int>(outcome_published): ++grants().published; break;
            case static_cast<int>(outcome_cancelled): ++grants().cancelled; break;
            case static_cast<int>(outcome_failed):    ++grants().failed;    break;
            case static_cast<int>(outcome_read):      ++grants().reads;     break;
            default:                                  ++grants().external;  break;
        }
        ++grants().generation;

        if (bytes) { grants().leases[index].cancel_pending(bytes); }
        else { grants().leases[index] = common_admission_lease(); }
        return true;
    } catch (...) {
        return false;
    }
}

void sign_cancel_unclaimed(uint64_t token, uint64_t bytes) noexcept {
    try {
        auto & state = grants();
        std::lock_guard<std::mutex> guard(state.mutex);
        for (int i = 0; i < 64; ++i) {
            if (state.live[i] && state.batch[i] && !state.claimed[i] && state.tokens[i] == token &&
                    state.pids[i] == authority_pid() && state.bytes[i] == bytes) {
                state.leases[i].cancel_pending(bytes);
                state.live[i] = false;
                state.used[i] = true;
                state.pending_bytes -= bytes;
                ++state.cancelled;
                ++state.generation;
                return;
            }
        }
    } catch (...) {}
}

const ggml_safe_io::authority::ops the_ops = {
    &sign_acquire, &sign_release, &sign_acquire_batch, &sign_claim_batch, &sign_cancel_unclaimed
};

bool registrar() noexcept {
    return ggml_safe_io::authority::authority_register(&the_ops);
}

[[maybe_unused]] const bool registered = registrar();

}

bool settle_grant(uint64_t token, uint64_t bytes) noexcept {
    if (!token) { return false; }
    const int64_t pid = authority_pid();
    if (!pid) { return false; }
    try {
        std::lock_guard<std::mutex> guard(grants().mutex);
        int index = -1;
        for (int j = 0; j < 64; ++j) {
            if (grants().live[j] && grants().tokens[j] == token &&
                grants().pids[j] == pid && grants().bytes[j] == bytes) {
                index = j;
                break;
            }
        }
        if (index < 0 || grants().used[index]) {
            std::fprintf(stderr, "pack settle rejected: grant identity token=%llu bytes=%llu\n",
                (unsigned long long) token, (unsigned long long) bytes);
            return false;
        }

        const common_admission_status st = grants().leases[index].settle_pin(bytes);
        if (st.error != common_admission_error::ok) {
            std::fprintf(stderr, "pack settle rejected token=%llu gate_rc=%d bytes=%llu lease_pending=%llu\n",
                (unsigned long long) token, int(st.error), (unsigned long long) bytes,
                (unsigned long long) grants().leases[index].pending_bytes());
            return false;
        }
        ++grants().settled_pins;
        ++grants().generation;
        return true;
    } catch (...) {
        return false;
    }
}

bool release_pin_grant(uint64_t token, uint64_t bytes) noexcept {
    if (!token) { return false; }
    const int64_t pid = authority_pid();
    if (!pid) { return false; }
    try {
        std::lock_guard<std::mutex> guard(grants().mutex);
        int index = -1;
        for (int j = 0; j < 64; ++j) {
            if (grants().live[j] && grants().tokens[j] == token &&
                grants().pids[j] == pid && grants().bytes[j] == bytes) {
                index = j;
                break;
            }
        }
        if (index < 0 || grants().used[index]) { return false; }
        grants().leases[index].release_pin(bytes);
        ++grants().generation;
        return true;
    } catch (...) {
        return false;
    }
}

bool release(uint64_t token, uint64_t bytes, outcome how) noexcept {
    return sign_release(token, bytes, static_cast<int>(how));
}

bool force_link() noexcept {
    const ggml_safe_io::authority::ops * table = ggml_safe_io::authority::authority_table();
    return table != nullptr && table->acquire == &sign_acquire && table->release == &sign_release;
}

grant_observation observe_grants() noexcept {
    grant_observation out;
    try {
        auto & state = grants();
        std::lock_guard<std::mutex> guard(state.mutex);
        const uint64_t before = state.generation;
        out.pending_grants = state.pending_bytes;
        out.released_bytes = state.released_bytes;
        out.published_grants = state.published;
        out.cancelled_grants = state.cancelled;
        out.failed_grants = state.failed;
        out.external_grants = state.external;
        out.read_grants = state.reads;
        out.settled_pins = state.settled_pins;
        out.attempts = 1;
        out.error = (before == state.generation) ? common_admission_error::ok
                                                 : common_admission_error::unstable;
        return out;
    } catch (...) {
        out.error = common_admission_error::internal;
        return out;
    }
}

}
