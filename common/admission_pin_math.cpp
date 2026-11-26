#include "admission_pin_math.h"
#include <algorithm>

namespace admission_pin_math {

settle_result settle(const settle_input & in) noexcept {
    settle_result out;
    if (in.pin_bytes == 0) {
        out.status = error::invalid_input;
        return out;
    }
    if (in.pending < in.pin_bytes) {

        out.status = error::uncovered;
        return out;
    }
    uint64_t registered = 0;
    if (in.owned_pinned > UINT64_MAX - in.pin_bytes) {
        out.status = error::overflow;
        return out;
    }
    registered = in.owned_pinned + in.pin_bytes;

    if (in.rss_file < registered) {
        out.status = error::unproven;
        out.vmlck_moved = in.vm_locked > 0 && in.rss_file < in.vm_locked;
        return out;
    }
    out.status = error::ok;
    out.retire = true;
    out.retire_bytes = in.pin_bytes;
    out.registered_after = registered;
    out.overhead = in.rss_file - registered;
    out.vmlck_moved = in.vm_locked > 0;
    return out;
}

release_result release(const release_input & in) noexcept {
    release_result out;
    if (in.pin_bytes == 0) {
        out.status = error::invalid_input;
        return out;
    }
    out.status = error::ok;
    out.release_bytes = std::min(in.pin_bytes, in.owned_pinned);
    return out;
}

form_result classify_form(const form_input & in) noexcept {
    form_result out;
    if (in.pin_bytes == 0) {
        out.matched = form::observed;
        return out;
    }
    const uint64_t x = in.pin_bytes;

    const uint64_t eps = in.delta_rss_file > x ? in.delta_rss_file - x : 0;
    out.structural = x + eps;
    out.ledger_exact = (in.pending_after == 0);
    out.bound_held = (in.delta_rss_file >= x);

    const bool vmlck_form = (in.delta_vm_locked >= x);
    if (vmlck_form) {
        out.matched = form::mlock_migration;

        out.structural = 2 * x + eps;
        out.formula_double = true;
    } else if (in.delta_usable >= x + eps - (x / 16) && in.delta_usable <= x + eps + (x / 16)) {

        out.matched = form::gup;
    } else {

        out.matched = form::observed;
    }
    if (out.structural > in.delta_usable) {
        out.residual = out.structural - in.delta_usable;
    }
    if (x > in.delta_usable) {

        out.naive_miss = x - in.delta_usable;
    }
    return out;
}

}
