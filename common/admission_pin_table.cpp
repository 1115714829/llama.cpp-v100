#include "admission_pin_table.h"

#include "admission_authority.h"
#include "admission_gate.h"
#include "../ggml/src/ggml-pin-settle.h"

namespace {

int settle_pin(uint64_t token, uint64_t bytes) noexcept {

    return common_admission_authority::settle_grant(token, bytes) ? 0 : 1;
}

void release_pin(uint64_t token, uint64_t bytes) noexcept {
    common_admission_authority::release_pin_grant(token, bytes);
}

const ggml_pin_settle::ops the_ops = { &settle_pin, &release_pin };

bool registrar() noexcept {
    return ggml_pin_settle::register_ops(&the_ops);
}

[[maybe_unused]] const bool registered = registrar();

}

namespace common_admission_pin_table {

bool force_link() noexcept {
    return registrar() || ggml_pin_settle::table() != nullptr;
}

}
