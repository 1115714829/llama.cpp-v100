#include "ggml-pin-settle.h"

#include <mutex>

namespace ggml_pin_settle {
namespace {
std::mutex register_mutex;
const ops * settled_table = nullptr;
}

bool register_ops(const ops * table) noexcept {
    if (!table || !table->settle || !table->release) { return false; }
    std::lock_guard<std::mutex> guard(register_mutex);
    if (settled_table) { return false; }
    settled_table = table;
    return true;
}

const ops * table() noexcept {
    std::lock_guard<std::mutex> guard(register_mutex);
    return settled_table;
}
}
