#pragma once
#include <cstddef>
#include <cstdint>

namespace admission_ancestor_scan {

constexpr size_t max_levels = 64;
constexpr size_t max_path = 4096;
enum class error { ok, invalid_path, inaccessible, malformed, capacity, changed, hierarchy_unknown };
struct level {
    uint64_t device, inode, limit, current;
    uint32_t version;
    bool finite, hard_usage, visible_root;
};
struct snapshot {
    level levels[max_levels];
    size_t count;
    error status;
};

#ifdef COMMON_ADMISSION_TESTING
using between_passes_hook = void (*)(void *) noexcept;
void test_between_passes(between_passes_hook hook, void * context) noexcept;
#endif

error scan_fd(int controller_fd, const char * relative_leaf,
              unsigned version, snapshot & output) noexcept;
error scan(const char * controller_root, const char * relative_leaf,
           unsigned version, snapshot & output) noexcept;
}
