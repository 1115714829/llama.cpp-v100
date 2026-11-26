#pragma once
#include <cstddef>
#include <cstdint>
namespace admission_cgroup_resolver {
constexpr size_t path_capacity = 4096;
constexpr size_t line_capacity = 16384;
constexpr size_t entry_capacity = 128;
constexpr size_t input_capacity = 1024 * 1024;
enum class error { ok, malformed, capacity, missing, ambiguous, namespace_unproven, membership_outside_visible_root };
struct result {
    error status;
    unsigned version;
    uint64_t mount_id;
    uint64_t parent_id;
    uint64_t device_major, device_minor;
    bool widest_visible_mapping;
    char mount_root[path_capacity];
    char mountpoint[path_capacity];
    char member[path_capacity];
    char relative_leaf[path_capacity];
    bool visible_root;
    bool global_ancestry_proven;
};

error resolve(const char * mountinfo, size_t mountinfo_size,
              const char * cgroup, size_t cgroup_size, result & out) noexcept;
}
