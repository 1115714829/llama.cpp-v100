#pragma once
#include "admission_cgroup_resolver.h"
#include "admission_ancestor_scan.h"
namespace admission_cgroup_observe {
enum class error { ok_local, io, capacity, embedded_nul, truncated, parse, identity, changed, scanner_unknown, unsupported };

struct workspace {
    char mounts[admission_cgroup_resolver::input_capacity];
    char membership[admission_cgroup_resolver::input_capacity];
    char reread[admission_cgroup_resolver::input_capacity];
};
struct result {
    error status;
    admission_cgroup_resolver::result mapping;
    admission_ancestor_scan::snapshot constraints;
    bool global_ancestry_proven;
};

error observe(workspace & storage, result & out) noexcept;
#ifdef COMMON_ADMISSION_TESTING
struct test_hooks {
    const char * proc_root = nullptr;
    const char * filesystem_root = nullptr;
    long (*filesystem_type)(int fd, long actual) noexcept = nullptr;
    void (*stat_identity)(int fd, unsigned long long & device, unsigned long long & inode) noexcept = nullptr;
    const char * (*fdinfo_bytes)(int fd, size_t & length) noexcept = nullptr;
    void (*after_scan)() noexcept = nullptr;
    size_t read_chunk = SIZE_MAX;
    unsigned interrupted_reads = 0;
    int read_error = 0;
    size_t read_calls = 0;
    size_t interrupt_call = SIZE_MAX;
};
void set_test_hooks(test_hooks * hooks) noexcept;

error test_read(int directory, const char * name, char * buffer, size_t capacity,
                size_t & used, size_t max_read, bool inject_eintr) noexcept;
#endif
}
