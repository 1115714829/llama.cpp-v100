#pragma once
#include "admission_numa.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_admission_host_class {
    COMMON_ADMISSION_HOST_OWN      = 0,
    COMMON_ADMISSION_HOST_BORROWED = 1,
};

struct common_admission_node {
    int      id          = -1;
    uint64_t total       = 0;
    uint64_t free        = 0;
    uint64_t reclaimable = 0;
    uint64_t os_reserve  = 0;
    uint64_t avail       = 0;
};

struct common_admission_supply {
    bool      probe_ok     = false;
    std::string probe_error;

    bool      cgroup_seen        = false;
    uint64_t  cgroup_limit       = 0;
    uint64_t  cgroup_usage       = 0;
    uint64_t  cgroup_reclaimable = 0;

    bool      numa_seen = false;
    std::vector<common_admission_node> nodes;

    uint64_t  mem_total     = 0;
    uint64_t  mem_available = 0;

    uint64_t  own_page_cache = 0;
    uint64_t  own_locked     = 0;
};

struct common_admission_demand {
    uint64_t committed_bytes = 0;
    uint64_t resident_bytes   = 0;
    uint64_t transient        = 0;

};

enum common_admission_verdict {
    COMMON_ADMISSION_ADMIT   = 0,
    COMMON_ADMISSION_DEGRADE = 1,
    COMMON_ADMISSION_REJECT  = 2,
};

struct common_admission_result {
    int      verdict    = COMMON_ADMISSION_REJECT;
    uint64_t demand     = 0;
    uint64_t resident   = 0;
    uint64_t available  = 0;
    uint64_t headroom   = 0;
    uint64_t usable     = 0;
    uint64_t deficit    = 0;
    uint64_t shed_bytes = 0;
    std::vector<common_admission_node> nodes;
    std::string detail;
};

struct common_admission_config {
    int      host_class        = COMMON_ADMISSION_HOST_BORROWED;
    uint64_t headroom_override = 0;
    bool     degrade_allowed   = true;
};

struct common_admission_paths {
    std::string proc_dir    = "/proc";
    std::string sys_dir     = "/sys";
    std::string cgroup_root = "/sys/fs/cgroup";
};

uint64_t common_admission_headroom_bytes(uint64_t mem_total, int host_class);

int common_admission_host_class_from_env();

common_admission_config common_admission_config_from_env();

common_admission_supply common_admission_probe(const common_admission_paths & paths = common_admission_paths());

common_admission_result common_admission_check(
        const common_admission_supply & supply,
        const common_admission_demand & demand,
        const common_admission_config & config);

std::string common_admission_detail(const common_admission_supply & supply, const common_admission_result & result);

bool common_admission_drop_file_cache(const std::string & path);

enum class common_admission_error { ok, rejected, probe_failed, overflow, wrong_process, scope_required, busy, internal, not_integrated, mapping_failed, policy_failed, lock_failed, cleanup_failed, registry_full, wrong_thread, unstable, invalid_owner_observation, materializing };
struct common_admission_status {
    common_admission_error error = common_admission_error::ok;
    uint64_t requested = 0;
    uint64_t pending = 0;
    uint64_t usable = 0;
};

enum class common_admission_scope_kind { initialization, transaction };
class common_admission_scope {
public:
    common_admission_scope() noexcept = default;
    ~common_admission_scope() noexcept;
    common_admission_scope(const common_admission_scope &) = delete;
    common_admission_scope & operator=(const common_admission_scope &) = delete;
    common_admission_status enter(common_admission_scope_kind kind) noexcept;
    void leave() noexcept;
private:
    bool entered_ = false;
    int64_t owner_pid_ = 0;
};

class common_admission_lease {
public:
    common_admission_lease() noexcept = default;
    ~common_admission_lease() noexcept;
    common_admission_lease(const common_admission_lease &) = delete;
    common_admission_lease & operator=(const common_admission_lease &) = delete;
    common_admission_lease(common_admission_lease && other) noexcept;
    common_admission_lease & operator=(common_admission_lease && other) noexcept;

    common_admission_status grow(uint64_t additional) noexcept;

    bool split_pending(uint64_t bytes, common_admission_lease & child) noexcept;

    void cancel_pending(uint64_t bytes) noexcept;

    common_admission_status settle_pin(uint64_t bytes) noexcept;

    void release_pin(uint64_t bytes) noexcept;
    uint64_t pending_bytes() const noexcept { return pending_; }

    uint64_t settled_pinned_bytes() const noexcept { return settled_pinned_; }
private:
    friend class common_admission_owner;
    uint64_t pending_ = 0;
    uint64_t settled_pinned_ = 0;
    int64_t owner_pid_ = 0;
};

class common_admission_owner {
public:
    common_admission_owner() noexcept = default;
    ~common_admission_owner() noexcept;
    common_admission_owner(const common_admission_owner &) = delete;
    common_admission_owner & operator=(const common_admission_owner &) = delete;
    common_admission_owner(common_admission_owner && other) noexcept;
    common_admission_owner & operator=(common_admission_owner &&) = delete;
    common_admission_error create(uint64_t bytes) noexcept;
    common_admission_error close() noexcept;
    admission_numa::status numa_status() const noexcept { return numa_status_; }
    void * data() const noexcept { return published_ ? mapping_ : nullptr; }
    uint64_t size() const noexcept { return published_ ? bytes_ : 0; }
private:
    void * mapping_ = nullptr;
    bool published_ = false;
    admission_numa::status numa_status_{admission_numa::code::ok,0,"not-started"};
    uint64_t bytes_ = 0;
    int slot_ = -1;
    int64_t process_ = 0;
};
#ifdef COMMON_ADMISSION_OWNER_TESTING
struct common_admission_owner_test_input {
    common_admission_supply supply;
    common_admission_config config;
    int fail_stage = 0;
    bool fail_cleanup = false;
    unsigned fail_touch_page = 1;
    const admission_numa::operations * numa_operations = nullptr;

    bool override_process_status = false;
    uint64_t override_rss_file = 0;
    uint64_t override_vm_locked = 0;

    void (*observe_point)() = nullptr;

    void (*materialize_point)() = nullptr;

    bool override_paths = false;
    common_admission_paths paths;
    bool throw_from_status = false;
};
void common_admission_owner_test_set(common_admission_owner_test_input *) noexcept;
void common_admission_owner_test_counts(uint64_t & pending, uint64_t & owned, unsigned & entries) noexcept;

void common_admission_owner_test_materializing(unsigned & materializing) noexcept;
#endif

struct common_admission_observation {
    common_admission_error error = common_admission_error::unstable;
    uint64_t rss_file          = 0;
    uint64_t vm_locked         = 0;
    uint64_t owned_locked_anon = 0;
    uint64_t unclassified_locked = 0;
    uint64_t held_bytes        = 0;
    uint64_t legacy_resident_cap = 0;
    unsigned attempts          = 0;
    bool repeated              = false;
};

common_admission_observation common_admission_observe() noexcept;

#ifdef COMMON_ADMISSION_OWNER_TESTING

void common_admission_pin_counts(uint64_t & registered, unsigned & settles) noexcept;
#endif
