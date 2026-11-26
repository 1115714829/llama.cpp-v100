#pragma once

#include "ggml.h"

#include "ggml-safe-io-authority.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(GGML_SHARED)
# if defined(_WIN32) && !defined(__MINGW32__)
#  if defined(GGML_BUILD)
#   define GGML_SAFE_IO_MEMBER __declspec(dllexport)
#  else
#   define GGML_SAFE_IO_MEMBER __declspec(dllimport)
#  endif
# else
#  define GGML_SAFE_IO_MEMBER __attribute__((visibility("default")))
# endif
#else
# define GGML_SAFE_IO_MEMBER
#endif

namespace ggml_safe_io {

enum class error : uint8_t {
    none, not_integrated, authority_refused, unsupported, invalid_argument, invalid_state,
    path_changed, unsafe_path, unsupported_attributes, unhealthy, probe_failed,
    no_space, system, zero_write, length_mismatch, cancelled, wrong_process,
};

enum class stage : uint8_t {
    idle, authorization, directory, health, space, lock_open, lock_acquired,
    target, temp_open, preallocate, write, window_sync, window_confirmed,
    attributes, file_sync, rename, directory_sync, cleanup, complete,

    open, identity, read, mmap, count,
};

struct status {
    error code;
    stage where;
    int sys_errno;
    int cleanup_errno;
    uint64_t written;
    uint64_t confirmed;
    bool published;
    bool durability_confirmed;
};

#ifdef GGML_SAFE_IO_WRITER_TEST

namespace testing {
struct fault {
    stage where;
    unsigned occurrence;
    int sys_errno;
    size_t short_count;
};
struct event {
    stage where;
    uint64_t written;
    uint64_t confirmed;
    uint64_t amount;
};
struct hooks {
    fault faults[16];
    size_t fault_count;
    unsigned calls[static_cast<size_t>(stage::count)];
    event events[1024];
    size_t event_count;
    bool overflow;
};
struct access;
}
#endif

class reader;

class writer {
public:
    GGML_SAFE_IO_MEMBER writer() noexcept;
    GGML_SAFE_IO_MEMBER ~writer() noexcept;
    writer(const writer &) = delete;
    writer & operator=(const writer &) = delete;
    writer(writer &&) = delete;
    writer & operator=(writer &&) = delete;

    GGML_SAFE_IO_MEMBER status begin(const char * directory, const char * basename,
                          uint64_t length, uint64_t window) noexcept;

    GGML_SAFE_IO_MEMBER status begin(const char * directory, const char * basename,
                          uint64_t length, uint64_t window, uint64_t footprint) noexcept;
    GGML_SAFE_IO_MEMBER status begin_pack(const char * directory, const char * basename,
                           uint64_t length, uint64_t window, uint64_t token, bool manifest = false) noexcept;
    GGML_SAFE_IO_MEMBER status append(const void * data, size_t length) noexcept;

    GGML_SAFE_IO_MEMBER status write_at(uint64_t offset, const void * data, size_t length) noexcept;
    GGML_SAFE_IO_MEMBER status finish() noexcept;
    GGML_SAFE_IO_MEMBER status finish_pack() noexcept;
    GGML_SAFE_IO_MEMBER bool handoff_pack_lock(reader & target) noexcept;
    GGML_SAFE_IO_MEMBER status cancel() noexcept;
    GGML_SAFE_IO_MEMBER status result() const noexcept;

    GGML_SAFE_IO_MEMBER int temp_fd() const noexcept;

    GGML_SAFE_IO_MEMBER uint64_t authority_bytes() const noexcept;

private:
#ifdef GGML_SAFE_IO_WRITER_TEST
    friend struct testing::access;
    testing::hooks * hooks_;
#endif

    uint64_t authority_token_ = 0;
    uint64_t authority_bytes_ = 0;
    status s_;
    int dir_, lock_, temp_;
    int64_t pid_;
    uint64_t total_, window_, page_;
    uint64_t dir_dev_, dir_ino_, lock_dev_, lock_ino_, temp_dev_, temp_ino_;
    uint32_t mode_;
    bool active_;

    std::vector<std::pair<uint64_t, uint64_t>> spans_;
    bool explicit_ = false;
    bool spans_no_memory_ = false;
    char path_[4096], target_[181], lock_name_[208], temp_name_[256];

    status finish_impl(bool retain_name_lock) noexcept;
    status start_private(const char *, const char *, uint64_t, uint64_t) noexcept;

    void release_authority(authority::outcome) noexcept;
    status fail(error, stage, int = 0) noexcept;
    void cleanup() noexcept;

    bool cover(uint64_t offset, uint64_t length) noexcept;

    bool whole_file() const noexcept;
    bool identity() noexcept;
    bool health() noexcept;
    bool space() noexcept;
    bool confirm_window() noexcept;
    int64_t inject(stage, uint64_t = 0) noexcept;
};

class reader {
public:
    GGML_SAFE_IO_MEMBER reader() noexcept;
    GGML_SAFE_IO_MEMBER ~reader() noexcept;
    reader(const reader &) = delete;
    reader & operator=(const reader &) = delete;
    reader(reader &&) = delete;
    reader & operator=(reader &&) = delete;

    GGML_SAFE_IO_MEMBER status open(const char * path, uint64_t resident_bound = 0,
                                    bool direct = false) noexcept;
    GGML_SAFE_IO_MEMBER status open_pack(const char * path, uint64_t length, uint64_t token,
                                         bool writer_handoff = false) noexcept;
    GGML_SAFE_IO_MEMBER status open_manifest(const char * path, uint64_t bound, uint64_t token) noexcept;
    GGML_SAFE_IO_MEMBER uint64_t pack_token() const noexcept;

    GGML_SAFE_IO_MEMBER status read(void * ptr, size_t len, uint64_t offset) noexcept;

    GGML_SAFE_IO_MEMBER status mmap(void ** addr, size_t len, uint64_t offset) noexcept;

    GGML_SAFE_IO_MEMBER status reopen_buffered() noexcept;

    GGML_SAFE_IO_MEMBER void close() noexcept;
    GGML_SAFE_IO_MEMBER status result() const noexcept;

    GGML_SAFE_IO_MEMBER int fd() const noexcept;

    GGML_SAFE_IO_MEMBER int detach_fd() noexcept;

    GGML_SAFE_IO_MEMBER size_t alignment() const noexcept;
    GGML_SAFE_IO_MEMBER uint64_t size() const noexcept;
    GGML_SAFE_IO_MEMBER bool qualified() const noexcept;
    GGML_SAFE_IO_MEMBER bool matches_path(const char * path, int held_fd) noexcept;

private:
    status open_impl(const char *, uint64_t, bool, uint64_t, bool manifest = false) noexcept;
    friend class writer;
    int pack_name_lock_ = -1;
    static const unsigned max_maps = 8;

    uint64_t authority_token_ = 0;
    uint64_t authority_bytes_ = 0;
    status s_;
    int fd_ = -1;
    bool fd_owned_ = false;
    bool direct_ = false;
    int64_t pid_ = 0;
    uint64_t size_ = 0;
    size_t alignment_ = 1;
    void * map_addr_[max_maps] = {};
    size_t map_len_[max_maps] = {};
    unsigned map_count_ = 0;
    uint64_t file_dev_ = 0, file_ino_ = 0;

    int64_t  mtim_sec_ = 0;
    int64_t  mtim_nsec_ = 0;
    char path_[4096] = {};
    bool active_ = false;

    void release_authority(authority::outcome how) noexcept;
    void release_resources() noexcept;
    status fail(error code, stage at, int e = 0) noexcept;
    bool identity() noexcept;
    bool health() noexcept;
};

GGML_API const char * error_name(error code) noexcept;

#ifdef GGML_SAFE_IO_WRITER_TEST
namespace testing {
struct access {

    static status begin(writer &, const char *, const char *, uint64_t,
                        uint64_t, hooks * = nullptr) noexcept;
    static bool rounded_pages(uint64_t, uint64_t, uint64_t &) noexcept;
};
}
#endif

}

#undef GGML_SAFE_IO_MEMBER
