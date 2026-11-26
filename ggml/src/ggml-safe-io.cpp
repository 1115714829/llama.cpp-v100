#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml-safe-io.h"
#include "ggml-safe-io-authority.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/xattr.h>
#include <time.h>
#include <unistd.h>
#endif

namespace ggml_safe_io {

namespace authority {
namespace {
std::mutex register_mutex;
const ops * signed_table = nullptr;
}

bool authority_register(const ops * table) noexcept {
    if (!table || !table->acquire || !table->release) { return false; }
    std::lock_guard<std::mutex> guard(register_mutex);
    if (signed_table) { return false; }
    signed_table = table;
    return true;
}

const ops * authority_table() noexcept {
    std::lock_guard<std::mutex> guard(register_mutex);
    return signed_table;
}
}

namespace {
bool rounded(uint64_t bytes, uint64_t page, uint64_t & out) noexcept {
    if (!page || (page & (page - 1))) { return false; }
    const uint64_t extra = bytes % page ? page - bytes % page : 0;
    if (bytes > UINT64_MAX - extra) { return false; }
    out = bytes + extra;
    return true;
}

#if defined(__linux__)
int open_directory(const char * path) noexcept {
    if (!path || !*path) { errno = EINVAL; return -1; }
    int fd = open(*path == '/' ? "/" : ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) { return -1; }
    const char * p = path;
    while (*p) {
        while (*p == '/') { ++p; }
        if (!*p) { break; }
        char component[256];
        size_t n = 0;
        while (*p && *p != '/') {
            if (n == sizeof(component) - 1) { close(fd); errno = ENAMETOOLONG; return -1; }
            component[n++] = *p++;
        }
        component[n] = 0;
        if (!std::strcmp(component, "..")) { close(fd); errno = EINVAL; return -1; }
        const int next = openat(fd, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        const int saved = errno;
        close(fd);
        if (next < 0) { errno = saved; return -1; }
        fd = next;
    }
    return fd;
}

bool ensure_directory(const char * path) noexcept {
    if (!path || !*path) { errno = EINVAL; return false; }
    char cur[4096];
    size_t n = 0;
    if (*path == '/') { cur[n++] = '/'; }
    const char * p = path;
    while (*p) {
        while (*p == '/') { ++p; }
        if (!*p) { break; }
        if (n && n < sizeof(cur) - 1) { cur[n++] = '/'; }
        while (*p && *p != '/') {
            if (n == sizeof(cur) - 1) { errno = ENAMETOOLONG; return false; }
            cur[n++] = *p++;
        }
        cur[n] = 0;
        if (std::strcmp(cur, "/") != 0) {
            if (mkdir(cur, 0755) != 0 && errno != EEXIST) { return false; }
        }
    }
    return true;
}

bool plain_attributes(int fd) noexcept {
    const ssize_t n = flistxattr(fd, nullptr, 0);
    if (n < 0) { return false; }
    if (n != 0) { errno = ENOTSUP; return false; }
    return true;
}

bool ordinary(const struct stat & st) noexcept {
    return S_ISREG(st.st_mode) && st.st_uid == geteuid() && st.st_gid == getegid() &&
           st.st_nlink == 1 && !(st.st_mode & 07000);
}

bool dirty_sample(uint64_t & dirty, uint64_t & writeback) noexcept {
    const int fd = open("/proc/meminfo", O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return false; }
    char buf[8192];
    size_t used = 0;
    while (used < sizeof(buf) - 1) {
        const ssize_t n = read(fd, buf + used, sizeof(buf) - 1 - used);
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0) { const int e = errno; close(fd); errno = e; return false; }
        if (!n) { break; }
        used += static_cast<size_t>(n);
    }
    close(fd);
    buf[used] = 0;
    bool found[2] = {false, false};
    const char * keys[2] = {"Dirty:", "Writeback:"};
    uint64_t * values[2] = {&dirty, &writeback};
    for (size_t k = 0; k != 2; ++k) {
        const char * p = buf;
        while (*p) {
            if (!std::strncmp(p, keys[k], std::strlen(keys[k]))) {
                p += std::strlen(keys[k]);
                while (*p == ' ' || *p == '\t') { ++p; }
                uint64_t v = 0;
                bool digit = false;
                while (*p >= '0' && *p <= '9') {
                    const unsigned d = static_cast<unsigned>(*p++ - '0');
                    if (v > (UINT64_MAX - d) / 10) { errno = EOVERFLOW; return false; }
                    v = v * 10 + d;
                    digit = true;
                }
                while (*p == ' ' || *p == '\t') { ++p; }
                if (!digit || std::strncmp(p, "kB", 2) || v > UINT64_MAX / 1024) {
                    errno = EIO; return false;
                }
                *values[k] = v * 1024;
                found[k] = true;
                break;
            }
            while (*p && *p != '\n') { ++p; }
            if (*p) { ++p; }
        }
    }
    if (!found[0] || !found[1]) { errno = EIO; return false; }
    return true;
}

void temp_basename(char * dst, const char * target, const unsigned char * random) noexcept {
    static const char digits[] = "0123456789abcdef";
    *dst++ = '.';
    while (*target) { *dst++ = *target++; }
    const char * suffix = ".safeio-tmp-";
    while (*suffix) { *dst++ = *suffix++; }
    for (unsigned i = 0; i != 16; ++i) {
        *dst++ = digits[random[i] >> 4];
        *dst++ = digits[random[i] & 15];
    }
    *dst = 0;
}
#endif
}

writer::writer() noexcept :
#ifdef GGML_SAFE_IO_WRITER_TEST
    hooks_(nullptr),
#endif
    s_{error::none, stage::idle, 0, 0, 0, 0, false, false}, dir_(-1), lock_(-1), temp_(-1),
    pid_(0), total_(0), window_(0), page_(0), dir_dev_(0), dir_ino_(0),
    lock_dev_(0), lock_ino_(0), temp_dev_(0), temp_ino_(0), mode_(0600), active_(false), path_{}, target_{},
    lock_name_{}, temp_name_{} {}

writer::~writer() noexcept {

    cleanup();
    release_authority(authority::outcome_failed);
}
status writer::result() const noexcept { return s_; }

status writer::begin(const char * directory, const char * basename, uint64_t length,
                     uint64_t window) noexcept {

    return begin(directory, basename, length, window, length);
}

status writer::begin(const char * directory, const char * basename, uint64_t length,
                     uint64_t window, uint64_t footprint) noexcept {

    if (active_) { return fail(error::invalid_state, stage::authorization); }
    if (authority_token_) {

        return fail(error::invalid_state, stage::authorization);
    }
    const authority::ops * table = authority::authority_table();
    if (!table || !table->acquire) {

        s_ = {error::not_integrated, stage::authorization, 0, 0, 0, 0, false, false};
        return s_;
    }

    uint64_t padded = 0;
    uint64_t charge = 0;
#if defined(__linux__)
    const long p = sysconf(_SC_PAGESIZE);
    if (p <= 0) { return fail(error::invalid_argument, stage::authorization); }
    if (length && !rounded(length, static_cast<uint64_t>(p), padded)) {
        return fail(error::invalid_argument, stage::authorization);
    }
    if (footprint && !rounded(footprint, static_cast<uint64_t>(p), charge)) {
        return fail(error::invalid_argument, stage::authorization);
    }
    if (charge > padded) { return fail(error::invalid_argument, stage::authorization); }

    if (charge < padded && window > charge) {
        return fail(error::invalid_argument, stage::authorization);
    }
    uint64_t usable = 0;
    const uint64_t token = table->acquire(charge, &usable);
#else

    (void) footprint;
    const uint64_t token = table->acquire(0, nullptr);
#endif
    if (!token) {

        s_ = {error::authority_refused, stage::authorization, 0, 0, 0, 0, false, false};
        return s_;
    }
    authority_token_ = token;
    authority_bytes_ = charge;
    const status s = start_private(directory, basename, length, window);
    if (s.code != error::none) {

        release_authority(authority::outcome_failed);
    }
    return s;
}

status writer::begin_pack(const char * directory, const char * basename, uint64_t length,
                          uint64_t window, uint64_t token, bool manifest) noexcept {
#if defined(__linux__)
    if (active_ || authority_token_) { return fail(error::invalid_state, stage::authorization); }
    const long page = sysconf(_SC_PAGESIZE);
    uint64_t padded = 0;
    if (page <= 0 || !length || !window || !rounded(length, static_cast<uint64_t>(page), padded) ||
            window > padded || window % static_cast<uint64_t>(page) ||
            (manifest ? window != padded : length != padded)) {
        return fail(error::invalid_argument, stage::authorization);
    }
    const auto * table = authority::authority_table();
    if (!table || !table->claim_batch || !table->claim_batch(token, window, manifest ? authority::ops::batch_use::pack_manifest : authority::ops::batch_use::pack_write)) {
        return fail(error::authority_refused, stage::authorization);
    }
    authority_token_ = token;
    authority_bytes_ = window;
    const auto s = start_private(directory, basename, length, window);
    if (s.code != error::none) { release_authority(authority::outcome_failed); }
    return s;
#else
    (void) directory; (void) basename; (void) length; (void) window; (void) token; (void) manifest;
    return fail(error::unsupported, stage::authorization);
#endif
}

void writer::release_authority(authority::outcome how) noexcept {
    const uint64_t token = authority_token_;
    if (!token) { return; }
    if (s_.cleanup_errno) {

        std::fputs("safeio writer cleanup failed; authority charge retained\n", stderr);
        return;
    }
    const authority::ops * table = authority::authority_table();
    if (!table || !table->release) { return; }
    if (table->release(token, authority_bytes_, static_cast<int>(how))) {
        authority_token_ = 0;
    }
}

int64_t writer::inject(stage at, uint64_t amount) noexcept {
#ifdef GGML_SAFE_IO_WRITER_TEST
    if (hooks_) {
        const size_t idx = static_cast<size_t>(at);
        const unsigned call = ++hooks_->calls[idx];
        if (hooks_->event_count < sizeof(hooks_->events) / sizeof(hooks_->events[0])) {
            hooks_->events[hooks_->event_count++] = {at, s_.written, s_.confirmed, amount};
        } else { hooks_->overflow = true; }
        for (size_t i = 0; i < hooks_->fault_count && i < 16; ++i) {
            const testing::fault & f = hooks_->faults[i];
            if (f.where == at && f.occurrence == call) {
                if (f.sys_errno) { errno = f.sys_errno; return -1; }
                return static_cast<int64_t>(f.short_count);
            }
        }
    }
#else
    (void) at; (void) amount;
#endif
    return -2;
}

status writer::fail(error code, stage at, int e) noexcept {
    if (s_.code == error::none) { s_.code = code; s_.where = at; s_.sys_errno = e; }
    cleanup();
    return s_;
}

void writer::cleanup() noexcept {
#if defined(__linux__)

    const bool owner = pid_ == static_cast<int64_t>(getpid());
    if (owner && dir_ >= 0 && temp_name_[0] && !s_.published) {
        struct stat named, held;
        if (temp_ >= 0 && fstat(temp_, &held) == 0 &&
            fstatat(dir_, temp_name_, &named, AT_SYMLINK_NOFOLLOW) == 0) {
            if (held.st_dev == named.st_dev && held.st_ino == named.st_ino) {
                if (unlinkat(dir_, temp_name_, 0) && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
            } else if (!s_.cleanup_errno) { s_.cleanup_errno = ESTALE; }
        } else if (errno != ENOENT && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
    }
    if (temp_ >= 0) {
        if (close(temp_) && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
        temp_ = -1;
    }
    temp_name_[0] = 0;
    if (lock_ >= 0) {
        if (close(lock_) && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
        lock_ = -1;
    }
    if (dir_ >= 0) {
        if (close(dir_) && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
        dir_ = -1;
    }
#endif
    active_ = false;
}

bool writer::health() noexcept {
#if defined(__linux__)
    if (inject(stage::health) == -1) { fail(error::unhealthy, stage::health, errno); return false; }
    uint64_t d = 0, w = 0;
    if (!dirty_sample(d, w)) { fail(error::probe_failed, stage::health, errno); return false; }
    if (d > (uint64_t(8) << 30) && !w) {
        struct timespec delay = {2, 0};
        while (nanosleep(&delay, &delay) < 0) {
            if (errno != EINTR) { fail(error::probe_failed, stage::health, errno); return false; }
        }
        if (!dirty_sample(d, w)) { fail(error::probe_failed, stage::health, errno); return false; }
        if (d > (uint64_t(8) << 30) && !w) { fail(error::unhealthy, stage::health); return false; }
    }
    return true;
#else
    return false;
#endif
}

bool writer::space() noexcept {
#if defined(__linux__)
    struct statvfs v;
    if (inject(stage::space) == -1) { fail(error::no_space, stage::space, errno); return false; }
    if (fstatvfs(dir_, &v) < 0) { fail(error::probe_failed, stage::space, errno); return false; }

    uint64_t padded = 0;
    if (!rounded(total_, page_, padded) || page_ > UINT64_MAX / 2 ||
        padded > UINT64_MAX - padded / 2 || padded + padded / 2 > UINT64_MAX - 2 * page_) {
        fail(error::invalid_argument, stage::space, EOVERFLOW); return false;
    }
    const uint64_t need = padded + padded / 2 + 2 * page_;
    const uint64_t unit = v.f_frsize ? v.f_frsize : v.f_bsize;
    if (!unit) { fail(error::probe_failed, stage::space, EIO); return false; }
    const uint64_t free = v.f_bavail > UINT64_MAX / unit ? UINT64_MAX : v.f_bavail * unit;
    if (free < need) { fail(error::no_space, stage::space, ENOSPC); return false; }
    return true;
#else
    return false;
#endif
}

bool writer::identity() noexcept {
#if defined(__linux__)
    if (pid_ != static_cast<int64_t>(getpid())) { fail(error::wrong_process, stage::directory); return false; }
    const int fd = open_directory(path_);
    struct stat st;
    if (fd < 0) { fail(error::path_changed, stage::directory, errno); return false; }
    const int r = fstat(fd, &st);
    const int saved = errno;
    close(fd);
    if (r < 0 || static_cast<uint64_t>(st.st_dev) != dir_dev_ || static_cast<uint64_t>(st.st_ino) != dir_ino_) {
        fail(error::path_changed, stage::directory, r < 0 ? saved : 0); return false;
    }
    if (lock_ >= 0 && (fstatat(dir_, lock_name_, &st, AT_SYMLINK_NOFOLLOW) < 0 ||
        static_cast<uint64_t>(st.st_dev) != lock_dev_ || static_cast<uint64_t>(st.st_ino) != lock_ino_)) {
        fail(error::path_changed, stage::lock_acquired, errno); return false;
    }
    if (temp_ >= 0 && (fstatat(dir_, temp_name_, &st, AT_SYMLINK_NOFOLLOW) < 0 ||
        static_cast<uint64_t>(st.st_dev) != temp_dev_ || static_cast<uint64_t>(st.st_ino) != temp_ino_)) {
        fail(error::path_changed, stage::temp_open, errno); return false;
    }
    return true;
#else
    return false;
#endif
}

status writer::start_private(const char * directory, const char * basename, uint64_t length, uint64_t window) noexcept {
#if defined(__linux__)
    if (active_ || dir_ >= 0) { return fail(error::invalid_state, stage::idle); }
    s_ = {error::none, stage::idle, 0, 0, 0, 0, false, false};
    const long p = sysconf(_SC_PAGESIZE);
    if (!directory || !basename || !*directory || !*basename || std::strlen(directory) >= sizeof(path_) ||
        std::strlen(basename) >= sizeof(target_) || std::strchr(basename, '/') ||
        std::strstr(basename, ".safeio-lock") || std::strstr(basename, ".safeio-tmp-") ||
        !std::strcmp(basename, ".") || !std::strcmp(basename, "..") ||
        p <= 0 || !window || window % static_cast<uint64_t>(p) ||
        length > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        window > static_cast<uint64_t>(SSIZE_MAX)) {
        return fail(error::invalid_argument, stage::directory);
    }
    uint64_t padded = 0;
    if (!rounded(length, static_cast<uint64_t>(p), padded)) { return fail(error::invalid_argument, stage::directory); }
    std::strcpy(path_, directory);
    std::strcpy(target_, basename);
    std::strcpy(lock_name_, ".");
    std::strcat(lock_name_, basename);
    std::strcat(lock_name_, ".safeio-lock");
    total_ = length; window_ = window; page_ = static_cast<uint64_t>(p);

    spans_.clear();
    explicit_ = false;
    spans_no_memory_ = false;
    pid_ = static_cast<int64_t>(getpid());

    if (!ensure_directory(path_)) { return fail(error::unsafe_path, stage::directory, errno); }
    dir_ = open_directory(path_);
    if (dir_ < 0) { return fail(error::unsafe_path, stage::directory, errno); }
    struct stat st;
    if (fstat(dir_, &st) < 0) { return fail(error::system, stage::directory, errno); }
    if (st.st_uid != geteuid() || st.st_gid != getegid() || (st.st_mode & 07022) || !plain_attributes(dir_)) {
        return fail(error::unsupported_attributes, stage::directory, errno);
    }
    dir_dev_ = static_cast<uint64_t>(st.st_dev); dir_ino_ = static_cast<uint64_t>(st.st_ino);
    if (!health() || !space()) { return s_; }
    if (inject(stage::lock_open) == -1) { return fail(error::system, stage::lock_open, errno); }
    lock_ = openat(dir_, lock_name_, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_ < 0 && errno == EEXIST) {
        lock_ = openat(dir_, lock_name_, O_RDWR | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    }
    if (lock_ < 0) { return fail(error::unsafe_path, stage::lock_open, errno); }
    if (fstat(lock_, &st) < 0) { return fail(error::system, stage::lock_open, errno); }
    if (!ordinary(st) || (st.st_mode & 0777) != 0600 || !plain_attributes(lock_)) {
        return fail(error::unsupported_attributes, stage::lock_open, errno);
    }
    lock_dev_ = static_cast<uint64_t>(st.st_dev); lock_ino_ = static_cast<uint64_t>(st.st_ino);
    while (flock(lock_, LOCK_EX) < 0) {
        if (errno != EINTR) { return fail(error::system, stage::lock_acquired, errno); }
    }
    inject(stage::lock_acquired);
    if (!identity() || !health() || !space()) { return s_; }
    mode_ = 0600;
    if (fstatat(dir_, target_, &st, AT_SYMLINK_NOFOLLOW) == 0) {
        if (!ordinary(st)) { return fail(error::unsafe_path, stage::target); }
        const int old = openat(dir_, target_, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
        if (old < 0) { return fail(error::unsafe_path, stage::target, errno); }
        struct stat held;
        const bool ok = fstat(old, &held) == 0 && held.st_dev == st.st_dev && held.st_ino == st.st_ino && plain_attributes(old);
        const int saved = errno;
        close(old);
        if (!ok) { return fail(error::unsupported_attributes, stage::target, saved); }
        mode_ = st.st_mode & 0777;
    } else if (errno != ENOENT) { return fail(error::system, stage::target, errno); }
    unsigned char random[16];
    size_t have = 0;
    while (have < sizeof(random)) {
        const ssize_t n = syscall(SYS_getrandom, random + have, sizeof(random) - have, 0);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { return fail(error::system, stage::temp_open, n < 0 ? errno : EIO); }
        have += static_cast<size_t>(n);
    }
    char name[256];
    temp_basename(name, target_, random);
    if (inject(stage::temp_open) == -1) { return fail(error::system, stage::temp_open, errno); }
    temp_ = openat(dir_, name, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (temp_ < 0) { return fail(error::system, stage::temp_open, errno); }
    std::strcpy(temp_name_, name);
    if (fstat(temp_, &st) < 0 || !ordinary(st) || !plain_attributes(temp_)) {
        return fail(error::unsupported_attributes, stage::temp_open, errno);
    }
    temp_dev_ = static_cast<uint64_t>(st.st_dev); temp_ino_ = static_cast<uint64_t>(st.st_ino);

    if (length) {
        for (;;) {
            const int r = inject(stage::preallocate, length) == -1 ? -1 : fallocate(temp_, 0, 0, static_cast<off_t>(length));
            if (!r) { break; }
            if (errno != EINTR) { return fail(error::system, stage::preallocate, errno); }
        }
    }
    active_ = true;
    s_.where = stage::write;
    return s_;
#else
    (void) directory; (void) basename; (void) length; (void) window;
    return fail(error::unsupported, stage::directory);
#endif
}

bool writer::confirm_window() noexcept {
#if defined(__linux__)
    if (s_.written == s_.confirmed) { return true; }
    for (;;) {
        const int r = inject(stage::window_sync) == -1 ? -1 : fdatasync(temp_);
        if (!r) { break; }
        if (errno != EINTR) { fail(error::system, stage::window_sync, errno); return false; }
    }
    s_.confirmed = s_.written;
    inject(stage::window_confirmed);
    return true;
#else
    return false;
#endif
}

bool writer::cover(uint64_t offset, uint64_t length) noexcept {
    if (length == 0) { return true; }
    try {

        size_t i = 0;
        while (i < spans_.size() && spans_[i].second <= offset) { ++i; }
        if (i < spans_.size() && spans_[i].first < offset + length) {

            return false;
        }
        uint64_t begin = offset;
        uint64_t end   = offset + length;
        size_t lo = i, hi = i;
        if (i > 0 && spans_[i - 1].second == offset) { begin = spans_[i - 1].first; lo = i - 1; }
        if (i < spans_.size() && spans_[i].first == end) { end = spans_[i].second; hi = i + 1; }
        if (lo == hi) {

            spans_.insert(spans_.begin() + i, std::make_pair(begin, end));
            return true;
        }
        spans_[lo] = std::make_pair(begin, end);
        if (hi > lo + 1) { spans_.erase(spans_.begin() + lo + 1, spans_.begin() + hi); }
        return true;
    } catch (...) {

        spans_no_memory_ = true;
        return false;
    }
}

bool writer::whole_file() const noexcept {
    if (total_ == 0) { return true; }
    return spans_.size() == 1 && spans_[0].first == 0 && spans_[0].second == total_;
}

status writer::append(const void * data, size_t length) noexcept {

    if (explicit_) { return write_at(s_.written, data, length); }
    if (s_.code != error::none) { return s_; }
    if (!active_) { return fail(error::invalid_state, stage::write); }
#if defined(__linux__)
    if (!identity()) { return s_; }
    if ((!data && length) || length > total_ - s_.written) { return fail(error::length_mismatch, stage::write); }
    const unsigned char * p = static_cast<const unsigned char *>(data);
    while (length) {
        if (s_.written == s_.confirmed && !health()) { return s_; }
        const uint64_t room = window_ - (s_.written - s_.confirmed);
        size_t n = length;
        if (n > room) { n = static_cast<size_t>(room); }
        const int64_t injected = inject(stage::write, n);
        ssize_t done;
        if (injected == -1) { done = -1; }
        else {
            if (injected >= 0 && static_cast<uint64_t>(injected) < n) { n = static_cast<size_t>(injected); }
            done = pwrite(temp_, p, n, static_cast<off_t>(s_.written));
        }
        if (done < 0 && errno == EINTR) { continue; }
        if (done < 0) { return fail(error::system, stage::write, errno); }
        if (!done) { return fail(error::zero_write, stage::write, EIO); }
        p += done; length -= static_cast<size_t>(done); s_.written += static_cast<uint64_t>(done);
        if (s_.written - s_.confirmed == window_ && !confirm_window()) { return s_; }
    }
    return s_;
#else
    return fail(error::unsupported, stage::write);
#endif
}

status writer::write_at(uint64_t offset, const void * data, size_t length) noexcept {
    if (s_.code != error::none) { return s_; }
    if (!active_) { return fail(error::invalid_state, stage::write); }
#if defined(__linux__)
    if (!identity()) { return s_; }

    if ((!data && length) || offset > total_ || length > total_ - offset) {
        return fail(error::length_mismatch, stage::write);
    }
    explicit_ = true;
    if (!cover(offset, length)) {
        return fail(spans_no_memory_ ? error::system : error::length_mismatch, stage::write,
                    spans_no_memory_ ? ENOMEM : 0);
    }
    const unsigned char * p = static_cast<const unsigned char *>(data);
    while (length) {
        if (s_.written == s_.confirmed && !health()) { return s_; }
        const uint64_t room = window_ - (s_.written - s_.confirmed);
        size_t n = length;
        if (n > room) { n = static_cast<size_t>(room); }
        const int64_t injected = inject(stage::write, n);
        ssize_t done;
        if (injected == -1) { done = -1; }
        else {
            if (injected >= 0 && static_cast<uint64_t>(injected) < n) { n = static_cast<size_t>(injected); }
            done = pwrite(temp_, p, n, static_cast<off_t>(offset));
        }
        if (done < 0 && errno == EINTR) { continue; }
        if (done < 0) { return fail(error::system, stage::write, errno); }
        if (!done) { return fail(error::zero_write, stage::write, EIO); }
        p += done; length -= static_cast<size_t>(done); offset += static_cast<uint64_t>(done);
        s_.written += static_cast<uint64_t>(done);
        if (s_.written - s_.confirmed == window_ && !confirm_window()) { return s_; }
    }
    return s_;
#else
    (void) offset; (void) data; (void) length;
    return fail(error::unsupported, stage::write);
#endif
}

int writer::temp_fd() const noexcept {
    return temp_;
}

uint64_t writer::authority_bytes() const noexcept {
    return authority_bytes_;
}

status writer::finish() noexcept { return finish_impl(false); }
status writer::finish_pack() noexcept { return finish_impl(true); }
bool writer::handoff_pack_lock(reader & target) noexcept {
#if defined(__linux__)
    if (!s_.published || !s_.durability_confirmed || s_.cleanup_errno || lock_ < 0 ||
            !target.active_ || target.pack_name_lock_ >= 0 || target.file_dev_ != temp_dev_ ||
            target.file_ino_ != temp_ino_ || target.size_ != total_ || !identity()) {
        return false;
    }
    struct stat named;
    if (fstatat(dir_, target_, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
            static_cast<uint64_t>(named.st_dev) != temp_dev_ || static_cast<uint64_t>(named.st_ino) != temp_ino_) {
        return false;
    }
    target.pack_name_lock_ = lock_;
    lock_ = -1;
    return true;
#else
    (void) target;
    return false;
#endif
}

status writer::finish_impl(bool retain_name_lock) noexcept {
    if (s_.code != error::none) { return s_; }
    if (!active_) { return fail(error::invalid_state, stage::file_sync); }
#if defined(__linux__)
    if (!identity() || !health()) { return s_; }
    if (s_.written != total_) { return fail(error::length_mismatch, stage::file_sync); }

    if (explicit_ && !whole_file()) { return fail(error::length_mismatch, stage::file_sync); }
    if (!confirm_window()) { return s_; }
    struct stat st;
    if (fstatat(dir_, target_, &st, AT_SYMLINK_NOFOLLOW) == 0) {
        if (!ordinary(st)) { return fail(error::unsafe_path, stage::target); }
    } else if (errno != ENOENT) { return fail(error::system, stage::target, errno); }
    for (;;) {
        const int r = inject(stage::attributes) == -1 ? -1 : fchmod(temp_, mode_);
        if (!r) { break; }
        if (errno != EINTR) { return fail(error::system, stage::attributes, errno); }
    }
    for (;;) {
        const int r = inject(stage::file_sync) == -1 ? -1 : fsync(temp_);
        if (!r) { break; }
        if (errno != EINTR) { return fail(error::system, stage::file_sync, errno); }
    }
    if (!identity()) { return s_; }
    if (inject(stage::rename) == -1 || renameat(dir_, temp_name_, dir_, target_) < 0) {
        return fail(error::system, stage::rename, errno);
    }
    s_.published = true;
    for (;;) {
        const int r = inject(stage::directory_sync) == -1 ? -1 : fsync(dir_);
        if (!r) { break; }
        if (errno != EINTR) { return fail(error::system, stage::directory_sync, errno); }
    }
    s_.durability_confirmed = true;
    s_.where = stage::complete;
    if (retain_name_lock) {
        if (close(temp_) != 0) { s_.cleanup_errno = errno; }
        temp_ = -1;
        active_ = false;
    } else {
        cleanup();
    }

    release_authority(authority::outcome_published);
    return s_;
#else
    return fail(error::unsupported, stage::file_sync);
#endif
}

status writer::cancel() noexcept {
    if (active_) {
        const status s = fail(error::cancelled, stage::cleanup);

        release_authority(authority::outcome_cancelled);
        return s;
    }
    return s_;
}

const char * error_name(error code) noexcept {
    switch (code) {
#define SAFE_IO_NAME(x) case error::x: return #x
        SAFE_IO_NAME(none); SAFE_IO_NAME(not_integrated); SAFE_IO_NAME(authority_refused);
        SAFE_IO_NAME(unsupported);
        SAFE_IO_NAME(invalid_argument); SAFE_IO_NAME(invalid_state); SAFE_IO_NAME(path_changed);
        SAFE_IO_NAME(unsafe_path); SAFE_IO_NAME(unsupported_attributes); SAFE_IO_NAME(unhealthy);
        SAFE_IO_NAME(probe_failed); SAFE_IO_NAME(no_space); SAFE_IO_NAME(system);
        SAFE_IO_NAME(zero_write); SAFE_IO_NAME(length_mismatch); SAFE_IO_NAME(cancelled);
        SAFE_IO_NAME(wrong_process);
#undef SAFE_IO_NAME
    }
    return "unknown";
}

reader::reader() noexcept :
    s_{error::none, stage::idle, 0, 0, 0, 0, false, false}, fd_(-1), fd_owned_(false),
    direct_(false), pid_(0), size_(0), alignment_(1) {}

reader::~reader() noexcept {

    release_resources();
    release_authority(authority::outcome_failed);
}

status reader::result() const noexcept { return s_; }
int reader::fd() const noexcept { return fd_; }
size_t reader::alignment() const noexcept { return alignment_; }
uint64_t reader::size() const noexcept { return size_; }
bool reader::qualified() const noexcept { return active_ || authority_token_ != 0; }

int reader::detach_fd() noexcept {

    if (!fd_owned_) { return -1; }
    const int fd = fd_;
    fd_ = -1;
    fd_owned_ = false;
    return fd;
}

void reader::release_authority(authority::outcome how) noexcept {
    const uint64_t token = authority_token_;
    if (!token) { return; }
    if (s_.cleanup_errno) {
        std::fputs("safeio reader cleanup failed; authority charge retained\n", stderr);
        return;
    }
    const authority::ops * table = authority::authority_table();
    if (!table || !table->release) { return; }
    if (table->release(token, authority_bytes_, static_cast<int>(how))) {
        authority_token_ = 0;
    }
}

void reader::release_resources() noexcept {
#if defined(__linux__)
    for (unsigned i = 0; i < map_count_; ++i) {
        if (map_addr_[i] && ::munmap(map_addr_[i], map_len_[i]) != 0 && !s_.cleanup_errno) {
            s_.cleanup_errno = errno;
        }
        map_addr_[i] = nullptr;
        map_len_[i] = 0;
    }
    map_count_ = 0;
    if (fd_ >= 0 && fd_owned_) {

        (void) ::flock(fd_, LOCK_UN);
        if (::close(fd_) != 0 && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
    }
    fd_ = -1;
    fd_owned_ = false;
    if (pack_name_lock_ >= 0) {
        if (::close(pack_name_lock_) != 0 && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
        pack_name_lock_ = -1;
    }
#endif
    active_ = false;
}

status reader::fail(error code, stage at, int e) noexcept {
    if (s_.code == error::none) { s_.code = code; s_.where = at; s_.sys_errno = e; }
    release_resources();
    return s_;
}

bool reader::health() noexcept {
#if defined(__linux__)
    uint64_t d = 0, w = 0;
    if (!dirty_sample(d, w)) { fail(error::probe_failed, stage::health, errno); return false; }
    if (d > (uint64_t(8) << 30) && !w) {
        struct timespec delay = {2, 0};
        while (nanosleep(&delay, &delay) < 0) {
            if (errno != EINTR) { fail(error::probe_failed, stage::health, errno); return false; }
        }
        if (!dirty_sample(d, w)) { fail(error::probe_failed, stage::health, errno); return false; }
        if (d > (uint64_t(8) << 30) && !w) { fail(error::unhealthy, stage::health); return false; }
    }
    return true;
#else
    return true;
#endif
}

bool reader::identity() noexcept {
#if defined(__linux__)
    if (pid_ != static_cast<int64_t>(getpid())) { fail(error::wrong_process, stage::identity); return false; }
    struct stat st;
    if (::fstat(fd_, &st) < 0) { fail(error::system, stage::identity, errno); return false; }
    if (static_cast<uint64_t>(st.st_dev) != file_dev_ ||
        static_cast<uint64_t>(st.st_ino) != file_ino_ ||
        static_cast<uint64_t>(st.st_size) != size_ ||
        static_cast<int64_t>(st.st_mtim.tv_sec) != mtim_sec_ ||
        static_cast<int64_t>(st.st_mtim.tv_nsec) != mtim_nsec_) {
        fail(error::path_changed, stage::identity);
        return false;
    }
    return true;
#else
    return true;
#endif
}

bool reader::matches_path(const char * path, int held_fd) noexcept {
#if defined(__linux__)
    const auto matches = [&](const struct stat & st) {
        return S_ISREG(st.st_mode) && uint64_t(st.st_dev) == file_dev_ &&
            uint64_t(st.st_ino) == file_ino_ && uint64_t(st.st_size) == size_ &&
            st.st_mtim.tv_sec == mtim_sec_ && st.st_mtim.tv_nsec == mtim_nsec_;
    };
    struct stat held, named;
    return active_ && path && pid_ == static_cast<int64_t>(getpid()) && held_fd >= 0 &&
        ::fstat(held_fd, &held) == 0 && matches(held) &&
        ::lstat(path, &named) == 0 && matches(named);
#else
    (void) path;
    (void) held_fd;
    return false;
#endif
}

static void trace_read(const char * what, const char * path, uint64_t bytes) noexcept {
    const char * on = std::getenv("GGML_SAFE_IO_TRACE");
    if (!on || !*on || *on == '0') { return; }
    std::fprintf(stderr, "safeio reader %s path=%s bytes=%llu\n", what, path ? path : "(null)",
                 static_cast<unsigned long long>(bytes));
}

status reader::open(const char * path, uint64_t resident_bound, bool direct) noexcept {
    return open_impl(path, resident_bound, direct, 0);
}
status reader::open_pack(const char * path, uint64_t length, uint64_t token, bool writer_handoff) noexcept {
    if (!path || !length || !token || active_ || fd_ >= 0 || authority_token_ || pack_name_lock_ >= 0) {
        return fail(error::invalid_argument, stage::authorization);
    }
#if defined(__linux__)
    if (!writer_handoff) {
        const char * slash = std::strrchr(path, '/');
        const char * base = slash ? slash + 1 : path;
        char dir[4096];
        char lock_name[256];
        const size_t n = slash ? static_cast<size_t>(slash - path) : 0;
        if (!*base || n >= sizeof(dir) || std::strlen(base) + sizeof("..safeio-lock") > sizeof(lock_name)) {
            return fail(error::invalid_argument, stage::lock_open);
        }
        if (n) { std::memcpy(dir, path, n); dir[n] = 0; }
        else { std::strcpy(dir, slash ? "/" : "."); }
        std::snprintf(lock_name, sizeof(lock_name), ".%s.safeio-lock", base);
        const int parent = open_directory(dir);
        if (parent < 0) { return fail(error::unsafe_path, stage::directory, errno); }
        pack_name_lock_ = openat(parent, lock_name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
        struct stat held, named;
        bool ok = pack_name_lock_ >= 0 && fstat(pack_name_lock_, &held) == 0 && ordinary(held) &&
            (held.st_mode & 0777) == 0600 && plain_attributes(pack_name_lock_);
        if (ok) {
            int rc;
            do { rc = flock(pack_name_lock_, LOCK_SH | LOCK_NB); } while (rc < 0 && errno == EINTR);
            ok = rc == 0 && fstatat(parent, lock_name, &named, AT_SYMLINK_NOFOLLOW) == 0 &&
                held.st_dev == named.st_dev && held.st_ino == named.st_ino;
        }
        ::close(parent);
        if (!ok) { return fail(error::unsafe_path, stage::lock_acquired); }
    }
#else
    (void) writer_handoff;
#endif
    const auto result = open_impl(path, length, false, token);
    if (result.code != error::none) { release_resources(); }
    return result;
}
uint64_t reader::pack_token() const noexcept { return authority_token_; }

status reader::open_manifest(const char * path, uint64_t bound, uint64_t token) noexcept {
    if (!bound || !token) { return fail(error::invalid_argument, stage::authorization); }
    return open_impl(path, bound, false, token, true);
}

status reader::open_impl(const char * path, uint64_t resident_bound, bool direct, uint64_t reserved, bool manifest) noexcept {
#if defined(__linux__)
    if (active_ || fd_ >= 0 || authority_token_) { return fail(error::invalid_state, stage::open); }
    s_ = {error::none, stage::idle, 0, 0, 0, 0, false, false};
    if (!path || !*path) { return fail(error::invalid_argument, stage::authorization); }
    if (std::strlen(path) >= sizeof(path_)) { return fail(error::invalid_argument, stage::authorization); }

    const authority::ops * table = authority::authority_table();
    if (!table || !table->acquire) {
        s_ = {error::not_integrated, stage::authorization, 0, 0, 0, 0, false, false};
        return s_;
    }
    std::strcpy(path_, path);

    struct stat pre;
    if (::stat(path, &pre) < 0) { return fail(error::unsafe_path, stage::open, errno); }
    if (!S_ISREG(pre.st_mode) || pre.st_uid != geteuid() || pre.st_gid != getegid()) {
        return fail(error::unsupported_attributes, stage::open, EINVAL);
    }
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) { return fail(error::invalid_argument, stage::authorization); }
    uint64_t real = 0;
    if (!rounded(static_cast<uint64_t>(pre.st_size), static_cast<uint64_t>(page), real)) {
        return fail(error::invalid_argument, stage::authorization);
    }
    uint64_t footprint = real;
    if (resident_bound) {

        uint64_t bound = 0;
        if (!rounded(resident_bound, static_cast<uint64_t>(page), bound)) {
            return fail(error::invalid_argument, stage::authorization);
        }
        if (bound < footprint) { footprint = bound; }
    }
    uint64_t usable = 0;
    if (reserved && ((manifest ? real > resident_bound : real != resident_bound) || !table->claim_batch ||
            !table->claim_batch(reserved, footprint, manifest ? authority::ops::batch_use::manifest_read : authority::ops::batch_use::pack_pin))) {
        return fail(error::authority_refused, stage::authorization);
    }
    const uint64_t token = reserved ? reserved : table->acquire(footprint, &usable);
    if (!token) {

        s_ = {error::authority_refused, stage::authorization, 0, 0, 0, 0, false, false};
        release_resources();
        return s_;
    }
    authority_token_ = token;
    authority_bytes_ = footprint;

    if (!health()) { release_authority(authority::outcome_failed); return s_; }

    int flags = O_RDONLY | O_NOFOLLOW | O_CLOEXEC;
    if (direct) { flags |= O_DIRECT; }
    fd_ = ::open(path_, flags, 0);
    if (fd_ < 0 && direct && (errno == EINVAL || errno == EOPNOTSUPP)) {

        direct = false;
        fd_ = ::open(path_, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0);
    }
    if (fd_ < 0) {
        release_authority(authority::outcome_failed);
        return fail(error::unsafe_path, stage::open, errno);
    }
    fd_owned_ = true;

    struct stat st;
    if (::fstat(fd_, &st) < 0) {
        release_authority(authority::outcome_failed);
        return fail(error::system, stage::open, errno);
    }

    if (st.st_dev != pre.st_dev || st.st_ino != pre.st_ino || st.st_size != pre.st_size ||
        st.st_mtim.tv_sec != pre.st_mtim.tv_sec || st.st_mtim.tv_nsec != pre.st_mtim.tv_nsec) {
        release_authority(authority::outcome_failed);
        return fail(error::path_changed, stage::identity);
    }

    while (::flock(fd_, LOCK_SH) < 0) {
        if (errno != EINTR) {
            release_authority(authority::outcome_failed);
            return fail(error::system, stage::lock_acquired, errno);
        }
    }

    file_dev_ = static_cast<uint64_t>(st.st_dev);
    file_ino_ = static_cast<uint64_t>(st.st_ino);
    size_ = static_cast<uint64_t>(st.st_size);
    mtim_sec_ = static_cast<int64_t>(st.st_mtim.tv_sec);
    mtim_nsec_ = static_cast<int64_t>(st.st_mtim.tv_nsec);
    direct_ = direct;
    alignment_ = direct ? static_cast<size_t>(st.st_blksize) : 1;
    if (alignment_ == 0) { alignment_ = 1; }
    pid_ = static_cast<int64_t>(getpid());
    active_ = true;

    s_.written = 0;
    s_.where = stage::complete;
    trace_read("admitted", path_, footprint);
    if (footprint != real) { trace_read("bounded-demand", path_, real); }
    return s_;
#else
    (void) path; (void) resident_bound; (void) direct; (void) reserved; (void) manifest;
    return fail(error::unsupported, stage::open);
#endif
}

status reader::reopen_buffered() noexcept {
#if defined(__linux__)
    if (s_.code != error::none) { return s_; }
    if (!active_) { return fail(error::invalid_state, stage::open); }
    if (alignment_ == 1 && fd_ >= 0) { return s_; }
    if (!fd_owned_) { return fail(error::invalid_state, stage::open); }

    const int again = ::open(path_, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (again < 0) { return fail(error::unsafe_path, stage::open, errno); }
    struct stat st;
    if (::fstat(again, &st) < 0) { const int e = errno; ::close(again); return fail(error::system, stage::open, e); }

    if (static_cast<uint64_t>(st.st_dev) != file_dev_ || static_cast<uint64_t>(st.st_ino) != file_ino_ ||
        static_cast<uint64_t>(st.st_size) != size_ ||
        static_cast<int64_t>(st.st_mtim.tv_sec) != mtim_sec_ ||
        static_cast<int64_t>(st.st_mtim.tv_nsec) != mtim_nsec_) {
        ::close(again);
        return fail(error::path_changed, stage::identity);
    }
    while (::flock(again, LOCK_SH) < 0) {
        if (errno != EINTR) { ::close(again); return fail(error::system, stage::lock_acquired, errno); }
    }

    if (::close(fd_) != 0 && !s_.cleanup_errno) { s_.cleanup_errno = errno; }
    fd_ = -1;
    fd_owned_ = false;
    fd_ = again;
    fd_owned_ = true;
    direct_ = false;
    alignment_ = 1;
    trace_read("buffered", path_, size_);
    return s_;
#else
    return fail(error::unsupported, stage::open);
#endif
}

status reader::read(void * ptr, size_t len, uint64_t offset) noexcept {
#if defined(__linux__)
    if (s_.code != error::none) { return s_; }
    if (!active_ || fd_ < 0) { return fail(error::invalid_state, stage::read); }
    if (!identity()) { return s_; }
    if (offset > size_) { return fail(error::invalid_argument, stage::read); }
    const uint64_t n = std::min<uint64_t>(len, size_ - offset);
    if (n == 0) { s_.written = 0; return s_; }
    unsigned char * p = static_cast<unsigned char *>(ptr);
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd_, p + done, static_cast<size_t>(n - done),
                                  static_cast<off_t>(offset + done));
        if (r < 0 && errno == EINTR) { continue; }
        if (r < 0) { return fail(error::system, stage::read, errno); }
        if (r == 0) { break; }
        done += static_cast<size_t>(r);
    }
    s_.written = done;
    return s_;
#else
    (void) ptr; (void) len; (void) offset;
    return fail(error::unsupported, stage::read);
#endif
}

status reader::mmap(void ** addr, size_t len, uint64_t offset) noexcept {
#if defined(__linux__)
    if (s_.code != error::none) { return s_; }
    if (!active_ || fd_ < 0) { return fail(error::invalid_state, stage::mmap); }
    if (!addr) { return fail(error::invalid_argument, stage::mmap); }
    if (!identity()) { return s_; }
    const long p = sysconf(_SC_PAGESIZE);

    if (p <= 0 || (offset % static_cast<uint64_t>(p)) != 0 || offset > size_) {
        return fail(error::invalid_argument, stage::mmap);
    }
    if (map_count_ >= max_maps) { return fail(error::invalid_state, stage::mmap); }
    const uint64_t n = std::min<uint64_t>(len, size_ - offset);
    if (n == 0) { *addr = nullptr; return s_; }
    void * mapped = ::mmap(nullptr, n, PROT_READ, MAP_SHARED, fd_, static_cast<off_t>(offset));
    if (mapped == MAP_FAILED) { return fail(error::system, stage::mmap, errno); }

    map_addr_[map_count_] = mapped;
    map_len_[map_count_] = static_cast<size_t>(n);
    ++map_count_;
    s_.where = stage::mmap;
    *addr = mapped;
    return s_;
#else
    (void) addr; (void) len; (void) offset;
    return fail(error::unsupported, stage::mmap);
#endif
}

void reader::close() noexcept {
    release_resources();

    release_authority(authority::outcome_read);
    s_ = {error::none, stage::idle, 0, 0, 0, 0, false, false};
}

#ifdef GGML_SAFE_IO_WRITER_TEST
status testing::access::begin(writer & w, const char * directory, const char * basename,
                              uint64_t length, uint64_t window, hooks * h) noexcept {
    if (length > (uint64_t(1) << 20) || window > (uint64_t(1) << 20)) {
        return w.fail(error::invalid_argument, stage::authorization);
    }
    w.hooks_ = h;
    return w.start_private(directory, basename, length, window);
}
bool testing::access::rounded_pages(uint64_t n, uint64_t p, uint64_t & out) noexcept { return rounded(n, p, out); }
#endif

}
