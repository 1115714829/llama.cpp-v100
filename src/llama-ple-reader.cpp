// Portions from Strata and its AC922 fork; Copyright (c) 2026 Niko1221 and the Strata contributors (MIT); see licenses/LICENSE-strata.

#include "llama-ple-reader.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define LLAMA_PLE_POSIX 0
#else
#  define LLAMA_PLE_POSIX 1
#  include <dirent.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <pthread.h>
#  include <sched.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#if LLAMA_PLE_POSIX

namespace {

constexpr int PLE_READER_THREADS = 64;

constexpr size_t PLE_BLOCK_SIZES[] = { 512, 4096 };

struct ple_copy {
    uint32_t src_rel;
    uint32_t dst_rel;
    uint32_t len;
};

struct ple_task {
    uint64_t off;
    size_t   len;
    uint8_t * buf;
    uint32_t slot;
    uint32_t first_copy;
    uint32_t n_copies;
};

struct ple_group {
    uint64_t off;
    size_t   len;
    uint32_t first_copy;
    uint32_t n_copies;
};

#if defined(__linux__)

static std::vector<int> ple_parse_cpu_list(const char * p) {
    std::vector<int> cpus;

    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        if (*p < '0' || *p > '9') {
            break;
        }
        int first = 0;
        while (*p >= '0' && *p <= '9') {
            first = first*10 + (*p - '0');
            ++p;
        }
        int last = first;
        if (*p == '-') {
            ++p;
            last = first;
            while (*p >= '0' && *p <= '9') {
                last = last*10 + (*p - '0');
                ++p;
            }
        }
        for (int c = first; c <= last; ++c) {
            cpus.push_back(c);
        }
    }

    return cpus;
}

static std::vector<int> ple_cpuset_cpus() {
    std::vector<int> cpus;

    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "Cpus_allowed_list:", 18) == 0) {
                cpus = ple_parse_cpu_list(line + 18);
                break;
            }
        }
        fclose(f);
    }

    if (cpus.empty()) {

        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &set)) {
                    cpus.push_back(c);
                }
            }
        }
    }

    return cpus;
}

static std::vector<std::vector<int>> ple_cpus_by_node(const std::vector<int> & allowed) {
    std::vector<std::pair<int, std::vector<int>>> nodes;

    DIR * dir = opendir("/sys/devices/system/node");
    if (dir) {
        struct dirent * entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (strncmp(entry->d_name, "node", 4) != 0) {
                continue;
            }
            const int node = atoi(entry->d_name + 4);
            if (node < 0) {
                continue;
            }
            char path[512];
            snprintf(path, sizeof(path), "/sys/devices/system/node/%s/cpulist", entry->d_name);
            FILE * f = fopen(path, "r");
            if (!f) {
                continue;
            }
            char line[1024] = { 0 };
            if (!fgets(line, sizeof(line), f)) {
                fclose(f);
                continue;
            }
            fclose(f);

            std::vector<int> cpus;
            for (int c : ple_parse_cpu_list(line)) {
                if (std::find(allowed.begin(), allowed.end(), c) != allowed.end()) {
                    cpus.push_back(c);
                }
            }
            if (!cpus.empty()) {
                nodes.emplace_back(node, cpus);
            }
        }
        closedir(dir);
    }

    std::sort(nodes.begin(), nodes.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    std::vector<std::vector<int>> res;
    res.reserve(nodes.size());
    for (auto & n : nodes) {
        res.push_back(std::move(n.second));
    }

    return res;
}

static std::vector<int> ple_thread_cpus(const std::vector<int> & allowed, int n_threads, std::vector<int> & threads_per_node) {
    std::vector<int> cpus(n_threads, -1);
    threads_per_node.clear();

    if (allowed.empty()) {
        return cpus;
    }

    const std::vector<std::vector<int>> by_node = ple_cpus_by_node(allowed);

    if (by_node.empty()) {

        for (int i = 0; i < n_threads; ++i) {
            cpus[i] = allowed[i % allowed.size()];
        }
        threads_per_node.push_back(n_threads);
        return cpus;
    }

    const size_t n_nodes = by_node.size();
    threads_per_node.assign(n_nodes, 0);
    for (int i = 0; i < n_threads; ++i) {
        const size_t node = i % n_nodes;
        const size_t k    = (size_t) (i / n_nodes) % by_node[node].size();
        cpus[i] = by_node[node][k];
        threads_per_node[node]++;
    }

    return cpus;
}

static void ple_set_affinity(const std::vector<int> & cpus) {
    if (cpus.empty()) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) {
        CPU_SET(c, &set);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

#endif

static ssize_t ple_read_range(int fd, uint8_t * buf, size_t len, uint64_t off, uint64_t file_size, bool direct, size_t block_size, int & err) {
    size_t done = 0;

    while (done < len) {
        const ssize_t n = pread(fd, buf + done, len - done, (off_t) (off + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            err = errno;
            return -1;
        }
        if (n == 0) {
            break;
        }
        done += (size_t) n;
        if (done < len) {
            if (off + done >= file_size) {
                break;
            }
            if (direct) {

                if (block_size == 0 || done % block_size != 0) {
                    break;
                }
                continue;
            }
        }
    }

    return (ssize_t) done;
}

class ple_reader_impl {
public:
    ~ple_reader_impl() {
        {

            std::lock_guard<std::mutex> batch_lock(mtx_batch);
            closed = true;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        for (uint8_t * p : slots) {
            free(p);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    bool init(const llama_ple_reader & spec) {
        file      = spec.file;
        offset    = spec.offset;
        n_rows    = spec.n_rows;
        row_bytes = spec.row_bytes;

        if (row_bytes == 0 || n_rows == 0) {
            LLAMA_LOG_ERROR("%s: the n-gram table description is empty (rows=%zu, row_bytes=%zu)\n",
                    __func__, n_rows, row_bytes);
            return false;
        }

        if (!open_table()) {
            return false;
        }

        if (!start_workers()) {
            return false;
        }

        ok = true;

        {

            std::string nodes;
            for (size_t i = 0; i < threads_per_node.size(); ++i) {
                nodes += (i ? ", " : "") + std::to_string(threads_per_node[i]);
            }
            LLAMA_LOG_INFO("%s: n-gram table '%s' at offset %zu, %zu rows of %zu bytes, %s reads of %zu bytes, "
                    "%d threads on %zu NUMA nodes (%s threads per node)\n",
                    __func__, file.c_str(), offset, n_rows, row_bytes,
                    direct ? "O_DIRECT" : "buffered", block_size, PLE_READER_THREADS,
                    threads_per_node.size(), nodes.c_str());
        }

        return true;
    }

    bool matches(const llama_ple_reader & spec) const {
        return file      == spec.file &&
               offset    == spec.offset &&
               n_rows    == spec.n_rows &&
               row_bytes == spec.row_bytes;
    }

    bool is_direct() const {
        return direct;
    }

    bool read_rows(const int32_t * rows, size_t n, uint8_t * dst) {

        std::lock_guard<std::mutex> batch_lock(mtx_batch);

        if (!ok || closed) {
            return false;
        }

        if (n == 0) {
            return true;
        }

        if (!rows || !dst) {
            LLAMA_LOG_ERROR("%s: invalid arguments (rows=%p, dst=%p)\n", __func__, (const void *) rows, (void *) dst);
            return false;
        }

        const auto t_start = std::chrono::steady_clock::now();

        if (!build_plan(rows, n)) {
            return false;
        }

        cur_dst = dst;

        if (!dispatch()) {
            return false;
        }

        if (failed.load()) {
            LLAMA_LOG_ERROR("%s: reading the n-gram table failed: %s\n", __func__, strerror(read_error.load()));
            failed.store(false);
            read_error.store(0);
            return false;
        }

        if (!direct) {

#ifdef POSIX_FADV_DONTNEED
            for (const auto & g : groups) {
                posix_fadvise(fd, (off_t) g.off, (off_t) g.len, POSIX_FADV_DONTNEED);
            }
#endif
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        st.rows     += n;
        st.distinct += n_unique;
        st.reads    += groups.size();
        st.bytes    += bytes_read.exchange(0);
        st.wall_ms += wall_ms;
        st.max_latency_ms = std::max(st.max_latency_ms, max_latency_ms.exchange(0.0));

        return true;
    }

    llama_ple_reader::stats get_stats() const {
        std::lock_guard<std::mutex> batch_lock(mtx_batch);
        return st;
    }

    bool is_ok() const {
        return ok;
    }

private:
    bool open_table() {
        int  fd_try    = -1;
        bool have_direct = false;
#ifdef O_DIRECT
        fd_try = open(file.c_str(), O_RDONLY | O_DIRECT);
        have_direct = fd_try >= 0;
#endif
        if (have_direct) {
            size_t bs = 0;
            if (probe_block_size(fd_try, bs)) {
                fd         = fd_try;
                direct     = true;
                block_size = bs;
            } else {

                close(fd_try);
                fd_try = open(file.c_str(), O_RDONLY);
                if (fd_try < 0) {
                    LLAMA_LOG_ERROR("%s: cannot open the n-gram table file '%s': %s\n",
                            __func__, file.c_str(), strerror(errno));
                    return false;
                }
                fd         = fd_try;
                direct     = false;
                block_size = PLE_BLOCK_SIZES[sizeof(PLE_BLOCK_SIZES)/sizeof(PLE_BLOCK_SIZES[0]) - 1];
                LLAMA_LOG_WARN("%s: '%s' accepts O_DIRECT but no aligned read worked, using buffered reads\n",
                        __func__, file.c_str());
            }
        } else {
#ifdef O_DIRECT
            const int err = errno;
            if (err != EINVAL && err != ENOTSUP) {
                LLAMA_LOG_ERROR("%s: cannot open the n-gram table file '%s': %s\n",
                        __func__, file.c_str(), strerror(err));
                return false;
            }

            LLAMA_LOG_WARN("%s: '%s' does not support O_DIRECT (%s), reading through the page cache and dropping the pages of every batch\n",
                    __func__, file.c_str(), strerror(err));
#else

            LLAMA_LOG_WARN("%s: '%s' is read through the page cache, the platform has no O_DIRECT\n",
                    __func__, file.c_str());
#endif
            fd_try = open(file.c_str(), O_RDONLY);
            if (fd_try < 0) {
                LLAMA_LOG_ERROR("%s: cannot open the n-gram table file '%s': %s\n",
                        __func__, file.c_str(), strerror(errno));
                return false;
            }
            fd         = fd_try;
            direct     = false;
            block_size = PLE_BLOCK_SIZES[sizeof(PLE_BLOCK_SIZES)/sizeof(PLE_BLOCK_SIZES[0]) - 1];
        }

        struct stat st_buf;
        if (fstat(fd, &st_buf) != 0) {
            LLAMA_LOG_ERROR("%s: cannot stat the n-gram table file '%s': %s\n",
                    __func__, file.c_str(), strerror(errno));
            return false;
        }

        const uint64_t table_bytes = (uint64_t) n_rows * row_bytes;
        if ((uint64_t) offset + table_bytes > (uint64_t) st_buf.st_size) {
            LLAMA_LOG_ERROR("%s: the n-gram table does not fit into '%s' (offset=%zu, table=%llu, file=%llu)\n",
                    __func__, file.c_str(), offset, (unsigned long long) table_bytes, (unsigned long long) st_buf.st_size);
            return false;
        }

        file_size = (uint64_t) st_buf.st_size;

        return true;
    }

    static bool probe_block_size(int fd, size_t & block) {
        for (size_t bs : PLE_BLOCK_SIZES) {
            void * buf = nullptr;
            if (posix_memalign(&buf, bs, bs) != 0) {
                return false;
            }
            const ssize_t n = pread(fd, buf, bs, 0);
            const int err = errno;
            free(buf);
            if (n >= 0) {
                block = bs;
                return true;
            }
            if (err != EINVAL) {
                return false;
            }
        }
        return false;
    }

    bool start_workers() {

        const size_t span       = (row_bytes + block_size - 1) / block_size;
        const size_t slot_bytes = (span + 1)*block_size;

        slots.resize(PLE_READER_THREADS, nullptr);
        for (int i = 0; i < PLE_READER_THREADS; ++i) {
            void * p = nullptr;
            if (posix_memalign(&p, block_size, slot_bytes) != 0) {
                LLAMA_LOG_ERROR("%s: cannot allocate the read buffers\n", __func__);
                return false;
            }
            slots[i] = (uint8_t *) p;
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            for (int i = 0; i < PLE_READER_THREADS; ++i) {
                free_slots.push_back(i);
            }
        }

#if defined(__linux__)

        cpuset       = ple_cpuset_cpus();
        thread_cpus  = ple_thread_cpus(cpuset, PLE_READER_THREADS, threads_per_node);
#else

        threads_per_node.assign(1, PLE_READER_THREADS);
#endif

        workers.reserve(PLE_READER_THREADS);
        for (int i = 0; i < PLE_READER_THREADS; ++i) {
            workers.emplace_back([this, i]{ worker(i); });
        }

        return true;
    }

    bool build_plan(const int32_t * rows, size_t n) {
        order.clear();
        order.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (rows[i] < 0 || (size_t) rows[i] >= n_rows) {
                LLAMA_LOG_ERROR("%s: row %d is outside the n-gram table (%zu rows)\n", __func__, rows[i], n_rows);
                return false;
            }
            order.emplace_back(rows[i], (uint32_t) i);
        }
        std::sort(order.begin(), order.end());

        groups.clear();
        copies.clear();
        n_unique = 0;

        size_t i = 0;
        while (i < order.size()) {
            const uint64_t b0 = (offset + (uint64_t) order[i].first * row_bytes) / block_size;

            ple_group g;
            g.off        = b0*block_size;
            g.len        = 0;
            g.first_copy = (uint32_t) copies.size();
            g.n_copies   = 0;

            uint64_t b1_max = b0;

            while (i < order.size()) {
                const int32_t  row      = order[i].first;
                const uint64_t file_off = offset + (uint64_t) row * row_bytes;

                if (file_off / block_size != b0) {
                    break;
                }

                b1_max = std::max(b1_max, (file_off + row_bytes - 1) / block_size);

                const uint32_t src_rel = (uint32_t) (file_off - g.off);

                while (i < order.size() && order[i].first == row) {
                    const uint32_t pos = order[i].second;
                    ple_copy cp;
                    cp.src_rel = src_rel;
                    cp.dst_rel = pos*row_bytes;
                    cp.len     = (uint32_t) row_bytes;
                    copies.push_back(cp);
                    ++i;
                }

                ++n_unique;
            }

            g.len      = (size_t) ((b1_max + 1)*block_size - g.off);
            g.n_copies = (uint32_t) (copies.size() - g.first_copy);

            groups.push_back(g);
        }

        return true;
    }

    bool dispatch() {
        std::unique_lock<std::mutex> lock(mtx);

        for (const auto & g : groups) {
            cv_done.wait(lock, [this]{ return !free_slots.empty(); });

            ple_task t;
            t.off        = g.off;
            t.len        = g.len;
            t.slot       = free_slots.back();
            t.first_copy = g.first_copy;
            t.n_copies   = g.n_copies;
            t.buf        = slots[t.slot];

            free_slots.pop_back();
            queue.push_back(t);
            n_pending++;

            lock.unlock();
            cv_work.notify_one();
            lock.lock();
        }

        cv_done.wait(lock, [this]{ return n_pending == 0; });

        return true;
    }

    void worker(int id) {
#if defined(__linux__)

        ple_set_affinity(cpuset);
        if ((size_t) id < thread_cpus.size() && thread_cpus[id] >= 0) {
            ple_set_affinity({ thread_cpus[id] });
        }
#endif

        for (;;) {
            ple_task t;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_work.wait(lock, [this]{ return stop || !queue.empty(); });
                if (queue.empty()) {
                    if (stop) {
                        return;
                    }
                    continue;
                }
                t = queue.front();
                queue.pop_front();
            }

            int err = 0;
            const auto t0 = std::chrono::steady_clock::now();
            const ssize_t nread = ple_read_range(fd, t.buf, t.len, t.off, file_size, direct, block_size, err);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            double cur = max_latency_ms.load();
            while (ms > cur && !max_latency_ms.compare_exchange_weak(cur, ms)) {

            }

            if (nread < 0) {
                read_error.store(err);
                failed.store(true);
            } else {
                bytes_read += (uint64_t) nread;
                for (uint32_t c = 0; c < t.n_copies; ++c) {
                    const ple_copy & cp = copies[t.first_copy + c];
                    if ((size_t) cp.src_rel + cp.len > (size_t) nread) {
                        LLAMA_LOG_ERROR("%s: short read at offset %llu (%zd of %zu bytes)\n",
                                __func__, (unsigned long long) t.off, nread, t.len);
                        read_error.store(EIO);
                        failed.store(true);
                        break;
                    }
                    memcpy(cur_dst + cp.dst_rel, t.buf + cp.src_rel, cp.len);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                --n_pending;
                free_slots.push_back(t.slot);
            }
            cv_done.notify_all();
        }
    }

    std::string file;
    size_t      offset    = 0;
    size_t      n_rows    = 0;
    size_t      row_bytes = 0;

    int    fd         = -1;
    bool   direct     = false;
    bool   ok         = false;
    bool   closed     = false;
    size_t block_size = 0;
    uint64_t file_size = 0;

    std::vector<std::thread> workers;
    std::vector<uint8_t *>   slots;
    std::deque<ple_task>     queue;
    std::vector<uint32_t>    free_slots;
    std::mutex               mtx;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    int                      n_pending = 0;
    bool                     stop      = false;

    mutable std::mutex   mtx_batch;
    std::vector<std::pair<int32_t, uint32_t>> order;
    std::vector<ple_group>   groups;
    std::vector<ple_copy>    copies;
    uint8_t *                cur_dst   = nullptr;
    size_t                   n_unique  = 0;

    std::atomic<uint64_t> bytes_read    { 0 };
    std::atomic<double>   max_latency_ms { 0.0 };
    std::atomic<bool>     failed        { false };
    std::atomic<int>      read_error    { 0 };

    llama_ple_reader::stats st;

#if defined(__linux__)
    std::vector<int> cpuset;
    std::vector<int> thread_cpus;
#endif
    std::vector<int> threads_per_node;
};

std::mutex g_impl_mtx;
std::unordered_map<const llama_ple_reader *, std::unique_ptr<ple_reader_impl>> g_impls;

static ple_reader_impl * ple_impl_for(const llama_ple_reader * reader, bool create) {
    std::lock_guard<std::mutex> lock(g_impl_mtx);

    auto it = g_impls.find(reader);
    if (it != g_impls.end()) {
        if (it->second->matches(*reader)) {
            return it->second.get();
        }

        g_impls.erase(it);
    }

    if (!create) {
        return nullptr;
    }

    auto impl = std::make_unique<ple_reader_impl>();
    impl->init(*reader);

    auto * res = impl.get();
    g_impls[reader] = std::move(impl);

    return res;
}

}

#endif

bool llama_ple_reader::read_rows(const int32_t * rows, size_t n, uint8_t * dst) {
#if !LLAMA_PLE_POSIX
    LLAMA_LOG_ERROR("%s: the n-gram table reader needs O_DIRECT and is not available on this platform\n", __func__);
    GGML_UNUSED(rows);
    GGML_UNUSED(n);
    GGML_UNUSED(dst);
    return false;
#else
    ple_reader_impl * impl = ple_impl_for(this, true);
    if (!impl || !impl->is_ok()) {
        return false;
    }

    return impl->read_rows(rows, n, dst);
#endif
}

bool llama_ple_reader::init() {
#if !LLAMA_PLE_POSIX
    LLAMA_LOG_ERROR("%s: the n-gram table reader needs O_DIRECT and is not available on this platform\n", __func__);
    return false;
#else
    ple_reader_impl * impl = ple_impl_for(this, true);
    return impl != nullptr && impl->is_ok();
#endif
}

llama_ple_reader::stats llama_ple_reader::get_stats() const {
#if !LLAMA_PLE_POSIX
    return stats();
#else
    ple_reader_impl * impl = ple_impl_for(this, false);
    if (!impl) {
        return stats();
    }

    return impl->get_stats();
#endif
}

bool llama_ple_reader::is_direct() const {
#if !LLAMA_PLE_POSIX
    return false;
#else
    ple_reader_impl * impl = ple_impl_for(this, false);
    if (!impl) {
        return false;
    }

    return impl->is_direct();
#endif
}

llama_ple_reader::~llama_ple_reader() {
#if LLAMA_PLE_POSIX

    std::lock_guard<std::mutex> lock(g_impl_mtx);
    g_impls.erase(this);
#endif
}
