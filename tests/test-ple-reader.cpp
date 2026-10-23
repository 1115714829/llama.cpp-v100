// The PLE n-gram table reader must return the exact bytes of the requested rows, in the order
// they were asked for. The rows of a batch repeat often, they cross the block boundaries and the
// table starts at an arbitrary offset inside the file.
//
// The same test runs on a normal file system (O_DIRECT) and, if it is writable, on tmpfs, which
// has no O_DIRECT and takes the capability branch of the reader. The test checks the bytes and
// also that the reader took the path the file system offers.
//
// The last case is a self check of the comparison itself: one byte of the file is changed behind
// the reader's back and the case must fail. It runs in a child process, so the failure is the
// expected result: the parent checks that the child failed and that it reported the difference.

#include "llama-ple-reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <errno.h>
#  include <fcntl.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

// the mode the file system offers: O_DIRECT or the page cache. This mirrors the probe of the
// reader: the smallest aligned read the kernel accepts.
#if !defined(_WIN32) && defined(O_DIRECT)
static bool fs_has_direct(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        return false;
    }

    bool ok = false;
    for (size_t bs : { (size_t) 512, (size_t) 4096 }) {
        void * buf = nullptr;
        if (posix_memalign(&buf, bs, bs) != 0) {
            break;
        }
        const ssize_t n = pread(fd, buf, bs, 0);
        const int err = errno;
        free(buf);
        if (n >= 0) {
            ok = true;
            break;
        }
        if (err != EINVAL) {
            break;
        }
    }

    close(fd);

    return ok;
}
#else
static bool fs_has_direct(const std::string & path) {
    (void) path;
    return false;
}
#endif

#if !defined(_WIN32)

static bool write_file(const std::string & path, const std::vector<uint8_t> & data) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    if (!ok) {
        fprintf(stderr, "short write on %s\n", path.c_str());
    }
    return ok;
}

// the path of this program, so that the self check can run it again as a child process
static std::string self_path(const char * argv0) {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return std::string(argv0 ? argv0 : "");
}

// Compare what the reader returned with the bytes of the file: the rows of the batch are the
// ones of the table, the file holds them at the offset of the table. Returns the number of the
// row that differs, or -1 when all of them match. The cases and the self check share it, so a
// comparison that does not see a difference breaks them both.
static int compare_rows(const std::vector<uint8_t> & data, size_t prefix, size_t row_bytes,
                        const std::vector<int32_t> & rows, const std::vector<uint8_t> & dst) {
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint8_t * ref = data.data() + prefix + (size_t) rows[i]*row_bytes;
        if (memcmp(dst.data() + i*row_bytes, ref, row_bytes) != 0) {
            return (int) i;
        }
    }
    return -1;
}

// Compare the rows of a batch with the bytes of the file: 0 when every row matches, 1 when a row
// differs. The cases and the self check of the comparison go through it, so a comparison that does
// not see a difference, and a difference that is not reported as a failure, break them both.
static int read_and_compare(const std::vector<uint8_t> & data, size_t prefix, size_t row_bytes,
                             const std::vector<int32_t> & rows, const std::vector<uint8_t> & dst,
                             const std::string & path) {
    const int diff = compare_rows(data, prefix, row_bytes, rows, dst);
    if (diff >= 0) {
        fprintf(stderr, "row %d (table row %d) differs on %s\n", diff, rows[diff], path.c_str());
        return 1;
    }
    return 0;
}

// 0: the case passed, 1: the case failed, 2: the file could not be written (the file system has
//    no room for it), the case was skipped, 3: the self check did not see the change it made
// With corrupt set, the last byte of the file is changed behind the reader's back after the
// first read: the same comparison must see the change and the case must fail. The self check
// runs in a child process that way, so it goes through the comparison and the failure of this
// function, not through a comparison of its own.
static int run_case(const std::string & path, size_t prefix, size_t n_rows, size_t row_bytes, unsigned seed, bool corrupt) {
    std::vector<uint8_t> data(prefix + n_rows*row_bytes);
    std::mt19937 rng(seed);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t) (rng() & 0xff);
    }

    if (!write_file(path, data)) {
        return 2;
    }

    llama_ple_reader reader;
    reader.file      = path;
    reader.offset    = prefix;
    reader.n_rows    = n_rows;
    reader.row_bytes = row_bytes;

    // a batch with repeats, in random order, plus the rows at the block boundaries
    std::mt19937 rng_rows(seed + 1);
    const size_t n = 4096;
    std::vector<int32_t> rows(n);
    for (size_t i = 0; i < n; ++i) {
        rows[i] = (int32_t) (rng_rows() % n_rows);
    }
    // the first and the last row of the table are always in the batch: the last one ends at the end
    // of the file, where the read path is different (a short read at the end of the file is normal)
    rows[0] = 0;
    rows[1] = (int32_t) (n_rows - 1);
    for (size_t b = 0; b < 4; ++b) {
        const size_t row = (4096*b) / row_bytes;
        if (row < n_rows) {
            rows[4 + b] = (int32_t) row;
        }
    }
    // the last row of the table: it ends at the end of the file
    rows[4] = (int32_t) (n_rows - 1);

    std::vector<uint8_t> dst(n*row_bytes, 0);
    if (!reader.read_rows(rows.data(), n, dst.data())) {
        fprintf(stderr, "read_rows failed on %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }

    if (read_and_compare(data, prefix, row_bytes, rows, dst, path) != 0) {
        remove(path.c_str());
        return 1;
    }

    // the last row of the file, checked on its own: it ends at the end of the file
    {
        const std::vector<int32_t> last_row = { (int32_t) (n_rows - 1) };
        const std::vector<uint8_t> last_dst(dst.begin() + row_bytes, dst.begin() + 2*row_bytes);
        if (read_and_compare(data, prefix, row_bytes, last_row, last_dst, path) != 0) {
            remove(path.c_str());
            return 1;
        }
        printf("%s: the last row (%zu) of the file matches byte for byte\n", path.c_str(), n_rows - 1);
    }

    if (corrupt) {
        // change the last byte of the file behind the reader's back: the comparison must see it
        const size_t   byte_off = data.size() - 1;
        const uint8_t  old_byte = data[byte_off];
        const uint8_t  new_byte = (uint8_t) (old_byte ^ 0x5a);
        {
            FILE * f = fopen(path.c_str(), "r+b");
            if (!f) {
                fprintf(stderr, "cannot reopen %s\n", path.c_str());
                remove(path.c_str());
                return 2;
            }
            const bool ok = fseek(f, (long) byte_off, SEEK_SET) == 0 && fwrite(&new_byte, 1, 1, f) == 1;
            fclose(f);
            if (!ok) {
                fprintf(stderr, "cannot change the byte at %zu of %s\n", byte_off, path.c_str());
                remove(path.c_str());
                return 2;
            }
        }

        // the last row again: the reader must see the changed byte
        const std::vector<int32_t> last_row = { (int32_t) (n_rows - 1) };
        std::vector<uint8_t> again(row_bytes, 0);
        if (!reader.read_rows(last_row.data(), 1, again.data())) {
            fprintf(stderr, "the second read of the corrupt case failed\n");
            remove(path.c_str());
            return 2;
        }

        const int rc = read_and_compare(data, prefix, row_bytes, last_row, again, path);
        printf("SELFCHECK: the byte at %zu of %s changed from 0x%02x to 0x%02x in the file\n",
                byte_off, path.c_str(), old_byte, new_byte);
        printf("SELFCHECK: the read result differs from the original bytes: %s\n", rc != 0 ? "yes" : "no");
        if (rc == 0) {
            printf("SELFCHECK-NOT-DETECTED: the comparison did not see the changed byte\n");
            remove(path.c_str());
            return 3;
        }
        printf("SELFCHECK-DETECTED: the comparison saw the changed byte\n");
        remove(path.c_str());

        return 1; // the case failed on purpose: that is the expected result
    }

    // the reader must have taken the path the file system offers
    const bool expect_direct = fs_has_direct(path);
    if (reader.is_direct() != expect_direct) {
        fprintf(stderr, "the reader on %s used %s, the file system offers %s\n", path.c_str(),
                reader.is_direct() ? "O_DIRECT" : "the page cache",
                expect_direct ? "O_DIRECT" : "the page cache");
        remove(path.c_str());
        return 1;
    }

    const llama_ple_reader::stats st = reader.get_stats();
    if (st.rows != n || st.reads == 0 || st.bytes == 0 || st.distinct == 0 || st.distinct > n) {
        fprintf(stderr, "unexpected statistics on %s: rows=%llu distinct=%llu reads=%llu bytes=%llu\n",
                path.c_str(), (unsigned long long) st.rows, (unsigned long long) st.distinct,
                (unsigned long long) st.reads, (unsigned long long) st.bytes);
        remove(path.c_str());
        return 1;
    }

    // an empty batch is a no-op
    if (!reader.read_rows(nullptr, 0, nullptr)) {
        fprintf(stderr, "an empty batch failed on %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }

    // a row outside the table is an error, not a silent read
    std::vector<int32_t> bad = { (int32_t) n_rows };
    if (reader.read_rows(bad.data(), bad.size(), dst.data())) {
        fprintf(stderr, "a row outside the table was accepted on %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }

    printf("%s: %zu rows of %zu bytes at offset %zu, %s: %llu distinct rows, %llu reads, %llu bytes, %.3f ms\n",
            path.c_str(), n_rows, row_bytes, prefix,
            reader.is_direct() ? "O_DIRECT" : "page cache",
            (unsigned long long) st.distinct, (unsigned long long) st.reads, (unsigned long long) st.bytes, st.wall_ms);

    remove(path.c_str());

    return 0;
}

// The self check of the comparison: the file is changed behind the reader's back and the case must
// fail. It runs in a child process that calls run_case with the corrupt flag, so it goes through
// the comparison and the failure of run_case itself: the child exits with 1 when it saw the change
// (the expected failure) and with 3 when it did not (that would mean the comparison is broken).

// Run the corrupt case in a child process: the case must fail with a non zero exit code and it must
// say that it saw the changed byte. The output of the child is printed, so the log shows it.
static int run_selfcheck(const std::string & dir, const std::string & self) {
    const std::string path = dir + "/test-ple-reader-corrupt.bin";

    int fds[2] = { -1, -1 };
    if (pipe(fds) != 0) {
        fprintf(stderr, "cannot create the pipe of the self check\n");
        return 1;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "cannot fork the self check\n");
        close(fds[0]);
        close(fds[1]);
        return 1;
    }

    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        std::vector<char *> args;
        args.push_back(const_cast<char *>(self.c_str()));
        args.push_back(const_cast<char *>("--corrupt-case"));
        args.push_back(const_cast<char *>(path.c_str()));
        args.push_back(nullptr);
        execv(self.c_str(), args.data());
        _exit(127);
    }

    close(fds[1]);
    std::string out;
    char buf[4096];
    while (true) {
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, (size_t) n);
            continue;
        }
        if (n == 0) {
            break; // the child closed the pipe
        }
        if (errno != EINTR) {
            break;
        }
        // a signal interrupted the read: read again
    }
    close(fds[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            break; // ECHILD: there is no child to wait for
        }
    }

    printf("%s", out.c_str());

    const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    printf("SELFCHECK: the child exited with %d (a non zero code is the expected result)\n", rc);
    if (rc == 0) {
        fprintf(stderr, "the self check did not fail: the comparison may be always true\n");
        return 1;
    }
    if (rc != 1) {
        fprintf(stderr, "the self check exited with %d instead of 1\n", rc);
        return 1;
    }
    if (out.find("SELFCHECK-DETECTED") == std::string::npos) {
        fprintf(stderr, "the self check did not report the changed byte\n");
        return 1;
    }
    printf("SELFCHECK: OK, the comparison really sees a changed byte\n");

    return 0;
}

#endif // !_WIN32

int main(int argc, char ** argv) {
#if defined(_WIN32)
    (void) argc;
    (void) argv;
    fprintf(stderr, "the PLE table reader needs O_DIRECT, skipping\n");
    return 0;
#else
    // the child process of the self check: it changes one byte of the file and must fail
    if (argc >= 3 && strcmp(argv[1], "--corrupt-case") == 0) {
        const size_t n_rows    = (4*1024*1024) / 90;
        const size_t row_bytes = 90;
        return run_case(argv[2], 12345, n_rows, row_bytes, 97531, /*corrupt=*/true);
    }

    const std::string dir = argc > 1 ? argv[1] : ".";

    // 64 MiB of table data, at an offset that is not a multiple of any block size
    const size_t n_rows    = (64*1024*1024) / 90;
    const size_t row_bytes = 90;

    if (run_case(dir + "/test-ple-reader-a.bin", 0,     n_rows, row_bytes, 1234, /*corrupt=*/false) != 0) {
        return 1;
    }
    if (run_case(dir + "/test-ple-reader-b.bin", 12345, n_rows, row_bytes, 4321, /*corrupt=*/false) != 0) {
        return 1;
    }

    // a table with F16 rows (160 bytes): the row size of the tables that are not quantized
    {
        const size_t n_rows_f16 = (16*1024*1024) / 160;
        if (run_case(dir + "/test-ple-reader-d.bin", 333, n_rows_f16, 160, 97531, /*corrupt=*/false) != 0) {
            return 1;
        }
    }

    // tmpfs has no O_DIRECT: the reader falls back to the page cache and must still be correct
    {
        FILE * f = fopen("/dev/shm/test-ple-reader-c.bin", "wb");
        if (f) {
            fclose(f);
            remove("/dev/shm/test-ple-reader-c.bin");
            const int ret = run_case("/dev/shm/test-ple-reader-c.bin", 777, n_rows, row_bytes, 2468, /*corrupt=*/false);
            if (ret == 1) {
                return 1;
            }
            if (ret == 2) {
                // tmpfs is often small: a file that does not fit is not a failure of the reader
                printf("/dev/shm has no room for 64 MiB, the capability branch was not covered\n");
            }
        } else {
            printf("/dev/shm is not writable, the capability branch was not covered\n");
        }
    }

    // the self check of the comparison: it must fail, and it must say that it saw the change
    if (run_selfcheck(dir, self_path(argv[0])) != 0) {
        return 1;
    }

    printf("OK\n");
    return 0;
#endif
}
