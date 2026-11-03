











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


    std::mt19937 rng_rows(seed + 1);
    const size_t n = 4096;
    std::vector<int32_t> rows(n);
    for (size_t i = 0; i < n; ++i) {
        rows[i] = (int32_t) (rng_rows() % n_rows);
    }


    rows[0] = 0;
    rows[1] = (int32_t) (n_rows - 1);
    for (size_t b = 0; b < 4; ++b) {
        const size_t row = (4096*b) / row_bytes;
        if (row < n_rows) {
            rows[4 + b] = (int32_t) row;
        }
    }

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

        return 1;
    }


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


    if (!reader.read_rows(nullptr, 0, nullptr)) {
        fprintf(stderr, "an empty batch failed on %s\n", path.c_str());
        remove(path.c_str());
        return 1;
    }


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
            break;
        }
        if (errno != EINTR) {
            break;
        }

    }
    close(fds[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            break;
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

#endif

int main(int argc, char ** argv) {
#if defined(_WIN32)
    (void) argc;
    (void) argv;
    fprintf(stderr, "the PLE table reader needs O_DIRECT, skipping\n");
    return 0;
#else

    if (argc >= 3 && strcmp(argv[1], "--corrupt-case") == 0) {
        const size_t n_rows    = (4*1024*1024) / 90;
        const size_t row_bytes = 90;
        return run_case(argv[2], 12345, n_rows, row_bytes, 97531, true);
    }

    const std::string dir = argc > 1 ? argv[1] : ".";


    const size_t n_rows    = (64*1024*1024) / 90;
    const size_t row_bytes = 90;

    if (run_case(dir + "/test-ple-reader-a.bin", 0,     n_rows, row_bytes, 1234, false) != 0) {
        return 1;
    }
    if (run_case(dir + "/test-ple-reader-b.bin", 12345, n_rows, row_bytes, 4321, false) != 0) {
        return 1;
    }


    {
        const size_t n_rows_f16 = (16*1024*1024) / 160;
        if (run_case(dir + "/test-ple-reader-d.bin", 333, n_rows_f16, 160, 97531, false) != 0) {
            return 1;
        }
    }


    {
        FILE * f = fopen("/dev/shm/test-ple-reader-c.bin", "wb");
        if (f) {
            fclose(f);
            remove("/dev/shm/test-ple-reader-c.bin");
            const int ret = run_case("/dev/shm/test-ple-reader-c.bin", 777, n_rows, row_bytes, 2468, false);
            if (ret == 1) {
                return 1;
            }
            if (ret == 2) {

                printf("/dev/shm has no room for 64 MiB, the capability branch was not covered\n");
            }
        } else {
            printf("/dev/shm is not writable, the capability branch was not covered\n");
        }
    }


    if (run_selfcheck(dir, self_path(argv[0])) != 0) {
        return 1;
    }

    printf("OK\n");
    return 0;
#endif
}
