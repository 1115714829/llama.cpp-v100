// Tests the handling of cudaHostRegister-ed host ranges in the CUDA backend.
//
// A cudaMemcpy whose host side starts inside a registered range but ends outside it fails
// with "invalid argument" (the runtime classifies the copy by the starting address). The
// registered mapped buffer type pins whole pages in place, so its first/last page can hold
// the neighboring tensor of the mmap-ed model file; loading that tensor then copies from a
// pointer inside the registered range. The backend must split such copies at the range end.
//
// The test uses only the public backend API: the registered mapped buffer and the host
// buffer registration are reached through the proc addresses of the CUDA backend. It skips
// (and passes) when no CUDA device is present or the backend is too old.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

int main() {
    printf("SKIP: this test maps a file with mmap\n");
    return 0;
}

#else

typedef ggml_backend_buffer_type_t (*cuda_dev_mapped_registered_buft_t)(ggml_backend_dev_t dev);
typedef ggml_backend_buffer_t (*cuda_mapped_buffer_from_host_ptr_buft_t)(ggml_backend_buffer_type_t buft, void * ptr, size_t size);
typedef bool (*cuda_register_host_buffer_t)(void * buffer, size_t size);
typedef void (*cuda_unregister_host_buffer_t)(void * buffer);

static bool check_bytes(const char * what, const uint8_t * got, const uint8_t * want, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL: %s: byte %zu: got %u, want %u\n", what, i, (unsigned) got[i], (unsigned) want[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t n_pages = 64;
    const size_t nbytes = n_pages * page;

    // find the CUDA backend through the public device API
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
        if (r != nullptr && strcmp(ggml_backend_reg_name(r), "CUDA") == 0) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto registered_buft = (cuda_dev_mapped_registered_buft_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_dev_mapped_registered_buft");
    auto from_host_ptr = (cuda_mapped_buffer_from_host_ptr_buft_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mapped_buffer_from_host_ptr_buft");
    auto register_host = (cuda_register_host_buffer_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
    auto unregister_host = (cuda_unregister_host_buffer_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
    if (registered_buft == nullptr || from_host_ptr == nullptr ||
        register_host == nullptr || unregister_host == nullptr) {
        printf("SKIP: the CUDA backend does not expose the registered host buffer procs\n");
        return 0;
    }
    if (registered_buft(dev) == nullptr) {
        printf("SKIP: host memory cannot be registered read-only\n");
        return 0;
    }

    // the host buffer registration of the backend is behind an environment switch
    setenv("GGML_CUDA_REGISTER_HOST", "1", 1);

    // a file-backed read-only mapping, the same kind of memory a model file is mmap-ed as
    char path[] = "/tmp/ggml-cuda-host-register-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, (off_t) nbytes) != 0) {
        fprintf(stderr, "FAIL: cannot create the test file\n");
        return 1;
    }
    unlink(path);
    uint8_t * file = (uint8_t *) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (file == MAP_FAILED) {
        fprintf(stderr, "FAIL: mmap failed\n");
        return 1;
    }
    for (size_t i = 0; i < nbytes; ++i) {
        file[i] = (uint8_t) ((i * 7 + 3) & 0xff);
    }
    if (mprotect(file, nbytes, PROT_READ) != 0) {
        fprintf(stderr, "FAIL: mprotect failed\n");
        return 1;
    }

    // a plain mapping used as the destination of the D2H test
    uint8_t * plain = (uint8_t *) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (plain == MAP_FAILED) {
        fprintf(stderr, "FAIL: anonymous mmap failed\n");
        return 1;
    }

    // an ordinary device buffer
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "FAIL: cannot initialize the CUDA backend\n");
        return 1;
    }
    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        return 1;
    }
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (nbytes / sizeof(float)));
    ggml_backend_buffer_t device_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (device_buffer == nullptr) {
        fprintf(stderr, "FAIL: cannot allocate the device tensor\n");
        return 1;
    }

    printf("device: %s, page size: %zu\n", ggml_backend_dev_description(dev), page);

    // pin pages [16, 32) of the file mapping in place through the registered mapped buffer
    ggml_backend_buffer_t registered = from_host_ptr(registered_buft(dev), file + 16 * page, 16 * page);
    if (registered == nullptr) {
        fprintf(stderr, "FAIL: cannot build the registered mapped buffer\n");
        return 1;
    }

    // ranges in pages; [30, 34) starts inside the registered range and ends outside it,
    // which is the case that fails with "invalid argument" without the splitting
    const struct {
        size_t first;
        size_t last;
        const char * name;
    } ranges[] = {
        { 17, 21, "inside"          },
        {  2,  6, "outside"         },
        { 14, 18, "enter from out"  },
        { 30, 34, "leave through"   },
        { 10, 40, "span the whole"  },
    };

    std::vector<uint8_t> got(nbytes);
    for (const auto & r : ranges) {
        const uint8_t * src = file + r.first * page;
        const size_t n = (r.last - r.first) * page;

        ggml_backend_tensor_set(tensor, src, 0, n);
        memset(got.data(), 0, n);
        ggml_backend_tensor_get(tensor, got.data(), 0, n);
        if (!check_bytes(r.name, got.data(), src, n)) {
            return 1;
        }
    }

    // 2D copies: the host-side span of the whole copy decides whether it can stay 2D.
    // (a) the span overlaps the registered range and its first row starts inside the
    // range and ends outside it; (b) the span stays clear of the registered range.
    const size_t row_w    = 3 * page;  // bytes per row
    const size_t stride_d = 8 * page;  // host-side stride between rows
    const size_t stride_t = 12 * page; // tensor-side stride between rows
    const size_t n_rows   = 3;

    // (a) rows start at pages 30, 38, 46: the span [30, 49) overlaps [16, 32) and row 0
    // ends outside the range
    std::vector<uint8_t> round(nbytes, 0);
    ggml_backend_tensor_set_2d(tensor, file + 30 * page, 0, row_w, n_rows, stride_t, stride_d);
    ggml_backend_tensor_get_2d(tensor, round.data() + 30 * page, 0, row_w, n_rows, stride_t, stride_d);
    for (size_t y = 0; y < n_rows; ++y) {
        if (!check_bytes("2D overlap", round.data() + (30 + y * 8) * page, file + (30 + y * 8) * page, row_w)) {
            return 1;
        }
    }

    // (b) rows start at pages 40, 48, 56: the span [40, 59) is clear of [16, 32)
    ggml_backend_tensor_set_2d(tensor, file + 40 * page, 0, row_w, n_rows, stride_t, stride_d);
    ggml_backend_tensor_get_2d(tensor, round.data() + 40 * page, 0, row_w, n_rows, stride_t, stride_d);
    for (size_t y = 0; y < n_rows; ++y) {
        if (!check_bytes("2D outside", round.data() + (40 + y * 8) * page, file + (40 + y * 8) * page, row_w)) {
            return 1;
        }
    }

    // D2H: register the middle pages of the plain mapping, then read back into a range
    // that starts inside the registered range and ends outside it
    std::vector<uint8_t> pattern(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        pattern[i] = (uint8_t) ((i * 13 + 5) & 0xff);
    }
    ggml_backend_tensor_set(tensor, pattern.data(), 0, nbytes);

    // the range is registered read-only, so it is only written before the registration
    memset(plain, 0, nbytes);
    if (!register_host(plain + 16 * page, 16 * page)) {
        fprintf(stderr, "FAIL: cannot register the plain host range\n");
        return 1;
    }
    ggml_backend_tensor_get(tensor, plain + 30 * page, 0, 4 * page);
    if (!check_bytes("D2H leave through", plain + 30 * page, pattern.data() + 30 * page, 4 * page)) {
        return 1;
    }

    // 2D D2H into the registered range: the target span [30, 49) overlaps [16, 32) and
    // row 0 starts inside the range and ends outside it
    ggml_backend_tensor_get_2d(tensor, plain + 30 * page, 0, row_w, n_rows, stride_t, stride_d);
    for (size_t y = 0; y < n_rows; ++y) {
        if (!check_bytes("2D D2H overlap", plain + (30 + y * 8) * page, pattern.data() + y * 12 * page, row_w)) {
            return 1;
        }
    }

    // release the registrations: the same ranges must copy directly afterwards
    ggml_backend_buffer_free(registered);
    registered = nullptr;
    unregister_host(plain + 16 * page);

    ggml_backend_tensor_set(tensor, file + 30 * page, 0, 4 * page);
    memset(got.data(), 0, 4 * page);
    ggml_backend_tensor_get(tensor, got.data(), 0, 4 * page);
    if (!check_bytes("released file range", got.data(), file + 30 * page, 4 * page)) {
        return 1;
    }

    memset(plain, 0, nbytes);
    ggml_backend_tensor_get(tensor, plain + 30 * page, 0, 4 * page);
    if (!check_bytes("released plain range", plain + 30 * page, file + 30 * page, 4 * page)) {
        return 1;
    }

    munmap(plain, nbytes);
    munmap(file, nbytes);
    close(fd);
    ggml_backend_buffer_free(device_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("ALL_OK\n");
    return 0;
}

#endif // defined(_WIN32)
