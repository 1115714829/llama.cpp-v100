// Tests the CUDA mapped weights buffer type: pinned host memory owned by the buffer that the
// device reads in place over UVA.
//
// The weights that GPU kernels read in place (the token embedding table, the PLE table) use
// this buffer type. The buffer is allocated with cudaHostAlloc mapped, the host side is
// written through the ordinary backend API (set_tensor), and the data is copied in by the
// model loader, which then drops the page cache of the source range in the order of the
// weights. The test checks allocation, host writes, a GET_ROWS graph computed on the device
// with the weights read over UVA, whose output is F32 and is compared as the exact F16 to
// F32 conversion, and release.
//
// The test uses only the public backend API: the buffer type is reached through the proc
// address of the CUDA backend. It skips (and passes) when no CUDA device is present or the
// backend is too old.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef ggml_backend_buffer_type_t (*cuda_dev_mapped_weights_buft_t)(ggml_backend_dev_t dev);

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
    auto mapped_weights_buft = (cuda_dev_mapped_weights_buft_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_dev_mapped_weights_buft");
    if (mapped_weights_buft == nullptr) {
        printf("SKIP: the CUDA backend does not expose the mapped weights buffer proc\n");
        return 0;
    }
    ggml_backend_buffer_type_t buft = mapped_weights_buft(dev);
    if (buft == nullptr) {
        printf("SKIP: the CUDA backend does not offer a mapped weights buffer type\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "FAIL: cannot initialize the CUDA backend\n");
        return 1;
    }

    const int64_t n_embd   = 64;
    const int64_t n_rows   = 32;
    const int64_t n_tokens = 4;

    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_w = ggml_init(params); // the table, in the mapped weights buffer
    ggml_context * ctx_g = ggml_init(params); // ids, output and the graph, on the device
    if (ctx_w == nullptr || ctx_g == nullptr) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        return 1;
    }

    printf("device: %s\n", ggml_backend_dev_description(dev));

    ggml_tensor * table = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, n_embd, n_rows);
    ggml_backend_buffer_t table_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (table_buffer == nullptr) {
        fprintf(stderr, "FAIL: cannot allocate the mapped weights buffer\n");
        return 1;
    }

    // F16 bit patterns of 1.x .. 7.x values, so no comparison trips over NaN or denormals
    std::vector<uint16_t> rows((size_t) (n_embd * n_rows));
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = (uint16_t) ((((i % 7) + 1) << 10) | (i & 0x3ff));
    }

    // write through the backend API and read the same bytes back: the host side of the buffer
    // must hold, and the device address must map, exactly the bytes that were written
    ggml_backend_tensor_set(table, rows.data(), 0, rows.size() * sizeof(uint16_t));
    std::vector<uint16_t> readback(rows.size(), 0);
    ggml_backend_tensor_get(table, readback.data(), 0, rows.size() * sizeof(uint16_t));
    if (!check_bytes("host round trip", (const uint8_t *) readback.data(), (const uint8_t *) rows.data(), rows.size() * sizeof(uint16_t))) {
        return 1;
    }

    // GPU reads the table over UVA: a GET_ROWS graph on the device, ids and output in an
    // ordinary device buffer
    ggml_tensor * ids = ggml_new_tensor_1d(ctx_g, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "ids");
    ggml_tensor * out = ggml_get_rows(ctx_g, table, ids);
    ggml_set_name(out, "out");
    if (out->type != GGML_TYPE_F32) {
        fprintf(stderr, "FAIL: GET_ROWS output is not F32\n");
        return 1;
    }
    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_g, ggml_backend_dev_buffer_type(dev));
    if (graph_buffer == nullptr) {
        fprintf(stderr, "FAIL: cannot allocate the device buffer\n");
        return 1;
    }

    const int32_t ids_data[n_tokens] = { 3, 0, 31, 7 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, 16, false);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: GET_ROWS graph compute failed\n");
        return 1;
    }

    std::vector<float> got((size_t) (n_embd * n_tokens), 0.0f);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    for (int64_t t = 0; t < n_tokens; ++t) {
        const uint16_t * src = rows.data() + (size_t) ids_data[t] * n_embd;
        std::vector<float> want((size_t) n_embd);
        for (int64_t j = 0; j < n_embd; ++j) {
            want[j] = ggml_fp16_to_fp32((ggml_fp16_t) src[j]);
        }
        const float * have = got.data() + (size_t) t * n_embd;
        if (!check_bytes("GET_ROWS over UVA", (const uint8_t *) have, (const uint8_t *) want.data(), n_embd * sizeof(float))) {
            return 1;
        }
    }

    // release both buffers: the mapped one frees its host allocation
    ggml_backend_buffer_free(table_buffer);
    ggml_backend_buffer_free(graph_buffer);
    ggml_free(ctx_w);
    ggml_free(ctx_g);
    ggml_backend_free(backend);

    printf("ALL_OK\n");
    return 0;
}
