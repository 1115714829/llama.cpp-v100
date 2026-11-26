// Portions from Strata and its AC922 fork; Copyright (c) 2026 Niko1221 and the Strata contributors (MIT); see licenses/LICENSE-strata.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct llama_ple_reader {
    std::string file;
    size_t      offset;
    size_t      n_rows;
    size_t      row_bytes;
    bool        read_rows(const int32_t * rows, size_t n, uint8_t * dst);
    struct stats { uint64_t rows = 0, distinct = 0, reads = 0, bytes = 0; double wall_ms = 0, max_latency_ms = 0; };
    stats        get_stats() const;

    bool        init();

    ~llama_ple_reader();

    bool        is_direct() const;
};
