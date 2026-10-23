#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Reader for the raw rows of the PLE n-gram table.
//
// The table stays in the GGUF shard on disk. The rows an ubatch needs are read with O_DIRECT and
// handed to the graph as a plain input tensor, so the GPU keeps running the same GET_ROWS kernel
// on the same bytes.
//
// Adapted from Strata (https://github.com/Niko1221/Strata @ 011c9e600, MIT, see licenses/LICENSE-strata):
//   src/ngram/ple_reader.cpp             - block level dedup, offset ordering, row copy-out
//   src/platform/direct_file.cpp:263-366 - blocking pread pool, the thread count is the queue depth
// and from Strata-AC922 (https://github.com/eelgaev/Strata-AC922 @ 8eef05e18, MIT, see licenses/LICENSE-strata):
//   include/strata/platform/thread_affinity.hpp - release_inherited_pin (the helper threads must not
//   inherit the single CPU pin of the host thread)
struct llama_ple_reader {
    std::string file;       // 表所在 GGUF 分片文件名
    size_t      offset;     // 表在该分片内的字节偏移
    size_t      n_rows;     // 行数
    size_t      row_bytes;  // 90
    bool        read_rows(const int32_t * rows, size_t n, uint8_t * dst);
    struct stats { uint64_t rows = 0, distinct = 0, reads = 0, bytes = 0; double wall_ms = 0, max_latency_ms = 0; };
    stats        get_stats() const;

    // open the file of the table and start the read threads. the load calls it, so a table that
    //   can not be opened, or a machine that can not run the threads, stops the load instead of
    //   the first decode
    bool        init();

    // the file descriptor and the read threads are released when the reader is destroyed, that is
    //   when the model that owns it goes away
    ~llama_ple_reader();

    // true when the rows are read with O_DIRECT, false when the file system has no O_DIRECT and
    //   the reader reads through the page cache instead. the mode is known after the first read
    bool        is_direct() const;
};
