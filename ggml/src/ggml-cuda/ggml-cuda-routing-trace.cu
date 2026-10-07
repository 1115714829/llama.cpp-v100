#include "ggml-cuda-routing-trace.cuh"

#include "common.cuh"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "../ggml-expert-store.h"
#include "../ggml-impl.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)

// Per-token routing trace (RTRC format).
//
// The ids of every MUL_MAT_ID launch are copied into a per-device ring buffer by a tiny
// device kernel; the host queues a descriptor plus a CUDA event and a background thread
// drains the ring (event wait, device-to-host copy, file append) without ever synchronizing
// the decode stream.
//
// One trace file per device: with a layer split every device appends only its own layers,
// and the step counter is derived from the layer order inside one device, so the files stay
// independent (docs/design/r200-routing-trace.md 3.4). The path gets a ".devN" suffix when
// the process has more than one device.
//
// The trace records the raw (global) expert ids of the ids tensor, before the expert offset
// is applied, so an eventual expert-parallel run keeps global ids.
//
// Nothing is allocated, no kernel is launched and no thread exists while the trace is off.

namespace {

// header is 32 bytes, little endian, layout: "RTRC", version, header_bytes, n_layer,
// n_expert, k, flags, n_records (see the replay reader HEADER_FMT)
constexpr uint32_t TRACE_VERSION      = 1;
constexpr uint32_t TRACE_HEADER_BYTES = 32;

// per-record flags beyond the file header: bit0 is the gating weights (unused here, the
// format reserves it), bit1 marks a prefill record (n_tokens > 1). The reader ignores
// unknown bits, and n_tokens already tells the two apart.
constexpr uint32_t TRACE_REC_PREFILL = 1u << 1;

// records in flight per device; the pool of events bounds the memory and the drop rate
constexpr size_t TRACE_MAX_EVENTS = 1024;

struct trace_record {
    uint32_t step;
    uint32_t layer;
    uint32_t n_tokens;
    int64_t  off;     // ring element offset, already masked
    uint32_t n;       // number of ids
    cudaEvent_t ev;
};

struct trace_device_state {
    int     device = -1;
    int32_t * ring_dev = nullptr;
    int64_t cap  = 0;   // ring elements, power of two
    int64_t mask = 0;

    std::mutex      m;
    std::condition_variable cv;
    int64_t         head = 0; // next element offset assigned by the launch thread
    int64_t         tail = 0; // first element still owned by a queued record
    std::deque<trace_record> queue;
    std::vector<cudaEvent_t> ev_all;
    std::vector<cudaEvent_t> ev_free;
    std::thread     thread;
    bool            stop = false;
    int64_t         dropped = 0;

    uint32_t step = 0;          // forward pass counter, incremented when the layer order restarts
    uint32_t k    = 0;          // n_expert_used, fixed for the run
    uint32_t step_of_layer = 0; // step value the current layer belongs to
    int      last_layer = -1;   // last layer recorded, -1 before the first record
    bool     has_last   = false;

    std::vector<int32_t> stage; // host copy buffer of one record

    // output file, owned by the drain thread after init
    FILE *   file = nullptr;
    uint32_t n_layer  = 0;
    uint32_t n_expert = 0;
    uint64_t n_records = 0;
};

std::mutex g_trace_mutex;
trace_device_state * g_trace_dev[GGML_CUDA_MAX_DEVICES] = {};

// 1: on, 0: off. The flag lives in the expert store module and is set by the CLI before the
// model loads, so this is a lock-free load on the per-launch path.
bool trace_enabled_cached() {
    return ggml_expert_store_trace_enabled();
}

void put16(uint8_t * p, uint16_t v) { memcpy(p, &v, 2); }
void put32(uint8_t * p, uint32_t v) { memcpy(p, &v, 4); }
void put64(uint8_t * p, uint64_t v) { memcpy(p, &v, 8); }

// rewrite the 32-byte header of the device file. The drain thread is the only writer after
// init, so no lock is needed; fseek flushes the buffered records first.
void trace_write_header(trace_device_state * state) {
    if (state->file == nullptr) {
        return;
    }
    uint8_t hdr[TRACE_HEADER_BYTES] = {};
    memcpy(hdr, "RTRC", 4);
    put16(hdr + 4, (uint16_t) TRACE_VERSION);
    put16(hdr + 6, (uint16_t) TRACE_HEADER_BYTES);
    put32(hdr + 8, state->n_layer);
    put32(hdr + 12, state->n_expert);
    put32(hdr + 16, state->k);
    put32(hdr + 20, 0); // no gating weights in this version
    put64(hdr + 24, state->n_records);

    fflush(state->file);
    if (fseek(state->file, 0, SEEK_SET) != 0 ||
            fwrite(hdr, 1, sizeof(hdr), state->file) != sizeof(hdr) ||
            fseek(state->file, 0, SEEK_END) != 0) {
        GGML_LOG_ERROR("%s: cannot update the routing trace header of device %d\n", __func__, state->device);
    }
    fflush(state->file);
}

// open the trace file of one device. A file that already holds a valid header is appended to,
// so several short runs (llama-cli, restarted server) accumulate one trace.
void trace_open_file(trace_device_state * state, bool multi_device) {
    const std::string base = ggml_expert_store_get_trace_path();
    if (base.empty()) {
        return;
    }
    std::string path = base;
    if (multi_device) {
        path += ".dev" + std::to_string(state->device);
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (f != nullptr) {
        uint8_t hdr[TRACE_HEADER_BYTES];
        const bool ok = fread(hdr, 1, sizeof(hdr), f) == sizeof(hdr) && memcmp(hdr, "RTRC", 4) == 0;
        uint16_t version = 0, hbytes = 0;
        uint32_t n_layer = 0, n_expert = 0, k = 0;
        uint64_t n_records = 0;
        if (ok) {
            memcpy(&version, hdr + 4, 2);
            memcpy(&hbytes, hdr + 6, 2);
            memcpy(&n_layer, hdr + 8, 4);
            memcpy(&n_expert, hdr + 12, 4);
            memcpy(&k, hdr + 16, 4);
            memcpy(&n_records, hdr + 24, 8);
        }
        fclose(f);
        if (ok && version == TRACE_VERSION && hbytes == TRACE_HEADER_BYTES && k > 0) {
            state->file = fopen(path.c_str(), "r+b");
            if (state->file != nullptr) {
                state->n_layer   = n_layer;
                state->n_expert  = n_expert;
                state->k         = k;
                state->n_records = n_records;
                if (fseek(state->file, 0, SEEK_END) != 0) {
                    GGML_LOG_ERROR("%s: cannot seek to the end of '%s'\n", __func__, path.c_str());
                }
                GGML_LOG_INFO("%s: device %d: appending a routing trace to '%s' (%u layers, %u experts, k=%u, %llu records)\n",
                    __func__, state->device, path.c_str(), n_layer, n_expert, k, (unsigned long long) n_records);
                return;
            }
        }
    }

    state->file = fopen(path.c_str(), "wb+");
    if (state->file == nullptr) {
        GGML_LOG_ERROR("%s: device %d: cannot create routing trace '%s'\n", __func__, state->device, path.c_str());
        return;
    }
    trace_write_header(state);
    GGML_LOG_INFO("%s: device %d: writing a routing trace to '%s'\n", __func__, state->device, path.c_str());
}

// drain one descriptor: wait for its kernel, copy the ids out of the ring (handling a wrap)
// and append the record to the device file. Runs on the device drain thread.
void drain_record(trace_device_state * state, const trace_record & rec) {
    if (rec.ev != nullptr) {
        CUDA_CHECK(cudaEventSynchronize(rec.ev));
    }

    state->stage.resize(rec.n);
    const int64_t first = std::min<int64_t>(rec.n, state->cap - rec.off);
    CUDA_CHECK(cudaMemcpy(state->stage.data(), state->ring_dev + rec.off,
        (size_t) first * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (rec.n > first) {
        CUDA_CHECK(cudaMemcpy(state->stage.data() + first, state->ring_dev,
            (size_t) (rec.n - first) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    if (state->file != nullptr) {
        // grow the header bounds first: a process killed after this point still leaves a file
        // whose header covers every record that reaches the disk. The header is only rewritten
        // when the shape grows, so the drain thread does not fflush per record.
        const uint32_t n_layer  = std::max(state->n_layer, rec.layer + 1);
        uint32_t n_expert = state->n_expert;
        for (uint32_t i = 0; i < rec.n; ++i) {
            if (state->stage[i] >= 0) {
                n_expert = std::max(n_expert, (uint32_t) state->stage[i] + 1);
            }
        }
        if (n_layer != state->n_layer || n_expert != state->n_expert) {
            state->n_layer  = n_layer;
            state->n_expert = n_expert;
            trace_write_header(state);
        }

        uint32_t hdr[4] = { rec.step, rec.layer, rec.n_tokens,
                            rec.n_tokens > 1 ? TRACE_REC_PREFILL : 0 };
        if (fwrite(hdr, sizeof(uint32_t), 4, state->file) == 4 &&
                fwrite(state->stage.data(), sizeof(int32_t), rec.n, state->file) == rec.n) {
            state->n_records++;
        } else {
            GGML_LOG_ERROR("%s: device %d: cannot write a routing trace record\n", __func__, state->device);
        }
    }

    // the ring segment is free again once the bytes are on the host
    {
        std::lock_guard<std::mutex> slock(state->m);
        state->tail += rec.n;
        if (rec.ev != nullptr) {
            state->ev_free.push_back(rec.ev);
        }
    }
}

void drain_thread(trace_device_state * state) {
    ggml_cuda_set_device(state->device);
    for (;;) {
        trace_record rec;
        {
            std::unique_lock<std::mutex> lock(state->m);
            state->cv.wait(lock, [state] { return state->stop || !state->queue.empty(); });
            if (state->queue.empty()) {
                if (state->stop) {
                    break;
                }
                continue;
            }
            rec = state->queue.front();
            state->queue.pop_front();
        }
        drain_record(state, rec);
    }
}

trace_device_state * get_device_state(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    trace_device_state * state = g_trace_dev[device];
    if (state != nullptr) {
        return state;
    }

    const int size_mib = ggml_expert_store_get_trace_size_mib();
    // ring elements as a power of two, so the kernel masks instead of dividing
    int64_t cap = ((int64_t) size_mib * 1024 * 1024) / (int64_t) sizeof(int32_t);
    int64_t pow2 = 1;
    while (pow2 * 2 <= cap) {
        pow2 *= 2;
    }
    if (pow2 < 1) {
        pow2 = 1;
    }

    ggml_cuda_set_device(device);
    state = new trace_device_state();
    state->device = device;
    state->cap    = pow2;
    state->mask   = pow2 - 1;
    if (cudaMalloc((void **) &state->ring_dev, (size_t) pow2 * sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: cannot allocate the routing trace ring on device %d\n", __func__, device);
        delete state;
        return nullptr;
    }

    trace_open_file(state, ggml_backend_cuda_get_device_count() > 1);

    state->thread = std::thread(drain_thread, state);
    g_trace_dev[device] = state;
    GGML_LOG_INFO("%s: device %d: routing ring %lld elements (%.2f MiB)\n",
        __func__, device, (long long) pow2, (double) pow2 * sizeof(int32_t) / 1024.0 / 1024.0);
    return state;
}

// kernel: copy the raw ids of the launch into the ring. One thread per (token, slot).
__global__ void routing_trace_kernel(
        const int32_t * __restrict__ ids, const int64_t n_entries, const int n_expert_used,
        const int64_t ids_stride, int32_t * __restrict__ ring, const int64_t base, const int64_t mask) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) {
        return;
    }
    const int64_t token = i / n_expert_used;
    const int     slot  = (int) (i - token * n_expert_used);
    ring[(base + i) & mask] = ids[token * ids_stride + slot];
}

// layer number of an expert tensor from its name ("blk.<n>.ffn_*_exps"), the same rule the
// expert store uses; -1 when the name carries none
int trace_tensor_layer(const ggml_tensor * t) {
    const char * p = strstr(t->name, "blk.");
    if (p == nullptr) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    return layer;
}

} // namespace

bool ggml_cuda_routing_trace_enabled(void) {
    return trace_enabled_cached();
}

bool ggml_cuda_routing_trace_device_enabled(void) {
    return trace_enabled_cached();
}

void ggml_cuda_routing_trace_record(const ggml_tensor * src0, const ggml_tensor * ids, cudaStream_t stream) {
    if (!trace_enabled_cached()) {
        return;
    }
    if (src0 == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return;
    }
    // expert weight layout: one expert per ne[2] slice
    if (src0->ne[2] <= 1) {
        return;
    }
    const int layer = trace_tensor_layer(src0);
    if (layer < 0) {
        return;
    }
    const int64_t k = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n = k * n_tokens;
    if (k <= 0 || n_tokens <= 0 || n <= 0) {
        return;
    }

    const int device = ggml_cuda_get_device();
    trace_device_state * state = get_device_state(device);
    if (state == nullptr) {
        return;
    }

    // One lock for the whole booking: the ring offset and the queue entry must be created in
    // the same order, otherwise an out-of-order drain would free the wrong segment. The lock
    // covers a small kernel launch only, never a wait.
    std::lock_guard<std::mutex> lock(state->m);

    if (state->k == 0) {
        state->k = (uint32_t) k;
    } else if (state->k != (uint32_t) k) {
        // one trace file carries one k; a changing top-k would make the records unreadable,
        // so this is a configuration error and not something to paper over
        GGML_ABORT("%s: n_expert_used changed from %u to %lld during the trace\n",
            __func__, state->k, (long long) k);
    }
    // gate/up/down of one layer share the routing; record each forward-pass layer once
    if (state->has_last && layer == state->last_layer) {
        return;
    }
    // One forward pass executes the layers of one device in ascending order exactly once, so a
    // layer below the previous one marks the start of the next step. This groups the layers of
    // one forward pass without a host hook into llama_decode.
    if (state->has_last && layer < state->last_layer) {
        state->step++;
    }
    state->last_layer    = layer;
    state->step_of_layer = state->step;
    state->has_last      = true;

    if (n > state->cap) {
        state->dropped++;
        return;
    }

    // acquire an event from the pool; the pool bounds how many records may be in flight
    cudaEvent_t ev = nullptr;
    if (!state->ev_free.empty()) {
        ev = state->ev_free.back();
        state->ev_free.pop_back();
    } else if (state->ev_all.size() < TRACE_MAX_EVENTS) {
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            state->dropped++;
            return;
        }
        state->ev_all.push_back(ev);
    } else {
        state->dropped++;
        return;
    }

    const int64_t free_elems = state->cap - (state->head - state->tail);
    if (free_elems < n) {
        // ring full: drop this batch instead of waiting on the decode path
        state->dropped++;
        state->ev_free.push_back(ev);
        return;
    }
    const int64_t off = state->head & state->mask;
    state->head += n;

    const int nthreads = 256;
    const int nblocks = (int) ((n + nthreads - 1) / nthreads);
    routing_trace_kernel<<<nblocks, nthreads, 0, stream>>>(
        (const int32_t *) ids->data, n, (int) k, ids->nb[1] / sizeof(int32_t),
        state->ring_dev, off, state->mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev, stream));

    state->queue.push_back({ state->step_of_layer, (uint32_t) layer, (uint32_t) n_tokens, off, (uint32_t) n, ev });
    state->cv.notify_one();
}

void ggml_cuda_routing_trace_shutdown_device(int device) {
    if (!trace_enabled_cached()) {
        return;
    }
    trace_device_state * state = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_trace_mutex);
        if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
            return;
        }
        state = g_trace_dev[device];
        g_trace_dev[device] = nullptr;
    }
    if (state == nullptr) {
        return;
    }

    ggml_cuda_set_device(device);
    {
        std::lock_guard<std::mutex> lock(state->m);
        state->stop = true;
    }
    state->cv.notify_one();
    if (state->thread.joinable()) {
        state->thread.join();
    }

    if (state->dropped > 0) {
        GGML_LOG_WARN("%s: device %d: %lld routing trace records were dropped (ring too small or drain too slow)\n",
            __func__, device, (long long) state->dropped);
    }

    if (state->file != nullptr) {
        trace_write_header(state);
        fclose(state->file);
        state->file = nullptr;
        GGML_LOG_INFO("%s: device %d: routing trace done: %llu records, %u layers, %u experts, k=%u\n",
            __func__, device, (unsigned long long) state->n_records, state->n_layer, state->n_expert, state->k);
    }
    for (cudaEvent_t ev : state->ev_all) {
        CUDA_CHECK(cudaEventDestroy(ev));
    }
    if (state->ring_dev != nullptr) {
        CUDA_CHECK(cudaFree(state->ring_dev));
    }
    delete state;
}

void ggml_cuda_routing_trace_shutdown_all(void) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_routing_trace_shutdown_device(i);
    }
}

#else // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)

// HIP/MUSA: the routing trace is part of the CUDA expert path and is not implemented there.
bool ggml_cuda_routing_trace_enabled(void) {
    return false;
}

bool ggml_cuda_routing_trace_device_enabled(void) {
    return false;
}

void ggml_cuda_routing_trace_record(const ggml_tensor * src0, const ggml_tensor * ids, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(ids);
    GGML_UNUSED(stream);
}

void ggml_cuda_routing_trace_shutdown_device(int device) {
    GGML_UNUSED(device);
}

void ggml_cuda_routing_trace_shutdown_all(void) {
}

#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
