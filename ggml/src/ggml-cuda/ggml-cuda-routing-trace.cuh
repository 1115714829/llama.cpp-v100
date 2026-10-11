#pragma once

// Per-token expert routing trace of the CUDA backend (RTRC format, the file layout is described
// in ggml-cuda-routing-trace.cu). Every MUL_MAT_ID of the process can record the raw
// top-k expert ids of each token into a per-device ring buffer; a background thread drains
// the ring with events (no stream synchronization on the decode path) and writes one RTRC
// file per device.
//
// The trace is off by default: while disabled no ring is allocated and no kernel is launched,
// so a normal run is unchanged. It is enabled by --dump-routing-trace FILE (see common/arg.cpp),
// which sets the process-wide path through ggml_expert_store_set_trace_path().
//
// This records expert store tensors and plain CUDA expert tensors alike: the trace only needs
// the ids tensor of the launch, so a model whose experts all fit in VRAM still produces a
// usable profile.

#include "common.cuh"

#include <cstdint>

// true when a trace path is configured. Cheap enough to call on every MUL_MAT_ID.
bool ggml_cuda_routing_trace_enabled(void);

// true when a trace is enabled. Callers use this to disable CUDA graphs for MUL_MAT_ID while
// tracing: the ring offset is a per-launch kernel argument and cannot be baked into a graph.
bool ggml_cuda_routing_trace_device_enabled(void);

// record the ids of one MUL_MAT_ID launch. src0 is the expert weight tensor (store or plain),
// ids the I32 [k, n_tokens] routing tensor. The ids are written to the device ring on the
// given stream and an event is queued for the drain thread; the call never synchronizes.
void ggml_cuda_routing_trace_record(const struct ggml_tensor * src0, const struct ggml_tensor * ids, cudaStream_t stream);

// drain every pending record of one device and close its file, then join its drain thread.
// Called when the CUDA backend of that device is freed.
void ggml_cuda_routing_trace_shutdown_device(int device);

// drain and close every device. For process teardown paths that do not free the backend.
void ggml_cuda_routing_trace_shutdown_all(void);
