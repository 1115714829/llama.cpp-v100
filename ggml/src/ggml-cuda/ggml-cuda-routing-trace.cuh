#pragma once















#include "common.cuh"

#include <cstdint>


bool ggml_cuda_routing_trace_enabled(void);



bool ggml_cuda_routing_trace_device_enabled(void);




void ggml_cuda_routing_trace_record(const struct ggml_tensor * src0, const struct ggml_tensor * ids, cudaStream_t stream);



void ggml_cuda_routing_trace_shutdown_device(int device);


void ggml_cuda_routing_trace_shutdown_all(void);
