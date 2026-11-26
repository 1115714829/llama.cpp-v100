// Portions from 1Cat-vLLM, v100-skinny and haohervchb/sglang-V100; see licenses/LICENSE-1Cat-vLLM and licenses/LICENSE-v100-skinny.

#pragma once

#include "common.cuh"

static __device__ __forceinline__ int qpn8_col_from_lane(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

static __device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

static __device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}
