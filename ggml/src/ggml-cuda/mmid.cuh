#pragma once

#include "common.cuh"

// Linear MoE expert layout (adapted from upstream 69ee2ecc0, "CUDA: linearize
// MoE data layout for MMQ/MMF", PR #31047), re-derived for this tree's
// expert-parallel compact space.
//
// The grouping pass (mmid.cu) sorts the (token, slot) pairs routed to the local
// experts into one compact column space: expert e owns the contiguous run
// expert_bounds[e] .. expert_bounds[e+1], and the total is the number of pairs
// routed to this device, which only the device knows. The linear layout turns
// that compact space into a flat list of column tiles, one entry per tile:
//
//   tile_spec[slot] = {expert, col_low, col_high, 0}
//
// where [col_low, col_high) is the run of compact columns the tile owns: at most
// tile_width columns, the first tile of an expert starts at its compact range
// start, every later tile starts where the previous one ended, and the last tile
// of an expert may be shorter than tile_width. Every compact column of every
// local expert is covered by exactly one entry, so the kernels never recompute a
// tile's columns from expert_bounds: they read col_low/col_high and iterate only
// over the entries the tile actually owns.
//
// The slot list is dense: the builder assigns the slots with one atomic cursor,
// so slot j holds a real tile for every j below the number of tiles, and the
// caller sizes the allocation with ggml_cuda_mmid_tile_spec_count, an upper
// bound that also covers the slots no expert claims. Those keep the zeroed
// value the caller memset; col_low == col_high is the empty-tile marker every
// consumer checks before touching the ids or the fixup buffer (the 9826
// contract).
struct ggml_cuda_mm_id_tile_spec {
    int32_t expert;
    int32_t col_low;
    int32_t col_high;

    int32_t padding;
};
static_assert(sizeof(ggml_cuda_mm_id_tile_spec) == 16, "bad ggml_cuda_mm_id_tile_spec size");

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * expert_fill,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse,
        const ggml_cuda_expert_lut lut, cudaStream_t stream);

void ggml_cuda_mmid_build_tile_spec(
        const int32_t * expert_bounds, ggml_cuda_mm_id_tile_spec * tile_spec, int32_t * tile_spec_cursor,
        int n_experts, int tile_width, cudaStream_t stream);

static inline int ggml_cuda_mmid_tile_spec_count(int n_experts, int n_tokens, int n_expert_used, int tile_width) {
    return (n_tokens*n_expert_used + tile_width - 1) / tile_width + n_experts;
}

void ggml_cuda_mmid_zero_nonlocal(
        void * dst, const int32_t * ids, const ggml_cuda_expert_lut lut,
        int64_t n_tokens, int n_expert_used, int64_t ne0,
        int64_t ids_stride, int64_t stride_slot, int64_t stride_token, cudaStream_t stream, bool dst_f16 = false);
