// QSA (Qwen sparse attention) selection and attention over the selected cells.
//
// Portions ported from Strata (https://github.com/Niko1221/strata), MIT License,
// Copyright (c) 2026 Niko1221 and the Strata contributors:
//   - qsa_topk_reg_kernel:    src/kernels/cuda/qsa_select.cu, block_topk_reg_kernel (radix select, 1024 threads)
//   - qsa_topk_wide_kernel:   src/kernels/cuda/qsa_select.cu, block_topk_wide_kernel (radix select, 1024 threads)
//   - qsa_attn_chunk_kernel:  src/kernels/cuda/qsa_decode_attn.cu, attn_chunk_kernel_pre75, reduce12, head_of_lane
//   - qsa_attn_merge_kernel:  src/kernels/cuda/qsa_decode_attn.cu, attn_merge_kernel
//   - qsa_attn_prompt_kernel: src/kernels/cuda/qsa_prompt_attn.cu, prompt_attn_v70_kernel (mma.m8n8k4)
// Adapted to ggml: K/V rows are read from the ggml cache (F16 or Q8_0, row strides per KV head, cell and
// stream) instead of Strata's paged pools, the selection comes from QSA_SELECT, a cell id of -1 is skipped.
// The prompt kernel keeps the v70 numerics: hi/lo mma chains and Q8_0 scales applied in FP32.

#include "qsa.cuh"
#include "lightning-indexer.cuh"

#include <cfloat>

#define QSA_HD 256  // head size
#define QSA_G  12   // query heads per KV head

// tokens scored per pass of QSA_SELECT: the score scratch is n_pool*QSA_SELECT_TILE floats, independent of the batch
#define QSA_SELECT_TILE 64

// the 1024-thread top-k kernels (register and wide) hold one query per block, as in Strata
#define QSA_TOPK_T 1024
#define QSA_TOPK_PER 33
#define QSA_TOPK_WARPS (QSA_TOPK_T/32)

// dispatch of the top-k by the number of pools a query can scan: the register kernel holds QSA_TOPK_T*PER pools, with
// PER the smallest of 4, 8, 16, 24, 33 that fits (a register kernel sized for more pools than it holds runs its
// unrolled loops idle: on V100 4096 pools take 15.5 us with PER 4 and 25 us with PER 33); the wide kernel takes the
// pools past QSA_TOPK_T*QSA_TOPK_PER. Every kernel selects the same ids.

// QSA_ATTN: batches up to this size take the chunked kernel, larger ones the prompt kernel
#define QSA_ATTN_CHUNK_MAX_TOKENS 16
#define QSA_CHUNK 64          // cells per block of the chunked kernel
#define QSA_CHUNK_THREADS 256
#define QSA_PROMPT_CH 32      // cells per step of the prompt kernel
#define QSA_PROMPT_THREADS 128

// ---------------------------------------------------------------------------------------------------------------------
// selection

// total order of the scores as unsigned keys, NaN lowest
static __device__ __forceinline__ uint32_t qsa_order_key(const float s) {
    const float v = s + 0.0f;
    if (!(v == v)) {
        return 0u;
    }
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ---- radix select shared by the top-k kernels. The threshold is the n_top-th largest key, found digit by digit from the
// top: three digits of 11, 11 and 10 bits (the first one holds the sign, the exponent and two mantissa bits, so the
// scores of a query spread over more bins than with an 8-bit first digit, and there is one pass less). The threshold
// does not depend on how the key is cut into digits, and every count is an integer, so the selection is the same as
// with Strata's four 8-bit digits.
#define QSA_RS_BINS   2048
#define QSA_RS_PASSES 3

static __device__ __forceinline__ int qsa_rs_shift(const int p) {
    return p == 0 ? 21 : (p == 1 ? 10 : 0);
}

static __device__ __forceinline__ int qsa_rs_bins(const int p) {
    return p == 2 ? 1024 : 2048;
}

// the bits fixed by the passes before p
static __device__ __forceinline__ uint32_t qsa_rs_hi_mask(const int p) {
    return p == 0 ? 0u : (p == 1 ? 0xffe00000u : 0xfffffc00u);
}

// exclusive prefix sum over a block of NW warps, per warp then across warps; total gets the block total
template <int NW>
static __device__ __forceinline__ int qsa_excl_scan(const int v, int * s_warp, int & total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const int w = lane < NW ? s_warp[lane] : 0;
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        if (lane < NW) s_warp[lane] = z - w; // exclusive per warp
        if (lane == 31) s_warp[32] = z;      // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

// The digit of a pass: the largest d in [1, nbins) with above + hist[d] + ... + hist[nbins - 1] >= n_top, else 0, and in
// cum the pools above it (above + hist[d + 1] + ... ; for d = 0 above + hist[1] + ...): the scan from the top bin down,
// done by the whole block. Thread t holds the bins top - bpt + 1 .. top (top = nbins - 1 - t*bpt); a block scan of the
// thread sums finds the one thread whose bins carry the count over n_top, and it walks them.
template <int NT>
static __device__ __forceinline__ void qsa_find_digit(const int * hist, const int nbins, const int above, const int n_top,
                                                      int * s_warp, int & digit, int & cum) {
    const int t   = threadIdx.x;
    const int bpt = (nbins + NT - 1)/NT;
    const int top = nbins - 1 - t*bpt;

    int sum = 0;
    for (int i = 0; i < bpt; ++i) {
        if (top - i >= 0) {
            sum += hist[top - i];
        }
    }
    int total;
    const int excl = qsa_excl_scan<NT/32>(sum, s_warp, total);
    if (above + excl < n_top && above + excl + sum >= n_top) {
        int c = above + excl;
        int d = 0;
        for (int i = 0; i < bpt; ++i) {
            const int b = top - i;
            if (b <= 0) {
                break; // bin 0 is taken untested, as at the end of the serial scan
            }
            if (c + hist[b] >= n_top) {
                d = b;
                break;
            }
            c += hist[b];
        }
        s_warp[0] = d;
        s_warp[1] = c;
    } else if (t == 0 && above + total < n_top) {
        s_warp[0] = 0;
        s_warp[1] = above + total - hist[0];
    }
    __syncthreads();
    digit = s_warp[0];
    cum   = s_warp[1];
    __syncthreads();
}

// The tail cells and the -1 padding after the kept pools, shared by the 1024-thread kernels. n_kept pools have been
// written by the caller; thread 0 appends tidx[qi] then the block fills the rest.
static __device__ __forceinline__ void qsa_topk_epilogue(const int32_t * __restrict__ tidx, const int64_t s_tidx,
                                                         const int kpool, const int sentinel, const int n_kept,
                                                         const int qi, int32_t * __restrict__ o, const int n_sel,
                                                         int * s_digit) {
    const int t = threadIdx.x;
    __syncthreads();
    if (t == 0) {
        int n = n_kept*kpool;
        for (int c = 0; c < kpool - 1; ++c) {
            const int32_t cell = tidx[qi*s_tidx + c];
            if (cell != sentinel) {
                o[n++] = cell;
            }
        }
        *s_digit = n;
    }
    __syncthreads();
    for (int i = *s_digit + t; i < n_sel; i += QSA_TOPK_T) {
        o[i] = -1;
    }
}

// ---- one block per token: keep the n_top highest scores of the pools [lo, hi), equal scores by ascending pool, and write
// their cells in ascending pool order, then the tail cells, then -1. Each query's pools are read once: 1,024 threads
// hold up to PER consecutive pools' keys in registers; radix threshold, ties to the lowest pool, block-wide scans.
// Ported from Strata's block_topk_reg_kernel.
template <int PER>
static __global__ void __launch_bounds__(QSA_TOPK_T) qsa_topk_reg_kernel(
        const float * __restrict__ scores, const int64_t n_pool,
        const int32_t * __restrict__ rng, const int64_t s_rng,
        const int32_t * __restrict__ pidx, const int kpool,
        const int32_t * __restrict__ tidx, const int64_t s_tidx,
        const int n_top, const int sentinel,
        int32_t * __restrict__ out, const int n_sel) {
    __shared__ int hist[QSA_RS_BINS];
    __shared__ int s_warp[33];
    __shared__ int s_digit;

    const int qi = blockIdx.x;
    const int t  = threadIdx.x;

    const int64_t lo = rng[qi*s_rng + 0];
    const int64_t hi = rng[qi*s_rng + 1];
    const int64_t nv = hi - lo;

    const float * sc = scores + qi*n_pool;
    int32_t     * o  = out    + (int64_t) qi*n_sel;

    if (nv <= n_top) {                     // every pool is selected: write their cells, ascending
        for (int64_t b = lo + t; b < hi; b += QSA_TOPK_T) {
            for (int c = 0; c < kpool; ++c) {
                o[(b - lo)*kpool + c] = pidx[b*kpool + c];
            }
        }
        qsa_topk_epilogue(tidx, s_tidx, kpool, sentinel, (int) nv, qi, o, n_sel, &s_digit);
        return;
    }

    const int64_t per = (nv + QSA_TOPK_T - 1)/QSA_TOPK_T;   // <= PER (the caller checks)
    const int64_t b0  = lo + t*per;
    const int64_t b1  = min(b0 + per, hi);

    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        key[j] = (b0 + j < b1) ? qsa_order_key(sc[b0 + j]) : 0u;
    }

    uint32_t prefix = 0;
    int above = 0;
    for (int p = 0; p < QSA_RS_PASSES; ++p) {
        const int      shift   = qsa_rs_shift(p);
        const int      nbins   = qsa_rs_bins(p);
        const uint32_t hi_mask = qsa_rs_hi_mask(p);
        for (int i = t; i < nbins; i += QSA_TOPK_T) {
            hist[i] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            if (b0 + j < b1 && (key[j] & hi_mask) == (prefix & hi_mask)) {
                atomicAdd(&hist[(key[j] >> shift) & (nbins - 1)], 1);
            }
        }
        __syncthreads();
        int d, c;
        qsa_find_digit<QSA_TOPK_T>(hist, nbins, above, n_top, s_warp, d, c);
        prefix |= (uint32_t) d << shift;
        above = c;
    }
    const uint32_t thr       = prefix;
    const int64_t  eq_budget = n_top - above;   // pools equal to thr that fit, lowest index first

    // per-thread counts above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) {
            break;
        }
        if (key[j] > thr) {
            gt++;
        } else if (key[j] == thr) {
            eq++;
        }
    }
    int tot;
    const int eq_before = qsa_excl_scan<QSA_TOPK_WARPS>(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    my_eq = my_eq < 0 ? 0 : (my_eq > eq ? eq : my_eq);
    const int sel = gt + (int) my_eq;
    int64_t wpos = qsa_excl_scan<QSA_TOPK_WARPS>(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) {
            break;
        }
        if (key[j] > thr) {
            for (int c = 0; c < kpool; ++c) {
                o[wpos*kpool + c] = pidx[b*kpool + c];
            }
            wpos++;
        } else if (key[j] == thr && eq_left > 0) {
            for (int c = 0; c < kpool; ++c) {
                o[wpos*kpool + c] = pidx[b*kpool + c];
            }
            wpos++;
            eq_left--;
        }
    }
    qsa_topk_epilogue(tidx, s_tidx, kpool, sentinel, tot, qi, o, n_sel, &s_digit);
}

// ---- a query with more pools than the register kernel holds: the register kernel's threads and block scans, with each
// key read from memory again on every pass, QSA_TOPK_UNROLL independent loads in flight per thread. A histogram has no
// order, so on the radix passes thread t reads pools t, t + 1024, ...: a warp reads 32 neighbours at a time. The cells
// are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows of 32 consecutive pools from pool
// w * 32 * per. The selection rule is the register kernel's (radix threshold, ties to the lowest pool): identical ids.
// Ported from Strata's block_topk_wide_kernel.
#define QSA_TOPK_UNROLL 4
static __global__ void __launch_bounds__(QSA_TOPK_T) qsa_topk_wide_kernel(
        const float * __restrict__ scores, const int64_t n_pool,
        const int32_t * __restrict__ rng, const int64_t s_rng,
        const int32_t * __restrict__ pidx, const int kpool,
        const int32_t * __restrict__ tidx, const int64_t s_tidx,
        const int n_top, const int sentinel,
        int32_t * __restrict__ out, const int n_sel) {
    constexpr int U = QSA_TOPK_UNROLL;

    __shared__ int hist[QSA_RS_BINS];
    __shared__ int s_warp[33];
    __shared__ int s_gt[QSA_TOPK_WARPS], s_eq[QSA_TOPK_WARPS];
    __shared__ int s_digit, s_nkept;

    const int qi = blockIdx.x;
    const int t  = threadIdx.x, lane = t & 31, warp = t >> 5;

    const int64_t lo = rng[qi*s_rng + 0];
    const int64_t hi = rng[qi*s_rng + 1];
    const int64_t nv = hi - lo;

    const float * sc = scores + qi*n_pool;
    int32_t     * o  = out    + (int64_t) qi*n_sel;

    if (nv <= n_top) {                     // every pool is selected: write their cells, ascending
        for (int64_t b = lo + t; b < hi; b += QSA_TOPK_T) {
            for (int c = 0; c < kpool; ++c) {
                o[(b - lo)*kpool + c] = pidx[b*kpool + c];
            }
        }
        qsa_topk_epilogue(tidx, s_tidx, kpool, sentinel, (int) nv, qi, o, n_sel, &s_digit);
        return;
    }

    const int64_t per = (nv + QSA_TOPK_T - 1)/QSA_TOPK_T;   // rows of 32 pools per warp

    uint32_t prefix = 0;
    int above = 0;
    for (int p = 0; p < QSA_RS_PASSES; ++p) {
        const int      shift   = qsa_rs_shift(p);
        const int      nbins   = qsa_rs_bins(p);
        const uint32_t hi_mask = qsa_rs_hi_mask(p);
        for (int i = t; i < nbins; i += QSA_TOPK_T) {
            hist[i] = 0;
        }
        __syncthreads();
        for (int64_t bu = lo + t; bu < hi; bu += U*QSA_TOPK_T) {
            uint32_t k[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t b = bu + u*QSA_TOPK_T;
                k[u] = b < hi ? qsa_order_key(sc[b]) : 0u;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (bu + u*QSA_TOPK_T < hi && (k[u] & hi_mask) == (prefix & hi_mask)) {
                    atomicAdd(&hist[(k[u] >> shift) & (nbins - 1)], 1);
                }
            }
        }
        __syncthreads();
        int d, c;
        qsa_find_digit<QSA_TOPK_T>(hist, nbins, above, n_top, s_warp, d, c);
        prefix |= (uint32_t) d << shift;
        above = c;
    }
    const uint32_t thr       = prefix;
    const int      eq_budget = n_top - above;   // pools equal to thr that fit, lowest index first

    const int64_t w0   = lo + (int64_t) warp*32*per;
    const int64_t wend = min(hi, w0 + 32*per);
    int gt = 0, eq = 0;
    for (int64_t bu = w0 + lane; bu < wend; bu += U*32) {
        uint32_t k[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t b = bu + u*32;
            k[u] = b < wend ? qsa_order_key(sc[b]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (bu + u*32 < wend) {
                if (k[u] > thr) {
                    gt++;
                } else if (k[u] == thr) {
                    eq++;
                }
            }
        }
    }
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o2);
        eq += __shfl_xor_sync(0xffffffffu, eq, o2);
    }
    if (lane == 0) {
        s_gt[warp] = gt;
        s_eq[warp] = eq;
    }
    __syncthreads();
    if (t == 0) {                          // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < QSA_TOPK_WARPS; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
        s_nkept = sb;
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t ru = w0; ru < wend; ru += U*32) {
        // the keys of U rows first, so that their loads are in flight together
        uint32_t k[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t b = ru + u*32 + lane;
            k[u] = b < wend ? qsa_order_key(sc[b]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t b = ru + u*32 + lane;
            const bool has = b < wend;
            const int my_gt = (has && k[u] > thr) ? 1 : 0, my_eq = (has && k[u] == thr) ? 1 : 0;
            if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) {
                continue;                  // a row without a selected pool
            }
            int pe = my_eq;                // tied cells up to and with this lane
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, pe, o2);
                if (lane >= o2) pe += y;
            }
            const int left = eq_budget - (run_eq + pe - my_eq);
            const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
            const int my_sel = my_gt + take;
            int ps = my_sel;
#pragma unroll
            for (int o2 = 1; o2 < 32; o2 <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, ps, o2);
                if (lane >= o2) ps += y;
            }
            int32_t * dst = o + (int64_t) (run_sel + ps - my_sel)*kpool;
            for (int j = 0; j < my_sel; ++j) {
                for (int c = 0; c < kpool; ++c) {
                    dst[j*kpool + c] = pidx[b*kpool + c];
                }
            }
            run_eq  += __shfl_sync(0xffffffffu, pe, 31);
            run_sel += __shfl_sync(0xffffffffu, ps, 31);
        }
    }
    qsa_topk_epilogue(tidx, s_tidx, kpool, sentinel, s_nkept, qi, o, n_sel, &s_digit);
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1]; // key table, read in place through pool_cells
    const ggml_tensor * pidx  = dst->src[2];
    const ggml_tensor * tidx  = dst->src[3];
    const ggml_tensor * prng  = dst->src[4];
    const ggml_tensor * pcell = dst->src[5];

    const float head_scale = ggml_get_op_params_f32(dst, 0);
    const int   n_top      = ggml_get_op_params_i32(dst, 1);
    const int   sentinel   = ggml_get_op_params_i32(dst, 2);

    const int64_t n_tokens = q->ne[2];
    const int64_t n_pool   = pcell->ne[0];
    const int     kpool    = (int) pidx->ne[0];
    const int     n_sel    = (int) dst->ne[0];

    cudaStream_t stream = ctx.stream();

    const int64_t tile = std::min<int64_t>(n_tokens, QSA_SELECT_TILE);
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), n_pool*tile);

    for (int64_t t0 = 0; t0 < n_tokens; t0 += tile) {
        const int64_t nt = std::min<int64_t>(tile, n_tokens - t0);

        const float * q_d = (const float *) ((const char *) q->data + t0*q->nb[2]);
        ggml_cuda_lightning_indexer_scores(q_d, q->nb[1], q->nb[2], nt, k, (const int32_t *) pcell->data, n_pool,
                                           head_scale, scores.get(), n_pool*sizeof(float), stream);

        // n_pool bounds every query's visible range, so it picks the kernel
        const int32_t * rng = (const int32_t *) ((const char *) prng->data + t0*prng->nb[1]);
        const int32_t * tx  = (const int32_t *) ((const char *) tidx->data + t0*tidx->nb[1]);
        int32_t       * o   = (int32_t *) ((char *) dst->data + t0*dst->nb[1]);
#define QSA_TOPK_ARGS scores.get(), n_pool, rng, prng->nb[1]/sizeof(int32_t), (const int32_t *) pidx->data, kpool, \
                      tx, tidx->nb[1]/sizeof(int32_t), n_top, sentinel, o, n_sel
        if (n_pool <= (int64_t) QSA_TOPK_T*4) {
            qsa_topk_reg_kernel<4><<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        } else if (n_pool <= (int64_t) QSA_TOPK_T*8) {
            qsa_topk_reg_kernel<8><<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        } else if (n_pool <= (int64_t) QSA_TOPK_T*16) {
            qsa_topk_reg_kernel<16><<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        } else if (n_pool <= (int64_t) QSA_TOPK_T*24) {
            qsa_topk_reg_kernel<24><<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        } else if (n_pool <= (int64_t) QSA_TOPK_T*QSA_TOPK_PER) {
            qsa_topk_reg_kernel<QSA_TOPK_PER><<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        } else {
            qsa_topk_wide_kernel<<<nt, QSA_TOPK_T, 0, stream>>>(QSA_TOPK_ARGS);
        }
#undef QSA_TOPK_ARGS
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_qsa_select_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * pidx  = dst->src[2];
    const ggml_tensor * tidx  = dst->src[3];
    const ggml_tensor * prng  = dst->src[4];
    const ggml_tensor * pcell = dst->src[5];

    // the scoring kernels are written for 128-wide keys and 4 heads
    if (q->ne[0] != 128 || q->ne[1] != 4 || q->nb[0] != sizeof(float) || q->nb[1] % 16 != 0 || q->nb[2] % 16 != 0) {
        return false;
    }
    // the key rows are read in place: contiguous values in a row, rows nb[2] apart; F32 rows as float4
    if (k->nb[0] != ggml_type_size(k->type) || (k->type == GGML_TYPE_F32 && k->nb[2] % 16 != 0)) {
        return false;
    }
    if (!ggml_is_contiguous(pcell)) {
        return false;
    }
    switch (k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            break;
        default:
            return false;
    }
    return ggml_is_contiguous(pidx) && tidx->nb[0] == sizeof(int32_t) && prng->nb[0] == sizeof(int32_t) &&
        ggml_is_contiguous(dst);
}

// ---------------------------------------------------------------------------------------------------------------------
// attention: K/V rows of the ggml cache

struct qsa_kv {
    const char * k;
    const char * v;
    int64_t nbk1, nbk2, nbk3;
    int64_t nbv1, nbv2, nbv3;
};

// 8 consecutive values [d0, d0 + 8) of a row
template <ggml_type T>
static __device__ __forceinline__ void qsa_load8(const char * row, const int d0, float * out) {
    if constexpr (T == GGML_TYPE_F16) {
        const uint4 raw = *(const uint4 *) (row + d0*sizeof(half));
        const half2 * h2 = (const half2 *) &raw;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2*j + 0] = f.x;
            out[2*j + 1] = f.y;
        }
    } else {
        static_assert(T == GGML_TYPE_Q8_0, "unsupported type");
        const block_q8_0 * blk = (const block_q8_0 *) row + d0/QK8_0;
        const float d = __half2float(blk->d);
        const int8_t * qs = blk->qs + d0 % QK8_0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[j] = (float) qs[j] * d;
        }
    }
}

// one value d of a row
template <ggml_type T>
static __device__ __forceinline__ float qsa_load1(const char * row, const int d) {
    if constexpr (T == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[d]);
    } else {
        static_assert(T == GGML_TYPE_Q8_0, "unsupported type");
        const block_q8_0 * blk = (const block_q8_0 *) row + d/QK8_0;
        return (float) blk->qs[d % QK8_0] * __half2float(blk->d);
    }
}

// smem element of a KV row: half values for F16, the raw int8 codes for Q8_0 (the scales stay FP32)
template <ggml_type T> struct qsa_row_elem { using type = half; };
template <> struct qsa_row_elem<GGML_TYPE_Q8_0> { using type = int8_t; };

// a KV row [0, QSA_HD) copied into smem by the threads of a block: F16 verbatim, Q8_0 as int8 codes with the per-32
// scale written to dsc[]. part in [0, n_part) is the thread's piece of the row.
template <ggml_type T>
static __device__ __forceinline__ void qsa_row_to_smem(
        const char * row, typename qsa_row_elem<T>::type * dst, float * dsc, const int part, const int n_part) {
    if constexpr (T == GGML_TYPE_F16) {
        for (int i = part; i < QSA_HD*(int) sizeof(half)/16; i += n_part) {
            ((uint4 *) dst)[i] = ((const uint4 *) row)[i];
        }
        dsc[part] = 1.0f;
    } else {
        static_assert(T == GGML_TYPE_Q8_0, "unsupported type");
        const block_q8_0 * blk = (const block_q8_0 *) row + part;
        // both sides are only 2-byte aligned (the row carries 2-byte scales between the code blocks)
#pragma unroll
        for (int j = 0; j < QK8_0/2; ++j) {
            ((uint16_t *) dst)[part*QK8_0/2 + j] = ((const uint16_t *) blk->qs)[j];
        }
        dsc[part] = __half2float(blk->d);
    }
}

// two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152
static __device__ __forceinline__ uint32_t qsa_i8x2_to_h2(const uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    half2 h = *reinterpret_cast<half2 *>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t *>(&h);
}

static __device__ __forceinline__ float qsa_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

static __device__ __forceinline__ float qsa_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// The 12 heads' lane sums (part[12..15] zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1).
// Lane l ends holding head qsa_head_of_lane(l); lanes l and l^1 agree.
static __device__ __forceinline__ float qsa_reduce12(const float (&part)[16], const int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}

static __device__ __forceinline__ int qsa_head_of_lane(const int lane) {
    return ((lane >> 4) & 1)*8 + ((lane >> 3) & 1)*4 + ((lane >> 2) & 1)*2 + ((lane >> 1) & 1);
}

// chunked attention, one block per (chunk of QSA_CHUNK selected cells, KV head, token): partial accumulators,
// maxima and sums per chunk, combined by qsa_attn_merge_kernel
template <ggml_type TK, ggml_type TV>
static __global__ void __launch_bounds__(QSA_CHUNK_THREADS) qsa_attn_chunk_kernel(
        const float * __restrict__ q, const int64_t s_qh, const int64_t s_qt,
        const qsa_kv kv, const int32_t * __restrict__ ids, const int n_sel, const int n_tps, const float scale,
        float * __restrict__ part_acc, float * __restrict__ part_m, float * __restrict__ part_l,
        const int n_chunks, const int64_t scratch_stride) {
    constexpr int WARPS = QSA_CHUNK_THREADS/32;

    const int qi    = blockIdx.z;
    const int chunk = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int t     = threadIdx.x;
    const int lane  = t & 31;
    const int warp  = t >> 5;

    q        += qi*s_qt + (int64_t) kvh*QSA_G*s_qh;
    ids      += (int64_t) qi*n_sel;
    part_acc += qi*scratch_stride;
    part_m   += qi*scratch_stride;
    part_l   += qi*scratch_stride;

    const int64_t s = qi / n_tps;
    const char * k_base = kv.k + kvh*kv.nbk1 + s*kv.nbk3;
    const char * v_base = kv.v + kvh*kv.nbv1 + s*kv.nbv3;

    __shared__ __align__(16) float sq[QSA_G][QSA_HD]; // this KV head's query heads, then the probabilities
    __shared__ float sp[QSA_G][QSA_CHUNK];            // scores
    __shared__ int   scell[QSA_CHUNK];                // cell of each slot, -1 for none

    // probabilities cell-major, sharing the storage of sq once the scores are done
    float (*const spt)[QSA_G] = reinterpret_cast<float (*)[QSA_G]>(&sq[0][0]);
    static_assert(QSA_CHUNK*QSA_G <= QSA_G*QSA_HD, "spt must fit in sq");

    const int c0     = chunk*QSA_CHUNK;
    const int n_here = min(QSA_CHUNK, n_sel - c0);
    const int slot   = kvh*n_chunks + chunk;

    if (t < QSA_CHUNK) {
        scell[t] = t < n_here ? ids[c0 + t] : -1;
    }
    for (int i = t; i < QSA_G*QSA_HD; i += QSA_CHUNK_THREADS) {
        sq[i / QSA_HD][i % QSA_HD] = q[(i / QSA_HD)*s_qh + i % QSA_HD];
    }
    __syncthreads();

    // the valid ids are a prefix: a chunk that starts with -1 is empty
    if (scell[0] < 0) {
        if (t < QSA_G) {
            part_m[slot*QSA_G + t] = -FLT_MAX;
            part_l[slot*QSA_G + t] = 0.0f;
        }
        return;
    }

    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions, NC cells per step
    constexpr int NC = 2;
    const int hd = qsa_head_of_lane(lane);
    for (int i = 0; i < QSA_CHUNK/WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS*(i + j);
            all_ok = all_ok && scell[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                qsa_load8<TK>(k_base + scell[cc[j]]*kv.nbk2, lane*8, kk[j]);
            }
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < QSA_G; ++h) {
                const float4 qa = *reinterpret_cast<const float4 *>(&sq[h][lane*8]);
                const float4 qb = *reinterpret_cast<const float4 *>(&sq[h][lane*8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    pp[j][h] = kk[j][0]*qa.x + kk[j][1]*qa.y + kk[j][2]*qa.z + kk[j][3]*qa.w +
                               kk[j][4]*qb.x + kk[j][5]*qb.y + kk[j][6]*qb.z + kk[j][7]*qb.w;
                }
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = QSA_G; h < 16; ++h) {
                    pp[j][h] = 0.0f;
                }
                const float sj = qsa_reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < QSA_G) {
                    sp[hd][cc[j]] = sj*scale;
                }
            }
        } else {
            for (int j = 0; j < NC; ++j) {
                const int c = cc[j];
                if (scell[c] < 0) {
                    if (lane < QSA_G) {
                        sp[lane][c] = -FLT_MAX;
                    }
                    continue;
                }
                float k8[8];
                qsa_load8<TK>(k_base + scell[c]*kv.nbk2, lane*8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < QSA_G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4 *>(&sq[h][lane*8]);
                    const float4 qb = *reinterpret_cast<const float4 *>(&sq[h][lane*8 + 4]);
                    part[h] = k8[0]*qa.x + k8[1]*qa.y + k8[2]*qa.z + k8[3]*qa.w +
                              k8[4]*qb.x + k8[5]*qb.y + k8[6]*qb.z + k8[7]*qb.w;
                }
#pragma unroll
                for (int h = QSA_G; h < 16; ++h) {
                    part[h] = 0.0f;
                }
                const float sj = qsa_reduce12(part, lane);
                if ((lane & 1) == 0 && hd < QSA_G) {
                    sp[hd][c] = sj*scale;
                }
            }
        }
    }
    __syncthreads();

    // per-head chunk max and exp-sum: warp w handles heads w and w+8
    for (int h = warp; h < QSA_G; h += WARPS) {
        const float a = sp[h][lane];
        const float b = sp[h][lane + 32];
        const float m = qsa_warp_max(fmaxf(a, b));
        const float ea = scell[lane]      >= 0 ? __expf(a - m) : 0.0f;
        const float eb = scell[lane + 32] >= 0 ? __expf(b - m) : 0.0f;
        spt[lane][h]      = ea;
        spt[lane + 32][h] = eb;
        const float l = qsa_warp_sum(ea + eb);
        if (lane == 0) {
            part_m[slot*QSA_G + h] = m;
            part_l[slot*QSA_G + h] = l;
        }
    }
    __syncthreads();

    // values: thread t owns dimension t for all 12 heads, cells folded in ascending order
    float acc[QSA_G];
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        acc[h] = 0.0f;
    }
    const auto fold = [&](const int c, const float v) {
        const float4 * w4 = reinterpret_cast<const float4 *>(spt[c]);
        const float4 w0 = w4[0];
        const float4 w1 = w4[1];
        const float4 w2 = w4[2];
        acc[0]  = fmaf(w0.x, v, acc[0]);  acc[1]  = fmaf(w0.y, v, acc[1]);
        acc[2]  = fmaf(w0.z, v, acc[2]);  acc[3]  = fmaf(w0.w, v, acc[3]);
        acc[4]  = fmaf(w1.x, v, acc[4]);  acc[5]  = fmaf(w1.y, v, acc[5]);
        acc[6]  = fmaf(w1.z, v, acc[6]);  acc[7]  = fmaf(w1.w, v, acc[7]);
        acc[8]  = fmaf(w2.x, v, acc[8]);  acc[9]  = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    int c = 0;
    for (; c + 4 <= n_here; c += 4) {
        const int r0 = scell[c];
        const int r1 = scell[c + 1];
        const int r2 = scell[c + 2];
        const int r3 = scell[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) {
            break;
        }
        const float v0 = qsa_load1<TV>(v_base + r0*kv.nbv2, t);
        const float v1 = qsa_load1<TV>(v_base + r1*kv.nbv2, t);
        const float v2 = qsa_load1<TV>(v_base + r2*kv.nbv2, t);
        const float v3 = qsa_load1<TV>(v_base + r3*kv.nbv2, t);
        fold(c + 0, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (scell[c] < 0) {
            continue;
        }
        fold(c, qsa_load1<TV>(v_base + scell[c]*kv.nbv2, t));
    }
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        part_acc[((int64_t) slot*QSA_G + h)*QSA_HD + t] = acc[h];
    }
}

static __global__ void __launch_bounds__(QSA_HD) qsa_attn_merge_kernel(
        const float * __restrict__ part_acc, const float * __restrict__ part_m, const float * __restrict__ part_l,
        const int n_chunks, const int64_t scratch_stride,
        float * __restrict__ dst, const int64_t s_dh, const int64_t s_dt) {
    const int qi = blockIdx.y;
    const int h  = blockIdx.x; // query head
    const int d  = threadIdx.x;

    part_acc += qi*scratch_stride;
    part_m   += qi*scratch_stride;
    part_l   += qi*scratch_stride;

    const int kvh = h / QSA_G;
    const int hl  = h % QSA_G;

    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) {
        M = fmaxf(M, part_m[(kvh*n_chunks + c)*QSA_G + hl]);
    }
    float L   = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh*n_chunks + c;
        const float m = part_m[slot*QSA_G + hl];
        if (m == -FLT_MAX) {
            continue;
        }
        const float w = __expf(m - M);
        L   = fmaf(part_l[slot*QSA_G + hl], w, L);
        acc = fmaf(part_acc[((int64_t) slot*QSA_G + hl)*QSA_HD + d], w, acc);
    }
    dst[qi*s_dt + h*s_dh + d] = L > 0.0f ? acc / L : 0.0f;
}

// ---------------------------------------------------------------------------------------------------------------------
// attention, prompt kernel for Volta: mma.m8n8k4 (FP16 in, FP32 accumulate), one block per (token, KV head)
//
// Volta's tensor cores have m8n8k4 only: per warp four independent 8x8x4 products, one per quad-pair (QP q = lanes
// 4q..4q+3 and 4q+16..4q+19; thread j = (lane & 3) + 4*(lane >> 4) within it). The thread holds row j of A (k 0..3 as
// two registers) and column j of B; C register i is row (lane & 1) + 4*(lane >> 4) + 2*((i >> 1) & 1), column
// (lane & 2) + 4*(i >> 2) + (i & 1) of its QP's 8x8 tile.
//   q.k: warp w owns dims [64w, 64w+64); QP q takes cells 8q..8q+7 of the 32-cell chunk; 16 k-steps of 4 dims, two
//        8-row blocks (12 heads + 4 zero rows), q as hi + lo halves in separate chains.
//   p.v: warp w owns dims [64w, 64w+64); QP q its 16 dims 64w+16q.. (two n-tiles of 8); k = the 32 cells in steps of
//        4; p split into hi + lo halves.
// F16 cache rows are gathered as half. Q8_0 rows keep their int8 codes (exact in FP16) and their per-32 scales,
// which are applied on the FP32 partials.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING
#define QSA_VOLTA_MMA 1
#else
#define QSA_VOLTA_MMA 0
#endif

static __device__ __forceinline__ void qsa_mma884(float * c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if QSA_VOLTA_MMA
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, b0, b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ uint32_t qsa_pack_h2(const float lo_k, const float hi_k) { // element k in the low half
    half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t *>(&h);
}

#define QSA_QS (QSA_HD + 8) // q row stride in halves, bank-conflict-free fragment loads

// K/V smem rows: half values or int8 codes, 16-byte pieces and bank spread
template <ggml_type T> struct qsa_kv_smem {
    using elem = typename qsa_row_elem<T>::type;
    static constexpr int ROW    = T == GGML_TYPE_F16 ? QSA_HD + 8 : QSA_HD + 16;
    static constexpr int PIECES = QSA_HD*(int) sizeof(elem)/16; // 16-byte pieces of the values
};

template <ggml_type TK, ggml_type TV>
struct qsa_prompt_smem {
    half  qh[16][QSA_QS];
    half  ql[16][QSA_QS];
    typename qsa_kv_smem<TK>::elem k[QSA_PROMPT_CH][qsa_kv_smem<TK>::ROW];
    typename qsa_kv_smem<TV>::elem v[QSA_PROMPT_CH][qsa_kv_smem<TV>::ROW];
    float ks[QSA_PROMPT_CH][QSA_HD/32];   // K scale per 32 values (1 for F16)
    float vs[QSA_PROMPT_CH][QSA_HD/32];   // V scale per 32 values (1 for F16)
    float part[4][16][QSA_PROMPT_CH + 4]; // q.k per dim group
    float s[16][QSA_PROMPT_CH + 4];       // the chunk's probabilities
    float qmax[QSA_PROMPT_THREADS/32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    int   cell[QSA_PROMPT_CH];
};

template <ggml_type TK, ggml_type TV>
static __global__ void __launch_bounds__(QSA_PROMPT_THREADS) qsa_attn_prompt_kernel(
        const float * __restrict__ q, const int64_t s_qh, const int64_t s_qt,
        const qsa_kv kv, const int32_t * __restrict__ ids, const int n_sel, const int n_tps, const float scale_log2,
        float * __restrict__ dst, const int64_t s_dh, const int64_t s_dt) {
#if QSA_VOLTA_MMA
    constexpr int CH = QSA_PROMPT_CH;

    extern __shared__ __align__(16) unsigned char qsa_smem_raw[];
    qsa_prompt_smem<TK, TV> & S = *reinterpret_cast<qsa_prompt_smem<TK, TV> *>(qsa_smem_raw);

    const int qi  = blockIdx.x;
    const int kvh = blockIdx.y;

    q   += qi*s_qt + (int64_t) kvh*QSA_G*s_qh;
    dst += qi*s_dt + (int64_t) kvh*QSA_G*s_dh;
    ids += (int64_t) qi*n_sel;

    const int64_t s = qi / n_tps;
    const char * k_base = kv.k + kvh*kv.nbk1 + s*kv.nbk3;
    const char * v_base = kv.v + kvh*kv.nbv1 + s*kv.nbv3;

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int qp   = (lane >> 2) & 3;                 // quad-pair
    const int jj   = (lane & 3) + 4*(lane >> 4);      // thread within it
    const int crow = (lane & 1) + 4*(lane >> 4);      // C row of register 0 (+2 per register pair)
    const int ccol = lane & 2;                        // C column of register 0 (+4 per half, +1 per register)

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14, split into hi + lo
    float qm = 0.0f;
    for (int i = t; i < QSA_G*QSA_HD; i += QSA_PROMPT_THREADS) {
        qm = fmaxf(qm, fabsf(q[(i / QSA_HD)*s_qh + i % QSA_HD]));
    }
    qm = qsa_warp_max(qm);
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) {
        frexpf(qm, &qe); // qm < 2^qe
    }
    const float qup   = ldexpf(1.0f, 14 - qe);
    const float qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16*QSA_HD; i += QSA_PROMPT_THREADS) {
        const int h = i / QSA_HD;
        const int d = i % QSA_HD;
        const float x = h < QSA_G ? q[h*s_qh + d]*qup : 0.0f;
        const half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }

    float acc[2][2][8];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[rb][nt][i] = 0.0f;
            }
        }
    }

    for (int c0 = 0; c0 < n_sel; c0 += CH) {
        // the valid ids are a prefix: stop at the first chunk that starts with -1
        if (__ldg(ids + c0) < 0) {
            break;
        }
        const int nh = min(CH, n_sel - c0);
        if (t < CH) {
            S.cell[t] = t < nh ? ids[c0 + t] : -1;
        }
        __syncthreads(); // cells ready; the previous chunk's p.v is done with k, v, s

        // gather the chunk's K and V rows, with their per-32 scales; a missing cell is zero and masked below
        for (int i = t; i < CH*8; i += QSA_PROMPT_THREADS) {
            const int c    = i / 8;
            const int part = i % 8;
            const int cell = S.cell[c];
            if (cell >= 0) {
                qsa_row_to_smem<TK>(k_base + cell*kv.nbk2, S.k[c], S.ks[c], part, 8);
                qsa_row_to_smem<TV>(v_base + cell*kv.nbv2, S.v[c], S.vs[c], part, 8);
            } else {
                for (int d = part; d < qsa_kv_smem<TK>::PIECES; d += 8) {
                    ((uint4 *) S.k[c])[d] = make_uint4(0, 0, 0, 0);
                }
                for (int d = part; d < qsa_kv_smem<TV>::PIECES; d += 8) {
                    ((uint4 *) S.v[c])[d] = make_uint4(0, 0, 0, 0);
                }
                S.ks[c][part] = 0.0f;
                S.vs[c][part] = 0.0f;
            }
        }
        __syncthreads();

        // scores: warp w = dim group w (64 dims), QP q = cells 8q..8q+7. The q hi and lo halves accumulate in their
        // own m8n8k4 chains (tg, tgl), added in FP32 below: in one chain Volta's truncating accumulation dropped
        // most of the lo half correction. Q8_0 K codes enter the mma exactly, the per-32 scale stays in FP32.
        {
            constexpr int NG = TK == GGML_TYPE_F16 ? 1 : 2; // scale groups per 64-dim warp row
            float tg[NG][2][8], tgl[NG][2][8];
#pragma unroll
            for (int g = 0; g < NG; ++g) {
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        tg[g][rb][i] = 0.0f;
                        tgl[g][rb][i] = 0.0f;
                    }
                }
            }
            const int cell_b = qp*8 + jj;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int d0 = warp*64 + ks*4;
                const int g  = NG == 1 ? 0 : ks >> 3;
                uint32_t b0, b1;
                if constexpr (TK == GGML_TYPE_F16) {
                    const uint2 raw = *reinterpret_cast<const uint2 *>(&S.k[cell_b][d0]);
                    b0 = raw.x;
                    b1 = raw.y;
                } else {
                    const uint32_t raw = *reinterpret_cast<const uint32_t *>(&S.k[cell_b][d0]);
                    b0 = qsa_i8x2_to_h2(raw & 0xffffu);
                    b1 = qsa_i8x2_to_h2(raw >> 16);
                }
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const uint2 ah = *reinterpret_cast<const uint2 *>(&S.qh[rb*8 + jj][d0]);
                    const uint2 al = *reinterpret_cast<const uint2 *>(&S.ql[rb*8 + jj][d0]);
                    qsa_mma884(tg[g][rb], ah.x, ah.y, b0, b1);
                    qsa_mma884(tgl[g][rb], al.x, al.y, b0, b1);
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row  = rb*8 + crow + 2*((i >> 1) & 1);
                    const int cell = qp*8 + ccol + 4*(i >> 2) + (i & 1);
                    if constexpr (NG == 1) {
                        S.part[warp][row][cell] = tg[0][rb][i] + tgl[0][rb][i];
                    } else {
                        S.part[warp][row][cell] = (tg[0][rb][i] + tgl[0][rb][i])*S.ks[cell][2*warp + 0] +
                                                  (tg[1][rb][i] + tgl[1][rb][i])*S.ks[cell][2*warp + 1];
                    }
                }
            }
        }
        __syncthreads();

        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH/8;
            const int r   = t >> 3;
            const int sub = t & 7;
            float x[PER];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub*PER + j;
                const float sum4 = ((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c];
                x[j] = c < nh && S.cell[c] >= 0 ? sum4*qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub*PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r]  = fmaf(S.lsum[r], a, sum);
                S.mrow[r]  = m_new;
            }
        }
        __syncthreads();

        // p.v: warp w owns dims [64w, 64w+64), QP q its 16 dims 64w+16q.. The V scale is folded into p in FP32
        // relative to this warp's largest (times 2^14, the accumulator scales back). The p hi and lo halves
        // accumulate in their own chains (tmp, tmpl); Q8_0 V codes enter the mma exactly.
        {
            const int vg = 2*warp + (qp >> 1); // the 32-value scale group of this thread's dims
            float vmax = 1.0f;
            if constexpr (TV == GGML_TYPE_Q8_0) {
                vmax = 0.0f;
                for (int c = 0; c < CH; ++c) {
                    vmax = fmaxf(vmax, S.vs[c][vg]);
                }
            }
            const float vup   = vmax > 0.0f ? 16384.0f/vmax : 0.0f;
            const float vdown = vmax*(1.0f/16384.0f);

            float tmp[2][2][8], tmpl[2][2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        tmp[rb][nt][i]  = 0.0f;
                        tmpl[rb][nt][i] = 0.0f;
                    }
                }
            }
#pragma unroll
            for (int ks = 0; ks < CH/4; ++ks) {
                const int cs = ks*4;
                float w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    w[e] = S.vs[cs + e][vg]*vup;
                }
                uint2 ah[2];
                uint2 al[2];
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 pv = *reinterpret_cast<const float4 *>(&S.s[rb*8 + jj][cs]);
                    const float p0 = pv.x*w[0];
                    const float p1 = pv.y*w[1];
                    const float p2 = pv.z*w[2];
                    const float p3 = pv.w*w[3];
                    ah[rb].x = qsa_pack_h2(p0, p1);
                    ah[rb].y = qsa_pack_h2(p2, p3);
                    const float2 f01 = __half22float2(*reinterpret_cast<const half2 *>(&ah[rb].x));
                    const float2 f23 = __half22float2(*reinterpret_cast<const half2 *>(&ah[rb].y));
                    al[rb].x = qsa_pack_h2(p0 - f01.x, p1 - f01.y);
                    al[rb].y = qsa_pack_h2(p2 - f23.x, p3 - f23.y);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp*64 + qp*16 + nt*8 + jj;
                    uint32_t b0, b1;
                    if constexpr (TV == GGML_TYPE_F16) {
                        const half2 h0 = __halves2half2(S.v[cs + 0][d], S.v[cs + 1][d]);
                        const half2 h1 = __halves2half2(S.v[cs + 2][d], S.v[cs + 3][d]);
                        b0 = *reinterpret_cast<const uint32_t *>(&h0);
                        b1 = *reinterpret_cast<const uint32_t *>(&h1);
                    } else {
                        const uint32_t x0 = (uint8_t) S.v[cs + 0][d] | ((uint32_t) (uint8_t) S.v[cs + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cs + 2][d] | ((uint32_t) (uint8_t) S.v[cs + 3][d] << 8);
                        b0 = qsa_i8x2_to_h2(x0);
                        b1 = qsa_i8x2_to_h2(x1);
                    }
#pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        qsa_mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                        qsa_mma884(tmpl[rb][nt], al[rb].x, al[rb].y, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                const float a0 = S.alpha[rb*8 + crow];
                const float a1 = S.alpha[rb*8 + crow + 2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[rb][nt][i] = fmaf(acc[rb][nt][i], ((i >> 1) & 1) ? a1 : a0,
                                              (tmp[rb][nt][i] + tmpl[rb][nt][i])*vdown);
                    }
                }
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb*8 + crow + 2*h;
            if (row >= QSA_G) {
                continue;
            }
            const float l   = S.lsum[row];
            const float inv = l > 0.0f ? 1.0f/l : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh*4 + h*2; // registers i, i+1: row h, columns ccol + 4*hh + {0, 1}
                    const int d = warp*64 + qp*16 + nt*8 + ccol + 4*hh;
                    *reinterpret_cast<float2 *>(dst + row*s_dh + d) =
                        make_float2(acc[rb][nt][i]*inv, acc[rb][nt][i + 1]*inv);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, s_qh, s_qt, kv, ids, n_sel, n_tps, scale_log2, dst, s_dh, s_dt);
    NO_DEVICE_CODE;
#endif // QSA_VOLTA_MMA
}

template <ggml_type TK, ggml_type TV>
static void qsa_attn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int64_t n_head    = q->ne[1];
    const int64_t n_tokens  = q->ne[2];
    const int64_t n_head_kv = k->ne[1];
    const int64_t n_stream  = k->ne[3];
    const int     n_sel     = (int) ids->ne[0];
    const int     n_tps     = (int) (n_tokens/n_stream);

    const qsa_kv kv = {
        (const char *) k->data, (const char *) v->data,
        (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
        (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
    };

    const float * q_d   = (const float *)   q->data;
    const int32_t * i_d = (const int32_t *) ids->data;
    float * dst_d       = (float *)         dst->data;

    const int64_t s_qh = q->nb[1]/sizeof(float);
    const int64_t s_qt = q->nb[2]/sizeof(float);
    const int64_t s_dh = dst->nb[1]/sizeof(float);
    const int64_t s_dt = dst->nb[2]/sizeof(float);

    cudaStream_t stream = ctx.stream();

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // capability branch, not a fallback: the prompt kernel is written for Volta's mma.m8n8k4, other GPUs run the
    // chunked kernel for every batch size
    const bool prompt = n_tokens > QSA_ATTN_CHUNK_MAX_TOKENS && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;

    if (prompt) {
        const int smem = (int) sizeof(qsa_prompt_smem<TK, TV>);
        static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
        const int device = ggml_cuda_get_device();
        if (!attr_set[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(qsa_attn_prompt_kernel<TK, TV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set[device] = true;
        }
        const float scale_log2 = 1.4426950408889634f*scale;
        GGML_ASSERT(n_tokens <= 65535);
        qsa_attn_prompt_kernel<TK, TV><<<dim3((unsigned) n_tokens, (unsigned) n_head_kv), QSA_PROMPT_THREADS, smem, stream>>>(
            q_d, s_qh, s_qt, kv, i_d, n_sel, n_tps, scale_log2, dst_d, s_dh, s_dt);
    } else {
        const int n_chunks = (n_sel + QSA_CHUNK - 1)/QSA_CHUNK;
        // per token: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head]
        const int64_t stride = (int64_t) n_chunks*n_head*(QSA_HD + 2);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), stride*n_tokens);
        float * part_acc = scratch.get();
        float * part_m   = part_acc + (int64_t) n_chunks*n_head*QSA_HD;
        float * part_l   = part_m   + (int64_t) n_chunks*n_head;

        GGML_ASSERT(n_tokens <= 65535);
        qsa_attn_chunk_kernel<TK, TV><<<dim3((unsigned) n_chunks, (unsigned) n_head_kv, (unsigned) n_tokens), QSA_CHUNK_THREADS, 0, stream>>>(
            q_d, s_qh, s_qt, kv, i_d, n_sel, n_tps, scale, part_acc, part_m, part_l, n_chunks, stride);
        qsa_attn_merge_kernel<<<dim3((unsigned) n_head, (unsigned) n_tokens), QSA_HD, 0, stream>>>(
            part_acc, part_m, part_l, n_chunks, stride, dst_d, s_dh, s_dt);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    if (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0) {
        qsa_attn_launch<GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
    } else if (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
        qsa_attn_launch<GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
    } else if (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_F16) {
        qsa_attn_launch<GGML_TYPE_Q8_0, GGML_TYPE_F16>(ctx, dst);
    } else if (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_Q8_0) {
        qsa_attn_launch<GGML_TYPE_F16, GGML_TYPE_Q8_0>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_qsa_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    // the kernels are written for 256-wide heads and 12 query heads per KV head
    if (k->ne[0] != QSA_HD || v->ne[0] != QSA_HD || q->ne[1] != QSA_G*k->ne[1]) {
        return false;
    }
    for (const ggml_tensor * t : { k, v }) {
        if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0) {
            return false;
        }
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 16 != 0 || t->nb[2] % 2 != 0) {
            return false;
        }
    }
    if (q->nb[0] != sizeof(float) || q->nb[1] % 16 != 0 || q->nb[2] % 16 != 0) {
        return false;
    }
    return ids->nb[0] == sizeof(int32_t) && ggml_is_contiguous(ids) && dst->nb[0] == sizeof(float) && q->ne[2] <= 65535;
}
