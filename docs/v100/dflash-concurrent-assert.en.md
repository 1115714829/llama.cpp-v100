# DFlash2 speculative decoding crash under concurrent slots on multi-GPU tensor split: GGML_ASSERT(ggml_is_contiguous)

**Summary**: With 6× V100 `--split-mode tensor` + `--model-draft` (DFlash2) + `-np 4`, **concurrent** multi-slot load aborts the process at `ggml-backend-meta.cpp:4728` — `GGML_ASSERT(ggml_is_contiguous(tensor))` inside `ggml_backend_meta_set_tensor_async`. Under systemd this is a crash loop (~16 s per exit, status=1). A single request works fine. Reproduced on both the 2.0.1.1 and 2.0.1 builds.

## Environment

- Hardware: Lenovo AC922 (POWER9 ppc64le, 2 sockets, 176 cores, 221 GiB RAM), 6× Tesla V100-SXM2-16GB
  - NUMA 0: GPU 0,1,2; NUMA 8: GPU 3,4,5; NVLink only within a socket
- Driver / CUDA: 550.54.15 / CUDA 12.4
- Version: Llama.cpp-V100 **2.0.1.1** (2.0.1 build aborts identically)
- Main model: Qwen3.8-27B-Q8_0.gguf (~28 GB, 64 layers incl. a nextn/MTP block)
- Draft: Qwen3.8-27B-DFlash2-F16.gguf (~3.6 GB), `--spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 0`
  - logged params: block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true

## Repro command (6 cards, 524K pool, 4×131K slots)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./llama-server \
  -m Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 4 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 0 \
  --host 0.0.0.0 --port 8081 --metrics
```

Then send **4 concurrent** chat requests (one per slot).

## Crash sequence

The first two slots finish normally (draft acceptance = 0.714 / 0.560, mean len = 6.00 / 4.92); the next `pre_decode` of another slot aborts:

```
ggml/src/ggml-backend-meta.cpp:4728: GGML_ASSERT(ggml_is_contiguous(tensor)) failed
```

Backtrace, key frames (bottom-up):

```
server_queue::start_loop
server_context_impl::update_slots
server_context_impl::pre_decode
common_speculative_draft
common_speculative_impl_draft_dflash::draft
llama_process
llama_context::decode
llama_context::process_ubatch
llama_context::graph_compute_impl
ggml_backend_sched_graph_compute_async
ggml_backend_sched_compute_splits
ggml_backend_tensor_set_async          <- failure point
```

Failing source (`ggml_backend_meta_set_tensor_async`, ggml-backend-meta.cpp:4726-4728):

```c
static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));   // <- line 4728 fails
```

## Configuration matrix (same host, measured 2026-10-25)

| Config | Spec draft | Concurrency | Result |
|---|---|---|---|
| 6 cards `-np 4` | DFlash2 | 1 request | ✅ OK (acceptance 0.571) |
| 6 cards `-np 4` | DFlash2 | **4 concurrent** | ❌ **GGML_ASSERT abort (stable repro)** |
| 6 cards `-np 4` | none | long multi-task run | ✅ OK |
| 6 cards `-np 8` | none | up to 8 slots | ✅ OK (Merkyor Q8 main model, previous production config) |
| 4 cards (GPU 1,2,4,5) `-np 2` | DFlash2 | ≤2 concurrent | ✅ OK (in production for days, ~142 tok/s) |
| 6 cards `-np 6` | DFlash2 | — | not executed (test blocked by port :8080 conflict, no data) |
| 6 cards `-np 4` | MTP (nextn) | — | ❌ different failure: nextn "h sink" tensor missing (independent of this bug) |

## Hypothesis

The DFlash draft path (`common_speculative_impl_draft_dflash::draft`) hands a **non-contiguous (strided) tensor** to the meta backend's `set_tensor_async` when multiple slots are processed concurrently — likely a per-slot draft-state race/aliasing under `SPLIT_MODE_TENSOR`. Single-slot traffic and `-np 2` (≤2 concurrent) do not trigger it.

## Side observations

1. Startup warning: `common_fit_params: failed to fit params to free device memory: llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` — says "abort" but the process continues (fit_params unimplemented for tensor split).
2. With a main model that contains a nextn (MTP) block and a DFlash2 draft, load prints 11 unused-tensor warnings (`blk.64.nextn.*`, `blk.64.attn_*`, `blk.64.ffn_*` are ignored) — harmless but noisy.

## Workaround

Drop `--model-draft` (DFlash2) when `-np > 2`. Cost: no speculative speedup on the high-concurrency 6-card config (measured acceptance 0.56–0.71, mean len 5–6 — a real gain when it works).

## Suggested fix direction

1. In the dflash draft path, `ggml_contiguous` the target tensor before `tensor_set`, or track down where the strided view originates (which per-slot buffer).
2. At minimum, degrade the assert at line 4728 into an error return (`ggml_status`) instead of aborting the whole server.

## Attached raw logs

- Single request OK: `np4_dflash_assert_repro.log` (1.55 min load → 1 request, 61 tokens → clean exit)
- 4 concurrent crash: `np4_dflash_concurrent.log` (full backtrace)
- Service-mode crash loop: `v100_6gpu_np4.log` (same assert repeated at high task ids)
