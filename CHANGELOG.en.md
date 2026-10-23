# Changelog

[简体中文](CHANGELOG.md) | **English** | [← Home README](README.en.md) | [Benchmark history](docs/v100/benchmarks.en.md)

All measured tables and test conditions of every version are in [Benchmark history](docs/v100/benchmarks.en.md).

## 2.0.1.1 (2026-10-23)

- **Qwen3.8-Flash-Next supports 2 × V100-16GB**: Qwen3.8-Flash-Next (UD-Q4_K_XL + MTP) can run on 2 × 16 GB GPUs with a context of 262144; for the launch command see [command C on the Qwen3.8-Flash-Next detail page](docs/v100/moe-2.0.0.en.md#c-2--16-gb-gpus-ac922-two-gpus-on-different-cpu-sockets). The previous version could not start on 2 GPUs.
- **Host memory needed on 2 GPUs**: about 63 GiB (context 262144; the smaller `-c`, the less is needed). Two GPUs on different CPU sockets are recommended; two GPUs on the same socket also run, but with very little host memory headroom, and each round is also slightly slower.
- **No regression on 6 GPUs or for 27B**: in a same-session A/B comparison with 2.0.1, the speed changes of 6-GPU Qwen3.8-Flash-Next and of 27B at every prompt length are all within measurement noise, and the temperature-0 output is the same.
- **VRAM**: the per-GPU VRAM peak of 6-GPU Qwen3.8-Flash-Next and of 27B is the same as in 2.0.1.
- **Web UI included in the source**: it is embedded directly into the program at build time, so a fresh clone can be built offline; the whole build needs neither network access nor Node.js. With the previous version, building it yourself required downloading the UI files from Hugging Face, and if they could not be downloaded, the resulting server had no web UI.
- Docker image updated to `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1` (multi-architecture: x86_64 and IBM Power AC922).
- Command-line parameters and defaults are the same as in 2.0.1.
- Docs: the README is trimmed down and a new [AI deployment guide](docs/v100/ai-deploy.en.md) is added; the detail pages keep only usage, and the measured data is collected in [Benchmark history](docs/v100/benchmarks.en.md).

Measured: 2 GPUs (thinking on + the model card's sampling parameters) and the same-session A/B comparison of 6 GPUs against 2.0.1 (temperature 0, thinking off) are in [Benchmark history · 2.0.1.1](docs/v100/benchmarks.en.md#2011).

## 2.0.1 (2026-10-18)

- **Qwen3.8-Flash-Next is faster**: in a same-session A/B comparison with 2.0.0, prefill +48% at ~200K tokens and +38% at ~16K tokens; decode +8% at ~200K tokens, +18% at ~16K tokens and +11% for a short prompt.
- **27B**: prefill +3% and decode +2% at ~200K tokens; short-prompt decode +5%.
- **VRAM**: the VRAM peak of every GPU is no higher than in 2.0.0; Qwen3.8-Flash-Next saves about 30–70 MiB per GPU.
- **Fixes**: 27B temperature-0 output not consistent across restarts on 6 GPUs at long context; the server aborting with DFlash2 / MTP speculative decoding at temperature 0 with several slots (`-np > 1`) (reported by Poqianhu7895, PR #4); extra latency in the first decode round of Qwen3.8-Flash-Next; a startup crash with single-GPU MTP, as well as several out-of-bounds errors and startup crashes in multi-GPU prefill; improved FFmpeg-related diagnostic messages for multimodal (umacro, PR #3).
- **New parameter** `--fit-mmproj on|off` (default `on`, same as the previous behavior): whether to set aside VRAM for the multimodal projector when planning VRAM with `--fit`. The defaults of all other parameters are the same as in 2.0.0.

Measured (temperature 0, thinking off): [Benchmark history · 2.0.1](docs/v100/benchmarks.en.md#201).

## 2.0.0 (2026-10-12)

- **License**: GPL-3.0 from this version on (full text in `LICENSE`); 1.1.1 and earlier were MIT. The MIT license text of upstream llama.cpp is kept as `licenses/LICENSE-llama.cpp`, and ported third-party code keeps its original license (see `licenses/README.md`).
- **New V100 support for Qwen3.8-Flash-Next**: 6 × V100-16GB (recommended) can run a context of 262144 with MTP speculative decoding; 4 × V100-32GB (x86) can run it. Whatever does not fit in VRAM automatically uses host memory, with no need for `--n-cpu-moe` / `-ot`; it cannot start when host memory is not enough. Usage on the [Qwen3.8-Flash-Next detail page](docs/v100/moe-2.0.0.en.md).
- **New parameters**: `--expert-profile` (expert routing profile: keeps frequently used experts in VRAM, which is faster), `--dump-expert-routing` (collects a profile) and the script `scripts/v100-expert-profile.sh`; `--spec-draft-window` (the MTP draft only looks at a number of the most recent positions, default 32768); `--spec-draft-ubatch` (a separate physical batch size for the draft; inherits `-ub` by default, the same behavior as 1.1.1).
- **27B is faster**: in a same-session A/B comparison with 1.1.1, short-prompt decode +13%, prefill at ~16K tokens +9%, decode at ~200K tokens +8%.
- Models with BF16 weights can be loaded on V100 (converted to F16 at load time; finite values outside the FP16 range are clamped to ±65504); multi-GPU prefill with layer split (`--split-mode layer`) is faster.
- **Fixes**: extra time added to every prefill batch when speculative decoding is enabled (Qwen3.8-Flash-Next + MTP is affected; the current 27B configurations are not); immediate startup failure of layer split combined with the DFlash2 draft (already present in 1.1.1); wrong IQ1_S results computed on the CPU on POWER9 (ppc64le) (only affects CPU computation); other stability fixes.
- The docs are reorganized into a home page plus two detail pages ([27B](docs/v100/27b.en.md) and [Qwen3.8-Flash-Next](docs/v100/moe-2.0.0.en.md)); Docker image updated to `ghcr.io/1115714829/llama.cpp-v100:2.0.0`.

Measured: on the same machine, Qwen3.8-Flash-Next's prefill at ~200K tokens is about 13% higher than 27B's, and its decode is about 87% of 27B's; all data in [Benchmark history · 2.0.0](docs/v100/benchmarks.en.md#200).

## 1.1.1 (2026-10-06)

- Merged upstream llama.cpp v0.6.0; all of this project's speculative decoding and server features (DFlash2, multi-slot concurrency, `--prefill-pace`, `-mmdev` for the vision module, etc.) are kept; speed, output and VRAM are the same as in 1.1.0.
- Loading and running support for Qwen3.8-Flash-Next (`qwen4exp` architecture) came in with upstream; V100-specific optimization starts in 2.0.0.
- Docker image updated to `ghcr.io/1115714829/llama.cpp-v100:1.1.1`.
- Parameter documentation added: how `--cache-ram` and `--ctx-checkpoints` relate to host memory usage, the per-request thinking effort `reasoning_effort`, and the `NCCL_P2P_LEVEL=SYS` set automatically on ppc64le.

Measured: [Benchmark history · 1.1.1](docs/v100/benchmarks.en.md#111); the measured tables and launch parameters of each configuration are also in the [27B results archive (1.1.1)](docs/v100/27b-results-1.1.1.en.md).

## 1.1.0 (2026-10-03)

- Docker image: `ghcr.io/1115714829/llama.cpp-v100:1.1.0`, one tag containing both x86_64 and IBM Power AC922 (ppc64le); the entry point is `llama-server`, which is started with `numactl --membind` when `LLAMA_NUMA_MEMBIND` is set; `.devops/v100-ac922-cdi.sh` generates the CDI spec on the AC922.
- Fixed crashes of the 6-GPU multi-slot server at long context (several slots each processing a long prompt of more than 32768 tokens at the same time, e.g. agent sessions with large images).
- Everything else is on par with 1.0.9.

Measured: [Benchmark history · 1.1.0](docs/v100/benchmarks.en.md#110).

## 1.0.9 (2026-10-02)

- Decode about 10% faster in the Q4 configuration (including Q2_K–Q6_K mixed quantization); Q5_K / Q6_K output differs slightly from 1.0.8. From external contributor ATIVX928 (PR #2); test cases were added and an edge case was fixed when merging; one change in the PR that showed no measured gain was not merged.
- Slightly faster decode with a q4_0 KV cache; fixed a crash with q4_1 / q5_1 KV caches on V100.
- The Q8 configuration is the same as in 1.0.8.

Measured: [Benchmark history · 1.0.9](docs/v100/benchmarks.en.md#109).

## 1.0.8 (2026-10-02)

- Faster long-context prefill on 6 GPUs: at a 524288 context, synthetic 209715 tokens +16% and 419430 tokens +24%; about 300 MiB more VRAM per GPU.
- Long-context prefill on 4 GPUs +7.4%; slightly faster decode with speculative decoding.
- The vision module is split across all GPUs used by the main model (2–6 GPUs); VRAM is more balanced across GPUs, and the VRAM peak of the highest GPU drops by about 0.4–0.6 GB.
- Time to first token with multi-slot concurrency is the same or faster.

Measured: [Benchmark history · 1.0.8](docs/v100/benchmarks.en.md#108).

## 1.0.7 (2026-10-01)

- 3-GPU load balancing: 3-GPU Q8 prefill +4%–+9%, decode +3%; the VRAM peak per GPU is about 0.3 GB higher.
- Less VRAM at long context: at a 262144 context the VRAM peak per GPU drops by about 0.46–0.54 GB, with prefill speed unchanged.
- Faster decode in non-thinking mode (presence penalty): time per round −17%.
- Faster processing of small, fragmented batches (including batches of 17–64 tokens in the Q4 configuration).
- 3, 5 and 6 GPUs no longer impose a divisibility requirement on the model's vocabulary size; multi-GPU communication optimized for 5 GPUs.
- Shared KV across slots (`--kv-unified`, with several slots): a new request can reuse a longer prefix from another slot; not enabled by default.
- When the middle of a long session is rewritten, it is no longer recomputed in full.
- Fixed crashes of a few Flash Attention shapes on V100.
- Draft model quantization comparison: F16 has a clearly higher acceptance rate on short tasks, and the three are close at long context; choose F16 when VRAM is sufficient; when it is tight, Q8_0 / Q4_K_M lose very little.

Measured: [Benchmark history · 1.0.7](docs/v100/benchmarks.en.md#107).

## 1.0.6 (2026-09-30)

- Faster prefill on 2, 3, 5 and 6 GPUs (6 GPUs 200K +5.4%, 2-GPU Q4 +8%–10%, 3-GPU Q8 +6%); perplexity unchanged.
- Faster decode in the Q4 configuration (−5%–9% per round); slight output changes, perplexity unchanged.
- Output, speed and VRAM of the 4-GPU Q8 configuration are the same as in 1.0.5.

Measured: [Benchmark history · 1.0.6](docs/v100/benchmarks.en.md#106).

## 1.0.5 (2026-09-30)

- New Q4 configuration (main model UD-Q4_K_M, KV cache q4_0): 2 GPUs can run 262144 (with the Q4_K_M draft), and 3-GPU Q4 can run 524288. From external contributor ATIVX928 (PR #1).
- Comprehensive measurements on 2–6 GPUs (single slot and two concurrent slots).
- Fixed errors and aborts when running on 2 GPUs with `GGML_CUDA_ALLREDUCE=internal` set (from PR #1).
- The 1.0.4 known issue "4-GPU 200K prefill about 0.2% slower" is closed: within measurement error, not a regression.
- The 4-GPU and 6-GPU Q8 configurations produce output token-for-token identical to 1.0.4; speed and VRAM unchanged.

Measured: [Benchmark history · 1.0.5](docs/v100/benchmarks.en.md#105).

## 1.0.4 (2026-09-30)

- New parameter `--prefill-pace P` (default 30): with several slots, while a request is decoding, prompt processing for the other requests takes at most P% of the time, so decoding requests are no longer slowed down by long prompts; 0 or 100 is the same as 1.0.3.
- Faster decode with several slots decoding at the same time (4 GPUs, two slots of 16K each: per slot 45.7 / 58.8 → 81.4 / 111.4 tok/s); less VRAM with several slots, so 4 GPUs with `-np 4` and 6 GPUs with `-c 524288 -np 2` can run.
- DFlash2 with several slots: fixed the draft possibly using another slot's data when several slots are in the same batch.
- Single slot: speed and VRAM the same as 1.0.3, short-prompt output token-for-token identical; long-prompt output may differ slightly, perplexity unchanged.
- Known issue: 4-GPU 200K prefill about 0.2% slower than 1.0.3 (closed in 1.0.5, not a regression).

Measured: [Benchmark history · 1.0.4](docs/v100/benchmarks.en.md#104).

## 1.0.3 (2026-09-28)

- Faster decode on 6 GPUs: time per round down 6%–9%; takes effect on machines where the 6 GPUs can be divided into two groups (3 GPUs each) by NVLink link speed (e.g. the AC922); other topologies are the same as 1.0.2; prefill and VRAM unchanged.
- 4 GPUs are not affected; output token-for-token identical to 1.0.2.

Measured: [Benchmark history · 1.0.3](docs/v100/benchmarks.en.md#103).

## 1.0.2 (2026-09-28)

- Fix: in VRAM-tight configurations (e.g. 6 GPUs at 512K), putting the vision module on the GPU made the very first request run out of VRAM and exit.
- Less VRAM at long context: about 1 GB less per GPU at a 256K context and about 2 GB less at 512K, and the same amount less host memory.
- Multimodal: 4 GPUs at 256K and 6 GPUs at 512K can both put the vision module on the GPU; time to first token for one image drops from 2–4 minutes to about 5 seconds; after an image appears in a conversation, later requests keep long-context speed.
- When requests with different sampling parameters alternate, there are no longer pauses between requests.

Measured: [Benchmark history · 1.0.2](docs/v100/benchmarks.en.md#102).

## 1.0.1 (2026-09-28)

- 6-GPU support (correct splitting even when the number of attention heads is not divisible by the GPU count).
- Automatic builds on GitHub; releases include prebuilt x86_64 Linux binaries.
- New script `scripts/v100-get-models.sh` for downloading the models and converting the draft model.
- Fixes: build errors with some compilers; shared library link failure on x86_64.

Measured: [Benchmark history · 1.0.1](docs/v100/benchmarks.en.md#101).

## 1.0.0 (2026-09-28)

- First release: V100 (SM70)-specific optimization of Qwen3.8-27B (Q8_0) + DFlash2 speculative decoding, 4-GPU tensor parallelism.

Measured: [Benchmark history · 1.0.0](docs/v100/benchmarks.en.md#100).
