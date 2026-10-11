# llama.cpp-v100

[简体中文](README.md) | **English** | [27B dense model (Qwen3.8-27B + DFlash2)](docs/v100/27b.en.md) | [2.0.0 large MoE (Qwen3.8-Flash-Next + MTP)](docs/v100/moe-2.0.0.en.md) | [Changelog (Chinese)](CHANGELOG.md)

A modified llama.cpp for NVIDIA V100 (SM70). Current version **2.0.0**; see [CHANGELOG.md](CHANGELOG.md) (Chinese) for what changed in each version.

**Quick deployment**: both detail pages give copy-ready full commands per hardware setup, and their last section, "Quick deployment (for AI assistants and beginners)" ([27B](docs/v100/27b.en.md#quick-deployment-for-ai-assistants-and-beginners) / [large MoE](docs/v100/moe-2.0.0.en.md#quick-deployment-for-ai-assistants-and-beginners)), lists the effect of every parameter on capability / speed / VRAM, decision steps for choosing a configuration, and a symptom-to-fix table. You can hand a detail page to an AI assistant and have it deploy from there.

---

## Introduction

Based on llama.cpp (upstream v0.6.0, `d81235049` of 2026-10-05), re-implemented and optimized for V100 from the compute kernels up to the runtime. External implementations used are listed in the acknowledgements at the end; their origin and license are in each file header and in the `licenses/` directory.

Systematic optimizations built around the hardware characteristics of V100 (SM70), from compute kernels to runtime:

- **Multi-GPU parallelism**: optimized scheduling and kernel launches for tensor parallelism, reducing the time GPUs wait for the host; all-reduce across GPUs is hierarchical, following the NVLink topology; optimized splitting when the number of attention heads is not divisible by the number of GPUs (e.g. 6 GPUs).
- **Long context**: prefill and decode speed stay stable at very long context, and compute buffers use less VRAM.
- **Speculative decoding**: the complete DFlash2 speculative decoding path, with optimized sampling and per-round overhead for faster decode.
- **Compute kernels**: optimized SM70 kernels adapted to the Q8_0 quantization format, plus operator fusion.
- **Platform**: adapted to the NVLink interconnect and NUMA memory architecture of the IBM AC922.
- **Concurrency**: with several requests, decode has priority and the time spent on prompt processing is capped; decode batches, speculative decoding and VRAM reservations scale with the number of slots.
- **Multimodal**: the vision module (mmproj) runs on GPU and handles images, PDFs and video.
- **Large MoE models** (2.0.0): multi-GPU split combining expert parallelism and tensor parallelism; experts that do not fit in VRAM live in pinned host memory, read directly by the GPU and placed per NUMA node; a routing profile decides which experts stay resident in VRAM and which experts each GPU gets; MTP speculative decoding and sparse attention optimized for V100.

---

## Acknowledgements

**Thanks to community members [umacro](https://github.com/umacro), 看透<!-- TODO: GitHub 链接待补 -->, [天语星落](https://github.com/ativx928) and [lsaac](https://github.com/isaac-bj).** Without the test machines and tokens they provided, this project would not have come this far this fast.

Development: the repository author (started the project, provides the hardware and test environment, sets the goals and trade-offs); external contributor [天语星落 (ATIVX928)](https://github.com/ativx928) ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1), [PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2)). Referenced projects and the AI assistants used are listed in the [acknowledgements at the end](#acknowledgements-projects-and-ai-assistants).

---

## What's new in 2.0.0

2.0.0 adds V100-specific support for the large MoE model Qwen3.8-Flash-Next; Qwen3.8-27B keeps the same features and gets the fixes in the next section. Large MoE configurations supported in this version: **6 × V100-16GB + Qwen3.8-Flash-Next UD-Q4_K_XL + MTP draft (Q4_K_M)** is the standard configuration, and all measurements on this page and the detail page come from it; **4 × V100-32GB (x86, PCIe)** runs, with all experts resident in VRAM and none in host memory (reference numbers on the [MoE detail page](docs/v100/moe-2.0.0.en.md#measured-results)); other GPU counts, other quantizations (e.g. a Q8_0 main model) and more models will follow.

- **Large MoE model Qwen3.8-Flash-Next**: with multi-GPU `--split-mode tensor`, routed experts are distributed across GPUs by expert (expert parallelism), and the dense parts — attention, linear attention, shared experts and so on — are tensor-parallel. Recommended configuration: 6 × 16 GB V100, context 262144, with MTP speculative decoding.
- **Experts in two tiers, VRAM and host memory**: at startup, the experts resident in VRAM for each layer are chosen automatically from the VRAM left on each GPU; the rest live in pinned host memory that the GPU reads directly, and are still computed on the GPU. `--n-cpu-moe` / `-ot` are not needed. The pinned memory is allocated and filled per CPU node (NUMA) the GPU is attached to.
- **Expert routing profile**: `--expert-profile` uses routing statistics to keep the most-used experts in VRAM and to rebalance which experts each GPU gets so the GPUs do similar amounts of work; `scripts/v100-expert-profile.sh` builds a profile from a calibration text.
- **n-gram embedding table in host memory**: the model's n-gram embedding table and token embedding table live in pinned host memory, shared by all GPUs as one copy and looked up directly by the GPU; they take no VRAM.
- **Sparse attention (QSA)**: full-attention layers attend only to the positions selected by the indexer (the previous upstream implementation computed over the whole context on V100); compute buffers no longer grow linearly with context, so a 262144 context fits on 16 GB GPUs.
- **V100 optimizations for MTP speculative decoding**: the draft chain runs back to back on the GPU and returns its result once per round; the draft layer attends only to the most recent positions (new parameter `--spec-draft-window`); the draft context can have its own physical batch size (new parameter `--spec-draft-ubatch`).
- **Kernels and runtime**: expert matrix multiplications in prefill use fused Volta tensor-core kernels; several small operators are fused in decode; the first run of a multi-GPU execution plan is submitted in parallel by per-GPU launch threads.

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB, build `4d50148cc`; context 262144, with MTP and the expert routing profile; about 200K tokens of real code, fresh server, temperature 0, thinking off, output ended naturally at 686 tokens): prefill 2334.8 / 2340.8 tok/s, decode 107.9 / 109.7 tok/s. On the same machine, the large MoE prefills ~200K tokens about 13% faster than 27B and decodes at about 87% of 27B's speed.

---

## 27B fixes

- **VRAM re-planned for every batch when top-k output is on** (present in 1.1.1): when the target model produces top-k output (device-side top-k or rejection sampling), a prefill batch without output rows did not match the reserved compute graph, so every batch re-planned VRAM and synchronized all GPUs. The current 27B DFlash2 configuration does not trigger it; measured, there is no difference before and after the fix. The fix prepares 27B for MTP support later.
- **Layer split + DFlash2 failed at startup** (present in 1.1.1): `--split-mode layer` with a DFlash2 draft failed to start; it now starts, and the startup VRAM plan includes the DFlash2 draft's hidden-state buffers.
- **Other**: the ggml allocator could overwrite neighboring memory when an input that only appears as a leaf grew; the CPU IQ1_S dot product gave wrong results on POWER9 (ppc64le) (CPU compute only; GPU inference was not affected).

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB; 1.1.1 and 2.0.0 build `4d50148cc` alternated in the same time window, two rounds each; context 262144, DFlash2; conditions on the [27B detail page](docs/v100/27b.en.md#measured-results)): short-prompt decode 186.2 / 188.9 → 218.3 / 206.9 tok/s; ~16K-token prefill 2863.4 / 2842.7 → 3119.1 / 3121.1 tok/s; ~200K-token prefill 2047.8 / 2047.1 → 2066.6 / 2063.2 tok/s and decode 114.5 / 115.6 → 122.2 / 127.0 tok/s.

---

## Test platforms

**Standard test machine** (most numbers on both detail pages come from it): IBM Power AC922, 2 × POWER9, 96 GB memory (48 GB on each of the two CPU nodes), 6 × Tesla V100-SXM2-16GB, CUDA 12.4. Each CPU has 3 GPUs attached; GPUs under the same CPU are connected directly by NVLink; GPU-to-CPU links are NVLink 2.0.

**Reference hosts** (x86): 4 × Tesla V100-SXM2-32GB (all-to-all NVLink between GPUs, PCIe 3.0 between GPU and host, 136 GB memory, CUDA 12.8) for the 4-GPU reference numbers of the large MoE; 2 × Tesla V100-SXM2-16GB (NVLink, PCIe 3.0, CUDA 12.8) for build and correctness tests.

Basic differences between the AC922 and x86 servers, and how they indirectly affect the speedups (especially for the large MoE):

| | IBM AC922 | x86 server |
|---|---|---|
| GPU to host | NVLink 2.0 (2 links per GPU in the 6-GPU configuration, 50 GB/s per direction in theory) | PCIe 3.0 (about 16 GB/s per direction in theory for x16) |
| GPU memory and system memory | GPU memory is onlined into system memory as NUMA nodes, one node per GPU | GPU memory is not part of system memory |
| CPU / GPU affinity | 2 CPU nodes with 3 GPUs each; reading memory on the other node crosses the inter-CPU bus | depends on the board (the 4-GPU reference host above has a single NUMA node) |

- **Page cache can occupy VRAM**: because the AC922 treats GPU memory as memory nodes, page cache from reading large files (e.g. loading a model) can land in a GPU's node; CUDA cannot reclaim it, and it shows up as out-of-memory. So on the AC922, start with `numactl --membind=<CPU nodes>` (e.g. `0,8`); x86 does not need this.
- **Slightly less usable VRAM**: when GPU memory is onlined into system memory, the driver reserves part of it per memory block, so each GPU on the AC922 has a bit less usable VRAM than the same GPU on x86; for the large MoE, less VRAM means more experts in host memory.
- **Experts in host memory are read over NVLink**: 16 GB GPUs cannot hold all experts, so during decode the experts in host memory must be copied into VRAM by the GPU before computing. The AC922's GPU-host bandwidth is about 3× PCIe 3.0 (theoretical), so keeping experts in host memory costs much less there; reading across CPU nodes is slower.
- **Placement of pinned memory**: the experts in host memory and the n-gram embedding table are in pinned memory. At startup the program allocates and fills it per CPU node the GPUs are attached to, and measures how fast each GPU reads each node to decide the placement; machines with a single NUMA node have no such difference.

This is a general-purpose version: the same code runs on the AC922 and on x86; architecture-specific optimizations will come later.

---

## Community group

Scan the QR code below to add the maintainer on WeChat and join the discussion group.

<img src="media/v100-wechat.jpg" width="260" alt="WeChat QR code">

---

## Versions and license

### Versions

- Current version 2.0.0 (released 2026-10-12); see [CHANGELOG.md](CHANGELOG.md) (Chinese) for what changed in each version.
- Build instructions, the prebuilt x86_64 Linux binaries on the Releases page, and the Docker image `ghcr.io/1115714829/llama.cpp-v100:2.0.0` (one tag covering x86_64 and IBM Power AC922) are described on the two detail pages.
- The full documentation of 1.1.1 and earlier (27B test tables for 2–6 GPUs, Q4 configurations, concurrency, multimodal, etc.) is in the README of the corresponding tag, e.g. [v100-1.1.1](https://github.com/1115714829/llama.cpp-v100/blob/v100-1.1.1/README.md) (Chinese).

### License

GPL-3.0 from 2.0.0 on (full text in `LICENSE`); 1.1.1 and earlier were MIT. The MIT license of upstream llama.cpp is kept in `licenses/LICENSE-llama.cpp`; ported third-party code keeps its original license, with origin and license in each file header and in `licenses/README.md`.

---

## Acknowledgements (projects and AI assistants)

- **Referenced projects**: [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) (the base framework of this project); [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (reference for V100 engineering practice; some early SM70 kernels were ported from it — origin and license in the file headers and `licenses/`); [Strata](https://github.com/Niko1221/Strata) and its AC922 fork [Strata-AC922](https://github.com/eelgaev/Strata-AC922) (the fused expert matrix multiplication, QSA selection and attention kernels for the large MoE are ported from them, and the expert cache and prefill ring follow their design; origin and license in the file headers and `licenses/`); [vLLM](https://github.com/vllm-project/vllm) (reference for the design of speculative decoding and more).
- **AI assistants**: development was done with Claude, DeepSeek, Xiaomi MiMo and MiniMax.
