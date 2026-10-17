# llama.cpp-v100

[简体中文](README.md) | **English** | [27B dense model detail page](docs/v100/27b.en.md) | [Large MoE detail page](docs/v100/moe-2.0.0.en.md) | [Changelog (Chinese)](CHANGELOG.md)

A modified llama.cpp for NVIDIA V100 (SM70). Current version **2.0.1** (2026-10-18).

---

## Performance at a glance

**Measurement conditions: temperature 0, thinking off**, not the sampling parameters recommended by the model card; numbers with thinking on and model-card sampling will be measured in a later version and labeled separately.

| Model (6 × V100-16GB, context 262144) | Scenario | Prefill | Decode | Measured on |
|---|---|---:|---:|---|
| Large MoE: Qwen3.8-Flash-Next UD-Q4_K_XL + MTP | ~200K tokens of real code | **3464 tok/s** | 118.5 tok/s | 2.0.1 |
| | ~16K tokens of real code | 2823 tok/s | **139.6 tok/s** | 2.0.1 |
| | Short prompt | — | **151.0 tok/s** | 2.0.1 |
| 27B: Qwen3.8-27B Q8_0 + DFlash2 | ~200K tokens of real code | 2140 tok/s | 128.9 tok/s | 2.0.1 |
| | ~16K tokens of real code | **3178 tok/s** | — | 2.0.1 |
| | Short prompt | — | **225.8 tok/s** | 2.0.1 |

<sub>All numbers are reproducible on the current version: 2026-10-17, 2.0.1 release build, IBM AC922 + 6 × V100-SXM2-16GB (see "Test platforms"), measured alternately with the previous version in the same window. Temperature 0, thinking off, 1024 / 768 / 256 output tokens; the 27B ~16K request ran at temperature 0.8 (prefill does not depend on temperature); the 27B ~200K request hit the cached prefix of the previous request in the same server and actually processed 195048 tokens. When a model does not change in a new version, its rows keep the last measured numbers and version. Full conditions and all measurements per version: [Measured results by version](docs/v100/benchmarks.en.md).</sub>

---

## Introduction

Based on llama.cpp (upstream v0.6.0, `d81235049` of 2026-10-05), re-implemented and optimized for V100 from the compute kernels up to the runtime. External implementations used are listed in the acknowledgements at the end; their origin and license are in each file header and in the `licenses/` directory.

**This project only optimizes the underlying framework (the inference runtime); it does not make or modify quantized models**: it uses the GGUF weights published by the model authors, with quantization formats and weights left untouched; what changes are the compute kernels, multi-GPU parallelism, scheduling and the speculative-decoding execution path, accepted only when outputs are bit-identical or match token-for-token at temperature 0. For the model's intelligence and capabilities, refer to the original model card and the quantization level you choose; this project only changes how fast it runs and how much VRAM it takes.

Systematic optimizations built around the hardware characteristics of V100 (SM70), from compute kernels to runtime:

- **Multi-GPU parallelism**: 2–6 GPUs work together efficiently, adapted to the NVLink topology, including GPU counts that do not divide the number of attention heads (e.g. 6 GPUs).
- **Long context**: prefill and decode speed stay stable at very long context, and compute buffers use less VRAM.
- **Speculative decoding**: the complete DFlash2 speculative decoding path, with optimized sampling and per-round overhead for faster decode.
- **Compute kernels**: optimized SM70 kernels adapted to the Q8_0 quantization format, plus operator fusion.
- **Platform**: adapted to the NVLink interconnect and NUMA memory architecture of the IBM AC922.
- **Concurrency**: with several requests, decode has priority and the time spent on prompt processing is capped; decode batches, speculative decoding and VRAM reservations scale with the number of slots.
- **Multimodal**: the vision module (mmproj) runs on GPU and handles images, PDFs and video.
- **Large MoE models**: 16 GB V100s run a 100B-class MoE model with a 262K context, automatically using host memory for what does not fit in VRAM; MTP speculative decoding and sparse attention optimized for V100.

---

## Supported models and configurations

| Model | Recommended setup | Detail page |
|---|---|---|
| Qwen3.8-27B (Q8_0) + DFlash2 speculative decoding | 2–6 × V100-16GB, context 262144; Q4 quantization, concurrency, multimodal and a 524288 context also work | [27B dense model](docs/v100/27b.en.md) |
| Qwen3.8-Flash-Next (UD-Q4_K_XL, large MoE) + MTP speculative decoding | 6 × V100-16GB or 4 × V100-32GB, context 262144 | [Large MoE](docs/v100/moe-2.0.0.en.md) |

Launch commands, parameters and known limitations are on the detail pages; all measured tables per version (GPU counts, Q4, concurrency, multimodal) are in [Measured results by version](docs/v100/benchmarks.en.md).

## Quick start

- **Prebuilt binaries**: x86_64 Linux builds on the [Releases](https://github.com/1115714829/llama.cpp-v100/releases) page.
- **Docker image**: `ghcr.io/1115714829/llama.cpp-v100:2.0.1`, one tag for x86_64 and IBM Power (ppc64le); usage in the [Docker section of the 27B page](docs/v100/27b.en.md#docker).
- **Build from source**: see the [build section of the 27B page](docs/v100/27b.en.md#build).
- **Launch commands**: both detail pages give copy-ready full commands per hardware setup; their last section, "Quick deployment (for AI assistants and beginners)", lists each parameter's effect on capability / speed / VRAM, decision steps and a symptom-to-fix table, so you can hand a detail page to an AI assistant to deploy from.

## Current version 2.0.1

Faster prefill and decode for the large MoE model on top of 2.0.0; fixes for 27B T=0 output differing across restarts at long context on 6 GPUs and for the server aborting with speculative decoding and multiple slots (`-np > 1`); the per-GPU VRAM peak is no higher than 2.0.0. Same-window comparison with 2.0.0 at ~200K tokens: large MoE prefill +48%, decode +8%; 27B prefill +3%, decode +2%. New parameter `--fit-mmproj` (default keeps the previous behavior).

Changes and the measurements of every version: [CHANGELOG.md](CHANGELOG.md) (2.0.1 back to 1.0.0) and [Measured results by version](docs/v100/benchmarks.en.md).

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

### Host memory and performance

When part of the experts of a large MoE model lives in host memory (because they do not fit in VRAM), prefill and decode speed depends on host memory bandwidth and capacity: performance is better when all memory channels are populated and the memory is large enough to hold the experts that are not in VRAM. How to configure host memory is up to you and your machine.

---

## Acknowledgements

**Thanks to community members [umacro](https://github.com/umacro), 看透<!-- TODO: GitHub 链接待补 -->, [天语星落](https://github.com/ativx928) and [lsaac](https://github.com/isaac-bj).** Without the test machines and tokens they provided, this project would not have come this far this fast.

Development: the repository author (started the project, provides the hardware and test environment, sets the goals and trade-offs); external contributors [天语星落 (ATIVX928)](https://github.com/ativx928) ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1), [PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2)), [Poqianhu7895](https://github.com/Poqianhu7895) ([PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4)). Referenced projects and the AI assistants used are listed in the [acknowledgements at the end](#acknowledgements-projects-and-ai-assistants).

- **Referenced projects**: [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) (the base framework of this project); [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (reference for V100 engineering practice; some early SM70 kernels were ported from it — origin and license in the file headers and `licenses/`); [Strata](https://github.com/Niko1221/Strata) and its AC922 fork [Strata-AC922](https://github.com/eelgaev/Strata-AC922) (the fused expert matrix multiplication, QSA selection and attention kernels for the large MoE are ported from them, and the expert cache and prefill ring follow their design; origin and license in the file headers and `licenses/`); [vLLM](https://github.com/vllm-project/vllm) (reference for the design of speculative decoding and more).
- **AI assistants**: development was done with Claude, DeepSeek, Xiaomi MiMo and MiniMax.

---

## Community

| WeChat | QQ group |
|---|---|
| <img src="media/v100-wechat.jpg" width="220" alt="WeChat QR code"> | <img src="media/v100-qq-group.jpg" width="220" alt="QQ group QR code"> |
| Scan to add the author on WeChat; the author will invite you to the WeChat group (WeChat group codes expire, so add the author first) | Scan to join the QQ group directly |

---

## License

GPL-3.0 from 2.0.0 on (full text in `LICENSE`); 1.1.1 and earlier were MIT. The MIT license of upstream llama.cpp is kept in `licenses/LICENSE-llama.cpp`; ported third-party code keeps its original license, with origin and license in each file header and in `licenses/README.md`.
