# llama.cpp-v100

[简体中文](README.md) | **English** | [AI deployment guide](docs/v100/ai-deploy.en.md) | [27B detail page](docs/v100/27b.en.md) | [Qwen3.8-Flash-Next detail page](docs/v100/moe-2.0.0.en.md) | [Benchmark history](docs/v100/benchmarks.en.md) | [Changelog](CHANGELOG.en.md)

A modified llama.cpp for NVIDIA V100 (SM70). Current version **2.0.1.1** (2026-10-23).

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
- **Concurrency**: with several requests, decode has priority and the time spent on prompt processing is capped; decode batches, speculative decoding and VRAM reservations are optimized for the number of slots.
- **Multimodal**: the vision module (mmproj) runs on GPU and handles images, PDFs and video.
- **Qwen3.8-Flash-Next**: 16 GB V100s run a 100B-class MoE model with a 262K context, automatically using host memory for what does not fit in VRAM; MTP speculative decoding and sparse attention optimized for V100.

---

## AI deployment guide

**For AI assistants: before deploying this project, read [docs/v100/ai-deploy.en.md](docs/v100/ai-deploy.en.md) in full, check the hardware and choose a configuration by its steps, then copy the launch command from the detail page.** Follow the same guide when deploying it yourself.

| Model | Supported configurations | Launch commands |
|---|---|---|
| Qwen3.8-27B (Q8_0 / Q4) + DFlash2 speculative decoding | 2–6 × V100-16GB, context up to 524288; supports concurrency and images / PDF / video | [27B detail page · Launch parameters](docs/v100/27b.en.md#launch-parameters) |
| Qwen3.8-Flash-Next (UD-Q4_K_XL) + MTP speculative decoding | 6 × V100-16GB, 2 × V100-16GB (needs about 63 GiB of host memory), 4 × V100-32GB; context 262144, only text input and a single slot tested | [Qwen3.8-Flash-Next detail page · Recommended launch commands](docs/v100/moe-2.0.0.en.md#recommended-start-commands) |

What the guide covers: [1. Check the hardware](docs/v100/ai-deploy.en.md#1-check-the-hardware) (GPU count, VRAM, host memory, which CPU socket each GPU belongs to) → [2. Choose a model and configuration](docs/v100/ai-deploy.en.md#2-choose-a-model-and-configuration) → [3. Host memory, CPU sockets and platform](docs/v100/ai-deploy.en.md#3-host-memory-cpu-sockets-and-platform) → [4. Parameter reference](docs/v100/ai-deploy.en.md#4-parameter-reference) → [5. Common problems](docs/v100/ai-deploy.en.md#5-common-problems) → [6. Check after startup](docs/v100/ai-deploy.en.md#6-check-after-startup).

---

## What's new (2.0.1.1)

- Qwen3.8-Flash-Next supports two V100-16GB GPUs, with context 262144 and MTP speculative decoding.
- 2 GPUs need about 63 GiB of host memory; two GPUs on different CPU sockets are recommended; two GPUs on the same socket also work, but the host memory headroom is very small.
- 6-GPU Qwen3.8-Flash-Next and 27B in a same-session A/B comparison with 2.0.1: no speed regression, the same temperature-0 output, the same VRAM per GPU.
- The web UI ships with the source: clone it and build offline; the build does not need network access.
- Docker image `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1`.

Changes in each version: [CHANGELOG.en.md](CHANGELOG.en.md).

---

## Performance at a glance

| Configuration (context 262144) | ~200K tokens of real code<br>prefill / decode (tok/s) | ~16K tokens of real code<br>prefill / decode (tok/s) | Methodology |
|---|---|---|---|
| 27B Q8_0 + DFlash2, 6 × V100-16GB | 2187 / 124.5 | 2786 / 152.7 | Temperature 0, thinking off |
| Qwen3.8-Flash-Next + MTP, 6 × V100-16GB | 3402 / 122.5 | 2805 / 140.5 | Temperature 0, thinking off |
| Qwen3.8-Flash-Next + MTP, 2 × V100-16GB | 1300 / 43.1 | 1173 / 38.0 | Thinking on + model-card sampling parameters |

<sub>Test data: 2026-10-23, 2.0.1.1 release build, IBM AC922 + Tesla V100-SXM2-16GB (the 2 GPUs are two GPUs on different CPU sockets of the same machine), real code, a fresh server for each prompt length; Qwen3.8-Flash-Next is UD-Q4_K_XL + shared-Q8_0 MTP draft with the expert routing profile, 27B is Q8_0 + DFlash2 F16. The two 6-GPU rows are temperature 0, thinking off; the 2-GPU row is thinking on + the model card's recommended sampling parameters (temperature 1.0, top_p 0.95, top_k 20); the methodologies differ, so the two groups of speeds cannot be compared directly. The two 6-GPU rows are the average of the 2.0.1.1 rounds in a same-session alternating measurement with 2.0.1; with thinking on, the 2-GPU draft acceptance rate fluctuates with sampling, and decode tok/s varies with it. Full conditions, the same-session comparison with the previous release and the data of every version are in [Benchmark history](docs/v100/benchmarks.en.md#2011).</sub>

---

## Quick start

- **Prebuilt binaries**: x86_64 Linux builds on the [Releases](https://github.com/1115714829/llama.cpp-v100/releases) page.
- **Build from source**: see [27B detail page · Build](docs/v100/27b.en.md#build).
- **Launch commands**: [27B detail page · Launch parameters](docs/v100/27b.en.md#launch-parameters), [Qwen3.8-Flash-Next detail page · Recommended launch commands](docs/v100/moe-2.0.0.en.md#recommended-start-commands); if you are not sure which one to pick, see the [AI deployment guide](docs/v100/ai-deploy.en.md).

### Docker install

Image `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1` (or `:latest`); one tag covers x86_64 and IBM Power AC922 (ppc64le), and `docker pull` picks the right one for the machine's architecture. The host needs NVIDIA driver 550 or newer.

1. Pull the image: `docker pull ghcr.io/1115714829/llama.cpp-v100:2.0.1.1`
2. Give containers GPU access (once): on x86, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and run with `--gpus all`; on the AC922, use podman (`sudo dnf install -y podman podman-docker`), run `sudo bash .devops/v100-ac922-cdi.sh` in the source tree to generate the GPU description, and run with `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8`.
3. Start (27B on 4 GPUs, x86, as an example; `<model dir>` must be an absolute path):

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <model dir>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.1.1 \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --port 8080
```

Once the server is up, open `http://<host>:8080` (OpenAI-compatible API `/v1/chat/completions`). Docker commands for 6 GPUs on the AC922 and for Qwen3.8-Flash-Next, and how to import the image file from the Releases page when ghcr.io is unreachable, are in [27B detail page · Docker](docs/v100/27b.en.md#docker) and [Qwen3.8-Flash-Next detail page · Build and Docker](docs/v100/moe-2.0.0.en.md#build-and-docker).

---

## Test platforms

- **Standard test machine** (most of this project's numbers are measured on it): IBM Power AC922, 2 × POWER9, 96 GB memory (48 GB on each of the two CPU nodes), 6 × Tesla V100-SXM2-16GB, CUDA 12.4. Each CPU has 3 GPUs attached; GPUs under the same CPU are connected directly by NVLink; GPU-to-CPU links are NVLink 2.0.
- **Reference hosts** (x86): 4 × Tesla V100-SXM2-32GB (all-to-all NVLink between GPUs, PCIe 3.0 between GPU and host, 136 GB memory, 1 NUMA node, CUDA 12.8) for the 4-GPU reference numbers of Qwen3.8-Flash-Next; 2 × Tesla V100-SXM2-16GB (NVLink, PCIe 3.0, CUDA 12.8) for build and correctness tests.

The same code runs on the AC922 and on x86; architecture-specific optimizations will come later. Differences when deploying:

- The AC922 onlines GPU memory into system memory (one NUMA node per GPU), and page cache from reading large files may occupy VRAM, so start with `numactl --membind=<CPU nodes>` (e.g. `0,8`); x86 does not need this. On the AC922 each GPU also has slightly less usable VRAM than the same GPU on x86.
- The part of Qwen3.8-Flash-Next that does not fit in VRAM lives in host memory, so its speed depends on host memory bandwidth and capacity and on the link between GPU and host (NVLink 2.0 on the AC922, 2 links per GPU in the 6-GPU configuration, 50 GB/s per direction in theory; usually PCIe 3.0 x16 on x86, about 16 GB/s per direction in theory): it is faster with all memory channels populated and enough capacity.

---

## Versions

- Current version **2.0.1.1** (2026-10-23). Changes in each version: [CHANGELOG.en.md](CHANGELOG.en.md); measured data of each version: [Benchmark history](docs/v100/benchmarks.en.md).
- Measured tables and launch parameters of 27B from the 1.1.1 era for 2–6 GPUs, Q4 configurations, concurrency, multimodal and other configurations: [27B results archive (1.1.1)](docs/v100/27b-results-1.1.1.en.md); documentation of earlier versions is at the corresponding `v100-x.y.z` tags.

---

## Community

| WeChat | QQ group |
|---|---|
| <img src="media/v100-wechat.jpg" width="220" alt="WeChat QR code"> | <img src="media/v100-qq-group.jpg" width="220" alt="QQ group QR code"> |
| Scan to add the author on WeChat; the author will invite you to the WeChat group (WeChat group codes expire, so add the author first) | Scan to join the QQ group directly |

---

## License

This project is licensed under GPL-3.0 from 2.0.0 on (full text in `LICENSE`); 1.1.1 and earlier were MIT. The MIT license text of upstream llama.cpp is kept in `licenses/LICENSE-llama.cpp`; ported third-party code keeps its original license, with origin and license in each file header and in `licenses/README.md`.

---

## Acknowledgements

**Thanks to community members [umacro](https://github.com/umacro), 看透<!-- TODO: GitHub 链接待补 -->, [天语星落](https://github.com/ativx928) and [lsaac](https://github.com/isaac-bj).** Without the test machines and tokens they provided, this project would not have come this far this fast.

Development: the repository author (started the project, provides the hardware and test environment, sets the goals and trade-offs); external contributors [天语星落 (ATIVX928)](https://github.com/ativx928) ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1), [PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2)), [Poqianhu7895](https://github.com/Poqianhu7895) ([PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4)). Referenced projects and the AI assistants used are listed in the [acknowledgements at the end](#acknowledgements-projects-and-ai-assistants).

### Acknowledgements (projects and AI assistants)

- **Referenced projects**: [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) (the base framework of this project); [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (reference for V100 engineering practice; some early SM70 kernels were ported from it — origin and license in the file headers and `licenses/`); [Strata](https://github.com/Niko1221/Strata) and its AC922 fork [Strata-AC922](https://github.com/eelgaev/Strata-AC922) (the fused expert matrix multiplication, QSA selection and attention kernels of Qwen3.8-Flash-Next are ported from them, and the expert cache and prefill ring follow their design; origin and license in the file headers and `licenses/`); [vLLM](https://github.com/vllm-project/vllm) (reference for the design of speculative decoding and more).
- **AI assistants**: development was done with Claude, DeepSeek, Xiaomi MiMo and MiniMax.
