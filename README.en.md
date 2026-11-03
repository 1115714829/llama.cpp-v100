# llama.cpp-v100

[简体中文](README.md) | **English** | [AI deployment guide](docs/v100/ai-deploy.en.md) | [27B detail page](docs/v100/27b.en.md) | [Qwen3.8-Flash-Next detail page](docs/v100/moe-2.0.0.en.md) | [Benchmark history](docs/v100/benchmarks.en.md) | [Changelog](CHANGELOG.en.md)

A modified llama.cpp for NVIDIA V100 (SM70). Current version **2.0.1.2** (2026-11-03).

---

## Preface

This project started with one goal: let people run large models on low-cost V100s. Running 27B on six V100s with a 1M-token context has gone from impossible to possible.

**Special thanks to community members 正博, 蓝莲 and 小王的王Ww for financial sponsorship**, and to everyone in the community who lent test machines and helped with testing. The project would not be where it is without you (see [Acknowledgements](#acknowledgements)).

The project is maintained by one person. Two things are scarcest in keeping up the current pace:

- **Time**: the project keeps growing and has to stay compatible with two model lines, 27B and Qwen3.8-Flash-Next (MoE). Testing and development time has multiplied and now cuts deep into my working hours. One release takes three machines running for about 12 hours to finish every acceptance check and produce the release data.
- **Tokens**: development, debugging and testing are done mainly with AI assistance, which consumes a lot of tokens. The free tokens are used up, the Claude 20x plan expires at the end of November, and the test machines are all on loan and may be returned at any time.

So updates will definitely slow down; please plan accordingly. The project will not stop, and there will be no pause in the short term: many ideas are still waiting, including implementations written from the ground up and items logged for later optimization when time ran short.

**Updates will be slower, but release quality will not be compromised.** Every release must pass all of the following; if any one fails, it does not ship:

- **Same-session comparison with the previous release**: the old and new versions are measured on the same machine in the same time window, interleaved A-B-B-A, taking the median of repeated runs; when the spread is high, more samples are taken before judging. Prefill, decode, per-GPU peak VRAM and peak host memory (process RSS, including pinned memory) may only stay equal or improve.
- **Measured the way it is used**: published numbers are taken with thinking on, the model card's official sampling parameters and the vision module loaded; each tier starts a fresh server and the prompt prefix never hits the cache.
- **Unchanged numerics**: every touched operator passes the full `test-backend-ops` cases; outputs are compared byte for byte at temperature 0; perplexity (PPL) and KL divergence (KLD) are checked to make sure model outputs have not quietly changed.
- **Neither model line holds the other back**: any change to shared code is regression-tested on 27B tensor parallel for speed, VRAM and output.
- **Clean, traceable measurement**: the machine is reserved exclusively while measuring (no builds, no other jobs), with load and foreign GPU processes sampled throughout; each build under test is built per commit into its own directory and frozen read-only, binary hashes and version strings are verified before measuring, and the GPUs a process actually uses are verified by PCI bus ID.
- **Every configuration that started before still starts**: 1, 2, 3 and 6 GPUs, multi-request concurrency and multimodal configurations are each verified.

---

## Introduction

Based on llama.cpp (upstream v0.7.0, `620d2225c` of 2026-10-15), re-implemented and optimized for V100 from the compute kernels up to the runtime. External implementations used are listed in the acknowledgements at the end; their origin and license are in each file header and in the `licenses/` directory.

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

## Supported models and current status

### Qwen3.8-27B (dense) + DFlash2

- **GPU count**: runs on 1 to 6 V100s. 2 to 6 GPUs have full measurements (Q8_0 and Q4 configurations; use Q4 on 2 GPUs); 1 GPU needs a lower-bit quantized model. More than 6 GPUs also works in code, but without more hardware at hand it has not been specifically optimized or measured.
- **Context and concurrency**: up to 524288 (YaRN); multi-request concurrency (two requests on 2, 4 and 6 GPUs, four on 4 GPUs) and image, PDF and video input are supported.
- **Results**: currently at a leading level on V100 for this stage.

| GPUs | Configuration | Context | ~16K tokens of real code<br>prefill / decode (tok/s) |
|---|---|---:|---|
| 6 | Q8_0 | 262144 | 2786 / 152.7 |
| 6 | Q8_0 | 524288 | 2774 / 114.1 |
| 5 | Q8_0 | 524288 | 2612 / 126.1 |
| 4 | Q8_0 | 262144 | 2584 / 131.7 |
| 3 | Q8_0 | 131072 | 2470 / 90.0 |
| 2 | Q4 (Q4_K_M draft) | 262144 | 1841 / 78.6 |

<sub>The 6-GPU 262144 row was measured on 2.0.1.1 (temperature 0, thinking off; at about 200K tokens prefill is 2187 and decode 124.5 tok/s); the other rows were measured on 1.0.6–1.1.1, conditions in the [27B results archive (1.1.1)](docs/v100/27b-results-1.1.1.en.md).</sub>

The model the author runs in production (with its matching DFlash2 draft, downloadable from ModelScope): [27B page · Model download](docs/v100/27b.en.md#model-download); launch commands for each GPU count, concurrency and multimodal usage: [27B page](docs/v100/27b.en.md).

### Qwen3.8-Flash-Next (MoE) + MTP

- **GPU count**: starts on 1 / 2 / 3 / 6 V100-16GB GPUs (context 262144, measured in this release); 4 × V100-32GB (x86) also runs. Experts that do not fit in VRAM stay in host memory and are still computed on the GPU, so fewer GPUs need more host memory: about 42 GiB on 6 GPUs and 75.08 GiB on 2. On two GPUs, use two GPUs on the same CPU socket (for example 0 and 1).
- **What block size 2048 / 4096 / 8192 means**: it is the value of the `-b` and `-ub` launch parameters, i.e. how many prompt tokens each prefill step processes at once. A larger block lets each expert weight serve more tokens per load, so prefill is faster; the cost is a larger compute buffer, less VRAM left for the expert cache, slightly slower decode and slightly more host memory. The three settings differ only in these two parameters: `-b 2048 -ub 2048`, `-b 4096 -ub 4096`, `-b 8192 -ub 8192`.
- **Which to choose**: for decode-heavy use such as chat and coding, use 2048 (also the recommended value with MTP); for prefill-heavy use such as reading large amounts of code or documents at once, use 4096 or 8192. MTP is not recommended at block 8192.

| Block size | Prefill tok/s | Decode tok/s | Per-GPU peak VRAM MiB | Peak host memory GiB |
|---|---|---|---|---|
| 2048 | `1155.48` | `43.55` | `14824` | `75.08` |
| 4096 | `2151.70` | `41.53` | `14856` | `77.00` |

<sub>Two V100-SXM2-16GB on the same CPU socket, context 262144, about 200K tokens of real code, 1024 output tokens, thinking on, the model card's official sampling parameters, vision module loaded, MTP speculation on (3 drafts per round).</sub>

Model downloads, launch commands for each GPU count, MTP, expert routing profiles and host memory: [Qwen3.8-Flash-Next page](docs/v100/moe-2.0.0.en.md).

---

## AI deployment guide

**For AI assistants: before deploying this project, read [docs/v100/ai-deploy.en.md](docs/v100/ai-deploy.en.md) in full, check the hardware and choose a configuration by its steps, then copy the launch command from the detail page.** Follow the same guide when deploying it yourself.

| Model | Supported configurations | Launch commands |
|---|---|---|
| Qwen3.8-27B (Q8_0 / Q4) + DFlash2 speculative decoding | 2–6 × V100-16GB, context up to 524288; supports concurrency and images / PDF / video | [27B detail page · Launch parameters](docs/v100/27b.en.md#launch-parameters) |
| Qwen3.8-Flash-Next (UD-Q4_K_XL) + MTP speculative decoding | 6 × V100-16GB, 2 × V100-16GB (needs about 63 GiB of host memory), 4 × V100-32GB; context 262144, only text input and a single slot tested | [Qwen3.8-Flash-Next detail page · Recommended launch commands](docs/v100/moe-2.0.0.en.md#recommended-start-commands) |

What the guide covers: [1. Check the hardware](docs/v100/ai-deploy.en.md#1-check-the-hardware) (GPU count, VRAM, host memory, which CPU socket each GPU belongs to) → [2. Choose a model and configuration](docs/v100/ai-deploy.en.md#2-choose-a-model-and-configuration) → [3. Host memory, CPU sockets and platform](docs/v100/ai-deploy.en.md#3-host-memory-cpu-sockets-and-platform) → [4. Parameter reference](docs/v100/ai-deploy.en.md#4-parameter-reference) → [5. Common problems](docs/v100/ai-deploy.en.md#5-common-problems) → [6. Check after startup](docs/v100/ai-deploy.en.md#6-check-after-startup).

---

## What's new (2.0.1.2)

- **Under the same conditions, prefill is about 10 % faster than Strata**: same machine, same two GPUs (same socket), same context and requests, speculative decoding on and the vision module loaded on both sides, thinking on, the model card's official sampling parameters; on about 200K tokens of real code, prefill is 10.0 % faster at block 2048 and 13.1 % faster at 4096 (see the same-session table below).
- **Two GPUs on the same CPU socket (0, 1)**: Qwen3.8-Flash-Next runs on two V100-16GB cards of the same socket at blocks 2048 and 4096 with context 262144; on ~200K tokens of real code (prompt 209,075 tokens, output 1024 tokens) **prefill is 1155.48 / 2151.70 tok/s and decode is 22.941 / 24.075 ms/token**. Caliber: **two GPUs of one socket + MTP-3 speculation + vision module + thinking on + model-card sampling** (temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0). Block 8192 is not recommended with MTP and is not listed.
- **Host memory split across the two CPU nodes**: the expert complement is placed half per node; the peak host memory (process RSS, pinned pages included) for two cards of one socket at context 262144 is 75.08 / 77.00 GiB (blocks 2048 / 4096).
- **Faster decode**: expert swapping is now adaptive (the swap window carries a barrier).
- **Faster prefill**: the expert complement is streamed over the copy engine in a ring overlapped with compute, and the decode staging alignment is fixed.
- **Compute kernels**: optimized SM70 compute kernels with Q8_0 quantization support and operator fusion; numeric criteria are in the [changelog](CHANGELOG.md).
- **1, 2, 3 and 6 GPUs all work**: Qwen3.8-Flash-Next starts on 1 / 2 / 3 / 6 V100-16GB cards at context 262144. If VRAM is short, lower the context or use a smaller quantization. See [measured data](docs/v100/benchmarks.en.md#2012) for the 1-card (block 2048) and 3-card (all blocks) runs.
- **27B multi-request concurrency**: 27B supports two concurrent requests (`-np 2`) on 2, 4 and 6 V100-16GB GPUs (Q8 and Q4 configurations; use Q4 on 2 GPUs), four concurrent requests (`-np 4`, 65536 per request) on 4 GPUs, and on 6 GPUs two requests with 262144 context each (`-c 524288 -np 2`, YaRN) while also handling images and video. Measured data for each configuration: [27B results archive (1.1.1)](docs/v100/27b-results-1.1.1.en.md).
- **Using multimodal and thinking modes**: both work, but they need the right setup —
  - **The vision module is a separate official download**: `mmproj-F16.gguf` (27B: 927 607 488 B, in the `Qwen3.8-27B-GGUF` repository; Flash-Next: 904 004 000 B, in the `Qwen3.8-Flash-Next-GGUF` repository — the two files are not the same one, do not mix them up). [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) ｜ [ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) and [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) ｜ [ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF); start the server with `--mmproj <file>`. The vision module goes to the GPU by default and is split across the GPUs on multi-GPU setups. Without it an image request returns `image input is not supported - hint: if this is unexpected, you may need to provide the mmproj`.
  - **Thinking content is in `reasoning_content`**: add `"chat_template_kwargs": {"enable_thinking": true}` to the request body to enable thinking (`false` disables it). By default the thinking part is taken out of the answer text and put into `message.reasoning_content`, so clients that only read `message.content` will not see it (use `--reasoning-format deepseek-legacy` to keep it in the text and still populate the field); **`--reasoning-format none` leaves the thinking in the answer text, unparsed, with no separate section**. The server side also has `--reasoning on/off`, `--reasoning-effort`, `--reasoning-budget` and `--reasoning-format`; the web UI renders the thinking section separately.
  - **Add `--image-min-tokens 1024` for small images**: models of this family need at least that many image tokens per picture; without it a small picture produces only about 280 image tokens and recognition degrades (the server prints the same hint at start-up). This is a recommendation only — the default is unchanged.
  - **Do not use `-fit off`**: for the large MoE models of this family, in expert-parallel mode the automatic parameter plan is also the expert placement planner — turning it off makes the model fail to load; keep the default automatic planning.
  - **The 27B line (`-sm tensor`) and the vision module**: under tensor parallelism the vision module is not part of the automatic plan; when VRAM is tight use `--fit-target` to reserve headroom, or put the vision module on the CPU.
- **Speculative decoding recommended on two GPUs**: with two GPUs of the same socket (0, 1) we recommend **MTP speculation (3 drafts per round) at block size 2048** (`-b 2048 -ub 2048 --spec-draft-n-max 3`, draft `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`): for ~200K tokens of real code each output token takes 22.941 ms and the draft acceptance rate is 0.4536; **speculation is not recommended at block size 8192**; MTP must be enabled explicitly — the defaults are unchanged. Use MTP speculation with `-np 1` for now.
- **VRAM**: two-card peak per GPU 14824 / 14856 MiB (blocks 2048 / 4096); 27B (4 GPUs) peak per GPU 10933 MiB.
- **27B no-regression conclusion**: 4 GPUs (Q8_0 + DFlash2, block 2048, 16K real code) show no regression against 2.0.1.1 - prefill +0.70 %, decode on par, per-GPU peak VRAM 10933 MiB (identical to the reference), peak host memory 12.49 GiB (reference 12.48 GiB, difference within measurement spread).
- **Merged upstream llama.cpp v0.7.0** (`620d2225c` of 2026-10-15): the V100-specific work of this project was carried over onto the new upstream version.
- Docker image `ghcr.io/1115714829/llama.cpp-v100:2.0.1.2`.
- Command-line parameters and defaults are the same as in 2.0.1.1.

**Same-session comparison with Strata (two GPUs on the same socket, speculative decoding on and vision module loaded on both sides)**

| Block size | Prefill tok/s<br>Strata / this project | Decode tok/s<br>Strata / this project | Per-GPU peak VRAM MiB<br>Strata / this project | Peak host memory GiB<br>Strata / this project |
|---|---|---|---|---|
| 2048 | `1050.80 / 1155.48` | `45.65 / 43.55` | `15101 / 14824` | `91.11 / 75.08` |
| 4096 | `1902.00 / 2151.70` | `48.30 / 41.53` | `15089 / 14856` | `91.22 / 77.00` |

<sub>Same IBM AC922, two V100-SXM2-16GB on the same CPU socket, measured interleaved in the same time window; context 262144, about 200K tokens of real code (prompt 209,075 tokens; the other engine 209,102 tokens), 1024 output tokens, thinking on, the model card's official sampling parameters; this project runs MTP speculation (3 drafts per round), Strata runs its own speculative decoding, both load the vision module; each cell is the median of 8 measurements. Decode tok/s is converted from time per token; host memory is process RSS including pinned memory.</sub>

Block 8192 is not recommended with MTP and is not listed.

Changes in each version: [CHANGELOG.en.md](CHANGELOG.en.md).

---

## Performance at a glance

| Configuration (context 262144) | ~200K tokens of real code<br>prefill / decode (tok/s) | ~16K tokens of real code<br>prefill / decode (tok/s) | Methodology |
|---|---|---|---|
| 27B Q8_0 + DFlash2, 6 × V100-16GB | 2187 / 124.5 | 2786 / 152.7 | Temperature 0, thinking off |
| Qwen3.8-Flash-Next + MTP, 6 × V100-16GB | 3402 / 122.5 | 2805 / 140.5 | Temperature 0, thinking off |
| Qwen3.8-Flash-Next + MTP, 2 × V100-16GB | 1300 / 43.1 | 1173 / 38.0 | Thinking on + model-card sampling parameters |
| Qwen3.8-Flash-Next + MTP, 2 × V100-16GB (same socket, block 2048, recommended configuration for this release) | `1155.48` / `43.55` | `1084.58` / `42.70` | Thinking on + model-card sampling parameters + vision module + **MTP-3 (3 drafts per round)** |

<sub>Test data: measured 2026-11-03, 2.0.1.2 release build, IBM AC922 + Tesla V100-SXM2-16GB; Qwen3.8-Flash-Next is UD-Q4_K_XL + MTP draft with the expert routing profile, 27B is Q8_0 + DFlash2 F16; real code, a fresh server for each prompt length, prefix not cached. The two 6-GPU rows are temperature 0, thinking off (same-session alternating measurement against 2.0.1.1; unchanged cases keep the numbers measured last time); **the 2-GPU row added by this release is the recommended configuration (two GPUs of the same socket + MTP-3 speculation + block 2048 + vision module + thinking on + model-card sampling parameters)**, which differs from the two 6-GPU rows and from the existing 2-GPU row, so their speeds are not directly comparable. Full conditions, the same-session comparison with the previous release and the data of every version are in [Benchmark history](docs/v100/benchmarks.en.md#2012).</sub>

---

## Quick start

- **Prebuilt binaries**: x86_64 Linux builds on the [Releases](https://github.com/1115714829/llama.cpp-v100/releases) page.
- **Build from source**: see [27B detail page · Build](docs/v100/27b.en.md#build).
- **Launch commands**: [27B detail page · Launch parameters](docs/v100/27b.en.md#launch-parameters), [Qwen3.8-Flash-Next detail page · Recommended launch commands](docs/v100/moe-2.0.0.en.md#recommended-start-commands); if you are not sure which one to pick, see the [AI deployment guide](docs/v100/ai-deploy.en.md).

### Docker install

Image `ghcr.io/1115714829/llama.cpp-v100:2.0.1.2` (or `:latest`); one tag covers x86_64 and IBM Power AC922 (ppc64le), and `docker pull` picks the right one for the machine's architecture. The host needs NVIDIA driver 550 or newer.

1. Pull the image: `docker pull ghcr.io/1115714829/llama.cpp-v100:2.0.1.2`
2. Give containers GPU access (once): on x86, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and run with `--gpus all`; on the AC922, use podman (`sudo dnf install -y podman podman-docker`), run `sudo bash .devops/v100-ac922-cdi.sh` in the source tree to generate the GPU description, and run with `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8`.
3. Start (27B on 4 GPUs, x86, as an example; `<model dir>` must be an absolute path):

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <model dir>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.1.2 \
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

- Current version **2.0.1.2** (2026-11-03). Changes in each version: [CHANGELOG.en.md](CHANGELOG.en.md); measured data of each version: [Benchmark history](docs/v100/benchmarks.en.md).
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

**Special thanks to community members 正博, 蓝莲 and 小王的王Ww for financial sponsorship.**

**Thanks to community members [umacro](https://github.com/umacro), 看透<!-- TODO: GitHub 链接待补 -->, [天语星落](https://github.com/ativx928) and [lsaac](https://github.com/isaac-bj).** Without the test machines and tokens they provided, this project would not have come this far this fast.

Development: the repository author (started the project, provides the hardware and test environment, sets the goals and trade-offs); external contributors [天语星落 (ATIVX928)](https://github.com/ativx928) ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1) two-GPU Q4 support, [PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2) small-batch kernels for Q2_K–Q6_K), [umacro](https://github.com/umacro) ([PR #3](https://github.com/1115714829/llama.cpp-v100/pull/3) multimodal FFmpeg diagnostics), [wangxso](https://github.com/wangxso) ([PR #9](https://github.com/1115714829/llama.cpp-v100/pull/9) KV split of grouped verify attention sized by the device SM count), [Poqianhu7895](https://github.com/Poqianhu7895) ([PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4), and reported and located two concurrency issues, [#8](https://github.com/1115714829/llama.cpp-v100/pull/8) and [#13](https://github.com/1115714829/llama.cpp-v100/issues/13)). Referenced projects and the AI assistants used are listed in the [acknowledgements at the end](#acknowledgements-projects-and-ai-assistants).

### Acknowledgements (projects and AI assistants)

- **Referenced projects** (origin and license of ported code in the file headers and `licenses/`): [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) (the base framework of this project); [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (reference for V100 engineering practice; SM70 kernels such as small-batch quantized matrix multiplication, grouped verify attention, D256 prefill attention and push all-reduce are ported from it); [Strata](https://github.com/Niko1221/Strata) and its AC922 fork [Strata-AC922](https://github.com/eelgaev/Strata-AC922) (the fused expert matrix multiplication, QSA selection and attention, hyper-connection fusion and n-gram table reading of Qwen3.8-Flash-Next are ported from them; the expert cache, prefill ring and MTP draft window follow their design); [SGLang-V100](https://github.com/haohervchb/sglang-V100) (origin of the push all-reduce and block-scale adaptation); [v100-skinny](https://github.com/dnv2003/v100-skinny) (QPN8 weight layout); [flash-attention-v100 (ai-bond)](https://github.com/ai-bond/flash-attention-v100) and [flash-attention-v100 (zhinianqin)](https://github.com/zhinianqin/flash-attention-v100) (Volta attention kernels); [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) (support layer of the attention kernels); [TileLang](https://github.com/tile-ai/tilelang) and FlashQLA (GDN chunked prefill kernels); [ninfer-v100](https://github.com/dollarwong/ninfer-v100) (design reference for the Q4 small-batch kernels); [unsloth's llama.cpp fork](https://github.com/unslothai/llama.cpp) (MTP draft borrowing the target model's tensors); [vLLM](https://github.com/vllm-project/vllm) (reference for the design of speculative decoding and more); [z-lab](https://huggingface.co/z-lab) (the DFlash2 speculative decoding method and draft models).
- **AI assistants**: development was done with Claude, DeepSeek, Xiaomi MiMo and MiniMax.
