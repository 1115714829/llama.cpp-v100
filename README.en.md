# llama.cpp-v100

[简体中文](README.md) | **English**

A modified llama.cpp for NVIDIA V100 (SM70). Current version **2.0.2**.

---

> [!IMPORTANT]
> # Every published benchmark is measured with strictly zero precision loss
>
> **Every performance number this project publishes is obtained under a strict zero-precision-loss policy.**
>
> - No speed is bought with lower quantization, lossy low-precision communication or approximate computation;
> - every operator that is touched passes the full operator test suite;
> - outputs are compared byte for byte at temperature 0, and perplexity (PPL) plus KL divergence (KLD) checks confirm the model output has not quietly changed;
> - speculative decoding is verified through both draft acceptance rate and output consistency.
>
> The speed is real, and the output is still the same model.

> [!NOTE]
> # Two model lines: dense 27B on DFlash2, large MoE on MTP — zero regression vs previous release (2.0.1.2)
>
> - **Qwen3.8-27B (dense) + DFlash2**: keeps the existing DFlash2 path; compared against the previous official release **2.0.1.2**, both prefill and decode stay within the measurement noise band, and temperature-0 outputs match the existing anchor byte for byte — **zero regression vs 2.0.1.2**.
> - **Qwen3.8-Flash-Next (large MoE) + MTP**: built around MTP speculative decoding and **requires no environment variables** — the code defaults are already the full-performance state, delivering across-the-board speedups over **2.0.1.2** with zero precision loss.
> - Both lines share the same runtime, so any change on one line must be regression-tested on the other for speed, VRAM and output.

---

## Preface

This project started with one goal: let people run large models on low-cost V100s. Running 27B on six V100s with a 1M-token context has gone from impossible to possible.

**Special thanks to community members 正博, 蓝莲 and 小王的王Ww for financial sponsorship**, and to everyone in the community who lent test machines and helped with testing. The project would not be where it is without you (see [Acknowledgements](#acknowledgements)).

The project is maintained by one person. Two things are scarcest in keeping up the current pace:

- **Time**: the project keeps growing and has to stay compatible with two model lines, 27B and Qwen3.8-Flash-Next (MoE). Testing and development time has multiplied and now cuts deep into my working hours. One release takes multiple machines running for hours to finish every acceptance check and produce the release data.
- **Tokens**: development, debugging and testing are done mainly with AI assistance, which consumes a lot of tokens. The free tokens are used up, ~~the Claude 20x plan expires at the end of November~~ (**Claude account banned on 2026-11-02 — best wishes to Anthropic's early bankruptcy**). **Now the main driver is Google Gemini ("American Doubao") — huge thanks to Google, Gemini rocks 🐂🍺**

So updates will definitely slow down; please plan accordingly. The project will not stop, and there will be no pause in the short term: many ideas are still waiting, including implementations written from the ground up and items logged for later optimization when time ran short.

**Updates will be slower, but release quality will not be compromised.** Every official release must pass strict acceptance checks (exclusive machine runs, zero environment variables, full operator tests, temperature-0 byte-for-byte & PPL/KLD zero-precision-loss checks, and dual-line regression); if any check fails, it does not ship.

---

## ✨ Key Highlights in 2.0.2

> [!TIP]
> ### Core Highlights
> **⚡ 10K-Class 200K Long Prefill** (`9,372–9,401` on 6 GPUs · **`11,240 tok/s`** on 8 GPUs) ｜ **🎯 100% Strictly Zero Precision Loss** (Full Op Suite + Byte-Exact T=0 + PPL/KLD) ｜ **🧠 Runs 111 GB MoE Down to 32 GB Host RAM** (1–8 GPUs, No Quant Downgrade, Host-Safe) ｜ **🚀 Zero Env Vars Out-of-the-Box** (Auto Hardware Adaptation + Offline Telemetry WebUI)

> This project focuses exclusively on the inference runtime and **never creates or modifies quantized model weights**: public GGUF weights are used verbatim while the SM70 kernels, multi-GPU coordination, 3-tier placement, and speculative decoding pipeline are deeply customized for NVIDIA V100.

| Key Feature | Measured Spec / Coverage | Details |
|---|---|---|
| **⚡ 10K-Class 200K Prefill** | **9,372–11,240 tok/s** pure prefill; **7,412–8,471 tok/s** with MTP-3 | Ingests 200K-token codebases in seconds while delivering **119–167 tok/s** code-editing decode with `MTP-3`, turning low-cost V100s into flagship long-context rigs |
| **🧠 32 GB Minimum Host RAM for 100+ GB MoE** | Runs full **111 GB MoE** on **≥ 32 GB RAM** (even on **1 × 16 GB V100**, 1–8 GPUs) | **Unified 3-tier adaptive placement (VRAM → Host RAM → Disk)** with **no quant downgrade and no context shrinkage**: full RAM residency on large hosts, graceful disk streaming on 32 GB hosts, plus strict OS headroom guards (never counts swap, never hangs or OOM-kills the host) |
| **🎯 Strictly Zero Precision Loss & Zero Regression** | Full operator tests + T=0 byte-exact checks + PPL/KLD | Rejects lossy low-precision inter-GPU reductions or fast-math shortcuts; both dense 27B (DFlash2) and large MoE (MTP) achieve strictly zero regression vs `2.0.1.2` |
| **🚀 Out-of-the-Box Full Speed** | **0 tuning env vars** required for published numbers | Auto-detects GPU count, VRAM, NVLink/PCIe topology, and host RAM at startup for optimal 3-tier placement and layer split; explicit CLI flags (including `-b`/`-ub`) always win |
| **🌌 Extreme Context & Decode-First Concurrency** | **256K** native on MoE; **512K–1M (1,048,576)** via YaRN on dense 27B | Holds full repositories and long-horizon agent histories; `-np N` concurrency includes **decode-first prefill pacing (`--prefill-pace`)** so incoming long prompts never stall active streams |
| **🔥 Dual Specialized Speculative Engines** | **MTP** (+ optional bounded lookup) for MoE; **DFlash2** for dense 27B | Purpose-built per architecture and tuned for production sampling penalties and Chinese/English code & prose — real **2–3×** lossless decode speedups |
| **🖥️ Cross-Arch · Any GPU Count · 16G & 32G** | **x86_64 + IBM POWER9 AC922 (ppc64le)**; **1/2/3/4/5/6/8 GPUs**; **16 GB / 32 GB** | One codebase across architectures, VRAM sizes, and non-divisor GPU counts (3/5/6 GPUs), shipping with multi-arch Docker / Podman images and x86 prebuilt binaries |
| **🧩 Modern Low-Bit Ecosystem on Volta** | **FP8 / NVFP4 / MXFP4 / Q8_0 / Q4_K** & upstream architectures | Synced with modern upstream llama.cpp formats, executing low-bit weights via native SM70 hardware capabilities (never slow CPU emulation) |
| **🖼️ GPU-Accelerated Multimodal** | Images / small-image boost / PDFs (rendered) / videos (`ffmpeg`) | Vision encoder (`mmproj`) runs directly on GPU over the standard OpenAI-compatible `/v1/chat/completions` API |
| **🌐 Built-in Chinese/English Telemetry WebUI + Clean Logs + AI Guardrails** | **70 prebuilt assets** embedded directly from source (**zero network, zero Node.js**) | Replaces the stock UI with a **Light/Dark bilingual interface** featuring real-time same-origin telemetry (**live prefill/decode tok/s & 60s charts, per-message absorption/generation stats, 6-level reasoning control, and per-GPU VRAM/util/temp/power W**); clean release logs by default; copy-paste AI deployment guardrails |

---

## Performance matrix

> [!TIP]
> ### 🚀 200K Long-Context Peak Pure Prefill (No Speculation · Zero Env Vars · Strictly Zero Precision Loss)
>
> | **IBM AC922 · 6 × 16 GB** *(16 GB Authoritative)* | **x86 · 6 × 32 GB** | **x86 · 8 × 32 GB** *(Flagship > 11K)* | **x86 · 4 × 32 GB** |
> |:---:|:---:|:---:|:---:|
> | **`9,372 tok/s`** | **`9,401 tok/s`** | **`11,240 tok/s`** | **`6,664 tok/s`** |
>
> *(6-GPU 200K pure prefill approaches 10,000 tok/s and 8-GPU breaks past 11,240 tok/s; Table 1 below shows full end-to-end prefill & decode across platforms with speculative decoding enabled.)*

### Table 1｜Qwen3.8-Flash-Next (MoE) + MTP-3 speculative decoding: full platform matrix (200K context)

> Common methodology: 262144 context, 4096 block, single slot, zero environment variables (code defaults), median of repeated runs; **prefill and decode are always reported together**.

| Platform Tier | GPUs × VRAM | Test platform (objective hardware) | 200K Prefill tok/s | 200K Decode tok/s | Code Edit / Code Gen / Prose Decode (tok/s) |
|---|---|---|---:|---:|---|
| **16 GB Authoritative** | **2 × 16 GB** | IBM POWER9 AC922, dual socket, 96 GB; same-socket NVLink2, GPU–CPU NVLink2; CUDA 12.4 (4096 block) | **2528.49** | **36.41** | 49.49 / 51.75 / 39.24 *(2048 block)* |
| **16 GB Authoritative** | **4 × 16 GB** | IBM POWER9 AC922, dual socket, 96 GB; dual-socket NVLink2; CUDA 12.4 | **4755.04** | **69.54** | 92.74 / 90.03 / 65.84 |
| **16 GB Authoritative** | **6 × 16 GB** | **IBM POWER9 AC922, dual socket, 96 GB; dual-socket 6-GPU NVLink2; CUDA 12.4** | **7411.83** | **60.57** | **118.78 / 111.12 / 64.47** |
| x86 16 GB Reference | **2 × 16 GB** | x86_64, 96 cores, 126 GB; 2 × V100-SXM2-16GB 300W intra-island NVLink2; PCIe 3.0; CUDA 12.8 | **681.76** | **22.02** | 20.40 / 20.31 / 20.24 *(2048 block)* |
| x86 16 GB Reference | **4 × 16 GB** | x86_64, 96 cores, 126 GB; 4 × V100-SXM2-16GB 300W intra-island NVLink2; PCIe 3.0; CUDA 12.8 | **1947.81** | **55.19** | 71.52 / 69.62 / 55.11 |
| x86 16 GB Reference | **6 × 16 GB** | x86_64, 96 cores, 126 GB; 6 × V100-SXM2-16GB 300W; PCIe 3.0; CUDA 12.8 | **4158.75** | **65.54** | 104.43 (bounded lookup **122.36**) / 98.22 / 57.21 |
| x86 32 GB Reference | **2 × 32 GB** | x86_64, 104 cores, 252 GB; 2 × V100-SXM2-32GB intra-island NVLink2; PCIe 3.0; CUDA 12.8 | **2318.28** | **65.20** | 77.20 / 84.49 / 62.56 *(2048 block)* |
| x86 32 GB Reference | **4 × 32 GB** | x86_64, 44 cores, 136 GB; 4 × V100-SXM2-32GB all-to-all NVLink2; PCIe 3.0; CUDA 12.8 | **5771.30** | **66.57** | 126.55 (bounded lookup **139.16**) / **116.82** / **66.18** |
| x86 32 GB Reference | **6 × 32 GB** | x86_64, 104 cores, 252 GB; 8 × V100-SXM2-32GB in two 4-GPU NVLink2 islands (6 GPUs measured); PCIe 3.0; CUDA 12.8 | **7597.04** | **89.53** | 130.40 (bounded lookup **166.69**) / **120.22** / **67.27** |
| x86 32 GB Reference | **8 × 32 GB** | x86_64, 104 cores, 252 GB; 8 × V100-SXM2-32GB in two 4-GPU NVLink2 islands (all 8 GPUs); PCIe 3.0; CUDA 12.8 | **8470.63** | **87.43** | 125.49 (bounded lookup **159.12**) / 115.95 / 65.81 |

**How to read the two decode columns and prefill numbers:**

- **Why is 200K decode 60–90 tok/s while Code Edit / Code Gen reaches 110–167 tok/s?**
  1. **Draft acceptance rate by task predictability**: Code editing (refactoring existing code) and code generation follow tight structural patterns, pushing MTP-3 acceptance to **87%–99.8%** (4–7 tokens per step with bounded lookup) for **110–167 tok/s**. Open-ended prose (under production sampling `temperature=0.6, top_p=0.95, presence_penalty=1.0`) and 200K long-context continuation have higher entropy (acceptance **42%–67%**, 2–3 tokens/step), yielding **60–90 tok/s**.
  2. **Context-length overhead**: Every decode step at 200K performs sparse-attention selection and long-context cache access across 200,000 tokens, whereas short-to-medium prompts carry far lighter attention overhead.
- **Why does 200K prefill change from 9,372 / 9,401 / 11,240 (banner) to 7,412 / 7,597 / 8,471 (Table 1)?**
  - The top banner shows pure 48-layer target-model prefill with speculation off; with `MTP-3` on, the draft layer must also prefill the 200K context state, trading a few extra seconds of prefill for a **2–3×** decode speedup.
- **Authoritative 16 GB standard machine & cross-platform differences**: IBM POWER9 AC922 (6 × V100-SXM2-16GB, 96 GB) is this project's authoritative 16 GB benchmark machine; x86 16 GB and 32 GB rows illustrate real-world scaling across host–GPU interconnect bandwidth (AC922 GPU–CPU NVLink2 vs x86 PCIe 3.0) and VRAM capacity.
- **Cold-prompt & zero-precision-loss cross-check**: All 200K prefill numbers were measured cold with prefix cache disabled (`cache_prompt=false`, multi-run medians, zero prefix-cache hits). An independent reproduction window on AC922 6 × 16 GB measured 7384.79 / 61.43 tok/s (within ≤ 1.42%, byte-identical outputs); x86 16 GB 2/4/6-GPU greedy 200K outputs are **byte-identical across GPU counts**; block-2048 200K measurements are 1375.39 / 24.70 (AC922 2×16 GB), 432.14 / 14.05 (x86 2×16 GB), and 1273.16 / 61.99 tok/s (x86 2×32 GB).
- **Chinese multi-scenario reference (x86 6 × 32 GB, MTP-3 default / bounded lookup)**: Chinese Q&A **89.40 / 90.24 tok/s**, Chinese Prose **63.88 / 63.38 tok/s**, Chinese Code **96.17 / 91.79 tok/s**; 8 × 32 GB leads 200K prefill (8470.63 / bounded lookup 8478.28), while 6 × 32 GB leads single-stream decode (89.53 vs 87.43).

### Table 2｜Qwen3.8-27B (dense) + DFlash2 speculative decoding (zero-regression gate)

| GPUs × VRAM | Test platform (objective hardware) | Methodology | Prefill tok/s | Decode | Zero-regression result |
|---|---|---|---:|---:|---|
| **2 × 16 GB** | IBM POWER9 AC922, dual socket, 96 GB (same-socket NVLink2) | context 32768, prompt 2048 → 128 tokens, temperature 0, thinking off | **2116.48** | **129.9 tok/s** (7.70 ms/tok) | **4/4 cold starts byte-identical to anchor, 101/210 drafts accepted, prefill +0.19%, decode −0.84% (within noise band — zero regression vs 2.0.1.2)** |

> The dense 27B line preserves the full tensor-parallel + DFlash2 path and multi-GPU configurations (2–6 GPUs, YaRN 524288, multi-slot concurrency, and vision multimodal); see the archive and deployment commands below.

---

## This version vs previous official release 2.0.1.2

> Compared strictly against this project's own previous official release; rows with differing methodology are shown side by side without artificial percentages.

| Configuration (IBM POWER9 AC922, authoritative 16 GB machine) | Previous release 2.0.1.2 | This version (zero env vars, MTP-3) | Notes |
|---|---|---|---|
| **2 × 16 GB MoE, 200K, 4096 block** | 2151.70 / 41.53 tok/s (thinking + model-card sampling + vision) | **2528.49 / 36.41 tok/s** (T=0, thinking off, zero env vars) | 4096-block 200K prefill climbs past 2528 tok/s (shown side by side) |
| **4 × 16 GB MoE, 200K, 4096 block** | *Not listed in 2.0.1.2* | **4755.04 / 69.54 tok/s** | New complete 4-GPU zero-env measurement |
| **6 × 16 GB MoE, 200K, 4096 block** | 3402 / 122.5 tok/s (legacy architecture caliber) | **7411.83 / 60.57 tok/s** (edit **118.78** / code **111.12**) | 200K prefill jumps from 3402 to **7411.83 tok/s (2.18×)** |
| **27B Q8_0 + DFlash2 (zero-regression gate)** | 2.0.1.2 anchor | **2116.48 tok/s / 129.9 tok/s (4/4 byte-identical anchor)** | **Strictly zero regression & zero precision loss** |

---

## Deep-Customization Notice & Copy-Paste AI Deployment Guardrails

> [!WARNING]
> **This project is a deeply customized inference engine built specifically for NVIDIA V100 (SM70), not a generic upstream build with tweaked flags.**
>
> - **Out-of-the-box defaults are already full speed**: the recommended launch commands are tuned on exclusive hardware; **launch directly with the default recommended parameters** to reproduce the published performance — do not pile on random internet environment variables or drop required flags.
> - **Install every missing dependency — never cripple the build**: required dependencies such as **NCCL** (both `lib/libnccl.so.2` and `include/nccl.h` for multi-GPU communication), **CUDA 12.x**, **GCC 12+**, **numactl** (mandatory on AC922), **nvidia-persistenced** (mandatory on AC922), and **ffmpeg** (for video input) **must be installed and properly configured** to achieve the published results. Disabling NCCL, dropping `CMAKE_CUDA_ARCHITECTURES=70`, or stripping speculative/profile flags to “get it compiling” will severely degrade speed or break execution.

### Copy-Paste Prompt for Your AI Assistant (Strict Anti-Improvisation Rules)

If you use an AI assistant (Claude Code, Codex, Cursor, OpenCode, DeepSeek, etc.) to deploy this project, **copy and paste the entire block below verbatim** so it follows the exact specifications instead of improvising:

```text
Please deploy llama.cpp-v100 on this server strictly following the rules below. This is a deeply customized NVIDIA V100 (SM70) engine — follow the README recommendations verbatim and do NOT improvise, weaken build flags, or apply workarounds:

[RULE 1: Inspect hardware first — never guess]
1. Run read-only checks first: nvidia-smi, nvidia-smi topo -m, numactl -H, free -h, df -h, nvcc --version, gcc --version, cmake --version, and locate libnccl.so.2 + nccl.h.
2. Report the detected hardware (arch x86_64/ppc64le, V100 count and 16GB/32GB VRAM, NVLink topology, host RAM and memory-bearing NUMA nodes, free disk space, and missing dependencies) before installing or building.

[RULE 2: Install missing dependencies — never cripple the build]
1. Build strictly with all required CMake flags from the README:
   -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=<real path> -DNCCL_LIBRARY=<real path>
2. If NCCL, GCC 12+, CUDA 12.x, or numactl is missing, install and verify it cleanly. NEVER disable GGML_CUDA_NCCL, GGML_CUDA_FA, or GGML_CUDA_GRAPHS, and NEVER remove CMAKE_CUDA_ARCHITECTURES=70 or replace it with native.
3. The offline WebUI is already bundled in tools/ui/dist/ and embedded automatically during offline compilation. Do NOT run npm install / npm build, do NOT download frontend assets from the internet, and do NOT modify any C++/CUDA source code.

[RULE 3: Use the exact recommended defaults — no unauthorized tweaks]
1. Copy the exact launch command from the README matching my GPU count, VRAM, and model, changing only the actual model/profile paths.
2. All core optimizations are enabled by default. Do NOT invent or export extra GGML_* / LLAMA_* environment variables.
3. Do NOT change the parallelism mode (Qwen3.8-Flash-Next MoE MUST use --split-mode layer + --fit on + --expert-profile + MTP flags; dense Qwen3.8-27B MUST use --split-mode tensor + DFlash2 flags). Do NOT shrink -c, arbitrarily change -b/-ub, remove speculation/profile flags, or swap in a lower-bit quantized model; download strictly from the exact verified repositories and filenames listed in the README (prefer ModelScope in mainland China) and never mix mismatched drafts or mmproj files.
4. On IBM AC922 (ppc64le): verify nvidia-persistenced is running (Persistence-M=On) and bind host memory with numactl --membind=<CPU nodes with real RAM, e.g. 0,8> — NEVER bind GPU HBM NUMA nodes (250-255) or empty RAM nodes. On standard x86 servers, omit the AC922 membind flags.

[RULE 4: Stop on real errors — never fake success]
1. On any compile error, missing dependency, checksum mismatch, or OOM, show me the exact error log and proper root-cause fix. NEVER silently disable features, lower precision, shrink context, or patch source code to fake a passing deployment.
2. After startup, verify /health returns 200, run a real request and report prefill/decode tok/s, check per-GPU VRAM via nvidia-smi, and confirm the built-in WebUI loads at http://<IP>:<port>. Keep default concise logging (add -v only while troubleshooting).
```

---

## Out of the box

**Prebuilt binaries**: [Releases](https://github.com/1115714829/llama.cpp-v100/releases) (x86_64 Linux).
**Build from source**:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/path/to/nccl/include -DNCCL_LIBRARY=/path/to/nccl/lib/libnccl.so.2
cmake --build build -j --target llama-server llama-quantize
```

### Built-in offline WebUI

The web interface ships with the source tree (`tools/ui/dist/`) and is embedded into `llama-server` at build time: clone the repository and build fully offline — no network download of interface assets and no Node.js (only needed if you change the interface sources). Once the server is up, open `http://<host>:8080` for the web chat; the OpenAI-compatible API is at `/v1/chat/completions`. Release builds keep the log concise; pass `-v` (or `-lv 5`) for detailed logs.

### Qwen3.8-Flash-Next (MoE) + MTP

The commands below **require no environment variables** — the defaults are already the full-performance state. Keep the `numactl --membind=<nodes>` part on the AC922 (it pins host memory to CPU-backed NUMA nodes); drop it on ordinary x86 servers.

**6 × 16 GB (AC922)**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 999 \
  --split-mode layer -c 262144 --fit on --fit-ctx 262144 -np 1 \
  -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 --jinja --reasoning off \
  --expert-profile /models/flash-next-profile.txt \
  --model-draft /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-ubatch 256 --spec-draft-window 32768 --ctx-checkpoints 0 \
  --host 127.0.0.1 --port 8080 --metrics
```

**4 × 32 GB (x86)**: drop the `numactl` part, set `CUDA_VISIBLE_DEVICES=0,1,2,3`, everything else unchanged.

**8 × 32 GB (x86)**: same, with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

**2 × 16 GB**: set `CUDA_VISIBLE_DEVICES=0,1` (preferably two GPUs of the same CPU socket), everything else unchanged; more host memory is required.

- For decode-heavy chat, change `-b 4096 -ub 4096` to `-b 2048 -ub 2048`.
- `--ctx-checkpoints 0` enables target/draft async prefill overlap on long prompts (omit it to keep the default `32` if multi-turn prefix checkpoint rollback is preferred); dropping `--model-draft` and `--spec-…` disables speculative decoding; omitting `--expert-profile` also works (usually slower).
- Per-GPU VRAM being nearly full is normal (the extra VRAM is used for speed); lower `-c` first if VRAM is short.

### Qwen3.8-27B (dense) + DFlash2

**4 × 16 GB**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

Change `CUDA_VISIBLE_DEVICES` and `--tensor-split` (one `1` per GPU) for 2 / 6 GPUs; on 2 GPUs use the Q4 configuration (main model `Qwen3.8-27B-UD-Q4_K_M.gguf`, `-ctk q4_0 -ctv q4_0`, draft `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`).

### Recommended Models & Download Links

> Users in mainland China should prefer **ModelScope** direct links (supports `wget -c` resume; on IBM AC922 wrap downloads with `numactl --membind=0,8` so page cache does not spill into GPU HBM nodes). Always use matching model, draft, and mmproj files from the verified lists below.

#### 1. Qwen3.8-Flash-Next (Large MoE) + MTP

All weight files live in the same unsloth repository ([ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) ｜ [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)):

| Role | Exact File Path | Size | Notes |
|---|---|---|---|
| Target model (4 shards) | `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf` | ~111 GB total | Place all 4 shards in one directory; pass shard 1 to `-m` |
| MTP draft (16 GB recommended) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | ~1.9 GB | **Must use a `shared` draft** paired with the target model |
| MTP draft (32 GB / alternative) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | ~2.8 GB | Recommended by unsloth; similar acceptance to Q4_K_M |
| Expert routing profile | `flash-next-profile.txt` | ~370 KB | Shipped in [Releases](https://github.com/1115714829/llama.cpp-v100/releases) to keep hot experts resident in VRAM |

```bash
# Download from ModelScope (swap base to https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main if preferred)
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O "models/$f" "$base/$f"
done
```

#### 2. Qwen3.8-27B (Dense) + DFlash2

- **★ Author's Production Bundle (Recommended — Ready-to-Use Q8_0 Trio, No Conversion Needed)**
  Repository: [ModelScope (recommended)](https://modelscope.cn/models/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF) ｜ [Hugging Face](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF) (all three files live under `Q8_0/`):

| Role | Exact File Name (under `Q8_0/`) | Size | Notes |
|---|---|---|---|
| Target model | `Qwen3.8-27B-EfficientThink-SimPO-Q8_0.gguf` | ~28.6 GB | Author's production 27B (SFT + SimPO) |
| DFlash2 draft | `dflash2-qwen38-27b-Q8_0.gguf` | ~2.1 GB | Matching Q8_0 draft — **ready to use without conversion** |
| Vision module (optional) | `mmproj-Qwen3.8-27B-Q8_0.gguf` | ~0.6 GB | For image / PDF / video input |

```bash
base=https://modelscope.cn/models/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/resolve/master/Q8_0
mkdir -p models/27b
for f in Qwen3.8-27B-EfficientThink-SimPO-Q8_0.gguf dflash2-qwen38-27b-Q8_0.gguf mmproj-Qwen3.8-27B-Q8_0.gguf; do
  wget -c -O "models/27b/$f" "$base/$f"
done
```

- **Standard Base 27B (Alternative / 2-GPU Q4 Config)**
  - Target & mmproj: [ModelScope `unsloth/Qwen3.8-27B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) ｜ [HF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — `Qwen3.8-27B-Q8_0.gguf` (~29 GB for 4–6 GPUs), `Qwen3.8-27B-UD-Q4_K_M.gguf` (~16.5 GB for 2 × 16 GB), `mmproj-F16.gguf` (~0.9 GB);
  - DFlash2 draft: prefer `dflash2-qwen38-27b-Q8_0.gguf` from the ModelScope bundle above (or check [ModelScope `z-lab/Qwen3.8-27B-DFlash2-GGUF`](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) for Q8_0/Q4_K_M; note some HF draft mirrors may be unavailable, and if you obtain BF16 convert it to F16 with `./build/bin/llama-quantize <in_bf16> <out_f16> F16` since V100 has no native BF16 tensor cores).

### More usage

- **Extended 27B context (512K / 1M YaRN)**:
  - **524288 (512K)**: `-c 524288 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 --override-kv qwen35.context_length=int:524288`;
  - **1048576 (1M on 6 × 16 GB)**: `-c 1048576 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 --override-kv qwen35.context_length=int:1048576` (the YaRN settings apply to all lengths).
- **Multi-slot concurrency**: `-np N`, with each slot getting `-c` ÷ N; keep `-np 1` when a single agent needs the full long context.
- **Multimodal**: add `--mmproj /models/mmproj-F16.gguf` (the 27B and Flash-Next mmproj files are not the same one — do not mix them), and add `--video-ffmpeg-dir <dir>` for video; requests go through the OpenAI-compatible `/v1/chat/completions`.
- **Thinking mode**: add `"chat_template_kwargs": {"enable_thinking": true}` to the request body to enable it and `false` to disable it; thinking content lives in `message.reasoning_content` by default, so clients that only read `message.content` will not see it.
- **Small images**: add `--image-min-tokens 1024`; otherwise a small image produces too few image tokens and recognition degrades.

### Docker

Image `ghcr.io/1115714829/llama.cpp-v100:2.0.2` (or `:latest`); one tag covers both x86_64 and IBM Power AC922 (ppc64le), and `docker pull` picks the right one for the host architecture. The host needs NVIDIA driver 550 or newer.

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <model dir>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.2 \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --port 8080
```

On the AC922 use podman (`sudo dnf install -y podman podman-docker`), run `sudo bash .devops/v100-ac922-cdi.sh` in the source tree to generate the GPU description, and add `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8` at runtime. Once the server is up, open `http://<host>:8080` (OpenAI-compatible API at `/v1/chat/completions`).

---

## Automatic Hardware Adaptation, 32 GB Low-RAM Support, and Three-Tier Placement

- **Zero-configuration placement defaults**: at start-up the server detects GPU count, VRAM, interconnect topology, and available host memory, then plans VRAM/RAM/Disk three-tier placement and layer splitting automatically (batch sizes `-b`/`-ub` default to `2048`/`512` in code; pass them explicitly per workload below); explicit CLI parameters always override defaults.
- **Unified three-tier placement (runs 111 GB MoE down to 32 GB host RAM)**: the model and its experts settle along "**VRAM → Host RAM → Disk**", always computed on GPU with **no quant downgrade and no context shrinkage**. Servers with ample RAM run fully in VRAM + Host RAM; when host RAM cannot hold all non-VRAM experts, pass `--expert-profile <profile>` with `--expert-disk-gib <per-GPU GiB>` (and `--expert-disk-dir <fast data disk>`) to enable the disk tier with full OS headroom and safe-flush guards.
- **Host safety guards**: built-in memory budgeting reserves OS headroom (never counting swap as usable memory) and fails loudly with clear guidance rather than squeezing or OOM-hanging the host.
- **Choosing the block size**: `-b` / `-ub` sets how many prompt tokens each prefill step processes. Larger blocks give faster prefill at slightly lower decode speed and higher host memory; use 2048 for chat and interactive coding, or 4096 for ingesting large codebases or long documents.

---

## Supported Quantizations & Advanced Parameters

### 1. Quantization Support (Native SM70 V100 Acceleration)

1. **KV cache quantization (`-ctk` / `-ctv`, and draft `-ctkd` / `-ctvd`)**
   - Accepted values: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`.
   - **Dedicated V100 fast kernels**:
     - **`q8_0` (recommended default)**: native SM70 tensor-core kernels for both dense 27B (D256 attention + grouped verify) and large MoE (QSA sparse attention), halving KV VRAM vs `f16` with negligible quality loss.
     - **`q4_0` (maximum VRAM savings)**: dedicated vectorized dequant attention kernels on SM70, ideal for 27B 256K on 2 × 16 GB or 1M context on 6 × 16 GB.
     - **`f16`**: natively accelerated everywhere. V100 has no native BF16 tensor cores, so `bf16` KV cache is not recommended.
2. **Model weight quantizations (native SM70 execution)**
   - **Primary & small-batch accelerated formats (Skinny / MMQ / Fused Expert)**: `Q8_0` (dedicated V100 `q8` skinny kernels, primary for 4–6 GPU 27B), `Q4_K` / `UD-Q4_K_XL` / `UD-Q4_K_M` (primary for MoE and 2-GPU 27B), `Q2_K` / `Q3_K` / `Q5_K` / `Q6_K` (full V100 skinny codec family), plus `Q4_0` / `Q4_1` / `Q5_0` / `Q5_1`.
   - **Modern low-bit formats**: `FP8` (E4M3), `NVFP4`, `MXFP4`, and `IQ1_S`–`IQ4_NL`, executed via native Volta tensor cores / `dp4a`.
   - **Draft model note**: convert BF16 draft checkpoints to `F16`, or use `Q8_0` / `Q4_K_M` directly on V100.

### 2. Advanced CLI Reference (Zero Env Vars Required; Explicit Flags Always Win)

| Category | Flags | Purpose & Recommendations |
|---|---|---|
| **KV Quant & Batching** | `-ctk <type> -ctv <type>`<br>`-ctkd <type> -ctvd <type>`<br>`-b <N> -ub <N>` | KV cache types for target and draft models (`q8_0` recommended, `q4_0` for tight VRAM); `-b`/`-ub` set prefill chunk size (`2048` interactive, `4096` long-context throughput). |
| **Multi-GPU & NUMA** | `--split-mode` (`-sm layer\|tensor\|row\|none`)<br>`--tensor-split` (`-ts N0,N1,...`)<br>`--numa <distribute\|isolate\|numactl>` | Use `layer` for MoE and `tensor` for dense 27B; `-ts` adjusts per-GPU proportions on asymmetric setups; `--numa` tunes dual-socket hosts. |
| **MoE Three-Tier & Disk Tier** | `--expert-profile <file>`<br>`--dump-expert-routing <file>`<br>`--expert-host-gib <GiB>`<br>`--expert-disk-gib <GiB/GPU>`<br>`--expert-disk-dir <dir>`<br>`--load-mode` (`-lm auto\|mmap\|dio\|mlock\|none`) | Places hot experts in VRAM, warm experts in host RAM, and cold experts on a fast data disk (`--expert-disk-gib` is **per GPU**); `--dump-expert-routing` captures custom routing profiles; `-cmoe` / `-ncmoe N` control CPU expert layers explicitly. |
| **Auto-Fit & VRAM Margin** | `--fit [on\|off]`<br>`--fit-target <MiB0,...>`<br>`--fit-ctx <N>` | Enabled (`on`) by default for automatic three-tier placement and context fitting while respecting explicit flags (**keep `--fit on` for MoE; do not pass `--fit off`**); `--fit-target` adjusts per-GPU free-VRAM headroom. |
| **Speculative Decoding & Lookup** | `--model-draft <GGUF>`<br>`--spec-type <draft-mtp\|draft-dflash\|ngram-*>`<br>`--spec-draft-n-max <N>` / `--draft-max <N>`<br>`--spec-draft-p-min <P>`<br>`--spec-draft-device` (`-devd`)<br>`--spec-draft-ngl` (`-ngld`)<br>`-lcs <file>` / `-lcd <file>` | `draft-mtp` for MoE (`--spec-draft-n-max 3`), `draft-dflash` for 27B (`7`); supports bounded suffix lookup and static/dynamic lookup caches. |
| **Multi-Turn Cache & Concurrency** | `--cache-ram <MiB>` (`-cram`)<br>`--ctx-checkpoints <N>` (`-ctxcp`)<br>`--checkpoint-min-step <N>` (`-cms`)<br>`-np <N>` (`--parallel`)<br>`--prefill-pace <0..100>` | Automatic prefix-cache reuse across turns (`-cram` caps MiB, `-1` unlimited, `0` off; `-ctxcp` keeps checkpoints for mid-conversation branching); under `-np N` concurrency, `--prefill-pace` caps incoming prefill wall-time share to keep active decoding smooth. |
| **Multimodal & Thinking** | `--mmproj <GGUF>`<br>`--mmproj-device` (`-mmdev <dev\|none>`)<br>`--image-min-tokens <N>`<br>`--video-ffmpeg-dir <dir>`<br>`--reasoning [on\|off\|auto]`<br>`--reasoning-effort <level>`<br>`--reasoning-budget <N>`<br>`--reasoning-preserve` | `-mmdev` pins the vision projector to a specific GPU (or `none` for CPU); use `--image-min-tokens 1024` for small images; fine-grained CLI and per-request controls for thinking mode, effort, and token budget. |

### 3. Operational Notes & Known Limitations

- **Use `-np 1` with MTP speculation**: please keep `-np 1` when running MoE with `MTP` enabled (without MTP, `-np > 1` works normally).
- **Do not combine `-devd` with `--split-mode tensor` + DFlash2**: the DFlash2 draft shares `token_embd` with the tensor-split target model and cannot currently be pinned to a single card via `-devd` (`-mmdev` alone is unaffected).
- **Older NCCL fallback**: no environment variables are needed on recommended stacks; only if a host ships an older NCCL library that fails multi-GPU collectives, set `GGML_CUDA_ALLREDUCE=internal` to use the built-in collective path.

---

## Test platforms and the boundary of the conclusions

- **Authoritative 16 GB standard machine**: IBM POWER9 AC922, dual socket, 96 GB memory, 6 × Tesla V100-SXM2-16GB; intra-island NVLink2 and GPU–CPU NVLink2; CUDA 12.4. All 16 GB authoritative numbers of this project come from this platform.
- **x86 16 GB reference environment**: 126 GB memory, 7 × V100-16GB 300W; PCIe 3.0; CUDA 12.8.
- **x86 4 × 32 GB reference environment**: V100-SXM2-32GB all-to-all NVLink2, PVE hardware passthrough, 136 GB memory; PCIe 3.0; CUDA 12.8.
- **x86 8 × 32 GB reference environment**: two 4-GPU NVLink2 islands, 252 GB memory; PCIe 3.0; CUDA 12.8.

The same code runs on POWER9 and on x86. Unless a row is explicitly marked "authoritative", numbers from the other platforms are multi-platform reference measurements that show cross-platform behaviour.

---

## Community

| WeChat | QQ group |
|---|---|
| <img src="media/v100-wechat.jpg" width="220" alt="WeChat QR code"> | <img src="media/v100-qq-group.jpg" width="220" alt="QQ group QR code"> |
| Scan to add the author on WeChat; the author will invite you to the WeChat group (WeChat group codes expire, so add the author first) | Scan to join the QQ group directly |

---

## License

This project is licensed under GPL-3.0 from 2.0.0 on (full text in `LICENSE`); 1.1.1 and earlier were MIT. The MIT license text of upstream llama.cpp is kept in `licenses/LICENSE-llama.cpp`; ported third-party code keeps its original license, with origin and license in each file header and in `licenses/README.md`.

Historical release notes and the full benchmark archive are preserved under [docs/v100/archive/](docs/v100/archive/README.md).

---

## Acknowledgements

**Special thanks to community members 正博, 蓝莲 and 小王的王Ww for their financial sponsorship.**

**Thanks to community members [umacro](https://github.com/umacro), 看透, [天语星落](https://github.com/ativx928), [Isaac](https://github.com/isaac-bj), 愚者生 (provided GLM-5.2), 并夕夕 (provided hardware), 协调员 (provided hardware), 东海龙, 孙阳, Kingsman and TangMonk5859.** Without the test machines, compute and tokens they provided, this project would not have come this far this fast.

**Development**: the repository author (started the project, provides the hardware and test environment, sets the goals and trade-offs); external contributors [天语星落 (ATIVX928)](https://github.com/ativx928) ([PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1) two-GPU Q4 support, [PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2) small-batch kernels for Q2_K–Q6_K), [umacro](https://github.com/umacro) ([PR #3](https://github.com/1115714829/llama.cpp-v100/pull/3) multimodal FFmpeg diagnostics), [wangxso](https://github.com/wangxso) ([PR #9](https://github.com/1115714829/llama.cpp-v100/pull/9) KV split of grouped verify attention sized by the device SM count), [Poqianhu7895](https://github.com/Poqianhu7895) ([PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4), and reported and located two concurrency issues, [#8](https://github.com/1115714829/llama.cpp-v100/pull/8) and [#13](https://github.com/1115714829/llama.cpp-v100/issues/13)).

### Acknowledgements (projects and AI assistants)

- **Referenced projects** (origin and license of ported code in the file headers and `licenses/`): [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) (the base framework of this project); [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (reference for V100 engineering practice); [Strata](https://github.com/Niko1221/Strata) and its AC922 fork [Strata-AC922](https://github.com/eelgaev/Strata-AC922) (the expert cache, prefill ring and MTP draft window follow their design); [SGLang-V100](https://github.com/haohervchb/sglang-V100) (origin of the push all-reduce and block-scale adaptation); [v100-skinny](https://github.com/dnv2003/v100-skinny) (QPN8 weight layout); [flash-attention-v100 (ai-bond)](https://github.com/ai-bond/flash-attention-v100) and [flash-attention-v100 (zhinianqin)](https://github.com/zhinianqin/flash-attention-v100) (Volta attention kernels); [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) (support layer of the attention kernels); [TileLang](https://github.com/tile-ai/tilelang) and FlashQLA (GDN chunked prefill kernels); [ninfer-v100](https://github.com/dollarwong/ninfer-v100) (design reference for the Q4 small-batch kernels); [unsloth's llama.cpp fork](https://github.com/unslothai/llama.cpp) (MTP draft borrowing the target model's tensors); [vLLM](https://github.com/vllm-project/vllm) (reference for the design of speculative decoding and more); [z-lab](https://huggingface.co/z-lab) (the DFlash2 speculative decoding method and draft models).
- **AI assistants**: development was done with Gemini 4 Neon, Claude, DeepSeek, GLM, Meituan LongCat, Xiaomi MiMo and MiniMax.
