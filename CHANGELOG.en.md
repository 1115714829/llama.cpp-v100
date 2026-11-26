# Changelog

[简体中文](CHANGELOG.md) | **English** | [← README](README.en.md) | [Archive](docs/v100/archive/README.md)

## 2.0.2 (2026-11-26)

Milestone release of the V100 (SM70) pure layer-split and dual-model runtime: **every published benchmark is strictly zero precision loss and achieved out of the box with zero tuning environment variables**, with strictly zero regression against `2.0.1.2` on both dense `27B (DFlash2)` and large MoE `Qwen3.8-Flash-Next (MTP)`.

### Headline Performance (200K Context, Zero Env Vars, Strictly Zero Precision Loss)

- **10K-Class 200K Pure Prefill**: **`9,372 tok/s`** on IBM POWER9 AC922 `6 × 16 GB`, **`9,401 tok/s`** on x86 `6 × 32 GB`, **`11,240 tok/s`** on flagship x86 `8 × 32 GB`, and **`6,664 tok/s`** on x86 `4 × 32 GB`.
- **Full 2 / 4 / 6 / 8 GPU × 16 GB & 32 GB `MTP-3` Matrix**:
  - **Authoritative 16 GB (IBM POWER9 AC922)**: `6 × 16 GB` 200K prefill jumps from `3,402` (`2.0.1.2`) to **`7,411.83 tok/s` (2.18×)**, with `60.57 tok/s` 200K decode and **`118.78 / 111.12 / 64.47 tok/s`** on code editing / code generation / prose; `4 × 16 GB` hits **`4,755.04 / 69.54 tok/s`**; `2 × 16 GB` hits **`2,528.49 / 36.41 tok/s`**.
  - **x86 16 GB (300W)**: `2 / 4 / 6 × 16 GB` reach `681.76 / 22.02`, `1,947.81 / 55.19`, and `4,158.75 / 65.54 tok/s` (code editing `104.43`, bounded lookup `122.36 tok/s`).
  - **x86 32 GB**: `2 / 4 / 6 / 8 × 32 GB` reach `2,318.28 / 65.20`, `5,771.30 / 66.57`, `7,597.04 / 89.53`, and `8,470.63 / 87.43 tok/s`; code editing with bounded lookup reaches **`139.16–166.69 tok/s`**, Chinese QA `89.40–90.24 tok/s`, Chinese code `96.17 tok/s`.
- **Dense `Qwen3.8-27B + DFlash2` Zero Regression**: byte-identical T=0 outputs matching the `2.0.1.2` anchor with prefill and decode within measurement noise; supports YaRN up to **512K–1M (1,048,576)** context and decode-first concurrency.

### Features & Improvements

- **32 GB Minimum Host RAM Support for 111 GB MoE (1–8 GPUs)**: unified automatic 3-tier placement (VRAM → Host RAM → Disk) with host-safety guards, preserving full quant fidelity and context length while preventing host OOM hangs.
- **Zero-Env Out-of-the-Box Speed**: all core compute, pipeline, and speculative optimizations are enabled by default with automatic hardware detection.
- **Modern Low-Bit Formats on Volta**: aligned with upstream llama.cpp to execute **FP8 / NVFP4 / MXFP4 / Q8_0 / Q4_K** using native SM70 hardware capabilities, plus multi-GPU lifecycle stability hardening.
- **Production Sampling & Speculative Prefill Synergy**: accelerated penalty sampling (`presence_penalty` / `repeat_penalty`) and long-context MTP prefill coordination.
- **Offline Built-in WebUI & Concise Release Logging**: 70 prebuilt frontend assets ship in `tools/ui/dist/` for zero-network, Node.js-free builds (including Docker); release logs stay clean by default (`-v` for verbose output).
- **Overhauled Bilingual README & Historical Archive**: refreshed main documentation with key highlights, verified ModelScope download lists, and copy-paste AI deployment guardrails; `2.0.1.2` and earlier docs archived intact under [`docs/v100/archive/v2.0.1.2/`](docs/v100/archive/v2.0.1.2/).
- **Multi-Arch Docker Images**: `ghcr.io/1115714829/llama.cpp-v100:2.0.2` (and `:latest`) for `linux/amd64` and `linux/ppc64le` (IBM Power AC922).

---

## Earlier Releases (1.0.0 – 2.0.1.2)

Complete earlier release notes and historical benchmarks are archived intact at [`docs/v100/archive/v2.0.1.2/CHANGELOG.en.md`](docs/v100/archive/v2.0.1.2/CHANGELOG.en.md) and [`docs/v100/archive/v2.0.1.2/benchmarks.en.md`](docs/v100/archive/v2.0.1.2/benchmarks.en.md).
