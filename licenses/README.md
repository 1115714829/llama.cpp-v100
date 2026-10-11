# 第三方代码与许可 / Third-party code and licenses

本仓库自 2.0.0 起以 GPL-3.0 发布（见仓库根目录的 `LICENSE`），基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT，原许可文本见本目录的 `LICENSE-llama.cpp`）。下表列出移植或改编自其他项目的文件、来源和许可，每个文件的文件头也写明了出处。许可原文是本目录的 `LICENSE-*`（构建出的 `llama` 程序用 `llama licenses` 也能显示）和随代码放在子目录里的几份许可文件（见最后一列），发布包的 `licenses/` 目录包含全部这些文件。上游 llama.cpp 自带的其余第三方代码（`vendor/` 等）沿用上游的许可安排。

This repository is released under GPL-3.0 from 2.0.0 on (see `LICENSE` in the repository root) and is based on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT; its license text is kept as `LICENSE-llama.cpp` in this directory). The tables below list the files that are ported or adapted from other projects, with their sources and licenses; each file also names its source in its header. The license texts are the `LICENSE-*` files in this directory (also printed by `llama licenses`) and the license files kept next to the vendored code (last column); the `licenses/` directory of a release package contains all of them. Other third-party code that comes with upstream llama.cpp (`vendor/` and so on) keeps upstream's arrangement.

## 移植或改编的代码 / Ported or adapted code

| 文件 / File | 来源 / Source | 许可 / License | 许可原文 / License text |
|---|---|---|---|
| `ggml/src/ggml-cuda/fused-expert.cu`, `fused-expert.cuh` | [Strata-AC922] `src/kernels/cuda/fused_expert.cu` | MIT | `licenses/LICENSE-strata` |
| `ggml/src/ggml-cuda/lightning-indexer.cu` (Volta scoring path) | [Strata-AC922] `src/kernels/cuda/qsa_select.cu` (`scores_volta_kernel`) | MIT | `licenses/LICENSE-strata` |
| `ggml/src/ggml-cuda/qsa.cu` (top-k, decode and prompt attention kernels) | [Strata] `src/kernels/cuda/qsa_select.cu`, `qsa_decode_attn.cu`, `qsa_prompt_attn.cu` | MIT | `licenses/LICENSE-strata` |
| `ggml/src/ggml-cuda/moe-grouped.cu`, `moe-grouped.cuh` | [Strata] `src/kernels/cuda/s2_expert_grouped.cu` (kernel structure) | MIT | `licenses/LICENSE-strata` |
| `ggml/src/ggml-cuda/dsv4-hc.cu` (gated post epilogue), `unary.cu` (scaled SiLU) | [Strata] `src/kernels/cuda/fused_gr.cu` | MIT | `licenses/LICENSE-strata` |
| `ggml/src/ggml-cuda/allreduce-push.cu`, `allreduce-push.cuh` | [1Cat-vLLM] `csrc/custom_all_reduce.cuh` (adapted there from [SGLang-V100]) | Apache-2.0 | `licenses/LICENSE-1Cat-vLLM` |
| `ggml/src/ggml-cuda/q8-skinny.cu`, `q8-skinny.cuh`, `q-skinny-common.cuh`, `q4k-skinny.cu`, `q4k-skinny.cuh` | [1Cat-vLLM] `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu`; QPN8 layout from [v100-skinny], block-scale adaptation from [SGLang-V100] | Apache-2.0; MIT (v100-skinny) | `licenses/LICENSE-1Cat-vLLM`; `ggml/src/ggml-cuda/LICENSE.v100-skinny` |
| `ggml/src/ggml-cuda/fattn-sm70-grouped.cuh` | [1Cat-vLLM] `csrc/attention/sm70_grouped_long/` (`include/fused_mma.h`, `kernel/grouped-attention.cu`), a directory under the license of [flash-attention-v100 (ai-bond)] | BSD-3-Clause (D.Skryabin); Apache-2.0 (1Cat-vLLM) | `licenses/LICENSE-ai-bond-flash-attention-v100`; `licenses/LICENSE-1Cat-vLLM` |
| `ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh` | [1Cat-vLLM] `csrc/attention/sm70_v37/tail.cu` (the same kernel ships as `cmake/patches/sm70_flash_attn_d256_pipeline.patch`); support layer from [flash-attention-v100 (zhinianqin)] and CuTe/[CUTLASS] | BSD-3-Clause (file header); Apache-2.0 (1Cat-vLLM) | `licenses/LICENSE-1Cat-vLLM`; `ggml/src/ggml-cuda/sm70-vendor/LICENSE-flash-attention`; `ggml/src/ggml-cuda/sm70-vendor/LICENSE-cute-cutlass` |
| `ggml/src/ggml-cuda/sm70-vendor/flash/` | [flash-attention-v100 (zhinianqin)] @ `c2eda5e` (FlashAttention lineage) | BSD-3-Clause | `ggml/src/ggml-cuda/sm70-vendor/LICENSE-flash-attention` |
| `ggml/src/ggml-cuda/sm70-vendor/cute/`, `sm70-vendor/cutlass/` | [CUTLASS] @ `62750a2b` | BSD-3-Clause | `ggml/src/ggml-cuda/sm70-vendor/LICENSE-cute-cutlass` |
| `ggml/src/ggml-cuda/gdn-chunk-sm70/gdn-chunk-cumsum-hv12.cuh`, `gdn-chunk-kkt.cuh`, `gdn-chunk-fwd.cuh` | FlashQLA kernels in [1Cat-vLLM] `flash_qla/`, compiled for sm_70 by [TileLang] 0.1.13 | MIT (FlashQLA, TileLang); Apache-2.0 (1Cat-vLLM) | `licenses/LICENSE-FlashQLA`; `ggml/src/ggml-cuda/gdn-chunk-sm70/LICENSE-tilelang`; `licenses/LICENSE-1Cat-vLLM` |
| `ggml/src/ggml-cuda/gdn-chunk-sm70/tl_templates/` | [TileLang] templates | MIT | `ggml/src/ggml-cuda/gdn-chunk-sm70/LICENSE-tilelang` |
| MTP draft borrowing the target's tensors: `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-context.cpp`, `src/llama-cparams.h`, `src/models/qwen4exp.cpp`, `include/llama.h`, `common/speculative.cpp`, `conversion/base.py`, `convert_hf_to_gguf.py` (comments "from unsloth PR #142/#240") | [unsloth llama.cpp] PR #142, #240 | MIT (same as llama.cpp) | `licenses/LICENSE-llama.cpp` |
| `vendor/nlohmann/` (from upstream llama.cpp) | nlohmann/json | MIT | `licenses/LICENSE-jsonhpp` |

Strata 与 Strata-AC922 用同一份 MIT 许可（Copyright (c) 2026 Niko1221 and the Strata contributors）。 / Strata and Strata-AC922 carry the same MIT license (Copyright (c) 2026 Niko1221 and the Strata contributors).

## 只参考了设计、没有移植代码 / Design references only (no code taken)

| 文件 / File | 参考 / Reference |
|---|---|
| `ggml/src/ggml-expert-store.cpp`, `ggml-expert-store.h`, `ggml/src/ggml-cuda/ggml-cuda-expert.cu`, `ggml-cuda-expert.cuh` | [Strata] expert cache and prefill ring: `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`, `src/core/expert_source.cpp`, `src/prefill/prefill.cpp` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (HC fusion matching) | [Strata] `src/kernels/cuda/fused_gr.cu` |
| `common/speculative.cpp` (MTP draft window) | [Strata] `src/core/mtp.cpp` (`first_needed`) |
| `ggml/src/ggml.c`, `tests/test-backend-ops.cpp` (BF16 to F16 clamp rule) | [Strata] `src/prefill/gemm.cu` (`bf16_to_f16_kernel`) |
| `tests/bench-pinned-read.cu` | [Strata-AC922] `docs/IBM_AC922.md` (the hardware rules it measures) |
| `ggml/src/ggml-cuda/q4k-skinny.cu`, `q4k-skinny.cuh` (int4 scale/bias folding) | [1Cat-vLLM] `csrc/sm70_turbomind/ops/awq_qpn_m1_sm70.cu`, the q4 Volta kernels of [ninfer-v100] (Apache-2.0) |

[Strata]: https://github.com/Niko1221/Strata
[Strata-AC922]: https://github.com/eelgaev/Strata-AC922
[1Cat-vLLM]: https://github.com/1CatAI/1Cat-vLLM
[SGLang-V100]: https://github.com/haohervchb/sglang-V100
[v100-skinny]: https://github.com/dnv2003/v100-skinny
[flash-attention-v100 (ai-bond)]: https://github.com/ai-bond/flash-attention-v100
[flash-attention-v100 (zhinianqin)]: https://github.com/zhinianqin/flash-attention-v100
[CUTLASS]: https://github.com/NVIDIA/cutlass
[TileLang]: https://github.com/tile-ai/tilelang
[ninfer-v100]: https://github.com/dollarwong/ninfer-v100
[unsloth llama.cpp]: https://github.com/unslothai/llama.cpp
