# Benchmark history

[简体中文](benchmarks.md) | **English** | [← Home README](../../README.en.md) | [AI deployment guide](ai-deploy.en.md) | [27B detail page](27b.en.md) | [Qwen3.8-Flash-Next detail page](moe-2.0.0.en.md) | [Changelog](../../CHANGELOG.en.md)

This page collects the measured data of every release at the time it was released, newest first. The numbers are recorded as they appeared in the detail pages, READMEs and changelog of the time, nothing changed or added; the text keeps only test conditions, methodology and results. The note under each table gives the test date, build/version, machine and methodology (thinking on or off, temperature and sampling parameters); when a release's table is identical to that of the following release (not re-measured at the time, only carried over), it is not repeated, and a single sentence says it is the same as in that release's section and was not re-measured. The "Performance at a glance" section of the home page uses the current release's numbers. Speeds measured under different methodologies (temperature 0, thinking off vs. thinking on + the model card's sampling parameters) cannot be compared directly.

Releases: [2.0.1.1](#2011) · [2.0.1](#201) · [2.0.0](#200) · [1.1.1](#111) · [1.1.0](#110) · [1.0.9](#109) · [1.0.8](#108) · [1.0.7](#107) · [1.0.6](#106) · [1.0.5](#105) · [1.0.4](#104) · [1.0.3](#103) · [1.0.2](#102) · [1.0.1](#101) · [1.0.0](#100) · [Appendix: launch parameters of the 1.1.1 era](#appendix-launch-parameters-of-the-111-era)

---

## 2.0.1.1

### Qwen3.8-Flash-Next, 2 × V100-16GB (new in this release)

Conditions: two Tesla V100-SXM2-16GB GPUs of the IBM AC922 standard test machine, measured both as two GPUs on different CPU sockets (0 and 3) and as two GPUs on the same socket (0 and 1); command C of the Qwen3.8-Flash-Next detail page (`CUDA_VISIBLE_DEVICES=0,1` for the same socket), context 262144, Qwen3.8-Flash-Next UD-Q4_K_XL + shared-Q8_0 MTP draft (3 per round) with the expert routing profile; real code, a fresh server for each request (no prefix cache hit); **thinking on + the model card's recommended sampling parameters** (temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repeat_penalty 1.0).

| Prompt length | GPU pair | Prompt tokens | Prefill (tok/s) | Decode (tok/s) | Time to first token (s) | Time per round (ms) | Tokens per round | Draft acceptance rate | VRAM peak per GPU: highest / lowest / spread (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ~16K tokens | Different sockets (0, 3) | 16070 | 1173.2 | 37.96 | 13.75 | 54.55 | 2.073 | 0.3576 | 14754 / 14746 / 8 |
| ~16K tokens | Same socket (0, 1) | 16070 | 1100.2 | 39.05 | 14.66 | 58.34 | 2.281 | 0.4278 | 14776 / 14771 / 5 |
| ~200K tokens | Different sockets (0, 3) | 209070 | 1300.1 | 43.13 | 161.44 | 56.61 | 2.444 | 0.4802 | 14752 / 14744 / 8 |
| ~200K tokens | Same socket (0, 1) | 209070 | 1237.5 | 36.70 | 169.58 | 61.81 | 2.271 | 0.4235 | 14776 / 14771 / 5 |
| Short prompt | — | ~130 | not measured | not measured | not measured | not measured | not measured | not measured | — |

<sub>Test data: 2026-10-23, 2.0.1.1 release build `7cb60b0ec`, IBM AC922 + 2 × Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, the model card's recommended sampling parameters (temperature 1.0, top_p 0.95, top_k 20, min_p 0). The methodology differs from the 6-GPU numbers on this page (temperature 0, thinking off), so the speeds cannot be compared directly.</sub>

- At temperature 1.0 the draft acceptance rate fluctuates with sampling, and decode tok/s varies with it; to compare the speed of the two GPU pairs, look at "Time per round": on the same socket each round takes about 7%–9% longer and prefill is about 5%–6% slower.
- **Temperature 0 comparison** (the same two prompt lengths, measured once with each GPU pair): draft acceptance rate 0.6937 at ~16K tokens and 0.6082 at ~200K tokens, exactly the same for both GPU pairs; the temperature-0 output is bit-identical between the two GPU pairs.
- **Host memory headroom** (48 GB on each of the two CPU nodes of the standard test machine, context 262144): with the two GPUs on different sockets, the two nodes still have about 10.3 GiB and 8.4 GiB free after startup, and no less than about 7.7 GiB each after the requests; with the two GPUs on the same socket, one of the nodes has only about 1.0–1.6 GiB left.
- 2.0.1 cannot run Qwen3.8-Flash-Next on two 16 GB GPUs, so there are no numbers from the previous release to compare with.

### 6 GPUs, same-session A/B comparison with 2.0.1

Conditions: IBM AC922 + 6 × Tesla V100-SXM2-16GB; 2.0.1 and 2.0.1.1 were measured on the same machine in the same time window, alternating 2.0.1 → 2.0.1.1 → 2.0.1 → 2.0.1.1, two rounds each, with the machine dedicated, each version built separately with the same build options; context 262144, **temperature 0, thinking off**; Qwen3.8-Flash-Next is UD-Q4_K_XL + shared-Q8_0 MTP draft (3 per round) with the expert routing profile, 27B is Q8_0 + DFlash2 F16 (7 per round); real code, a fresh server for each prompt length, no prefix cache hit; 1024 / 768 / 256 output tokens (~200K / ~16K / short prompt, stopping early at an end token); the table shows the average of the two rounds (for Qwen3.8-Flash-Next at ~16K tokens, the average of 7 rounds including the extra rounds).

| Case | Prefill (tok/s): 2.0.1 → 2.0.1.1 | Decode (tok/s) | Time per round (ms) | Tokens per round / draft acceptance rate (same in both versions) |
|---|---|---|---|---|
| 27B ~200K tokens | 2189.9 → 2187.4 (−0.1%) | 123.79 → 124.54 (+0.6%) | 26.83 → 26.67 (−0.6%) | 3.325 / 0.3310 |
| 27B ~16K tokens | 2791.3 → 2786.1 (−0.2%) | 153.23 → 152.73 (−0.3%) | 20.34 → 20.41 (+0.3%) | 3.122 / 0.3031 |
| 27B short prompt | — | 218.02 → 220.54 (+1.2%) | 24.89 → 24.60 (−1.2%) | 5.447 / 0.5971 |
| Qwen3.8-Flash-Next ~200K tokens | 3420.5 → 3402.3 (−0.5%) | 121.57 → 122.51 (+0.8%) | 22.85 → 22.67 (−0.8%) | 2.782 / 0.5962 |
| Qwen3.8-Flash-Next ~16K tokens (7 rounds each) | 2799.1 → 2804.8 (+0.2%) | 141.15 → 140.52 (−0.4%) | 21.73 → 21.83 (+0.5%) | 3.072 / 0.6879 |
| Qwen3.8-Flash-Next short prompt | — | 149.24 → 149.84 (+0.4%) | 24.06 → 23.98 (−0.4%) | 3.606 / 0.8565 |

<sub>Test data: 2026-10-23, 2.0.1 (`1d50fc67b`) vs the 2.0.1.1 release build (`7cb60b0ec`), IBM AC922 + 6 × Tesla V100-SXM2-16GB, CUDA 12.4; temperature 0, thinking off. The short prompt is about 30 tokens; its prefill is not counted.</sub>

- Every change is within the round-to-round variation of the same version, or better than 2.0.1. For Qwen3.8-Flash-Next at ~16K tokens, the first two rounds measured decode −1.1%, so both versions were measured for 5 more rounds each: the difference over those five rounds is −0.2% and over all 7 rounds −0.4%, both within the roughly 1% round-to-round variation of the same version.
- Tokens per round and draft acceptance rate are bit-identical between the two versions; the temperature-0 output of all six cases is bit-identical between the two versions.

VRAM peak per GPU (MiB, highest of all GPUs / difference between the highest and the lowest):

| Case | 2.0.1 | 2.0.1.1 |
|---|---|---|
| 27B ~200K tokens | 9955 / 187 | 9955–9956 / 187 |
| 27B ~16K tokens | 9763 / 207 | 9763–9764 / 207 |
| 27B short prompt | 9704–9714 / 202–204 | 9636–9696 / 212–214 |
| Qwen3.8-Flash-Next ~200K tokens | 14806 / 89 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 14806 / 89 | 14806 / 89 |
| Qwen3.8-Flash-Next short prompt | 14626 / 65 | 14626 / 65 |

<sub>Test data: same as the table above.</sub>

- VRAM is the same as 2.0.1: for Qwen3.8-Flash-Next the peak of every GPU is identical, GPU by GPU, in all three cases; for 27B it is the same or slightly lower.

<details>
<summary>Per-round details</summary>

| Case | Round | Prefill (tok/s) | Decode (tok/s) | Time per round (ms) | VRAM peak per GPU: highest / spread (MiB) |
|---|---|---:|---:|---:|---|
| 27B ~200K tokens | 1: 2.0.1 | 2194.44 | 123.38 | 26.92 | 9955 / 187 |
| 27B ~200K tokens | 2: 2.0.1.1 | 2190.40 | 124.73 | 26.63 | 9956 / 187 |
| 27B ~200K tokens | 3: 2.0.1 | 2185.39 | 124.19 | 26.74 | 9955 / 187 |
| 27B ~200K tokens | 4: 2.0.1.1 | 2184.50 | 124.35 | 26.71 | 9955 / 187 |
| 27B ~16K tokens | 1: 2.0.1 | 2795.11 | 152.68 | 20.42 | 9763 / 207 |
| 27B ~16K tokens | 2: 2.0.1.1 | 2806.87 | 152.57 | 20.44 | 9764 / 207 |
| 27B ~16K tokens | 3: 2.0.1 | 2787.44 | 153.79 | 20.27 | 9763 / 207 |
| 27B ~16K tokens | 4: 2.0.1.1 | 2765.36 | 152.89 | 20.39 | 9763 / 207 |
| 27B short prompt | 1: 2.0.1 | — | 217.41 | 24.96 | 9704 / 202 |
| 27B short prompt | 2: 2.0.1.1 | — | 218.93 | 24.78 | 9696 / 212 |
| 27B short prompt | 3: 2.0.1 | — | 218.64 | 24.82 | 9714 / 204 |
| 27B short prompt | 4: 2.0.1.1 | — | 222.16 | 24.42 | 9636 / 214 |
| Qwen3.8-Flash-Next ~200K tokens | 1: 2.0.1 | 3388.63 | 120.23 | 23.10 | 14806 / 89 |
| Qwen3.8-Flash-Next ~200K tokens | 2: 2.0.1.1 | 3389.46 | 122.36 | 22.70 | 14806 / 89 |
| Qwen3.8-Flash-Next ~200K tokens | 3: 2.0.1 | 3452.43 | 122.90 | 22.60 | 14806 / 89 |
| Qwen3.8-Flash-Next ~200K tokens | 4: 2.0.1.1 | 3415.18 | 122.67 | 22.64 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 1: 2.0.1 | 2793.64 | 141.53 | 21.68 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 2: 2.0.1.1 | 2841.97 | 140.81 | 21.79 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 3: 2.0.1 | 2810.64 | 142.01 | 21.60 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 4: 2.0.1.1 | 2830.24 | 139.66 | 21.97 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 5: 2.0.1 (extra) | 2806.97 | 140.59 | 21.82 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 6: 2.0.1.1 (extra) | 2803.56 | 140.66 | 21.81 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 7: 2.0.1 (extra) | 2790.75 | 141.49 | 21.68 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 8: 2.0.1.1 (extra) | 2792.23 | 141.33 | 21.71 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 9: 2.0.1 (extra) | 2814.77 | 140.13 | 21.89 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 10: 2.0.1.1 (extra) | 2780.97 | 140.69 | 21.81 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 11: 2.0.1 (extra) | 2803.31 | 141.38 | 21.70 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 12: 2.0.1.1 (extra) | 2791.04 | 140.82 | 21.79 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 13: 2.0.1 (extra) | 2773.32 | 140.95 | 21.77 | 14806 / 89 |
| Qwen3.8-Flash-Next ~16K tokens | 14: 2.0.1.1 (extra) | 2793.33 | 139.70 | 21.96 | 14806 / 89 |
| Qwen3.8-Flash-Next short prompt | 1: 2.0.1 | — | 150.23 | 23.91 | 14626 / 65 |
| Qwen3.8-Flash-Next short prompt | 2: 2.0.1.1 | — | 153.39 | 23.41 | 14626 / 65 |
| Qwen3.8-Flash-Next short prompt | 3: 2.0.1 | — | 148.26 | 24.22 | 14626 / 65 |
| Qwen3.8-Flash-Next short prompt | 4: 2.0.1.1 | — | 146.29 | 24.55 | 14626 / 65 |

</details>

## 2.0.1

<sub>Source: the two "2.0.1 measured results" sections of docs/v100/moe-2.0.0.en.md and 27b.en.md at the 2.0.1 release; methodology for each table (incl. thinking on/off and temperature) in the note below it.</sub>

### Qwen3.8-Flash-Next + MTP

Conditions: IBM AC922 (2 × POWER9, 96 GB of host memory) + 6 × Tesla V100-SXM2-16GB; Qwen3.8-Flash-Next UD-Q4_K_XL with an MTP draft (Q4_K_M, 3 per round), 6 GPUs, context 262144. The prompts are ~200K tokens of real code, ~16K tokens of real code and a short prompt, with 1024 / 768 / 256 output tokens; **temperature 0, thinking off**. 2.0.1 and 2.0.0 were measured alternately in the same time window (same machine, same set of prompts), with a fresh server per case and the machine dedicated to the measurement. The 2.0.0 column is a same-window measurement of the 2.0.0 release-line build and agrees with "Measured at the 2.0.0 release" below.

| Case | 2.0.0 | 2.0.1 | Change |
|---|---|---|---|
| Prefill, ~200K tokens (tok/s) | 2344 | 3464 | +48% |
| Prefill, ~16K tokens (tok/s) | 2053 | 2823 | +38% |
| Decode, ~200K tokens (tok/s) | 109.9 | 118.5 | +8% |
| Decode time per round, ~200K tokens (ms) | 25.60 | 23.04 | −10% |
| Decode, ~16K tokens (tok/s) | 117.9 | 139.6 | +18% |
| Decode time per round, ~16K tokens (ms) | 24.59 | 21.92 | −11% |
| Decode, short prompt (tok/s) | 135.5 | 151.0 | +11% |
| Decode time per round, short prompt (ms) | 26.62 | 23.89 | −10% |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (test conditions in the original).</sub>

Prefill of ~200K tokens is on par with the best figure measured for this project before (3459 tok/s). Acceptance was measured separately with 24 prompts and a Q4_K_M draft: accepted tokens per round went 2.752 → 2.734 at temperature 0 and 2.592 → 2.579 at temperature 0.6, both within measurement noise, while the time per round on the same prompt batch went 24.4 → 21.3 ms (temperature 0) and 26.1 → 23.2 ms (temperature 0.6).

VRAM peak per GPU (highest of all GPUs, MiB):

| Case | 2.0.0 | 2.0.1 |
|---|---|---|
| Qwen3.8-Flash-Next, ~200K tokens | 14838 | 14804 |
| Qwen3.8-Flash-Next, ~16K tokens | 14838 | 14804 |
| Qwen3.8-Flash-Next, short prompt | 14655 | 14624 |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (test conditions in the original).</sub>

### 27B (Qwen3.8-27B + DFlash2)

Conditions: IBM AC922 (2 × POWER9, 96 GB of host memory) + 6 × Tesla V100-SXM2-16GB; Qwen3.8-27B Q8_0 with DFlash2 F16, 6 GPUs with tensor parallelism, context 262144. 2.0.1 and 2.0.0 were measured alternately in the same time window (same machine, same set of prompts, machine dedicated); the 2.0.0 column is a same-window measurement of the 2.0.0 release-line build. The three 27B requests ran in one server in the order "short prompt → ~16K tokens (temperature 0.8, not listed in the table) → ~200K tokens"; the ~200K-token prompt hit the cached prefix left by the previous request, so 195048 tokens were actually processed; **temperature 0, thinking off** (except the 16K case).

| Case | 2.0.0 | 2.0.1 | Change |
|---|---|---|---|
| Prefill, ~200K tokens (tok/s) | 2073 | 2140 | +3% |
| Decode, ~200K tokens (tok/s) | 126.4 | 128.9 | +2% |
| Decode time per round, ~200K tokens (ms) | 26.22 | 25.96 | −1% |
| Decode, short prompt (tok/s) | 216.0 | 225.8 | +5% |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (the ~16K-token request used temperature 0.8 and is not listed; test conditions in the original).</sub>

For the short prompt, accepted tokens per round are bit-identical between 2.0.0 and 2.0.1; at ~200K tokens the output of 2.0.0 differed between its two rounds (accepted 712/2205 and 718/2163), while 2.0.1 gave the same output in both rounds (718/2142) — the non-reproducible T=0 output of 2.0.0 across restarts on 6 GPUs at long context is fixed in this version.

VRAM peak per GPU (highest of all GPUs, MiB):

| Case | 2.0.0 | 2.0.1 |
|---|---|---|
| 27B | 9969 | 9963 |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (the ~16K-token request used temperature 0.8 and is not listed; test conditions in the original).</sub>

### Measurements in the changelog

- The VRAM peak of every GPU is no higher than 2.0.0; Qwen3.8-Flash-Next saves about 30–70 MiB per GPU.

<sub>Source: the measurement part of the 2.0.1 section of CHANGELOG.md (2026-10-18); measured alternately with 2.0.0 in the same time window, temperature 0, thinking off (as stated in the changelog).</sub>

## 2.0.0

<sub>Source: "Measured at the 2.0.0 release" in docs/v100/moe-2.0.0.en.md, "Measured at the 2.0.0 release (vs 1.1.1)" in 27b.en.md and the README.en.md at tag v100-2.0.0; methodology for each table (incl. thinking on/off and temperature) in the note below it.</sub>

### Qwen3.8-Flash-Next + MTP

Standard test machine: 2026-10-11, IBM Power AC922 (2 × POWER9, 96 GB memory) + 6 × Tesla V100-SXM2-16GB, CUDA 12.4, release build `4d50148cc`; UD-Q4_K_XL + MTP shared-Q4_K_M (3 drafted tokens per round) with the expert routing profile, using command A below; non-streaming, temperature 0, thinking off, a fresh server for every request (cold prefix), two rounds ("a / b" in the table are the two rounds; the output was identical in both). Real code = the source of an open-source C++ project plus a one-sentence summary request.

| Request | Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) | Draft acceptance (accepted / drafted) | VRAM per GPU after the request |
|---|---:|---:|---:|---:|---:|---:|
| Short prompt (a one-sentence coding task) | 33 | 256 | — | 134.1 / 134.4 | 185 / 213 (87%) | 14648–14652 MiB |
| Real code, ~16K tokens | 16030 | 768 | 2022.2 / 2069.1 | 117.4 / 116.3 | 503 / 795 (63%) | 14810–14835 MiB |
| Real code, ~200K tokens | 209030 | 686 (limit 1024, ended naturally) | 2334.8 / 2340.8 | 107.9 / 109.7 | 442 / 729 (61%) | 14793–14838 MiB |

<sub>Test data: 2026-10-11, 2.0.0 release build `4d50148cc`, IBM AC922 + 6 × Tesla V100-SXM2-16GB, CUDA 12.4; non-streaming, temperature 0, thinking off, fresh server per request (test conditions in the original).</sub>

- **Stability** (2026-10-12, release build, same command): 3 cold starts (the first request being ~16K tokens) all succeeded, prefill 2055.4–2060.9 tok/s; 41 mixed requests over 20 minutes (short prompts up to ~200K tokens, alternating temperature 0 and 0.7) with 0 failures, peak VRAM 14974 MiB on a single GPU. About 3 minutes from start to ready.
- **Above temperature 0**: in a long run where one server alternated temperature-0 and temperature-0.7 requests, temperature 0.7 lowered draft acceptance from 0.58–0.66 to 0.54–0.58 and decode by 12%–17% (measured).
- **Comparison with 27B on the same machine**: same machine, same build, same ~200K-token real code: Qwen3.8-Flash-Next prefills about 13% faster than 27B (Q8_0 + DFlash2) and decodes at about 87% of 27B's speed.
- **Reference host: x86, 4 × V100-SXM2-32GB** (2026-10-11, all-to-all NVLink between GPUs, PCIe 3.0 between GPU and host, 136 GB memory, 1 NUMA node, CUDA 12.8; an earlier pre-release build `638de68da`, for reference only; command B, one server receiving three requests in turn, temperature 0): short-prompt decode 95.0 tok/s; ~16K tokens (768 output) prefill 2239, decode 88.2 tok/s; ~200K tokens (the first ~14K tokens hit the previous request's cache, 195048 tokens actually processed, ended naturally at 533 tokens) prefill 2434, decode 66.4 tok/s. All experts resident in VRAM; peak VRAM per GPU 24916–24970 MiB.

### Qwen3.8-Flash-Next: measurements for parameters and configurations

Previously in the parameter tables and symptom tables of the 2.0.0 and 2.0.1 detail pages; from 2.0.1.1 on, the detail pages only describe what each parameter does, and the numbers are collected here.

| Item | Measured |
|---|---|
| Context `-c 262144` | 1437 MiB of context per GPU |
| Physical batch `-ub` | 691 MiB compute buffer per GPU at 2048; without MTP, `-ub 4096` gives −27% prefill at ~200K tokens and `-ub 8192` −43% |
| MTP draft shared-Q8_0 vs shared-Q4_K_M | Same decode speed and acceptance rate; shared-Q8_0 uses about 150 MiB more VRAM per GPU and prefills ~200K tokens about 2% slower |
| VRAM of the MTP draft | Draft 316–330 MiB per GPU; draft, draft compute buffer and draft context together about 0.4–0.45 GB per GPU |
| `--spec-draft-ubatch` | 93 MiB draft compute buffer per GPU at 256; 170 MiB when inheriting `-ub 2048` |
| `--spec-draft-n-max` | 3 drafted tokens per round is fastest; drafting more costs more in verification than it gains |
| Expert routing profile (a profile with only the `count` column, without MTP) | With the profile, decode at ~16K tokens is 12% faster and cold prefill at ~200K tokens 8% faster |
| Corpus used to collect the profile | A profile collected from code: prefill 6.8% faster on code text, 3.6% faster on English documents |
| First request after the server starts | ~16K tokens: about 2000–2030 tok/s prefill for the first request of a fresh server, about 2200 tok/s for later requests |

<sub>Test data: around the 2.0.0 release (2026-10), standard test machine IBM AC922 + 6 × Tesla V100-SXM2-16GB, Qwen3.8-Flash-Next UD-Q4_K_XL, context 262144; temperature and thinking mode were not stated for these items (thinking mode not recorded).</sub>

### 27B (Qwen3.8-27B + DFlash2)

The changes in 2.0.0 apply to 27B as well; the table below is a same-session comparison with 1.1.1. Conditions: 2026-10-11, IBM Power AC922 (2 × POWER9, 96 GB memory) + 6 × Tesla V100-SXM2-16GB, CUDA 12.4. 1.1.1 (tag `v100-1.1.1`) and 2.0.0 (release build `4d50148cc`) alternated in the same time window, two rounds each, with a fresh server per round, using command A below (Q8_0 + DFlash2 F16, 7 drafted tokens per round); each server received three non-streaming requests with thinking off. Real code = the source of an open-source C++ project plus a one-sentence summary request.

| Request | Metric | 1.1.1 | 2.0.0 | Change |
|---|---|---:|---:|---:|
| Short prompt (33 tokens, temperature 0, 256 output) | Decode | 186.2 / 188.9 | 218.3 / 206.9 | +13% |
| Real code, 16030 tokens (temperature 0.8, 128 output) | Prefill | 2863.4 / 2842.7 | 3119.1 / 3121.1 | +9.4% |
| Same | Decode | 116.5 / 115.9 | 156.8 / 124.5 | +21% |
| Real code, 209030 tokens (temperature 0, 1024 output; the first ~14K tokens hit the previous request's cache, 195048 tokens actually processed) | Prefill | 2047.8 / 2047.1 | 2066.6 / 2063.2 | +0.9% |
| Same | Decode | 114.5 / 115.6 | 122.2 / 127.0 | +8.3% |
| Same | Draft acceptance (accepted / drafted) | 711 / 2219, 709 / 2205 | 704 / 2254, 715 / 2170 | — |
| After the three requests (a separate same-session measurement on the same day, pre-release build `95a708853`) | VRAM per GPU | 9840–10017 MiB | 9776–9953 MiB | −64 MiB per GPU |

<sub>Test data: 2026-10-11, 1.1.1 (tag `v100-1.1.1`) vs 2.0.0 release build `4d50148cc` (VRAM row: pre-release build `95a708853`, same day), IBM AC922 + 6 × Tesla V100-SXM2-16GB, CUDA 12.4; thinking off, short prompt and ~200K tokens at temperature 0, ~16K tokens at temperature 0.8 (test conditions in the original).</sub>

Speeds in tok/s; "a / b" are the two rounds; changes use the two-round average.

- Short-prompt temperature-0 output is identical between the two versions. With 6-GPU 27B the number of accepted draft tokens varies slightly from run to run, so long-input decode speed varies with it; the 16K row outputs only 128 tokens and is mainly a prefill measurement.
- Multimodal smoke test (2026-10-12, release build, 6 GPUs, context 524288, 2 slots, a fine-tuned 27B Q8_0 + DFlash2 Q8_0 + mmproj Q8_0): image, PDF and video requests all returned normally.
- Other configurations (4 GPUs, 2/3/5 GPUs and Q4 configurations, `-ub` comparison, 524288 context, concurrency, multimodal) were not re-measured on 2.0.x; all 1.1.1-era tables and their launch parameters are on the [1.1.1 section of this page](#111).

### Measurement sentences in the 2.0.0 README

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB, build `4d50148cc`; context 262144, with MTP and the expert routing profile; ~200K tokens of real code, fresh server, temperature 0, thinking off, output ended naturally at 686 tokens): prefill 2334.8 / 2340.8 tok/s, decode 107.9 / 109.7 tok/s. On the same machine, Qwen3.8-Flash-Next prefills ~200K tokens about 13% faster than 27B and decodes at about 87% of 27B's speed.

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB; 1.1.1 and 2.0.0 build `4d50148cc` alternated in the same time window, two rounds each; context 262144, DFlash2; conditions in the 27B table above in this section): short-prompt decode 186.2 / 188.9 → 218.3 / 206.9 tok/s; ~16K-token prefill 2863.4 / 2842.7 → 3119.1 / 3121.1 tok/s; ~200K-token prefill 2047.8 / 2047.1 → 2066.6 / 2063.2 tok/s and decode 114.5 / 115.6 → 122.2 / 127.0 tok/s.

<sub>Source: README.en.md at tag v100-2.0.0, verbatim (2026-10-12); the first sentence is Qwen3.8-Flash-Next, the second 27B; both temperature 0, thinking off (27B ~16K tokens at temperature 0.8, see the table above).</sub>

### Measurements in the changelog

- Fixed the extra time per prefill batch with speculative decoding on: measured at about 20 ms per batch for Qwen3.8-Flash-Next + MTP; the existing 27B DFlash2 configurations are not affected.

<sub>Source: the measurement part of the 2.0.0 section of CHANGELOG.md (2026-10-12); thinking mode not recorded.</sub>

## 1.1.1

<sub>Source: the former docs/v100/27b-results-1.1.1.en.md (the "Test data" section of the 1.1.1 README with notes added), 2026-10-06; thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of the 1.1.1 README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged. The 3-GPU balancing and long-context VRAM changes of 1.0.7 are in "What's new". Numbers measured on 1.0.8 are marked "1.0.8" in the tables (4- and 6-GPU 200K synthetic, 6-GPU 419430, vision-module VRAM); the rest were measured on earlier versions, and 6-GPU long-context prefill is faster from 1.0.8 on.
- Values marked "1.1.1" were measured on version 1.1.1 (2026-10-06, alternating with 1.1.0 in the same session; all items unchanged).
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---:|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens, 1.1.1) | 1772 tok/s | 282.0 tok/s | 28.3 ms | 12.9 GB |
| real code 209228 tokens (1024 output tokens) | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| real code 131184 tokens (1024 output tokens) | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| real code 16500 tokens (1024 output tokens, 1.1.1) | 2584 tok/s | 131.7 tok/s | 21.9 ms | 12.9 GB |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

**Draft model quantization** (4 GPUs, Q8, 2 runs each):

| Draft model | Tokens produced per round (short / 16K / 128K) | Time per speculative round (short / 16K / 128K) | Peak VRAM per GPU |
|---|---:|---:|---:|
| F16 | 3.23 / 2.51 / 2.46 | 21.3 / 22.1 / 26.0 ms | 13.3 GB |
| Q8_0 | 2.76 / 2.55 / 2.75 | 20.6 / 21.3 / 25.3 ms | 12.9 GB |
| Q4_K_M | 2.84 / 2.56 / 2.43 | 20.6 / 21.3 / 25.3 ms | 12.6 GB |

<sub>Test data: measured on 1.0.7 (2026-10-01, see the 1.0.7 section of CHANGELOG), 4 GPUs Q8, 2 runs each; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2552 tok/s | 116.5 tok/s | 0.22 | 22.0 ms |
| thinking mode | 128K | 1963 tok/s | 97.6 tok/s | 0.22 | 25.9 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2620 tok/s | 105.4 tok/s | 0.25 | 26.0 ms |
| non-thinking mode | 128K | 1971 tok/s | 89.1 tok/s | 0.24 | 29.8 ms |

<sub>Test data: measured on 1.0.4 (2026-09-29 to 30, see this section's test environment), 4 GPUs Q8, context 262144, real code, 1024 output tokens; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking and non-thinking modes; temperature and sampling per the "Mode" column (model-card recommended values).</sub>

From 1.0.7 on, the time per speculative round in non-thinking mode (presence penalty) went 26.2 → 21.8 ms (the table was measured on 1.0.4).

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1801 tok/s, decode 277.1 tok/s |
| 8192 | not enough VRAM, fails to start |

<sub>Test data: measured on 1.0.4 (2026-09-29 to 30, see this section's test environment), 4 GPUs Q8, context 262144; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens, 1.1.1) | 2102 tok/s | 300.5 tok/s | 26.6 ms | 10.0 GB |
| 2048 | real code 209227 tokens (1024 output tokens) | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 16494 tokens (1024 output tokens) | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16500 tokens (1024 output tokens) | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.1.1) | 278.9 s | 1504 tok/s | 238.8 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

The `-ub 4096` row was measured on an earlier version (prefill is faster from 1.0.8 on as well). `-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Concurrent requests

Real code, thinking mode (T=0.6), 1024 output tokens per request, default `--prefill-pace 30`. Measured sequentially in the same server; "concurrent" means the requests are sent at the same time.
- TTFT is counted from when the request is sent; decode per request = the speed of that request from its first token to the end.
- Aggregate decode = total output across all requests divided by (earliest first token to the end of the last request).
- All done = from sending the requests to the end of the last request.

**4 GPUs, `-c 262144 -np 2` (131072 per request)**, startup VRAM 13.5 GB per GPU, peak 13.7 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K (1.1.1) | 6.4 s | 120.9 tok/s | 121 tok/s | 14.9 s |
| two concurrent requests of 16K each (1.1.1) | 8.1 s / 22.2 s | 84.1 / 103.6 tok/s | 86 tok/s | 32.0 s |
| single request ~115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of ~115K tokens each (1.1.1) | 55.1 s / 118.9 s | 60.0 / 103.8 tok/s | 28 tok/s | 128.8 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each (1.1.1) | 8.2 / 23.2 / 39.0 / 52.5 s | 76.3 / 82.8 / 83.7 / 106.8 tok/s | 73 tok/s | 62.1 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4–12.8 GB per GPU, peak 12.7–13.0 GB (measured on 1.0.4; about 0.3 GB more per GPU from 1.0.8 on):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each (1.1.1) | 8.0 s / 20.1 s | 94.0 / 104.3 tok/s | 94 tok/s | 29.9 s |
| two concurrent requests of ~115K tokens each (1.1.1) | 47.4 s / 101.6 s | 75.0 / 88.4 tok/s | 31 tok/s | 113.2 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

The same configuration with each slot filled to close to 256K (1.0.9, vision module split across the 6 GPUs, thinking mode, 12.44–12.60 GB per GPU after startup):

| Scenario | TTFT | Prefill | Decode per request | Peak VRAM per GPU |
|---|---|---|---|---|
| two concurrent text requests of ~244K tokens each | 126 s / 251 s | 1964 / 1855 tok/s | 54.0 / 69.0 tok/s | 12.9–13.1 GB |
| two concurrent requests: ~232K tokens of text + image / ~235K tokens of text + video | 253.5 s / 146.4 s | — | 60.8 / 67.9 tok/s | 12.9–13.1 GB |

<sub>Test data: measured on 1.0.9 (2026-10-02), 6 GPUs Q8, `-c 524288 -np 2` (YaRN), vision module split across 6 GPUs; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on (original says "thinking mode"; temperature and sampling not recorded).</sub>

`-c 1048576 -np 2` (524288 per slot) also starts and text works, but there is not enough VRAM when processing images; to have 512K per slot together with images, put the vision module on the CPU with `--no-mmproj-offload` (encoding becomes much slower).

Compared with 1.0.3 (4 GPUs, same test):

| Scenario | 1.0.3 | 1.0.4 |
|---|---|---|
| `-np 2` two 16K requests: TTFT | 9.2 s / 16.7 s | 8.1 s / 22.7 s |
| `-np 2` two 16K requests: decode per request | 45.7 / 58.8 tok/s | 81.4 / 111.4 tok/s |
| `-np 2` two 16K requests: all done | 34.1 s | 31.9 s |
| `-np 2` two 115K requests: TTFT | 59.6 s / 133.0 s | 58.0 s / 122.6 s |
| `-np 2` two 115K requests: decode per request | 11.4 / 58.9 tok/s | 70.2 / 81.4 tok/s |
| `-np 2` two 115K requests: all done | 150.6 s | 135.2 s |
| `-np 2` startup VRAM per GPU | 14.0 GB | 13.5 GB |
| `-np 4` (64K per request) | fails at startup for lack of VRAM | runs |

<sub>Test data: 1.0.3 vs 1.0.4 (at the 1.0.4 release, 2026-09-29 to 30), 4 GPUs Q8, same test; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 → 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0–4; 6 GPUs 0–5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2584 / 131.7 tok/s / 21.9 ms (1.1.1) | 209715 / 118 s / 1772 / 282.0 tok/s (1.1.1) | 12.9 GB (1.1.1) |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 278.9 s / 1504 / 238.8 tok/s (1.1.1) | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 on, Q4 decode is about 10% faster (the Q4 rows in the table were measured on earlier versions).
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8 (the table was measured on 1.0.6); from 1.0.8 6-GPU long-context prefill is faster: the 6-GPU Q8 long synthetic input is measured on 1.0.8, the 6-GPU Q4 row on an earlier version.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

| GPUs | Config | Context (per request) | single 16K: TTFT / decode | two concurrent 16K: TTFT | Decode per request | Peak VRAM per GPU |
|---|---|---:|---|---|---|---:|
| 2 | Q4 | 131072 (65536) | 10.0 s / 67.2 tok/s | 12.5 s / 37.2 s | 44.6 / 64.7 tok/s | 14.4 GB |
| 3 | Q8 | 131072 (65536) | 7.3 s / 74.1 tok/s | 9.1 s / 26.4 s | 64.4 / 103.8 tok/s | 15.2 GB |
| 3 | Q4 | 524288 (262144) | 7.6 s / 82.0 tok/s | 9.5 s / 28.7 s | 57.5 / 62.8 tok/s | 14.7 GB |
| 4 | Q8 | 262144 (131072) | 6.5 s / 117.0 tok/s | 8.2 s / 23.9 s | 70.6 / 89.6 tok/s | 13.7 GB |
| 4 | Q4 | 524288 (262144) | 6.7 s / 78.9 tok/s | 8.5 s / 24.5 s | 69.4 / 87.9 tok/s | 12.0 GB |
| 5 | Q8 | 524288 (262144) | 6.7 s / 109.2 tok/s | 8.6 s / 22.8 s | 84.2 / 121.5 tok/s | 14.5 GB |
| 5 | Q4 | 524288 (262144) | 6.8 s / 76.5 tok/s | 8.4 s / 26.4 s | 61.4 / 98.2 tok/s | 10.6 GB |
| 6 | Q8 | 524288 (262144) | 6.4 s / 117.7 tok/s | 8.0 s / 23.1 s | 73.7 / 121.4 tok/s | 13.0 GB |
| 6 | Q4 | 524288 (262144) | 6.5 s / 81.9 tok/s | 8.0 s / 24.8 s | 66.1 / 78.7 tok/s | 9.7 GB |

<sub>Test data: measured on 1.0.5 (2026-09-30), 2–6 GPUs Q8 / Q4, `-np 2`; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking mode not recorded (temperature and sampling not recorded).</sub>

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure, 1.1.1) | 3078 | 4.8 s | 145 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides, 1.1.1) | 23104 | 18.7 s | 173 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on (original says "thinking mode"; temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6–11.8 GB (measured on 1.1.1; 1.0.7: 12.2 GB on GPU 0, 10.8–11.2 GB on the others). Unmarked rows in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.2 s and a 12-second video 18.5 s; peak VRAM 13.5 GB on every GPU, at most 26 MiB apart (measured on 1.1.1). PDF first 6 pages 3.2 s (measured on 1.1.0). 1.0.7: 14.2 GB on GPU 0.
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

### Measurements in the changelog

- Measured alternately with 1.1.0 in the same time window: on 4/6 GPUs the T=0 output is word-for-word identical (5.66 / 5.65 ms per token); 4 GPUs, real 16K prefill 2579.0 / 2585.4 tok/s; 2 GPUs Q4 262144, synthetic 209715 prefill 1082.6 / 1081.0 tok/s, speculative round 43.7 / 43.7 ms; 6 GPUs 524288, synthetic 419430 prefill 1505.9 / 1503.2 tok/s, decode 239.6 / 239.4 tok/s; time to first token with two concurrent requests the same on 4 and 6 GPUs; 6-GPU multimodal images and video work normally; per-GPU VRAM peaks the same.

<sub>Source: the measurement part of the 1.1.1 section of CHANGELOG.md (2026-10-06); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.1.0

<sub>Source: the "Test data" section of the README at tag v100-1.1.0 (2026-10-03); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged. The 3-GPU balancing and long-context VRAM changes of 1.0.7 are in "What's new". Numbers measured on 1.0.8 are marked "1.0.8" in the tables (4- and 6-GPU 200K synthetic, 6-GPU 419430, vision-module VRAM); the rest were measured on earlier versions, and 6-GPU long-context prefill is faster from 1.0.8 on.
- Values marked "1.1.0" were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session; all items unchanged).
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---:|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens, 1.1.0) | 1752 tok/s | 281.3 tok/s | 28.4 ms | 12.9 GB |
| real code 209228 tokens (1024 output tokens) | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| real code 131184 tokens (1024 output tokens) | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| real code 16500 tokens (1024 output tokens, 1.1.0) | 2561 tok/s | 126.7 tok/s | 22.0 ms | 12.9 GB |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking mode not recorded (temperature and sampling not recorded).</sub>

**Draft model quantization** (4 GPUs, Q8, 2 runs each):

(This table is the same as in the 1.1.1 section; not re-measured.)

On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens, 1.1.0) | 2082 tok/s | 300.3 tok/s | 26.6 ms | 10.0 GB |
| 2048 | real code 209227 tokens (1024 output tokens) | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 16494 tokens (1024 output tokens) | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16500 tokens (1024 output tokens) | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.1.0) | 285.9 s | 1467 tok/s | 238.4 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking mode not recorded (temperature and sampling not recorded).</sub>

The `-ub 4096` row was measured on an earlier version (prefill is faster from 1.0.8 on as well). `-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Concurrent requests

Real code, thinking mode (T=0.6), 1024 output tokens per request, default `--prefill-pace 30`. Measured sequentially in the same server; "concurrent" means the requests are sent at the same time.
- TTFT is counted from when the request is sent; decode per request = the speed of that request from its first token to the end.
- Aggregate decode = total output across all requests divided by (earliest first token to the end of the last request).
- All done = from sending the requests to the end of the last request.

**4 GPUs, `-c 262144 -np 2` (131072 per request)**, startup VRAM 13.5 GB per GPU, peak 13.7 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K (1.1.0) | 6.5 s | 123.1 tok/s | 123 tok/s | 14.8 s |
| two concurrent requests of 16K each (1.1.0) | 8.1 s / 23.0 s | 74.5 / 94.0 tok/s | 79 tok/s | 33.9 s |
| single request ~115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of ~115K tokens each (1.1.0) | 55.1 s / 119.1 s | 58.7 / 81.0 tok/s | 27 tok/s | 131.8 s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking on, T=0.6 (other sampling parameters not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each | 8.5 / 27.1 / 43.6 / 57.5 s | 64.0 / 69.9 / 81.9 / 124.4 tok/s | 72 tok/s | 65.7 s |

<sub>Test data: 2026-10-03 (date of tag v100-1.1.0), README at the 1.1.0 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on, T=0.6 (other sampling parameters not recorded).</sub>

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4–12.8 GB per GPU, peak 12.7–13.0 GB (measured on 1.0.4; about 0.3 GB more per GPU from 1.0.8 on):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each (1.1.0) | 8.1 s / 20.7 s | 89.7 / 122.5 tok/s | 98 tok/s | 29.0 s |
| two concurrent requests of ~115K tokens each (1.1.0) | 47.3 s / 103.0 s | 65.4 / 108.6 tok/s | 32 tok/s | 112.4 s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking on, T=0.6 (other sampling parameters not recorded).</sub>

The same configuration with each slot filled to close to 256K (1.0.9, vision module split across the 6 GPUs, thinking mode, 12.44–12.60 GB per GPU after startup):

(This table is the same as in the 1.1.1 section; not re-measured.)

`-c 1048576 -np 2` (524288 per slot) also starts and text works, but there is not enough VRAM when processing images; to have 512K per slot together with images, put the vision module on the CPU with `--no-mmproj-offload` (encoding becomes much slower).

Compared with 1.0.3 (4 GPUs, same test):

(This table is the same as in the 1.1.1 section; not re-measured.)

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 → 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0–4; 6 GPUs 0–5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2561 / 126.7 tok/s / 22.0 ms (1.1.0) | 209715 / 120 s / 1752 / 281.3 tok/s (1.1.0) | 12.9 GB (1.1.0) |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 285.9 s / 1467 / 238.4 tok/s (1.1.0) | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 on, Q4 decode is about 10% faster (the Q4 rows in the table were measured on earlier versions).
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8 (the table was measured on 1.0.6); from 1.0.8 6-GPU long-context prefill is faster: the 6-GPU Q8 long synthetic input is measured on 1.0.8, the 6-GPU Q4 row on an earlier version.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.1.1 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure, 1.1.0) | 3078 | 4.8 s | 171 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides, 1.1.0) | 23104 | 18.7 s | 148 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment"; thinking on (temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6–11.8 GB (measured on 1.1.0; 1.0.7: 12.2 GB on GPU 0, 10.8–11.2 GB on the others). Unmarked rows in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.3 s and the PDF first 6 pages 3.2 s; peak VRAM 13.5 GB on every GPU, at most 28 MiB apart (measured on 1.1.0; 1.0.7: 14.2 GB on GPU 0).
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.1.1 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context".

### Measurements in the changelog

- Fixed a crash of the 6-GPU multi-slot server at long context (several slots each processing a long prompt of more than 32768 tokens at the same time, e.g. agent sessions with large images; reported as `illegal memory access` or a segmentation fault). Regression case (6 GPUs `-np 2`, two slots of ~40K tokens each + 6 large images): 1.0.9 crashed at about 23 s both times, 1.1.0 completed both times.
- Measured alternately with 1.0.9 in the same time window: on 4/6 GPUs the T=0 output is word-for-word identical; 6 GPUs, 200K synthetic prefill 2085.5 → 2082.0 tok/s, decode 300.2 → 300.3 tok/s; 6 GPUs 419430: 1469.1 → 1467.0 tok/s; 4 GPUs, real 16K prefill 2554.5 → 2561.0 tok/s; 4 GPUs, 200K synthetic (average of 5 alternating rounds) prefill 1754.0 → 1752.2 tok/s, decode 281.3 → 281.3 tok/s; 3 GPUs, 2 GPUs, concurrency and multimodal on par; VRAM peaks identical GPU by GPU.

<sub>Source: the measurement part of the 1.1.0 section of CHANGELOG.md (2026-10-03); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.9

<sub>Source: the "Test data" section of the README at tag v100-1.0.9 (2026-10-02); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged. The 3-GPU balancing and long-context VRAM changes of 1.0.7 are in "What's new". Numbers measured on 1.0.8 are marked "1.0.8" in the tables (4- and 6-GPU 200K synthetic, 6-GPU 419430, vision-module VRAM); the rest were measured on earlier versions, and 6-GPU long-context prefill is faster from 1.0.8 on.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---:|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens, 1.0.8) | 1759 tok/s | 281.2 tok/s | 28.4 ms | 12.9 GB |
| real code 209228 tokens (1024 output tokens) | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| real code 131184 tokens (1024 output tokens) | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| real code 16500 tokens (1024 output tokens; prefill on 1.0.8) | 2578 tok/s | 119.5 tok/s | 22.0 ms | 13.3 GB |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

**Draft model quantization** (4 GPUs, Q8, 2 runs each):

(This table is the same as in the 1.1.0 section; not re-measured.)

On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens, 1.0.8) | 2091 tok/s | 300.3 tok/s | 26.6 ms | 10.0 GB |
| 2048 | real code 209227 tokens (1024 output tokens) | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 16494 tokens (1024 output tokens) | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16500 tokens (1024 output tokens) | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.0.8) | 281.5 s | 1490 tok/s | 238.0 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

The `-ub 4096` row was measured on an earlier version (prefill is faster from 1.0.8 on as well). `-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Concurrent requests

Real code, thinking mode (T=0.6), 1024 output tokens per request, default `--prefill-pace 30`. Measured sequentially in the same server; "concurrent" means the requests are sent at the same time.
- TTFT is counted from when the request is sent; decode per request = the speed of that request from its first token to the end.
- Aggregate decode = total output across all requests divided by (earliest first token to the end of the last request).
- All done = from sending the requests to the end of the last request.

**4 GPUs, `-c 262144 -np 2` (131072 per request)**, startup VRAM 13.5 GB per GPU, peak 13.7 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 115.7 tok/s | 116 tok/s | 15.4 s |
| two concurrent requests of 16K each | 8.1 s / 22.7 s | 81.4 / 111.4 tok/s | 86 tok/s | 31.9 s |
| single request ~115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of ~115K tokens each | 58.0 s / 122.6 s | 70.2 / 81.4 tok/s | 27 tok/s | 135.2 s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on, T=0.6 (other sampling parameters not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

(This table is the same as in the 1.1.0 section; not re-measured.)

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4–12.8 GB per GPU, peak 12.7–13.0 GB (measured on 1.0.4; about 0.3 GB more per GPU from 1.0.8 on):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each | 8.0 s / 21.9 s | 85.4 / 116.1 tok/s | 90 tok/s | 30.7 s |
| two concurrent requests of ~115K tokens each | 55.3 s / 118.8 s | 66.0 / 90.5 tok/s | 27 tok/s | 130.1 s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on, T=0.6 (other sampling parameters not recorded).</sub>

Compared with 1.0.3 (4 GPUs, same test):

(This table is the same as in the 1.1.0 section; not re-measured.)

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 → 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0–4; 6 GPUs 0–5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2549 / 131.7 tok/s / 22.0 ms | 209715 / 128 s / 1641 / 277.8 tok/s | 13.3 GB |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 281.5 s / 1490 / 238.0 tok/s (1.0.8) | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 on, Q4 decode is about 10% faster (the Q4 rows in the table were measured on earlier versions).
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8 (the table was measured on 1.0.6); from 1.0.8 6-GPU long-context prefill is faster: the 6-GPU Q8 long synthetic input is measured on 1.0.8, the 6-GPU Q4 row on an earlier version.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.1.0 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 182 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides) | 23104 | 19.7 s | 150 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on (temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6–11.8 GB (1.0.7: 12.2 GB on GPU 0, 10.8–11.2 GB on the others), with unchanged processing time. Times in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.3 s and the PDF first 6 pages 3.2 s; peak VRAM 13.5 GB on every GPU, at most 26 MiB apart (1.0.8; 1.0.7: 14.2 GB on GPU 0).
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.1.0 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context".

### Measurements in the changelog

- Faster decode with the Q4 configuration (including Q2_K–Q6_K mixed quantization). Measured alternately with 1.0.8 in the same time window: 2 GPUs Q4, synthetic 102400, decode 193.0 → 212.2 tok/s (+10%), prefill 1419 → 1438 tok/s; 3 GPUs Q4, decode 220.1 → 240.9 tok/s (+9.5%), prefill 1902 → 1929 tok/s; 2 GPUs Q4, real content, 8 prompts, time per token at T=0 11.58 → 10.31 ms and at T=0.6 11.07 → 10.43 ms; perplexity 2.4097 → 2.4057 (`-ub 8`), 2.4067 → 2.4047 (`-ub 32`). The output of Q5_K/Q6_K differs slightly from 1.0.8.
- Slightly faster decode with a q4_0 KV cache (output bit-identical): 2 GPUs Q4 100K decode about +1.2%; fixed a crash with q4_1 / q5_1 KV caches on V100.
- The Q8 configuration is the same as 1.0.8: the T=0 output on 4/6 GPUs is word-for-word identical and every speed is on par (4 GPUs, real 16K prefill over 5 alternating rounds 2576.4 / 2576.5 tok/s).

<sub>Source: the measurement part of the 1.0.9 section of CHANGELOG.md (2026-10-02); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.8

<sub>Source: the "Test data" section of the README at tag v100-1.0.8 (2026-10-02); thinking mode is stated per table in the notes below each table.</sub>

The tables in these subsections are the same as in the 1.0.9 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "GPU counts and quantization"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- Faster long-context prefill on 6 GPUs (takes effect once the context reaches 32768): 6 GPUs 524288, synthetic 209715 prefill 1800 → 2091 tok/s (+16%), 419430: 1200 → 1490 tok/s (+24%); about 300 MiB more VRAM per GPU.
- Faster long-context prefill: 4 GPUs, 200K synthetic prefill 1637 → 1759 tok/s (+7.4%), real 16K 2551 → 2578 tok/s.
- Slightly faster speculative decoding: 4 GPUs, 200K decode 278.4 → 281.2 tok/s.
- The vision module (mmproj) is split across all GPUs used by the main model (2–6 GPUs), so VRAM is more balanced across GPUs. VRAM peak of the highest GPU: 6 GPUs 12220 → 11788, 5 GPUs 13911 → 13444, 4 GPUs 14151 → 13547, 2 GPUs 14966 → 14528 MiB, with a difference between GPUs of ≤ 170 MiB; image and video encoding time unchanged (median of 6 repetitions of the same image/video), output identical token for token. For configurations with little VRAM headroom that run out of VRAM when processing images, lower the context, switch to Q4, or put the vision module on the CPU with `--no-mmproj-offload`.
- Baseline: measured alternately with 1.0.7 in the same time window, the T=0 output on 4/6 GPUs is word-for-word identical; 2 GPUs Q4 100K prefill 1415 → 1419 tok/s, 3 GPUs Q4 100K 1881 → 1902 tok/s, 6 GPUs 16K on par; time to first token with concurrent requests the same or faster.

<sub>Source: the measurement part of the 1.0.8 section of CHANGELOG.md (2026-10-02); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.7

<sub>Source: the "Test data" section of the README at tag v100-1.0.7 (2026-10-01); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged. The 3-GPU balancing and long-context VRAM changes of 1.0.7 are in "What's new"; the other numbers in the tables were not re-measured.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---:|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1638 tok/s | 278.9 tok/s | 28.6 ms | 13.3 GB |
| real code 209228 tokens (1024 output tokens) | 1635 tok/s | 88.0 tok/s | 28.6 ms | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1976 tok/s | 306.0 tok/s | 26.1 ms | 13.3 GB |
| real code 131184 tokens (1024 output tokens) | 1966 tok/s | 102.6 tok/s | 25.9 ms | 13.3 GB |
| real code 16500 tokens (1024 output tokens) | 2547 tok/s | 119.5 tok/s | 22.0 ms | 13.3 GB |

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

**Draft model quantization** (4 GPUs, Q8, 2 runs each):

(This table is the same as in the 1.0.8 section; not re-measured.)

On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1708 tok/s | 293.1 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 209227 tokens (1024 output tokens) | 1703 tok/s | 103.7 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 16494 tokens (1024 output tokens) | 2569 tok/s | 146.9 tok/s | 20.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1897 tok/s | 292.2 tok/s | 27.3 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 89.4 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16500 tokens (1024 output tokens) | 2690 tok/s | 121.4 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1905 tok/s | 291.4 tok/s | 27.4 ms | 13.8 GB |

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 363 s | 1154 tok/s | 233.1 tok/s | 12.8 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0–4; 6 GPUs 0–5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1841 / 78.6 tok/s / 35.8 ms | 209715 / 197 s / 1065 / 162.4 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1861 / 63.4 tok/s / 37.1 ms | 104857 / 74 s / 1419 / 184.4 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2470 / 90.0 tok/s / 28.2 ms | 104857 / 59 s / 1779 / 231.5 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2353 / 78.3 tok/s / 31.6 ms | 419430 / 530 s / 792 / 134.9 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2549 / 131.7 tok/s / 22.0 ms | 209715 / 128 s / 1641 / 277.8 tok/s | 13.3 GB |
| 4 | Q4 | 524288 | 2458 / 110.4 tok/s / 26.7 ms | 419430 / 379 s / 1107 / 198.5 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2612 / 126.1 tok/s / 22.2 ms | 419430 / 367 s / 1143 / 222.4 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2489 / 93.2 tok/s / 26.5 ms | 419430 / 374 s / 1122 / 198.7 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2774 / 114.1 tok/s / 20.5 ms | 419430 / 351 s / 1197 / 232.2 tok/s | 12.8 GB |
| 6 | Q4 | 524288 | 2599 / 110.6 tok/s / 25.5 ms | 419430 / 358 s / 1171 / 203.3 tok/s | 9.5 GB |

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8.
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8; see "What's new" for speed and VRAM (the table was measured on 1.0.6).

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.0.8 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

The tables in these subsections are the same as in the 1.0.8 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- Attention load balancing on 3 GPUs (enabled only with 3 GPUs and an even split). 3 GPUs Q8, 131072, measured alternately with 1.0.6: 100K prefill 1808 → 1972 tok/s (+9.0%), decode 232.8 → 239.7 tok/s (+3.0%); 16K prefill 2571 → 2675 tok/s (+4.1%), decode 282.8 → 291.5 tok/s (+3.1%); perplexity unchanged (1.6557 / 1.6554); VRAM peak per GPU 14800 → 15116 MiB.
- Less VRAM for long-context prefill. VRAM peak per GPU at a 262144 context: 2 GPUs Q4 14140 → 13678 MiB (−462), 4 GPUs Q8 13299 → 12835 MiB (−464), 6 GPUs Q8 10234 → 9694 MiB (−540); prefill speed unchanged (2 GPUs Q4 100K prompt processing 70753 → 70853 ms, within noise).
- Faster processing of small leftover batches: 4 GPUs, real 16K prompt processing 6332 → 6309 ms (1.0.6 → 1.0.7).
- Faster small leftover batches of 17–64 tokens with the Q4 configuration: 2 GPUs Q4 with `-ub 32`, perplexity 1.7340 → 1.7323.
- Faster decode in non-thinking mode (presence penalty): 4 GPUs (presence 1.5, temperature 0.7), 26.2 → 21.8 ms per round (−17%), decode 139 → 172 tok/s.
- Draft model quantization compared (4 GPUs Q8, 2 runs each): tokens per round (short/16K/128K) F16 3.23/2.51/2.46, Q8_0 2.76/2.55/2.75, Q4_K_M 2.84/2.56/2.43; time per speculative round F16 21.3/22.1/26.0 ms, Q8_0 and Q4_K_M about 20.6/21.3/25.3 ms; VRAM peak per GPU F16 13.3, Q8_0 12.9, Q4_K_M 12.6 GB. On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.
- Baseline: measured alternately with 1.0.0 and 1.0.6 in the same time window (4 GPUs Q8 262144, 3 rounds each): 200K synthetic prefill 1.0.0 1641.4 / 1.0.6 1640.9 / 1.0.7 1647.2 tok/s (+0.35%), time per speculative round 28.75 / 28.72 / 28.67 ms, VRAM peak per GPU 14333 / 13299 / 12835 MiB; real 16K prefill 2547 / 2551 / 2568 tok/s. 6 GPUs, 200K synthetic prefill 1.0.6 1804 / 1.0.7 1808 tok/s, decode 294.2 / 297.5 tok/s. The T=0 output on 4/6 GPUs is word-for-word identical to 1.0.6. Concurrency compared with 1.0.4 in the same time window: time to first token the same or faster in every scenario (6 GPUs, two slots of ~115K tokens each: 55.2 / 117.7 → 51.8 / 110.5 s; 4 GPUs, two slots of ~115K each: 57.7 / 123.5 → 57.9 / 124.6 s); VRAM per GPU at the end of the 6-GPU two-slot run about 12.99 → 12.02 GB; aggregate decode varies with each slot's acceptance rate (single measurement), e.g. 4 GPUs two slots of 16K each 82.4 → 86.9 tok/s, 6 GPUs two slots of 115K each 28.1 → 29.3 tok/s.

<sub>Source: the measurement part of the 1.0.7 section of CHANGELOG.md (2026-10-01); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.6

<sub>Source: the "Test data" section of the README at tag v100-1.0.6 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

(This table is the same as in the 1.0.7 section; not re-measured.)

The tables in these subsections are the same as in the 1.0.7 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "GPU counts and quantization"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- Faster prefill on 2, 3, 5 and 6 GPUs (4-GPU output bit-identical): 6 GPUs, 200K synthetic prefill 1710 → 1801 tok/s (+5.4%), 6 GPUs real 16K 2568 → 2774 tok/s, 2 GPUs Q4 +8–10%, 3 GPUs Q8 +6%; perplexity unchanged.
- Faster decode with the Q4 configuration: time per speculative round 5–9% lower, output changes slightly, perplexity unchanged (2 GPUs 1.7321 → 1.7320).
- The 4-GPU Q8 configuration has the same output, speed and VRAM as 1.0.5 (compared alternately with 1.0.0 in the same time window: 4 GPUs 200K prefill 1641.0 vs 1641.6 tok/s); concurrency on par with 1.0.4.

<sub>Source: the measurement part of the 1.0.6 section of CHANGELOG.md (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.5

<sub>Source: the "Test data" section of the README at tag v100-1.0.5 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The "GPU counts and quantization" section was measured on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30), and on those configurations 1.0.5 produces word-for-word identical output to 1.0.4 with unchanged speed.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### GPU counts and quantization

Measured on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0–4; 6 GPUs 0–5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

**Single request** (real code 16K input, 1024 output tokens; synthetic input is 80% of the context, 512 output tokens):

| GPUs | Config | Context | real 16K: prefill / decode / time per speculative round | synthetic long input: tokens / TTFT / prefill / decode | Peak VRAM per GPU |
|---|---|---:|---|---|---:|
| 2 | Q4 (draft Q4_K_M) | 262144 | 1633 / 63.2 tok/s / 39.1 ms | 209715 / 212 s / 988 / 152.5 tok/s | 14.1 GB |
| 2 | Q4 (draft F16) | 131072 | 1633 / 69.5 tok/s / 39.8 ms | 104857 / 82 s / 1286 / 173.9 tok/s | 13.7 GB |
| 3 | Q8 | 131072 | 2288 / 98.5 tok/s / 28.1 ms | 104857 / 63 s / 1678 / 232.4 tok/s | 14.8 GB |
| 3 | Q4 | 524288 | 2134 / 74.7 tok/s / 33.5 ms | 419430 / 546 s / 768 / 130.7 tok/s | 14.2 GB |
| 4 | Q8 | 262144 | 2545 / 111.0 tok/s / 22.0 ms | 209715 / 128 s / 1640 / 277.9 tok/s | 13.3 GB |
| 4 | Q4 | 524288 | 2445 / 99.6 tok/s / 28.4 ms | 419430 / 381 s / 1102 / 189.9 tok/s | 11.6 GB |
| 5 | Q8 | 524288 | 2439 / 126.3 tok/s / 22.2 ms | 419430 / 379 s / 1107 / 221.8 tok/s | 14.2 GB |
| 5 | Q4 | 524288 | 2395 / 89.2 tok/s / 28.2 ms | 419430 / 377 s / 1112 / 191.2 tok/s | 10.3 GB |
| 6 | Q8 | 524288 | 2568 / 118.1 tok/s / 20.5 ms | 419430 / 363 s / 1156 / 232.1 tok/s | 12.8 GB |
| 6 | Q4 | 524288 | 2509 / 85.5 tok/s / 26.9 ms | 419430 / 364 s / 1153 / 196.1 tok/s | 9.5 GB |

<sub>Test data: 2026-09-30 (date of tag v100-1.0.5), README at the 1.0.5 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is slower than with Q8.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.0.6 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

The tables in these subsections are the same as in the 1.0.6 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- Known issue of 1.0.4 closed: "4-GPU 200K prefill about 0.2% slower than 1.0.3" was measured alternately in the same time window (1.0.3 / 1.0.4 back to back, 4 rounds each); each 2048 batch takes only 0.9–1.6 ms longer (0.07–0.13%), within this machine's measurement error and not attributable to any single change, so it is judged not to be a regression.

<sub>Source: the measurement part of the 1.0.5 section of CHANGELOG.md (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.4

<sub>Source: the "Test data" section of the README at tag v100-1.0.4 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

The tables in these subsections are the same as in the 1.0.5 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- Concurrency: new `--prefill-pace P` (default 30). While a request is decoding, prompt processing for other requests takes at most P% of the time, so a decoding request is no longer dragged down to a dozen or so tok/s by long prompts. 4 GPUs `-np 2`, two concurrent requests of ~115K tokens each: the request that starts decoding first 11.4 → 70.2 tok/s, time to first token of the other 133.0 → 122.6 s. Setting it to 0 or 100 is the same as 1.0.3 (no throttling).
- Faster decode with several slots decoding at once: 4 GPUs `-np 2`, two concurrent requests of 16K each, decode per slot 45.7 / 58.8 → 81.4 / 111.4 tok/s.
- Less VRAM with several slots: 4 GPUs `-c 262144 -np 2`, startup VRAM per GPU 14031 → 13513 MiB; `-np 4` (64K per slot) goes from running out of VRAM at startup to running; 6 GPUs `-c 524288 -np 2` runs.
- Known issue: 4 GPUs, 256K context, prefill of 209715 synthetic tokens about 0.2% slower than 1.0.3 (4 measurements each in the same time window, average 1640.6 → 1637.6 tok/s, about 0.2 s more per request); decode time per round unchanged. Cause under investigation, fix planned for 1.0.5.

<sub>Source: the measurement part of the 1.0.4 section of CHANGELOG.md (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.3

<sub>Source: the "Test data" section of the README at tag v100-1.0.3 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "two concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- 4-GPU speeds were measured on 1.0.0–1.0.2 and peak VRAM on 1.0.2 (on 4 GPUs, 1.0.3 produces output identical to 1.0.2 at the same speed); 6-GPU data was measured on 1.0.3.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Peak VRAM per GPU |
|---|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1652 tok/s | 281.5 tok/s | 13.3 GB |
| real code 209115 tokens (1024 output tokens) | 1646 tok/s | 28.3 ms per speculative round | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1997 tok/s | 310.4 tok/s | 13.3 GB |
| real code 131188 tokens (1024 output tokens) | 1983 tok/s | 97.6 tok/s | 13.3 GB |
| real code 16499 tokens (1024 output tokens) | 2570 tok/s | 123.5 tok/s | 13.3 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| thinking mode | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| non-thinking mode | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking and non-thinking modes; temperature and sampling per the "Mode" column (model-card recommended values).</sub>

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1799 tok/s, decode 278.0 tok/s |
| 8192 | not enough VRAM, fails to start |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1710 tok/s | 293.6 tok/s | 27.2 ms | 10.2 GB |
| 2048 | real code 209233 tokens (1024 output tokens) | 1701 tok/s | 97.5 tok/s | 27.1 ms | 10.2 GB |
| 2048 | real code 16502 tokens (1024 output tokens) | 2556 tok/s | 128.9 tok/s | 20.5 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1895 tok/s | 290.9 tok/s | 27.4 ms | 11.4 GB |
| 4096 | real code 209236 tokens (1024 output tokens) | 1889 tok/s | 99.7 tok/s | 27.2 ms | 11.4 GB |
| 4096 | real code 16501 tokens (1024 output tokens) | 2716 tok/s | 118.9 tok/s | 20.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1906 tok/s | 293.0 tok/s | 27.2 ms | 13.8 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 363 s | 1155 tok/s | 232.9 tok/s | 12.7 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 232.2 tok/s | 13.9 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### 4 GPUs, two concurrent requests

`-c 262144 -np 2` (131072 per request), measured sequentially in the same server, peak VRAM 14.5–14.7 GB per GPU:

| Scenario | TTFT | Decode per request | Aggregate decode |
|---|---|---:|---:|
| single request 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| two concurrent requests of 16K each | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| single request ~115K tokens | 55.8 s | 101.5 tok/s | 102 tok/s |
| two concurrent requests of ~115K tokens each | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-c 524288 -np 2` (262144 per request) does not fit in VRAM and fails to start.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | TTFT | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.1 s | 148 tok/s |
| image (diagram) | 4069 | 5.7 s | 186 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.6 s | 182 tok/s |
| 12-second video (slides) | 23104 | 19.7 s | 181 tok/s |
| 10-second video | 6387 | 7.4 s | 179 tok/s |
| 200K tokens of text + 1 image | 206958 | 123 s | 100 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on (temperature and sampling not recorded).</sub>

- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed).
- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 232.6 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: a one-image request takes 7.6 s and the PDF first 6 pages 5.5 s (both include answer generation); peak VRAM about 14.7 GB on GPU 0.
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

### Measurements in the changelog

- Faster decode on 6 GPUs (takes effect when the 6 GPUs can be split into two groups of 3 by NVLink link speed, e.g. the AC922; other topologies behave as in 1.0.2): time per speculative round on 6 GPUs 6–9% lower (e.g. 256K context, 209715 synthetic tokens, 512 output: decode 272.4 → 293.6 tok/s); prefill speed and VRAM use unchanged.
- Decode time per round on 6 GPUs about 0.4% lower again.

<sub>Source: the measurement part of the 1.0.3 section of CHANGELOG.md (2026-09-28); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.2

<sub>Source: the "Test data" section of the README at tag v100-1.0.2 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 6 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "two concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- Speeds were measured on 1.0.0–1.0.2; peak VRAM was measured on 1.0.2.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 6 GPUs, 262144 context

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1720 tok/s | 272.9 tok/s | 29.3 ms | 10.2 GB |
| 2048 | real code 209233 tokens (1024 output tokens) | 1719 tok/s | 84.3 tok/s | 29.1 ms | 10.2 GB |
| 2048 | real code 16496 tokens (1024 output tokens) | 2592 tok/s | 135.3 tok/s | 22.6 ms | 10.2 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.6 tok/s | 29.2 ms | 11.4 GB |
| 4096 | real code 209229 tokens (1024 output tokens) | 1905 tok/s | 87.4 tok/s | 29.1 ms | 11.4 GB |
| 4096 | real code 16497 tokens (1024 output tokens) | 2731 tok/s | 115.7 tok/s | 22.6 ms | 11.4 GB |
| 8192 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.3 tok/s | 29.2 ms | 13.8 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 360 s | 1165 tok/s | 220.2 tok/s | 12.7 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1309 tok/s | 218.1 tok/s | 13.9 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | TTFT | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 136 tok/s |
| image (diagram) | 4069 | 5.6 s | 126 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 151 tok/s |
| 12-second video (slides) | 23104 | 19.5 s | 164 tok/s |
| 10-second video | 6387 | 7.3 s | 160 tok/s |
| 200K tokens of text + 1 image | 206958 | 122 s | 91 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking on (temperature and sampling not recorded).</sub>

- Appending 8030 tokens to the same session after 208K tokens: TTFT 7.7 s (only the new part is processed).
- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 217.9 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: a one-image request takes 7.6 s and the PDF first 6 pages 5.5 s (both include answer generation); peak VRAM about 14.7 GB on GPU 0.
- With the vision module on CPU (`--no-mmproj-offload`): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.0.3 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "4 GPUs, two concurrent requests".

### Measurements in the changelog

- Less VRAM at long context: about 1 GB less VRAM per GPU at a 256K context and about 2 GB at 512K, and the same amount less host memory.

<sub>Source: the measurement part of the 1.0.2 section of CHANGELOG.md (2026-09-28); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.1

<sub>Source: the "Performance" section of the README at tag v100-1.0.1 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Performance" section of that README)

Test environment:
- Server: IBM Power AC922 (2 × POWER9), 4 × Tesla V100-SXM2-16GB (NVLink), CUDA 12.4.
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model, drafts 7 tokens per round.
- Parameters: see "Recommended launch parameters" in that README.
- Every case is measured on a freshly started server.

| Input | Prefill | Decode | Peak VRAM per GPU |
|---|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1652 tok/s | 281.5 tok/s | 14.3 GB |
| real code 209115 tokens (1024 output tokens) | 1646 tok/s | 28.3 ms per speculative round | 14.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1997 tok/s | 310.4 tok/s | 14.3 GB |
| real code 131188 tokens (1024 output tokens) | 1983 tok/s | 97.6 tok/s | 14.3 GB |
| real code 16499 tokens (1024 output tokens) | 2570 tok/s | 123.5 tok/s | 14.3 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

Decode speed on real content fluctuates a lot with the acceptance rate, so the time per speculative round is given as well.

### Sampling parameters recommended by the model card

Measured with the two sampling presets recommended by the model card (real code, 1024 output tokens):

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| Thinking (T=1.0, top_p 0.95, top_k 20) | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| Thinking | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| Non-thinking (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| Non-thinking | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking and non-thinking modes; temperature and sampling per the "Mode" column (model-card recommended values).</sub>

- In thinking mode the time per round is the same as at T=0.6; the difference in decode speed comes only from the acceptance rate.
- Non-thinking mode (`presence_penalty`) is about 4.5 ms (17~21%) slower per round (improved from 1.0.7 on); this kind of content has a higher acceptance rate, so the decode speed ends up similar.

### Batch size (`-ub`)

4 GPUs, context 262144:

| `-ub` | Result |
|---|---|
| 2048 | Runs normally, peak VRAM per GPU 14.3 GB |
| 4096 | Out of VRAM, fails to start |
| 8192 | Out of VRAM, fails to start |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

With 4 GPUs at the full 256K context, 2048 is the limit; 6 GPUs hold fewer weights per GPU, see the next subsection.

### 6 GPUs

From 1.0.1 on, 6 GPUs are supported (`--tensor-split 1,1,1,1,1,1`, other parameters as above); 6 GPUs had no specific optimization yet at that time (decode is faster from 1.0.3 on, long-context prefill from 1.0.8 on). The state at that time (context 262144, every case on a freshly started server):

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1720 tok/s | 272.9 tok/s | 29.3 ms | 11.3 GB |
| 2048 | real code 209233 tokens (1024 output tokens) | 1719 tok/s | 84.3 tok/s | 29.1 ms | 11.3 GB |
| 2048 | real code 16496 tokens (1024 output tokens) | 2592 tok/s | 135.3 tok/s | 22.6 ms | 11.3 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.6 tok/s | 29.2 ms | 13.5 GB |
| 4096 | real code 209229 tokens (1024 output tokens) | 1905 tok/s | 87.4 tok/s | 29.1 ms | 13.5 GB |
| 4096 | real code 16497 tokens (1024 output tokens) | 2731 tok/s | 115.7 tok/s | 22.6 ms | 13.5 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

- Compared with 4 GPUs: 200K prefill is about 4% faster at `-ub 2048` and about 16% faster at `-ub 4096`; decode is about 3% slower per round.
- `-ub 6144` also starts (about 15.0 GB per GPU after startup); `-ub 8192` does not fit in VRAM.

**Single request, 512K context**: with 6 GPUs there is VRAM to spare per GPU, enough for 512K. Qwen natively supports 262,144 tokens; longer contexts need YaRN as the model card suggests, and the server-side limit "per-slot context must not exceed the training length" must be lifted:

```bash
-c 524288 -np 1 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288
```

Measured with synthetic 419430 tokens (80% of 512K, 512 output tokens): time to first token 360 s, prefill 1165 tok/s, decode 220.2 tok/s, peak VRAM per GPU 14.8 GB; `-ub 4096` does not fit in VRAM. Only speed and VRAM were measured; output quality above 256K was not evaluated. YaRN applies to all lengths, so it is not recommended unless you need more than 256K.

### Two concurrent requests

256K per slot (`-c 524288 -np 2`) does not fit in VRAM and fails to start. 128K per slot (`-c 262144 -np 2`) runs, with peak VRAM per GPU 14.5~14.7 GB:

| Scenario | Time to first token | Decode per slot | Total decode |
|---|---|---:|---:|
| single request 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| two requests of 16K each, sent together | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| single request of ~115K tokens | 55.8 s | 101.5 tok/s | 102 tok/s |
| two requests of ~115K tokens each, sent together | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

At that time two concurrent requests were slower than one: the second request waited until the first finished prefill, and while one request was decoding and the other was in prefill, the decoding request slowed down markedly (improved from 1.0.4 on, see the [1.0.4 section](#104)). A single slot (`-np 1`) was recommended at that time.

## 1.0.0

<sub>Source: the "Performance" section of the README at tag v100-1.0.0 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

The tables in these subsections are the same as in the 1.0.1 section and were not re-measured: test environment and main table; sampling parameters recommended by the model card; batch size (`-ub`); two concurrent requests.

---

## Appendix: launch parameters of the 1.1.1 era

Below are the launch commands corresponding to the test data. `numactl --membind=0,8` limits host memory to the two CPU nodes of this machine (the AC922 exposes GPU VRAM as NUMA nodes; without this limit, page cache may occupy VRAM); on other machines, use `numactl -H` to check the node numbers.

### 4 GPUs, 262144 context

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 GPUs, 262144 context

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### 6 GPUs, 524288 context (YaRN)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### Concurrent requests

Change `-np` on the single-request commands above; the total context is divided among the requests:

```bash
# 4 GPUs, two requests, 131072 each
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 2 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --prefill-pace 30 \
  --host 127.0.0.1 --port 8080 --metrics

# 4 GPUs, four requests, 65536 each: replace -np 2 above with -np 4

# 6 GPUs, two requests, 262144 each (YaRN)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 2 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --prefill-pace 30 \
  --host 127.0.0.1 --port 8080 --metrics
```

`--prefill-pace 30` is the default; it is written out only for clarity and can be omitted.

### Multimodal: 6 GPUs, 524288 context (YaRN), vision module on GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --mmproj ./models/mmproj-F16.gguf \
  --video-ffmpeg-dir /path/to/ffmpeg/bin \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080 --metrics
```

### Multimodal: 4 GPUs, 262144 context, vision module on GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --mmproj ./models/mmproj-F16.gguf \
  --video-ffmpeg-dir /path/to/ffmpeg/bin \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080 --metrics
```

Multimodal requests use the OpenAI-compatible `/v1/chat/completions`: images via `image_url`, videos via `input_video` (or `video_url`); videos are sampled at 4 frames per second by default. For PDF, the client renders each page into an image and sends it (about 600 tokens per page at 80 dpi).

### Other GPU counts and Q4 configuration

Change these parts of the commands above:
- GPU count: `CUDA_VISIBLE_DEVICES` and `--tensor-split` (one 1 for each GPU);
- Q4 configuration: `-m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf`, `-ctk q4_0 -ctv q4_0`;
- context: pick from the "GPU counts and quantization" table; for 524288 add the three YaRN parameters;
- 2 GPUs with Q4 at 262144: `--model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`.

For example, 2 GPUs with Q4, 262144 context:

```bash
CUDA_VISIBLE_DEVICES=0,1 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 262144 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048 \
  --model-draft ./models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

### Parameter description

| Parameter | Description |
|---|---|
| `numactl --membind=<nodes>` | keep host memory on the given NUMA nodes (placed before `llama-server`). For hosts that expose GPU memory as NUMA nodes (such as AC922, where the nodes with CPUs are `0,8`), to keep host memory and the page cache out of GPU memory; not needed on ordinary x86 servers. List the nodes with `numactl --hardware`; the nodes must exist, otherwise the server does not start. In Docker use the `LLAMA_NUMA_MEMBIND` environment variable |
| `-ngl 999` | all layers on GPU |
| `--split-mode tensor --tensor-split 1,…,1` | tensor parallelism, evenly split across GPUs |
| `-c` | context length |
| `-np` | number of concurrent requests; context per request = `-c` divided by the number of requests |
| `--prefill-pace` | with multiple requests, the maximum percentage of time prompt processing of other requests may take while a request is decoding (0–100, default 30; 0 or 100 means no throttling) |
| `-fa on` | Flash Attention (must be on) |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache; `-ctk q4_0 -ctv q4_0` is 4-bit, using half the VRAM |
| `-b 2048 -ub 2048` | batch size; the test data uses 2048 unless noted otherwise |
| `--model-draft …`, `--spec-type draft-dflash`, `--spec-draft-n-max 7` | DFlash2 speculative decoding, drafts 7 tokens per round |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` | use YaRN to extend the context to 524288 (the model's native length is 262144) |
| `--override-kv qwen35.context_length=int:524288` | allow a single-slot context beyond the model's training length |
| `--mmproj` | vision module |
| `--video-ffmpeg-dir` | directory containing `ffmpeg` and `ffprobe`, needed for video |
| `--no-mmproj-offload` | run the vision module on CPU (by default it is split across all GPUs used by the main model). Saves GPU memory, but image and video processing becomes much slower |
| `-mmdev <device>` | put the vision module on one given GPU only (see `--list-devices` for device names, e.g. `CUDA0`); `-mmdev none` is the same as CPU |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--chat-template-kwargs` | default sampling parameters and thinking mode switch (the values above are the model card's thinking mode parameters) |
| `--metrics` | Prometheus metrics (optional) |
| `--cache-ram <MiB>` | Upper limit of the prompt cache kept in host memory (upstream parameter, default 8192). When several conversations take turns, their state can be restored from host memory instead of being recomputed. Choose it from the host memory size: host memory use is roughly base use + `--cache-ram` + slots × `--ctx-checkpoints` × checkpoint size; going over the host memory gets the process killed by the system |
| `--ctx-checkpoints <N>` | Recurrent-state checkpoints kept per slot (upstream parameter, default 32), used to reuse prefixes; each checkpoint takes one recurrent state of host memory (size depends on the model), multiplied by the number of slots |
| `reasoning_effort` (request parameter) | Thinking effort, interpreted by the model's chat template. The Qwen3.8 template accepts `low`, `medium`, `high`, `xhigh` (default), and `none` turns thinking off; `--chat-template-kwargs '{"reasoning_effort":"medium"}'` sets a server default |
| `NCCL_P2P_LEVEL` (environment variable) | On ppc64le the program sets it to `SYS` automatically; a value you set yourself is not overridden |

Sampling: fastest when `top_k` is at most 64 and only top-k, top-p, min-p, temperature and repeat / presence / frequency penalties are used; with DRY or other samplers, or a larger `top_k`, decode becomes slower.

<sub>Parameter reference table, not measured data; source: the README at the 1.1.1 release (2026-10-06); thinking mode not applicable.</sub>
