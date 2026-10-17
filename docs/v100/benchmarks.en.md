# Benchmark history

[简体中文](benchmarks.md) | **English** | [← Home README](../../README.en.md) | [27B detail page](27b.en.md) | [Large MoE detail page](moe-2.0.0.en.md) | [Changelog (Chinese)](../../CHANGELOG.md)

This page collects the measured tables of every release, newest first. Numbers are moved verbatim from the detail pages, READMEs and changelog of the time, nothing changed or added. The note under each table gives the test date, build/version, machine and methodology (thinking on or off, temperature and sampling); when a release's table is identical to the next release's (not re-measured, only carried over) it is not repeated and a single sentence says so. The "Performance at a glance" section of the home page picks the best results reproducible with the current release from here.

Releases: [2.0.1](#201) · [2.0.0](#200) · [1.1.1](#111) · [1.1.0](#110) · [1.0.9](#109) · [1.0.8](#108) · [1.0.7](#107) · [1.0.6](#106) · [1.0.5](#105) · [1.0.4](#104) · [1.0.3](#103) · [1.0.2](#102) · [1.0.1](#101) · [1.0.0](#100) · [Appendix: launch parameters of the 1.1.1 era](#appendix-launch-parameters-of-the-111-era)

---

## 2.0.1

<sub>Source: the "2.0.1 measured results" sections of docs/v100/moe-2.0.0.en.md and 27b.en.md, verbatim; methodology for each table (incl. thinking on/off and temperature) in the note below it.</sub>

### Large MoE (Qwen3.8-Flash-Next + MTP)

Conditions: IBM AC922 (2 × POWER9, 96 GB of host memory) + 6 × Tesla V100-SXM2-16GB; Qwen3.8-Flash-Next UD-Q4_K_XL with an MTP draft (Q4_K_M, 3 per round), 6 GPUs, context 262144. The prompts are about 200K tokens of real code, about 16K tokens of real code and a short prompt, with 1024 / 768 / 256 output tokens; **temperature 0, thinking off**. 2.0.1 and 2.0.0 were measured alternately in the same time window (same machine, same set of prompts), with a fresh server per case and the machine dedicated to the measurement. The 2.0.1 column includes the BF16 storage of the routing weights and the VRAM peak fix; the 2.0.0 column is a same-window measurement of the 2.0.0 release-line build and agrees with "Measured at the 2.0.0 release" below.

| Case | 2.0.0 | 2.0.1 | Change |
|---|---|---|---|
| Prefill, about 200K tokens (tok/s) | 2344 | 3464 | +48% |
| Prefill, about 16K tokens (tok/s) | 2053 | 2823 | +38% |
| Decode, about 200K tokens (tok/s) | 109.9 | 118.5 | +8% |
| Decode time per round, about 200K tokens (ms) | 25.60 | 23.04 | −10% |
| Decode, about 16K tokens (tok/s) | 117.9 | 139.6 | +18% |
| Decode time per round, about 16K tokens (ms) | 24.59 | 21.92 | −11% |
| Decode, short prompt (tok/s) | 135.5 | 151.0 | +11% |
| Decode time per round, short prompt (ms) | 26.62 | 23.89 | −10% |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (test conditions in the original).</sub>

Prefill of about 200K tokens is on par with the best figure measured for this project before (3459 tok/s). Acceptance was measured separately with 24 prompts and a Q4_K_M draft: accepted tokens per round went 2.752 → 2.734 at temperature 0 and 2.592 → 2.579 at temperature 0.6, both within measurement noise, while the time per round on the same prompt batch went 24.4 → 21.3 ms (temperature 0) and 26.1 → 23.2 ms (temperature 0.6).

VRAM peak per GPU (highest of all GPUs, MiB):

| Case | 2.0.0 | 2.0.1 |
|---|---|---|
| Large MoE, about 200K tokens | 14838 | 14804 |
| Large MoE, about 16K tokens | 14838 | 14804 |
| Large MoE, short prompt | 14655 | 14624 |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (test conditions in the original).</sub>

### 27B (Qwen3.8-27B + DFlash2)

Conditions: IBM AC922 (2 × POWER9, 96 GB of host memory) + 6 × Tesla V100-SXM2-16GB; Qwen3.8-27B Q8_0 with DFlash2 F16, 6 GPUs with tensor parallelism, context 262144. 2.0.1 and 2.0.0 were measured alternately in the same time window (same machine, same set of prompts, machine dedicated); the 2.0.0 column is a same-window measurement of the 2.0.0 release-line build. The three 27B requests ran in one server in the order "short prompt → about 16K tokens (temperature 0.8, not listed in the table) → about 200K tokens"; the ~200K-token prompt hit the cached prefix left by the previous request, so 195048 tokens were actually processed; **temperature 0, thinking off** (except the 16K case).

| Case | 2.0.0 | 2.0.1 | Change |
|---|---|---|---|
| Prefill, about 200K tokens (tok/s) | 2073 | 2140 | +3% |
| Decode, about 200K tokens (tok/s) | 126.4 | 128.9 | +2% |
| Decode time per round, about 200K tokens (ms) | 26.22 | 25.96 | −1% |
| Decode, short prompt (tok/s) | 216.0 | 225.8 | +5% |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (the ~16K-token request used temperature 0.8 and is not listed; test conditions in the original).</sub>

For the short prompt, accepted tokens per round are bit-identical between 2.0.0 and 2.0.1; at about 200K tokens the output of 2.0.0 differed between its two rounds (accepted 712/2205 and 718/2163), while 2.0.1 gave the same output in both rounds (718/2142) — the non-reproducible T=0 output of 2.0.0 across restarts on 6 GPUs at long context is fixed in this version.

VRAM peak per GPU (highest of all GPUs, MiB):

| Case | 2.0.0 | 2.0.1 |
|---|---|---|
| 27B | 9969 | 9963 |

<sub>Test data: 2026-10-17, 2.0.1 release build vs the 2.0.0 release-line build, same machine, measured alternately in the same time window; the 2.0.0 column is that same-window measurement; IBM AC922 + 6 × Tesla V100-SXM2-16GB; temperature 0, thinking off (the ~16K-token request used temperature 0.8 and is not listed; test conditions in the original).</sub>

## 2.0.0

<sub>Source: "Measured at the 2.0.0 release" in docs/v100/moe-2.0.0.en.md, "Measured at the 2.0.0 release (vs 1.1.1)" in 27b.en.md and the README.en.md at tag v100-2.0.0, verbatim; methodology for each table (incl. thinking on/off and temperature) in the note below it.</sub>

### Large MoE (Qwen3.8-Flash-Next + MTP)

Standard test machine: 2026-10-11, IBM Power AC922 (2 × POWER9, 96 GB memory) + 6 × Tesla V100-SXM2-16GB, CUDA 12.4, release build `4d50148cc`; UD-Q4_K_XL + MTP shared-Q4_K_M (3 drafted tokens per round) with the expert routing profile, using command A below; non-streaming, temperature 0, thinking off, a fresh server for every request (cold prefix), two rounds ("a / b" in the table are the two rounds; the output was identical in both). Real code = the source of an open-source C++ project plus a one-sentence summary request.

| Request | Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) | Draft acceptance (accepted / drafted) | VRAM per GPU after the request |
|---|---:|---:|---:|---:|---:|---:|
| Short prompt (a one-sentence coding task) | 33 | 256 | — | 134.1 / 134.4 | 185 / 213 (87%) | 14648–14652 MiB |
| Real code, ~16K tokens | 16030 | 768 | 2022.2 / 2069.1 | 117.4 / 116.3 | 503 / 795 (63%) | 14810–14835 MiB |
| Real code, ~200K tokens | 209030 | 686 (limit 1024, ended naturally) | 2334.8 / 2340.8 | 107.9 / 109.7 | 442 / 729 (61%) | 14793–14838 MiB |

<sub>Test data: 2026-10-11, 2.0.0 release build `4d50148cc`, IBM AC922 + 6 × Tesla V100-SXM2-16GB, CUDA 12.4; non-streaming, temperature 0, thinking off, fresh server per request (test conditions in the original).</sub>

- **Stability** (2026-10-12, release build, same command): 3 cold starts (the first request being ~16K tokens) all succeeded, prefill 2055.4–2060.9 tok/s; 41 mixed requests over 20 minutes (short prompts up to ~200K tokens, alternating temperature 0 and 0.7) with 0 failures, peak VRAM 14974 MiB on a single GPU. About 3 minutes from start to ready.
- **Above temperature 0**: in a long run where one server alternated temperature-0 and temperature-0.7 requests, temperature 0.7 lowered draft acceptance from 0.58–0.66 to 0.54–0.58 and decode by 12%–17% (measured).
- **Comparison with 27B on the same machine**: same machine, same build, same ~200K-token real code: the large MoE prefills about 13% faster than 27B (Q8_0 + DFlash2) and decodes at about 87% of 27B's speed.
- **Reference host: x86, 4 × V100-SXM2-32GB** (2026-10-11, all-to-all NVLink between GPUs, PCIe 3.0 between GPU and host, 136 GB memory, 1 NUMA node, CUDA 12.8; an earlier pre-release build `638de68da`, for reference only; command B, one server receiving three requests in turn, temperature 0): short-prompt decode 95.0 tok/s; ~16K tokens (768 output) prefill 2239, decode 88.2 tok/s; ~200K tokens (the first ~14K tokens hit the previous request's cache, 195048 tokens actually processed, ended naturally at 533 tokens) prefill 2434, decode 66.4 tok/s. All experts resident in VRAM; peak VRAM per GPU 24916–24970 MiB.

### 27B (Qwen3.8-27B + DFlash2)

The general multi-GPU runtime and kernel changes in 2.0.0 apply to 27B as well; the table compares against 1.1.1 in the same session. Conditions: 2026-10-11, IBM Power AC922 (2 × POWER9, 96 GB memory) + 6 × Tesla V100-SXM2-16GB, CUDA 12.4. 1.1.1 (tag `v100-1.1.1`) and 2.0.0 (release build `4d50148cc`) alternated in the same time window, two rounds each, with a fresh server per round, using command A below (Q8_0 + DFlash2 F16, 7 drafted tokens per round); each server received three non-streaming requests with thinking off. Real code = the source of an open-source C++ project plus a one-sentence summary request.

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

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB, build `4d50148cc`; context 262144, with MTP and the expert routing profile; about 200K tokens of real code, fresh server, temperature 0, thinking off, output ended naturally at 686 tokens): prefill 2334.8 / 2340.8 tok/s, decode 107.9 / 109.7 tok/s. On the same machine, the large MoE prefills ~200K tokens about 13% faster than 27B and decodes at about 87% of 27B's speed.

Measured (2026-10-11, IBM AC922 + 6 × V100-SXM2-16GB; 1.1.1 and 2.0.0 build `4d50148cc` alternated in the same time window, two rounds each; context 262144, DFlash2; conditions on the [27B detail page](27b.en.md#measured-results)): short-prompt decode 186.2 / 188.9 → 218.3 / 206.9 tok/s; ~16K-token prefill 2863.4 / 2842.7 → 3119.1 / 3121.1 tok/s; ~200K-token prefill 2047.8 / 2047.1 → 2066.6 / 2063.2 tok/s and decode 114.5 / 115.6 → 122.2 / 127.0 tok/s.

<sub>Source: README.en.md at tag v100-2.0.0, verbatim (2026-10-12); the first sentence is the large MoE, the second 27B; both temperature 0, thinking off (27B ~16K tokens at temperature 0.8, see the table above).</sub>

## 1.1.1

<sub>Source: the former docs/v100/27b-results-1.1.1.en.md (the "Test data" section of the 1.1.1 README with notes added), 2026-10-06; thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of the 1.1.1 README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
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

From 1.0.7 on, non-thinking mode (presence penalty) no longer falls back to host sampling; time per speculative round 26.2 -> 21.8 ms (the table was measured on 1.0.4).

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
| single request about 115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of about 115K tokens each (1.1.1) | 55.1 s / 118.9 s | 60.0 / 103.8 tok/s | 28 tok/s | 128.8 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each (1.1.1) | 8.2 / 23.2 / 39.0 / 52.5 s | 76.3 / 82.8 / 83.7 / 106.8 tok/s | 73 tok/s | 62.1 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4-12.8 GB per GPU, peak 12.7-13.0 GB (measured on 1.0.4; from 1.0.8 about 0.3 GB more per GPU for the attention-assist workspace):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each (1.1.1) | 8.0 s / 20.1 s | 94.0 / 104.3 tok/s | 94 tok/s | 29.9 s |
| two concurrent requests of about 115K tokens each (1.1.1) | 47.4 s / 101.6 s | 75.0 / 88.4 tok/s | 31 tok/s | 113.2 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on, T=0.6 (per the opening of this subsection; other sampling not recorded).</sub>

The same configuration with each slot filled to close to 256K (1.0.9, vision module split across the 6 GPUs, thinking mode, 12.44-12.60 GB per GPU after start):

| Scenario | First token | Prefill | Decode per slot | Per-GPU peak memory |
|---|---|---|---|---|
| two concurrent text requests of about 244K tokens each | 126 s / 251 s | 1964 / 1855 tok/s | 54.0 / 69.0 tok/s | 12.9-13.1 GB |
| two concurrent requests: about 232K tokens of text + image / about 235K tokens of text + video | 253.5 s / 146.4 s | — | 60.8 / 67.9 tok/s | 12.9-13.1 GB |

<sub>Test data: measured on 1.0.9 (2026-10-02), 6 GPUs Q8, `-c 524288 -np 2` (YaRN), vision module split by layer across 6 GPUs; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on (original says "thinking mode"; temperature and sampling not recorded).</sub>

`-c 1048576 -np 2` (524288 per slot) also starts and text works, but the vision module cannot allocate its compute buffer for images; to have 512K per slot together with images, put the vision module on the CPU with `--no-mmproj-offload` (encoding becomes much slower).

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
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 -> 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

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
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 Q2_K-Q6_K all use the small-batch kernels with gate/up and multi-weight fusion, about 10% faster Q4 decoding (see "What's new"; the Q4 rows in the table were measured on earlier versions).
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

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure, 1.1.1) | 3078 | 4.8 s | 145 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides, 1.1.1) | 23104 | 18.7 s | 173 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment"; IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; thinking on (original says "thinking mode"; temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs by layer: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6-11.8 GB (measured on 1.1.1; 1.0.7: 12.2 GB on GPU 0, 10.8-11.2 GB on the others). Unmarked rows in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.2 s and a 12-second video 18.5 s; peak VRAM 13.5 GB on every GPU, at most 26 MiB apart (measured on 1.1.1). PDF first 6 pages 3.2 s (measured on 1.1.0). 1.0.7: 14.2 GB on GPU 0.
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

### Measurements in the changelog

- 合并过程中发现并已消除的一处额外开销（发布版本没有这项开销，性能与 1.1.0 相同）：上游 v0.6.0 对普通批也会构建混合批（嵌入 + token）的输入分支，每批多 3 个主机输入张量及其节点，4 卡张量切分下每个 2048 批预填充多约 10 ms。现在混合批分支只在批中确实混合时才构建，这项开销不再存在；上游对递归状态复制的修复（#29856）保持不变。
- 与 1.1.0 同一时段交替测量：4/6 卡 T=0 输出逐字相同（每 token 5.66 / 5.65 ms）；4 卡真实 16K 预填充 2579.0 / 2585.4 tok/s；2 卡 Q4 262144 合成 209715 预填充 1082.6 / 1081.0 tok/s、每轮投机 43.7 / 43.7 ms；6 卡 524288 合成 419430 预填充 1505.9 / 1503.2 tok/s、吐字 239.6 / 239.4 tok/s；4 卡、6 卡两路并发首字相同；6 卡多模态图片、视频正常；各卡显存峰值相同。

<sub>Source: the 1.1.1 section of CHANGELOG.md, verbatim (Chinese only) (2026-10-06); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.1.0

<sub>Source: the "Test data" section of the README at tag v100-1.1.0 (2026-10-03); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
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

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking mode not recorded (temperature and sampling not recorded).</sub>

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

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.1.0) | 285.9 s | 1467 tok/s | 238.4 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking mode not recorded (temperature and sampling not recorded).</sub>

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
| single request about 115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of about 115K tokens each (1.1.0) | 55.1 s / 119.1 s | 58.7 / 81.0 tok/s | 27 tok/s | 131.8 s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking on (temperature and sampling not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each | 8.5 / 27.1 / 43.6 / 57.5 s | 64.0 / 69.9 / 81.9 / 124.4 tok/s | 72 tok/s | 65.7 s |

<sub>Test data: 2026-10-03 (date of tag v100-1.1.0), README at the 1.1.0 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4-12.8 GB per GPU, peak 12.7-13.0 GB (measured on 1.0.4; from 1.0.8 about 0.3 GB more per GPU for the attention-assist workspace):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each (1.1.0) | 8.1 s / 20.7 s | 89.7 / 122.5 tok/s | 98 tok/s | 29.0 s |
| two concurrent requests of about 115K tokens each (1.1.0) | 47.3 s / 103.0 s | 65.4 / 108.6 tok/s | 32 tok/s | 112.4 s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking on (temperature and sampling not recorded).</sub>

The same configuration with each slot filled to close to 256K (1.0.9, vision module split across the 6 GPUs, thinking mode, 12.44-12.60 GB per GPU after start):

(This table is the same as in the 1.1.1 section; not re-measured.)

`-c 1048576 -np 2` (524288 per slot) also starts and text works, but the vision module cannot allocate its compute buffer for images; to have 512K per slot together with images, put the vision module on the CPU with `--no-mmproj-offload` (encoding becomes much slower).

Compared with 1.0.3 (4 GPUs, same test):

(This table is the same as in the 1.1.1 section; not re-measured.)

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 -> 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

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

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 Q2_K-Q6_K all use the small-batch kernels with gate/up and multi-weight fusion, about 10% faster Q4 decoding (see "What's new"; the Q4 rows in the table were measured on earlier versions).
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8 (the table was measured on 1.0.6); from 1.0.8 6-GPU long-context prefill is faster: the 6-GPU Q8 long synthetic input is measured on 1.0.8, the 6-GPU Q4 row on an earlier version.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.1.1 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure, 1.1.0) | 3078 | 4.8 s | 171 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides, 1.1.0) | 23104 | 18.7 s | 148 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Data source: values marked 1.1.0 were measured on version 1.1.0 (2026-10-03, alternating with 1.0.9 in the same session); unmarked values are from the earlier versions noted below the table or in "Test environment".; thinking on (temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs by layer: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6-11.8 GB (measured on 1.1.0; 1.0.7: 12.2 GB on GPU 0, 10.8-11.2 GB on the others). Unmarked rows in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.3 s and the PDF first 6 pages 3.2 s; peak VRAM 13.5 GB on every GPU, at most 28 MiB apart (measured on 1.1.0; 1.0.7: 14.2 GB on GPU 0).
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.1.1 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context".

### Measurements in the changelog

- 修复 6 卡多路服务长上下文崩溃：多路同时处理各自超过 32768 的长提示词时（如带大图片的 Agent 会话），空闲 GPU 分担注意力的工作缓冲随多路切换而重建，ggml-cuda 的 CUDA Graph 缓存只比较子图张量、仍重放旧捕获，向已释放的显存写入（`illegal memory access` 或段错误）。CUDA Graph 缓存现在把挂载的分担描述符纳入是否重新捕获的判断。回归用例（6 卡 `-np 2`，两路各约 4 万 token + 6 张大图）：1.0.9 两次都在约 23 s 时崩溃，1.1.0 两次都完成。
- 与 1.0.9 同一时段交替测量：4/6 卡 T=0 输出逐字相同；6 卡 200K 合成预填充 2085.5 → 2082.0 tok/s、吐字 300.2 → 300.3 tok/s，6 卡 419430 1469.1 → 1467.0 tok/s；4 卡真实 16K 预填充 2554.5 → 2561.0 tok/s；4 卡 200K 合成（5 轮交替平均）预填充 1754.0 → 1752.2 tok/s、吐字 281.3 → 281.3 tok/s；3 卡、2 卡、并发、多模态持平，显存峰值逐卡相同。

<sub>Source: the 1.1.0 section of CHANGELOG.md, verbatim (Chinese only) (2026-10-03); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.9

<sub>Source: the "Test data" section of the README at tag v100-1.0.9 (2026-10-02); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
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

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

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

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.0.8) | 281.5 s | 1490 tok/s | 238.0 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

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
| single request about 115K tokens | 55.8 s | 99.9 tok/s | 100 tok/s | 66.0 s |
| two concurrent requests of about 115K tokens each | 58.0 s / 122.6 s | 70.2 / 81.4 tok/s | 27 tok/s | 135.2 s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

(This table is the same as in the 1.1.0 section; not re-measured.)

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4-12.8 GB per GPU, peak 12.7-13.0 GB (measured on 1.0.4; from 1.0.8 about 0.3 GB more per GPU for the attention-assist workspace):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each | 8.0 s / 21.9 s | 85.4 / 116.1 tok/s | 90 tok/s | 30.7 s |
| two concurrent requests of about 115K tokens each | 55.3 s / 118.8 s | 66.0 / 90.5 tok/s | 27 tok/s | 130.1 s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

Compared with 1.0.3 (4 GPUs, same test):

(This table is the same as in the 1.1.0 section; not re-measured.)

Notes:
- Prompt processing is queued sequentially, so with concurrent requests the TTFT of the later requests waits for the earlier ones to finish processing. The total time for all requests to finish is about the same as processing them one after another; the benefit of concurrent requests is that a request that has started decoding is not blocked by a later long prompt, and a later request does not have to wait for the previous one to finish decoding.
- A larger `--prefill-pace` makes later requests reach TTFT faster but slows down the request that is decoding; setting it to 0 or 100 means no throttling (the same as 1.0.3). For short prompts like 16K, the default 30 delays the second request's TTFT by a few seconds (16.7 -> 22.7 s in the table above).
- For a single long-context request (e.g. one agent using the full 256K), `-np 1` is still recommended; with multiple requests each request's context is a fraction of the total context.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

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

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8; from 1.0.9 Q2_K-Q6_K all use the small-batch kernels with gate/up and multi-weight fusion, about 10% faster Q4 decoding (see "What's new"; the Q4 rows in the table were measured on earlier versions).
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8 (the table was measured on 1.0.6); from 1.0.8 6-GPU long-context prefill is faster: the 6-GPU Q8 long synthetic input is measured on 1.0.8, the 6-GPU Q4 row on an earlier version.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.1.0 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | Prompt processing | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 182 tok/s |
| image (diagram) | 4069 | 5.7 s | 163 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 190 tok/s |
| 12-second video (slides) | 23104 | 19.7 s | 150 tok/s |
| 10-second video | 6387 | 7.3 s | 131 tok/s |
| 200K tokens of text + 1 image | 206958 | 122.7 s | 87 tok/s |

<sub>Test data: 2026-10-02 (date of tag v100-1.0.9), README at the 1.0.9 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

- From 1.0.8 the vision module is split across all GPUs by layer: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6-11.8 GB (1.0.7: 12.2 GB on GPU 0, 10.8-11.2 GB on the others), with unchanged processing time. Times in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.3 s and the PDF first 6 pages 3.2 s; peak VRAM 13.5 GB on every GPU, at most 26 MiB apart (1.0.8; 1.0.7: 14.2 GB on GPU 0).
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.1.0 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context".

### Measurements in the changelog

- Q4 混合量化的小批量矩阵乘：Q4_K 的小批量内核按量化类型做成 codec，扩展到 Q2_K、Q3_K、Q5_K、Q6_K（加载时原地重排、字节数不变，M=1～64）；Q4 配置启用 gate/up 融合（SwiGLU 直接写出）与多权重融合（2～4 个共享输入的权重一次启动）。与 1.0.8 同一时段交替测量：2 卡 Q4 合成 102400 吐字 193.0 → 212.2 tok/s（+10%）、预填充 1419 → 1438 tok/s，3 卡 Q4 吐字 220.1 → 240.9 tok/s（+9.5%）、预填充 1902 → 1929 tok/s；2 卡 Q4 真实内容 8 条提示词每 token 耗时 T=0 11.58 → 10.31 ms、T=0.6 11.07 → 10.43 ms；困惑度 2.4097 → 2.4057（`-ub 8`）、2.4067 → 2.4047（`-ub 32`）。Q5_K/Q6_K 改为 f16 计算，输出与 1.0.8 有细微差别。逐项消融：gate/up 与多权重融合单独约 +3%。
- q4_0 KV 缓存反量化向量化（grouped 验证注意力与 vec 路径，逐位不变），2 卡 Q4 100K 吐字约 +1.2%；修复 q4_1 / q5_1 KV 缓存在 SM70 上 32 位未对齐读取导致的崩溃。
- Q8 配置与 1.0.8 相同：4/6 卡 T=0 输出逐字相同，各项速度持平（4 卡真实 16K 预填充 5 轮交替 2576.4 / 2576.5 tok/s）。

<sub>Source: the 1.0.9 section of CHANGELOG.md, verbatim (Chinese only) (2026-10-02); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.8

<sub>Source: the "Test data" section of the README at tag v100-1.0.8 (2026-10-02); thinking mode is stated per table in the notes below each table.</sub>

The tables in these subsections are the same as in the 1.0.9 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "GPU counts and quantization"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- 6 卡长上下文预填充：6 卡时每个全注意力层有 2 张 GPU 未分到注意力头。上下文达到 32768 且一批不少于 256 个 token 时，这 2 张 GPU 从相邻 GPU 直接读取 KV 后段（不复制）、计算部分注意力，在原 GPU 上按 LSE 合并；切分点在设备上按实际 KV 长度计算。6 卡 524288 合成 209715 预填充 1800 → 2091 tok/s（+16%），419430 1200 → 1490 tok/s（+24%）；每卡显存多约 300 MiB。
- Split-D 预填充注意力：补回 K 双缓冲；GPU 未占满（CTA 数少于 4 倍 SM 数）时每个 KV 窗口分两段同时计算后合并。4 卡 200K 合成预填充 1637 → 1759 tok/s（+7.4%），真实 16K 2551 → 2578 tok/s。
- grouped 投机验证注意力小幅优化（单次 326 → 306 µs），4 卡 200K 吐字 278.4 → 281.2 tok/s。
- 视觉模块（mmproj）按层分到主模型使用的全部 GPU（2～6 卡），按各 GPU 加载后剩余显存连续分配、每张 GPU 至少一层，调度器按权重位置自动插入跨卡拷贝。最高一张 GPU 的显存峰值：6 卡 12220 → 11788、5 卡 13911 → 13444、4 卡 14151 → 13547、2 卡 14966 → 14528 MiB，各卡相差 ≤ 170 MiB；图片、视频编码耗时不变（同一图片/视频各重复 6 次取中位数），输出逐 token 相同。显存余量不足、处理图片时显存不够的配置，可降低上下文、改用 Q4，或用 `--no-mmproj-offload` 把视觉模块放 CPU。
- 底线：与 1.0.7 同一时段交替测量，4/6 卡 T=0 输出逐字相同；2 卡 Q4 100K 预填充 1415 → 1419 tok/s，3 卡 Q4 100K 1881 → 1902 tok/s，6 卡 16K 持平；多路并发首字相同或更快。

<sub>Source: the 1.0.8 section of CHANGELOG.md, verbatim (Chinese only) (2026-10-02); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.7

<sub>Source: the "Test data" section of the README at tag v100-1.0.7 (2026-10-01); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
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

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

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

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 363 s | 1154 tok/s | 233.1 tok/s | 12.8 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### GPU counts and quantization

The single-request table was measured on 1.0.6, the two-request table on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

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

<sub>Test data: 2026-10-01 (date of tag v100-1.0.7), README at the 1.0.7 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is still slower than with Q8: the Q4_K small-batch kernel covers a single matrix multiplication (M=17-64 included from 1.0.7); gate/up fusion and multi-weight fusion showed no gain and still go through the generic path.
- From 1.0.7 the attention load is balanced on 3 GPUs with Q8; see "What's new" for speed and VRAM (the table was measured on 1.0.6).

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.0.8 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

The tables in these subsections are the same as in the 1.0.8 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- 3 卡注意力负载均衡：注意力的 Q/gate/输出从每卡 6/6/12 个头改为 8/8/8；K/V 权重与缓存在相邻卡之间重叠分配（每卡 2 个 KV 头）。预填充每卡做两组 FlashAttention 后拼接，投机验证批每卡一次 FlashAttention（验证注意力内核支持每卡不均匀的 Q 头分界）。只在 3 卡、等比例切分时启用。1.0.5 的已知限制“3 卡注意力负载不均”在本版结案。3 卡 Q8、131072 与 1.0.6 交替测量：100K 预填充 1808 → 1972 tok/s（+9.0%）、吐字 232.8 → 239.7 tok/s（+3.0%），16K 预填充 2571 → 2675 tok/s（+4.1%）、吐字 282.8 → 291.5 tok/s（+3.1%）；困惑度不变（1.6557 / 1.6554）；每卡显存峰值 14800 → 15116 MiB。
- 长上下文预填充少占显存：Split-D 预填充注意力的 K/V f16 镜像从整份改为最多 131072 行的窗口，超出时逐窗反量化、计算部分结果并在线合并。262144 上下文每卡显存峰值：2 卡 Q4 14140 → 13678 MiB（−462），4 卡 Q8 13299 → 12835 MiB（−464），6 卡 Q8 10234 → 9694 MiB（−540）；预填充速度不变（2 卡 Q4 100K 提示词处理 70753 → 70853 ms，噪声内）。窗口取 131072 而不是 65536：后者显存再省约 250 MiB，但 2 卡 Q4 100K 预填充慢 0.46%。
- 多卡建图的主机开销：多卡切分状态结构从约 4 KB 缩到约 1 KB（段数上限 16 → 4），零碎批次重建计算图更快。4 卡真实 16K 提示词处理 6332 → 6309 ms（1.0.6 → 1.0.7）。
- Q4_K 小批量矩阵乘覆盖 M=17～64（此前 M>16 展开成 f16 走通用路径）：2 卡 Q4、`-ub 32` 时替掉约 0.68 s 的 f16 展开，困惑度 1.7340 → 1.7323。
- 非思考模式（presence penalty）：penalty 改在设备端 top-k 之前扣分，不再每轮落到主机采样。4 卡（presence 1.5、温度 0.7）每轮 26.2 → 21.8 ms（−17%），吐字 139 → 172 tok/s。
- 草稿模型量化对比（4 卡 Q8，各 2 遍）：每轮产出 token（短/16K/128K）F16 3.23/2.51/2.46、Q8_0 2.76/2.55/2.75、Q4_K_M 2.84/2.56/2.43；每轮投机耗时 F16 21.3/22.1/26.0 ms、Q8_0 与 Q4_K_M 约 20.6/21.3/25.3 ms；每卡显存峰值 F16 13.3、Q8_0 12.9、Q4_K_M 12.6 GB。短任务 F16 接受率明显更高，长上下文三者在噪声内；显存够用选 F16，紧张时 Q8_0/Q4_K_M 损失很小。
- 底线：与 1.0.0、1.0.6 同一时段交替测量（4 卡 Q8 262144，各 3 轮）：200K 合成预填充 1.0.0 1641.4 / 1.0.6 1640.9 / 1.0.7 1647.2 tok/s（+0.35%），每轮投机耗时 28.75 / 28.72 / 28.67 ms，每卡显存峰值 14333 / 13299 / 12835 MiB；真实 16K 预填充 2547 / 2551 / 2568 tok/s。6 卡 200K 合成预填充 1.0.6 1804 / 1.0.7 1808 tok/s，吐字 294.2 / 297.5 tok/s。4/6 卡 T=0 输出与 1.0.6 逐字相同。并发与 1.0.4 同一时段对照：各场景首字相同或更快（6 卡两路各约 11.5 万 token：55.2 / 117.7 → 51.8 / 110.5 s；4 卡两路各约 11.5 万：57.7 / 123.5 → 57.9 / 124.6 s）；6 卡两路结束时每卡显存约 12.99 → 12.02 GB；合计吐字随各路接受率波动（单次测量），例如 4 卡两路各 16K 82.4 → 86.9 tok/s、6 卡两路各 11.5 万 28.1 → 29.3 tok/s。

<sub>Source: the 1.0.7 section of CHANGELOG.md, verbatim (Chinese only) (2026-10-01); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.6

<sub>Source: the "Test data" section of the README at tag v100-1.0.6 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The single-request table in "GPU counts and quantization" was measured on 1.0.6, the two-request table on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30). On 4 GPUs with Q8, 1.0.5 and 1.0.6 produce word-for-word identical output to 1.0.4 with unchanged speed; on 6 GPUs prefill is faster from 1.0.6 on (see "What's new"), while decode and VRAM in the 6-GPU tables are unchanged.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

(This table is the same as in the 1.0.7 section; not re-measured.)

The tables in these subsections are the same as in the 1.0.7 section and were not re-measured: "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "GPU counts and quantization"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- 2、3、5、6 卡预填充提速：GDN 分块预填充支持任意每卡头数（此前只有每卡 12 头的 4 卡走分块，其他卡数退回逐 token 递归）。4 卡仍用原来的 TileLang 生成代码，输出逐位不变；其他头数用运行时参数，cumsum 改为通用扫描。6 卡 200K 合成预填充 1710 → 1801 tok/s（+5.4%），6 卡真实 16K 2568 → 2774 tok/s，2 卡 Q4 +8～10%，3 卡 Q8 +6%；困惑度不变。
- Q4 配置吐字提速：新增 Q4_K 权重的小批量矩阵乘内核（M=1～16，m8n8k4 HMMA），加载时把 Q4_K 原地重排、字节数不变；M>16（含预填充）展开成 f16 走 cuBLAS。解码按量化类型做成 codec，将来可扩展到其他 K 量化。Q4 配置每轮投机耗时降低 5～9%，输出有细微变化，困惑度不变（2 卡 1.7321 → 1.7320）。gate/up 融合、多权重融合暂不支持 Q4_K。
- 4 卡 Q8 配置的输出、速度与显存与 1.0.5 相同（与 1.0.0 同一时段交替对照：4 卡 200K 预填充 1641.0 对 1641.6 tok/s）；并发与 1.0.4 持平。

<sub>Source: the 1.0.6 section of CHANGELOG.md, verbatim (Chinese only) (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.5

<sub>Source: the "Test data" section of the README at tag v100-1.0.5 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "Concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- The "GPU counts and quantization" section was measured on 1.0.5 (2026-09-30); the other sections were measured on 1.0.4 (2026-09-29 to 30), and on those configurations 1.0.5 produces word-for-word identical output to 1.0.4 with unchanged speed.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### GPU counts and quantization

Measured on 1.0.5. Two configurations:
- **Q8**: target model Q8_0, KV cache q8_0 (`-ctk q8_0 -ctv q8_0`);
- **Q4**: target model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`).

The draft model is DFlash2 F16 by default; for 2 GPUs with Q4 to run 262144, use DFlash2 Q4_K_M instead (with the F16 draft the maximum is 131072). GPU indices: 2 GPUs 0, 1; 3 GPUs 0, 1, 2; 4 GPUs 0, 1, 3, 4; 5 GPUs 0-4; 6 GPUs 0-5. Each combination was tried downward from 524288, and the tables below give the highest context that can start; 524288 needs YaRN (see "Launch parameters").

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

<sub>Test data: 2026-09-30 (date of tag v100-1.0.5), README at the 1.0.5 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

- On 2 GPUs use the Q4 configuration: the Q8_0 target model is about 29 GB and does not fit on two 16 GB GPUs.
- 3 GPUs with Q8 cannot fit 262144 in VRAM (not even with the Q4_K_M draft); the maximum is 131072.
- Decode with the Q4 configuration is slower than with Q8: Q4_K weights currently go through the generic matrix multiplication path.

**Two concurrent requests** (`-np 2`, context per request = context / 2; a single 16K request and two concurrent requests of 16K each, 1024 output tokens):

(This table is the same as in the 1.0.6 section; not re-measured.)

Decode for real content varies a lot with the acceptance rate (single measurement); to compare engine speed, look at "time per speculative round" in the single-request table.

The tables in these subsections are the same as in the 1.0.6 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- 1.0.4 已知问题结案：“4 卡 200K 预填充比 1.0.3 慢约 0.2%”经同一时段交替测量（1.0.3 / 1.0.4 紧挨着各 4 轮），每个 2048 批只多 0.9～1.6 ms（0.07～0.13%），在本机测量误差内，也无法归到任何一个改动，判定不是回退。

<sub>Source: the 1.0.5 section of CHANGELOG.md, verbatim (Chinese only) (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.4

<sub>Source: the "Test data" section of the README at tag v100-1.0.4 (2026-09-30); thinking mode is stated per table in the notes below each table.</sub>

The tables in these subsections are the same as in the 1.0.5 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "6 GPUs, 262144 context"; "6 GPUs, 524288 context (YaRN)"; "Concurrent requests"; "Multimodal (image / PDF / video)".

### Measurements in the changelog

- 多路并发：新增 `--prefill-pace P`（默认 30）。有请求在吐字时，其他请求的提示词处理最多占 P% 的时间，吐字的请求不再被长提示词拖到十几 tok/s。4 卡 `-np 2` 两路各约 11.5 万 token 同时请求：先开始吐字的一路 11.4 → 70.2 tok/s，另一路首字 133.0 → 122.6 s。设为 0 或 100 与 1.0.3 相同（不限流）。
- 多路同时吐字：解码调度器的批次上限按路数扩大（单路仍为 16 个 token，多路每路 8 个、最多 64 个）；Q8_0 小批量矩阵乘内核新增 M=32 两阶段版本，支持 17～64 行；区间注意力掩码支持每路一个 stream。4 卡 `-np 2` 两路各 16K 同时请求：每路吐字 45.7 / 58.8 → 81.4 / 111.4 tok/s。
- 多路少占显存：非 unified 的多路（每路一个 stream）计算缓冲也按区间掩码预留。4 卡 `-c 262144 -np 2` 每卡启动显存 14031 → 13513 MiB；`-np 4`（每路 64K）从启动即显存不足变为可以运行；6 卡 `-c 524288 -np 2` 可以运行。
- 已知问题：4 卡 256K 上下文、合成 209715 token 的预填充比 1.0.3 慢约 0.2%（同一时段各测 4 次，平均 1640.6 → 1637.6 tok/s，每个请求约多 0.2 s），吐字每轮耗时不变。原因待查，计划在 1.0.5 修复。

<sub>Source: the 1.0.4 section of CHANGELOG.md, verbatim (Chinese only) (2026-09-30); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.3

<sub>Source: the "Test data" section of the README at tag v100-1.0.3 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "two concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- 4-GPU speeds were measured on 1.0.0-1.0.2 and peak VRAM on 1.0.2 (on 4 GPUs, 1.0.3 produces output identical to 1.0.2 at the same speed); 6-GPU data was measured on 1.0.3.
- The decode speed for real content varies with the acceptance rate; some tables also give the time per speculative round.

### 4 GPUs, 262144 context

| Input | Prefill | Decode | Peak VRAM per GPU |
|---|---:|---:|---:|
| synthetic 209715 tokens (512 output tokens) | 1652 tok/s | 281.5 tok/s | 13.3 GB |
| real code 209115 tokens (1024 output tokens) | 1646 tok/s | 28.3 ms per speculative round | 13.3 GB |
| synthetic 131072 tokens (512 output tokens) | 1997 tok/s | 310.4 tok/s | 13.3 GB |
| real code 131188 tokens (1024 output tokens) | 1983 tok/s | 97.6 tok/s | 13.3 GB |
| real code 16499 tokens (1024 output tokens) | 2570 tok/s | 123.5 tok/s | 13.3 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2589 tok/s | 114.0 tok/s | 0.21 | 21.8 ms |
| thinking mode | 128K | 1985 tok/s | 99.1 tok/s | 0.22 | 25.7 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2412 tok/s | 114.6 tok/s | 0.29 | 26.4 ms |
| non-thinking mode | 128K | 1974 tok/s | 101.2 tok/s | 0.29 | 30.1 ms |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking and non-thinking modes; temperature and sampling per the "Mode" column (model-card recommended values).</sub>

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1799 tok/s, decode 278.0 tok/s |
| 8192 | not enough VRAM, fails to start |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

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

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 363 s | 1155 tok/s | 232.9 tok/s | 12.7 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 232.2 tok/s | 13.9 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### 4 GPUs, two concurrent requests

`-c 262144 -np 2` (131072 per request), measured sequentially in the same server, peak VRAM 14.5-14.7 GB per GPU:

| Scenario | TTFT | Decode per request | Total decode |
|---|---|---:|---:|
| single request 16K | 6.45 s | 117.1 tok/s | 117 tok/s |
| two concurrent requests of 16K each | 9.1 s / 16.5 s | 45.2 / 68.3 tok/s | 90 tok/s |
| single request about 115K tokens | 55.8 s | 101.5 tok/s | 102 tok/s |
| two concurrent requests of about 115K tokens each | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-c 524288 -np 2` (262144 per request) does not fit in VRAM and fails to start.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | TTFT | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.1 s | 148 tok/s |
| image (diagram) | 4069 | 5.7 s | 186 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.6 s | 182 tok/s |
| 12-second video (slides) | 23104 | 19.7 s | 181 tok/s |
| 10-second video | 6387 | 7.4 s | 179 tok/s |
| 200K tokens of text + 1 image | 206958 | 123 s | 100 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.3), README at the 1.0.3 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed).
- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 232.6 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: a one-image request takes 7.6 s and the PDF first 6 pages 5.5 s (both include answer generation); peak VRAM about 14.7 GB on GPU 0.
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

### Measurements in the changelog

multi-GPU communication optimized for the NVLink topology.0.2 的方式。6 卡每轮投机耗时降低 6～9%（例如 256K 上下文、合成 209715 token、输出 512：吐字 272.4 → 293.6 tok/s），预填充速度和显存占用不变。
- 6 卡输出层按 32 行对齐切分到各卡，解码时输出层改用 Q8_0 小批量矩阵乘内核，每轮耗时再降约 0.4%。

<sub>Source: the 1.0.3 section of CHANGELOG.md, verbatim (Chinese only) (2026-09-28); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.2

<sub>Source: the "Test data" section of the README at tag v100-1.0.2 (2026-09-28); thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Test data" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 6 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4. The 4-GPU tests use GPUs 0, 1, 3, 4 (two per CPU).
- Model: Qwen3.8-27B, Q8_0 GGUF; speculative decoding uses the DFlash2 draft model (F16), drafts 7 tokens per round.
- Launch parameters: see the corresponding configuration in the "Launch parameters" section.
- Except for "two concurrent requests" and "Multimodal", every case is measured on a freshly started server; the cases in those two sections are measured sequentially in the same server.
- Speeds were measured on 1.0.0-1.0.2; peak VRAM was measured on 1.0.2.
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

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens) | 360 s | 1165 tok/s | 220.2 tok/s | 12.7 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1309 tok/s | 218.1 tok/s | 13.9 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking mode not recorded (temperature and sampling not recorded).</sub>

`-ub 8192` does not fit in VRAM and fails to start. Only speed and VRAM were measured; output quality above 262144 was not tested. The YaRN setting applies to all lengths.

### Multimodal (image / PDF / video)

6 GPUs, 524288 context (YaRN), vision module (mmproj) on GPU, thinking mode, measured sequentially in the same server:

| Input | Tokens | TTFT | Decode |
|---|---:|---:|---:|
| image (paper figure) | 3078 | 5.0 s | 136 tok/s |
| image (diagram) | 4069 | 5.6 s | 126 tok/s |
| PDF first 6 pages (one image per page) | 3590 | 3.5 s | 151 tok/s |
| 12-second video (slides) | 23104 | 19.5 s | 164 tok/s |
| 10-second video | 6387 | 7.3 s | 160 tok/s |
| 200K tokens of text + 1 image | 206958 | 122 s | 91 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.2), README at the 1.0.2 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table.; thinking on (temperature and sampling not recorded).</sub>

- Appending 8030 tokens to the same session after 208K tokens: TTFT 7.7 s (only the new part is processed).
- With the vision module loaded, text-only synthetic 419430 tokens (512 output tokens): prefill 1154 tok/s, decode 217.9 tok/s; peak VRAM 13.6 GB on GPU 0 and 12.7 GB on the others.
- 4 GPUs, 262144 context, vision module on GPU: a one-image request takes 7.6 s and the PDF first 6 pages 5.5 s (both include answer generation); peak VRAM about 14.7 GB on GPU 0.
- With the vision module on CPU (`--no-mmproj-offload`): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

The tables in these subsections are the same as in the 1.0.3 section and were not re-measured: "4 GPUs, 262144 context"; "4 GPUs, sampling parameters recommended by the model card"; "4 GPUs, batch size (`-ub`), 262144 context"; "4 GPUs, two concurrent requests".

### Measurements in the changelog

- 长上下文少占显存：计算缓冲不再为用不到的部分预留空间。256K 上下文每卡少占约 1 GB 显存，512K 约 2 GB，主机锁页内存也少占同样多。

<sub>Source: the 1.0.2 section of CHANGELOG.md, verbatim (Chinese only) (2026-09-28); the changelog does not state thinking mode or sampling (thinking mode not recorded).</sub>

## 1.0.1

<sub>Source: the "性能" (Performance) section of the README at tag v100-1.0.1 (2026-09-28). That release had no English README; the text below is translated from the Chinese, numbers unchanged; thinking mode is stated per table in the notes below each table.</sub>

### Test environment (opening of the "Performance" section of that README)

Test environment:
- Server: IBM Power AC922 (2 x POWER9), 4 x Tesla V100-SXM2-16GB (NVLink), CUDA 12.4.
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

- Thinking mode takes the fast device-side sampling path. Time per round is the same as at T=0.6; the difference in decode speed comes only from the acceptance rate.
- In non-thinking mode, `presence_penalty` makes sampling fall back to the full vocabulary, about 4.5 ms (17~21%) slower per round; this kind of content has a higher acceptance rate, so the decode speed ends up similar.

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

From 1.0.1 on, 6 GPUs are supported (`--tensor-split 1,1,1,1,1,1`, other parameters as above). Qwen's attention has only 4 KV heads, so with 6 GPUs two GPUs get no head in each attention layer; this split is not optimized yet and is the next direction. Current state (context 262144, every case on a freshly started server):

| `-ub` | Input | Prefill | Decode | Time per speculative round | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 209715 tokens (512 output tokens) | 1720 tok/s | 272.9 tok/s | 29.3 ms | 11.3 GB |
| 2048 | real code 209233 tokens (1024 output tokens) | 1719 tok/s | 84.3 tok/s | 29.1 ms | 11.3 GB |
| 2048 | real code 16496 tokens (1024 output tokens) | 2592 tok/s | 135.3 tok/s | 22.6 ms | 11.3 GB |
| 4096 | synthetic 209715 tokens (512 output tokens) | 1912 tok/s | 273.6 tok/s | 29.2 ms | 13.5 GB |
| 4096 | real code 209229 tokens (1024 output tokens) | 1905 tok/s | 87.4 tok/s | 29.1 ms | 13.5 GB |
| 4096 | real code 16497 tokens (1024 output tokens) | 2731 tok/s | 115.7 tok/s | 22.6 ms | 13.5 GB |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

- Compared with 4 GPUs: 200K prefill is about 4% faster at `-ub 2048` and about 16% faster at `-ub 4096`; decode is about 3% slower per round (more GPUs means more communication, and two GPUs are idle in the attention layers).
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
| single request of about 115K tokens | 55.8 s | 101.5 tok/s | 102 tok/s |
| two requests of about 115K tokens each, sent together | 58.8 s / 131 s | 11.8 / 63.1 tok/s | 23 tok/s |

<sub>Test data: 2026-09-28 (date of tag v100-1.0.1), README at the 1.0.1 release, IBM AC922 + Tesla V100-SXM2-16GB, CUDA 12.4; methodology per this section's test environment and the notes below the table; thinking mode not recorded (temperature and sampling not recorded).</sub>

Two concurrent requests are currently slower than one:
- Prefill is essentially queued; the second request waits until the first finishes prefill.
- While one request is decoding and the other is in prefill, the decoding request waits for the other's prefill chunk every round, and its speed drops markedly.

So a single slot (`-np 1`) is currently recommended; concurrency optimization is listed under "Next steps".

## 1.0.0

<sub>Source: the "性能" (Performance) section of the README at tag v100-1.0.0 (2026-09-28); that release had no English README; thinking mode is stated per table in the notes below each table.</sub>

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
| `--split-mode tensor --tensor-split 1,...,1` | tensor parallelism, evenly split across GPUs |
| `-c` | context length |
| `-np` | number of concurrent requests; context per request = `-c` divided by the number of requests |
| `--prefill-pace` | with multiple requests, the maximum percentage of time prompt processing of other requests may take while a request is decoding (0-100, default 30; 0 or 100 means no throttling) |
| `-fa on` | Flash Attention (required by the SM70 attention kernels) |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache; `-ctk q4_0 -ctv q4_0` is 4-bit, using half the VRAM |
| `-b 2048 -ub 2048` | batch size; the test data uses 2048 unless noted otherwise |
| `--model-draft ...`, `--spec-type draft-dflash`, `--spec-draft-n-max 7` | DFlash2 speculative decoding, drafts 7 tokens per round |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` | use YaRN to extend the context to 524288 (the model's native length is 262144) |
| `--override-kv qwen35.context_length=int:524288` | allow a single-slot context beyond the model's training length |
| `--mmproj` | vision module |
| `--video-ffmpeg-dir` | directory containing `ffmpeg` and `ffprobe`, needed for video |
| `--no-mmproj-offload` | run the vision module on CPU (by default it is split by layer across all GPUs used by the main model). Saves GPU memory, but image and video processing becomes much slower |
| `-mmdev <device>` | put the vision module on one given GPU only (see `--list-devices` for device names, e.g. `CUDA0`); `-mmdev none` is the same as CPU |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--chat-template-kwargs` | default sampling parameters and thinking mode switch (the values above are the model card's thinking mode parameters) |
| `--metrics` | Prometheus metrics (optional) |
| `--cache-ram <MiB>` | Upper limit of the prompt cache kept in host memory (upstream parameter, default 8192). When several conversations take turns, their state can be restored from host memory instead of being recomputed. Choose it from the host memory size: host memory use is roughly base use + `--cache-ram` + slots × `--ctx-checkpoints` × checkpoint size; going over the host memory gets the process killed by the system |
| `--ctx-checkpoints <N>` | Recurrent-state checkpoints kept per slot (upstream parameter, default 32), used to reuse prefixes; each checkpoint takes one recurrent state of host memory (size depends on the model), multiplied by the number of slots |
| `reasoning_effort` (request parameter) | Thinking effort, interpreted by the model's chat template. The Qwen3.8 template accepts `low`, `medium`, `high`, `xhigh` (default), and `none` turns thinking off; `--chat-template-kwargs '{"reasoning_effort":"medium"}'` sets a server default |
| `NCCL_P2P_LEVEL` (environment variable) | On ppc64le the program sets it to `SYS` before initializing NCCL so that GPUs on different CPU sockets also use P2P; a value you set yourself is not overridden |

Sampling: when `top_k ≤ 64` and only top-k, top-p, min-p, and temperature are used, the top-k candidates are selected on the device; repeat/presence/frequency penalties also select top-k candidates on the device (candidate count = `top_k` plus the tokens inside the penalty window, capped at 256); when DRY or other samplers are used, or `top_k` is larger, sampling over the full vocabulary is still done on the host.

<sub>Parameter reference table, not measured data; source: the README at the 1.1.1 release (2026-10-06); thinking mode not applicable.</sub>
