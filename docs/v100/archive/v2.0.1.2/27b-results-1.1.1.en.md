# Qwen3.8-27B + DFlash2: measured results per configuration from the 1.1.1 era (archive)

[简体中文](27b-results-1.1.1.md) | **English** | [← 27B detail page](27b.en.md) | [← Home README](../../README.en.md) | [Changelog (Chinese)](../../CHANGELOG.md)

This page archives the 27B measurements taken at the 1.1.1 release across configurations: 4 GPUs / 6 GPUs / 6 GPUs with a 524288 context, the model card's sampling parameters, batch-size comparison, concurrency, the 2–6 GPU and Q4 quantization compatibility table, multimodal, plus the matching launch parameters. These configurations were **not re-measured one by one on 2.0.x** (the 2.0.x multi-GPU runtime and kernel changes apply to them as well; the current 6-GPU 262144 numbers are on the [27B detail page](27b.en.md#measured-results)). The small print under each table says which version and date the data comes from. Whenever a configuration is re-measured on a new version, the new table goes to the detail page and this page keeps the historical values.

---

## Test data (at the 1.1.1 release)

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

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

**Draft model quantization** (4 GPUs, Q8, 2 runs each):

| Draft model | Tokens produced per round (short / 16K / 128K) | Time per speculative round (short / 16K / 128K) | Peak VRAM per GPU |
|---|---:|---:|---:|
| F16 | 3.23 / 2.51 / 2.46 | 21.3 / 22.1 / 26.0 ms | 13.3 GB |
| Q8_0 | 2.76 / 2.55 / 2.75 | 20.6 / 21.3 / 25.3 ms | 12.9 GB |
| Q4_K_M | 2.84 / 2.56 / 2.43 | 20.6 / 21.3 / 25.3 ms | 12.6 GB |

On short tasks F16 has a clearly higher acceptance rate; at long context the three are within noise. Use F16 when VRAM allows; when it is tight, Q8_0/Q4_K_M lose little.

### 4 GPUs, sampling parameters recommended by the model card

Real code, 1024 output tokens:

| Mode | Input | Prefill | Decode | Acceptance rate | Time per speculative round |
|---|---|---:|---:|---:|---:|
| thinking mode (T=1.0, top_p 0.95, top_k 20) | 16K | 2552 tok/s | 116.5 tok/s | 0.22 | 22.0 ms |
| thinking mode | 128K | 1963 tok/s | 97.6 tok/s | 0.22 | 25.9 ms |
| non-thinking mode (T=0.7, top_p 0.8, top_k 20, presence_penalty 1.5) | 16K | 2620 tok/s | 105.4 tok/s | 0.25 | 26.0 ms |
| non-thinking mode | 128K | 1971 tok/s | 89.1 tok/s | 0.24 | 29.8 ms |

From 1.0.7 on, non-thinking mode (presence penalty) no longer falls back to host sampling; time per speculative round 26.2 -> 21.8 ms (the table was measured on 1.0.4).

### 4 GPUs, batch size (`-ub`), 262144 context

| `-ub` | Result |
|---|---|
| 2048 | runs normally, peak VRAM 13.3 GB per GPU |
| 4096 | runs normally, peak VRAM 14.5 GB per GPU; synthetic 209715 tokens (512 output tokens): prefill 1801 tok/s, decode 277.1 tok/s |
| 8192 | not enough VRAM, fails to start |

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

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

### 6 GPUs, 524288 context (YaRN)

| `-ub` | Input | TTFT | Prefill | Decode | Peak VRAM per GPU |
|---|---|---:|---:|---:|---:|
| 2048 | synthetic 419430 tokens (512 output tokens, 1.1.1) | 278.9 s | 1504 tok/s | 238.8 tok/s | 11.5 GB |
| 4096 | synthetic 419430 tokens (512 output tokens) | 320 s | 1312 tok/s | 230.3 tok/s | 13.9 GB |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

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

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

**4 GPUs, `-c 262144 -np 4` (65536 per request)**, startup VRAM 14.2 GB per GPU, peak 14.3 GB:

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.5 s | 116.0 tok/s | 116 tok/s | 15.3 s |
| four concurrent requests of 16K each (1.1.1) | 8.2 / 23.2 / 39.0 / 52.5 s | 76.3 / 82.8 / 83.7 / 106.8 tok/s | 73 tok/s | 62.1 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

**6 GPUs, `-c 524288 -np 2` (YaRN, 262144 per request)**, startup VRAM 12.4-12.8 GB per GPU, peak 12.7-13.0 GB (measured on 1.0.4; from 1.0.8 about 0.3 GB more per GPU for the attention-assist workspace):

| Scenario | TTFT | Decode per request | Aggregate decode | All done |
|---|---|---:|---:|---:|
| single request 16K | 6.4 s | 118.5 tok/s | 119 tok/s | 15.0 s |
| two concurrent requests of 16K each (1.1.1) | 8.0 s / 20.1 s | 94.0 / 104.3 tok/s | 94 tok/s | 29.9 s |
| two concurrent requests of about 115K tokens each (1.1.1) | 47.4 s / 101.6 s | 75.0 / 88.4 tok/s | 31 tok/s | 113.2 s |

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

The same configuration with each slot filled to close to 256K (1.0.9, vision module split across the 6 GPUs, thinking mode, 12.44-12.60 GB per GPU after start):

| Scenario | First token | Prefill | Decode per slot | Per-GPU peak memory |
|---|---|---|---|---|
| two concurrent text requests of about 244K tokens each | 126 s / 251 s | 1964 / 1855 tok/s | 54.0 / 69.0 tok/s | 12.9-13.1 GB |
| two concurrent requests: about 232K tokens of text + image / about 235K tokens of text + video | 253.5 s / 146.4 s | — | 60.8 / 67.9 tok/s | 12.9-13.1 GB |

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

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

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

<sub>Data source: values marked 1.1.1 were measured on version 1.1.1 (2026-10-06; alternating with 1.1.0 in the same session, all items unchanged); unmarked values are from the earlier versions noted below the table or in "Test environment".</sub>

- From 1.0.8 the vision module is split across all GPUs by layer: after image and video requests the per-GPU VRAM peak on 6 GPUs is 11.6-11.8 GB (measured on 1.1.1; 1.0.7: 12.2 GB on GPU 0, 10.8-11.2 GB on the others). Unmarked rows in the table were measured on 1.0.4.
- 4 GPUs, 262144 context, vision module on GPU: image prompt processing 4.2 s and a 12-second video 18.5 s; peak VRAM 13.5 GB on every GPU, at most 26 MiB apart (measured on 1.1.1). PDF first 6 pages 3.2 s (measured on 1.1.0). 1.0.7: 14.2 GB on GPU 0.
- Appending 8034 tokens to the same session after 209K tokens: TTFT 7.7 s (only the new part is processed; measured on 1.0.3).
- With the vision module on CPU (`--no-mmproj-offload`, measured on 1.0.1): the same two images take 150 s and 232 s to first token, and a 12-second video 649 s.

---

---

## Launch parameters (1.1.1 era, matching the tables above)

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

---
