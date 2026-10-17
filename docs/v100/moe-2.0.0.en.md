# Qwen3.8-Flash-Next (large MoE) + MTP

[简体中文](moe-2.0.0.md) | **English** | [← Home README](../../README.en.md) | [27B dense model](27b.en.md) | [large MoE](moe-2.0.0.en.md) | [Changelog (Chinese)](../../CHANGELOG.md)

2.0.0 adds V100-specific support for the large MoE model Qwen3.8-Flash-Next. This page covers: supported configurations and implementation overview, measured results, model download, official sampling recommendations, launch commands, the expert routing profile, host memory and VRAM, build and Docker, and known limitations. The last section, "[Quick deployment (for AI assistants and beginners)](#quick-deployment-for-ai-assistants-and-beginners)", gives a parameter-impact table, decision steps and a symptom-to-fix table.

> **Supported configurations**: the standard configuration is 6 × V100-16GB + the UD-Q4_K_XL main model + the shared-Q4_K_M MTP draft, context 262144 (every measurement below uses it); 4 × V100-32GB (x86) runs with all experts resident in VRAM; other GPU counts and quantizations will follow.

---

## Model and implementation overview

Structure from the official model card ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)): a 125B-parameter language model with about 6B activated per token; plus a 51B-parameter n-gram embedding table (at layer 2, looked up by bigrams and trigrams) and a 4B-parameter MTP layer (1 layer); 48 layers, each group of 4 being 3 Gated DeltaNet (linear attention) layers plus 1 Qwen Sparse Attention (QSA) layer; 512 experts per layer, 10 routed experts per token plus 1 shared expert; native context 262144. The official model has a vision encoder; this project only tested text input. What this project does on V100:

- **Multiple GPUs and memory**: GPUs work together under `--split-mode tensor`; at startup the program decides from each GPU's free VRAM what stays in VRAM, and the rest lives in host memory while still being computed on the GPU, with no need for `--n-cpu-moe` / `-ot`. An expert routing profile (`--expert-profile`) keeps the most used parts in VRAM first.
- **Sparse attention and MTP speculative decoding**: both have V100-specific implementations, so a 262144 context runs on 16 GB GPUs; the MTP draft window and draft physical batch can be set separately (`--spec-draft-window`, `--spec-draft-ubatch`).
---

## Measured results

### 2.0.1 measured results (alternated with 2.0.0 in the same window)

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

Prefill of about 200K tokens is on par with the best figure measured for this project before (3459 tok/s). Acceptance was measured separately with 24 prompts and a Q4_K_M draft: accepted tokens per round went 2.752 → 2.734 at temperature 0 and 2.592 → 2.579 at temperature 0.6, both within measurement noise, while the time per round on the same prompt batch went 24.4 → 21.3 ms (temperature 0) and 26.1 → 23.2 ms (temperature 0.6).

VRAM peak per GPU (highest of all GPUs, MiB):

| Case | 2.0.0 | 2.0.1 |
|---|---|---|
| Large MoE, about 200K tokens | 14838 | 14804 |
| Large MoE, about 16K tokens | 14838 | 14804 |
| Large MoE, short prompt | 14655 | 14624 |

### Measured at the 2.0.0 release

Standard test machine: 2026-10-11, IBM Power AC922 (2 × POWER9, 96 GB memory) + 6 × Tesla V100-SXM2-16GB, CUDA 12.4, release build `4d50148cc`; UD-Q4_K_XL + MTP shared-Q4_K_M (3 drafted tokens per round) with the expert routing profile, using command A below; non-streaming, temperature 0, thinking off, a fresh server for every request (cold prefix), two rounds ("a / b" in the table are the two rounds; the output was identical in both). Real code = the source of an open-source C++ project plus a one-sentence summary request.

| Request | Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) | Draft acceptance (accepted / drafted) | VRAM per GPU after the request |
|---|---:|---:|---:|---:|---:|---:|
| Short prompt (a one-sentence coding task) | 33 | 256 | — | 134.1 / 134.4 | 185 / 213 (87%) | 14648–14652 MiB |
| Real code, ~16K tokens | 16030 | 768 | 2022.2 / 2069.1 | 117.4 / 116.3 | 503 / 795 (63%) | 14810–14835 MiB |
| Real code, ~200K tokens | 209030 | 686 (limit 1024, ended naturally) | 2334.8 / 2340.8 | 107.9 / 109.7 | 442 / 729 (61%) | 14793–14838 MiB |

- **Stability** (2026-10-12, release build, same command): 3 cold starts (the first request being ~16K tokens) all succeeded, prefill 2055.4–2060.9 tok/s; 41 mixed requests over 20 minutes (short prompts up to ~200K tokens, alternating temperature 0 and 0.7) with 0 failures, peak VRAM 14974 MiB on a single GPU. About 3 minutes from start to ready.
- **Above temperature 0**: in a long run where one server alternated temperature-0 and temperature-0.7 requests, temperature 0.7 lowered draft acceptance from 0.58–0.66 to 0.54–0.58 and decode by 12%–17% (measured).
- **Comparison with 27B on the same machine**: same machine, same build, same ~200K-token real code: the large MoE prefills about 13% faster than 27B (Q8_0 + DFlash2) and decodes at about 87% of 27B's speed.
- **Reference host: x86, 4 × V100-SXM2-32GB** (2026-10-11, all-to-all NVLink between GPUs, PCIe 3.0 between GPU and host, 136 GB memory, 1 NUMA node, CUDA 12.8; an earlier pre-release build `638de68da`, for reference only; command B, one server receiving three requests in turn, temperature 0): short-prompt decode 95.0 tok/s; ~16K tokens (768 output) prefill 2239, decode 88.2 tok/s; ~200K tokens (the first ~14K tokens hit the previous request's cache, 195048 tokens actually processed, ended naturally at 533 tokens) prefill 2434, decode 66.4 tok/s. All experts resident in VRAM; peak VRAM per GPU 24916–24970 MiB.

---

## Model download

**Users in mainland China should download from ModelScope.** All files are in the same unsloth repository:

| Purpose | File | ModelScope | Hugging Face |
|---|---|---|---|
| Target model | the 4 shards `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` in `UD-Q4_K_XL/` (~111 GB total) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| MTP draft (recommended) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (~1.9 GB) | same | same |
| MTP draft (optional) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (~2.8 GB) | same | same |

- Put the 4 shards in one directory and point `-m` at the first shard (`-00001-of-00004`). The expert routing profile `flash-next-profile.txt` comes with the Release (see "Expert routing profile").
- An MTP draft with `shared` in its name must be used with the same main model. unsloth recommends shared-Q8_0; this project uses the smaller shared-Q4_K_M: measured, shared-Q8_0 gives the same decode speed and acceptance, uses about 150 MiB more VRAM per GPU, and prefills ~200K tokens about 2% slower.
- `scripts/v100-get-models.sh` only downloads the 27B files; download the large MoE with direct links (resumable; on the AC922, wrap it in `numactl --membind=0,8` so page cache does not land in GPU memory nodes):

```bash
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
# Hugging Face: base=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O models/$f $base/$f
done
```

---

## Official sampling recommendations

The official Qwen model card ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Best Practices section; the unsloth GGUF model card says the same) gives two sets of recommended values. Append them to the launch command to make them server defaults; sampling parameters in a request take precedence:

| Mode | Official values | llama-server parameters |
|---|---|---|
| Thinking mode (model default) | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` | `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0` |
| Non-thinking (Instruct) mode | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 --chat-template-kwargs '{"enable_thinking":false}'` |

- The model card notes that `presence_penalty` can be adjusted between 0 and 2 to reduce endless repetition; high values may occasionally cause language mixing and slightly lower quality.
- Thinking switch: to turn thinking off, pass `"chat_template_kwargs": {"enable_thinking": false}` in the request, or set it as a server default with `--chat-template-kwargs`; thinking effort is the request parameter `reasoning_effort`, which the model card lists as `xhigh` (default), `medium` and `low`.
- The model card gives no MTP recommendation. The official [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) drafts 3 tokens per round (same as this page); unsloth's [MTP notes](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) suggest 2 per round, which this project has not tested.

---

## Recommended launch commands

### A. 6 × 16 GB GPUs (AC922), context 262144, with MTP (used for the measurements on this page)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 999 \
  --split-mode tensor \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --jinja \
  --expert-profile ./models/flash-next-profile.txt \
  --model-draft ./models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ubatch 256 --spec-draft-sampling greedy \
  --host 127.0.0.1 --port 8080 --metrics
```

- `numactl --membind=0,8` restricts host memory to the NUMA nodes that have CPUs (the AC922 onlines GPU memory into system memory; without it, page cache may occupy VRAM); check node numbers with `numactl --hardware`; plain x86 servers drop this part.
- Without a profile yet, drop the `--expert-profile` line and it still starts (experts are placed the default way, usually slower); without MTP, drop the `--model-draft` and `--spec-…` lines; to use the official default sampling, append one set from the table in the previous section. What each parameter does and its impact are in the parameter table in the last section.

### B. 4 × 32 GB GPUs (x86)

Command A without `numactl --membind=0,8` and with `CUDA_VISIBLE_DEVICES=0,1,2,3`; everything else stays the same. Four 32 GB GPUs hold all experts, so host memory only holds the n-gram embedding table and the token embedding table (still pinned, see "Host memory, VRAM and startup"); the profile then only affects which experts each GPU gets.

**Other GPU counts and VRAM sizes** have not been fully tested. At startup the plan follows each GPU's VRAM: the less VRAM in total, the more experts go to host memory, the more host memory is needed and the slower decode gets; if host memory is not enough, the server does not start. Just change `CUDA_VISIBLE_DEVICES`; if VRAM is short, lower `-c` first.

---

## MTP draft window

Available since 2.0.0.

The draft layer of MTP speculative decoding (`--spec-type draft-mtp`) attends only to the most recent positions instead of the whole main context. The window size is set with `--spec-draft-window N`; the default is 32768, and 0 means every position. The draft only proposes candidates; the emitted tokens are decided by the target model after verification, so the draft may see less context than the target.

| | |
|---|---|
| Option | `--spec-draft-window N` (environment variable `LLAMA_ARG_SPEC_DRAFT_WINDOW`) |
| Default | `32768`; `0` = the draft layer attends to every position |
| Negative values | rejected as an invalid value at startup |

The window is enabled by default because:

- Both the compute and the cache of the draft grow with the window size. Without a limit, the per-step draft compute and its cache grow with the main context, which slows down decoding at contexts of hundreds of thousands of tokens and takes VRAM.
- With the window, the per-step draft compute and cache have a fixed upper bound that does not depend on the main context length.
- While the prompt is shorter than the window, every position is visible to the draft and the output is token-for-token identical to a run without the option, so the default changes nothing for common usage.
- The typical configuration of this project runs contexts of tens of thousands to hundreds of thousands of tokens; enabling it by default makes the default configuration a working configuration.

Effect on VRAM and decoding:

- **VRAM**: the draft context cache is sized to the window plus one physical batch of slack, so the longer the main context and the smaller the window, the more is saved. The target model's own KV cache is not affected. With 0 the draft cache is sized to the main context length.
- **Decoding**: each draft step computes only the positions inside the window; the longer the context, the more visible. There is no difference when the context is within the window.
- **Prefill**: the draft only replays the positions inside the window; the prefix beyond the window is no longer replayed position by position.
- **Acceptance rate**: positions inside the window stay fully visible to the draft; earlier positions beyond the window are not, which can lower the acceptance rate at long contexts, partly offsetting the decoding gain. Acceptance rate and decoding figures are in "Measured results" above.

To turn the window off, or to change its size, add the option to the server command line:

```bash
llama-server ... --spec-type draft-mtp -md <draft file> --spec-draft-window 0      # off: the draft layer attends to every position
llama-server ... --spec-type draft-mtp -md <draft file> --spec-draft-window 8192   # example value: smaller window, smaller draft cache
```

This option only applies to the draft context of MTP speculative decoding. Speculative decoding that uses a separate draft model (for example DFlash2 on 27B) does not use it, and passing it there changes nothing.

---

## --fit-mmproj

New in 2.0.1.

Whether `--fit` counts the estimated VRAM of the multimodal projector (mmproj) into the per-GPU margin it leaves free (environment variable `LLAMA_ARG_FIT_MMPROJ`). The default is `on`: with `--fit`, the projector's estimated VRAM is counted into the per-GPU margin as before; with `off` it is not, and you must reserve the projector's memory yourself with `--fit-target`. Under tensor parallelism (`--split-mode tensor`) fit does not run at all, so this option has no effect there.

---

## Expert routing profile

The expert weights of an MoE model often add up to more than the VRAM, so only some experts can stay resident. The default placement ignores how often experts are used and gives every layer the same number of VRAM slots, but in real requests the experts are chosen at very different rates; with several GPUs the work per GPU is also uneven. `--expert-profile FILE` addresses both with a plain-text profile of three or four numbers per line, `layer expert count [launches]`:

- `count`: how many times the expert in that layer was routed to. At startup the experts of each layer are redistributed across GPUs by it so the GPUs get similar amounts of work.
- `launches` (optional): the number of matrix-multiplication launches routed to the expert; one launch counts once however many tokens it carries. During decode, every launch that uses an expert in host memory must first copy it into VRAM, so when this column is present each layer keeps the experts with the most `launches` (ties broken by `count`) resident in VRAM; if any line has a 4th column, the whole file is used this way. With only 3 columns, `count` decides; experts that do not get a VRAM slot go to host memory.

The profile `flash-next-profile.txt` that comes with the Release was collected from real requests; download it and pass it to `--expert-profile`. Two ways to make your own:

1. **From a calibration text** (build `llama-perplexity` first):

   ```bash
   # -f calibration text, -o output profile path, -c context length for sampling; arguments after -- go to llama-perplexity as is
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 bash scripts/v100-expert-profile.sh \
       -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
       -f calibration.txt -o ./models/flash-next-profile.txt -c 4096 \
       -- -sm tensor -ngl 999 -ctk q8_0 -ctv q8_0
   ```

   The script merges the counts written by each GPU into one profile, fills in never-routed experts (as 0), sorts by `layer expert` and checks that the line count equals "layers × experts per layer", exiting with an error otherwise; finally it prints the total routing count, the number of routed experts and the per-layer distribution, so you can judge whether the calibration text is representative. `--help` shows the full usage.
2. **From real load**: start `llama-server` with `--dump-expert-routing FILE`, use it normally for a while and exit normally; each GPU writes `FILE.dev<GPU number>` (4 columns); merge them with `cat FILE.dev* > profile.txt` and pass the result to `--expert-profile`. Killing the process midway loses that run's statistics.

- **A profile only decides where experts live; it does not change the math.** With a profile, the order of the multi-GPU sum changes with the expert distribution, so output may differ very slightly from runs without it while being numerically equivalent; the same profile gives the same result on every start. With a profile, startup takes a little longer to rearrange the experts.
- **Effect**: measured (a profile with only the `count` column, without MTP), the profile made ~16K-token decode 12% faster and cold ~200K-token prefill 8% faster, and raised the expert hit rate during decode from 82.6% to about 98%.
- **Collect per model and use case.** A profile records "layer number, expert number", so a new model needs a new profile; a profile does not depend on the GPU count. A profile collected from code works best for coding (measured prefill: code text 6.8% faster, English documents 3.6% faster); keep the calibration text close to the main use case.

---

## Host memory, VRAM and startup

- **Host memory**: the n-gram embedding table and token embedding table (27466 MiB) and the experts that do not fit in VRAM all live in pinned host memory; the standard 6 × 16 GB configuration pins about 42 GiB in total, plus headroom for the system, page cache and `--cache-ram`. Fewer GPUs or less VRAM need more host memory; if it does not fit, the server does not start (this version cannot read experts from disk).
- **Pinned memory limit**: the user running the server needs a sufficient locked-memory limit (`ulimit -l`, `unlimited` recommended; `--ulimit memlock=-1` in Docker), otherwise the server does not start.
- **NUMA**: experts in host memory are placed per CPU node the GPUs are attached to; the pinned memory is allocated and filled by threads on that node, and the placement is decided automatically at startup by measuring how fast each GPU reads each node; on the AC922, `numactl --membind=0,8` is still required.
- **VRAM**: the VRAM left after the rest of the model, the KV cache and the compute buffers is all used for expert slots, so per-GPU VRAM usage is close to the available VRAM; this does not mean VRAM is short. Do not share these GPUs with other programs. If VRAM runs out after running for a long time, raise `-fitt` (default 800 MiB). The MTP draft also takes part of each GPU's VRAM; when VRAM is very tight, run without MTP or lower `--spec-draft-ubatch`.
- **Startup time**: all shards are read and the n-gram embedding table and host-memory experts are copied into pinned memory; with a profile the experts are also rearranged; the standard configuration takes about 3 minutes.

---

## Build and Docker

Build as on the [27B page](27b.en.md#build); the profile script also needs `llama-perplexity`: `cmake --build build -j --target llama-server llama-quantize llama-perplexity`.

Usage of the Docker image `ghcr.io/1115714829/llama.cpp-v100:2.0.0` is on the [27B page, Docker section](27b.en.md#docker); the large MoE has not been measured under Docker. `--ulimit memlock=-1` is required and the host needs enough host memory; put the profile file in the mounted model directory. The image has no `llama-perplexity`, so making a profile from a calibration text has to happen in a source build; `--dump-expert-routing` works inside the container.

- **C. Docker, 6 × V100-16GB, AC922 (podman)**: prepare podman and CDI as in command C on the 27B page, put the arguments of command A from `-m` onward after the image name, change the model and profile paths to `/models/...`, and drop `--host 127.0.0.1`.
- **D. Docker, 4 × V100-32GB, x86**: the same, following command D on the 27B page (`--gpus all`, no `LLAMA_NUMA_MEMBIND`).

---

## Known limitations

- Only text input (the official model has a vision encoder), a single slot (`-np 1`) and the native 262144 context were tested.
- Combinations other than 6 × 16 GB and 4 × 32 GB have not been fully tested; Docker has not been measured.
- Experts live only in VRAM and host memory in this version; if host memory is not enough, the server does not start (reading experts from disk is not supported).
- The n-gram lookup reads host memory directly from the GPU; the asynchronous prefetch overlapped with layer 1 from the official design is not implemented.

---

## Quick deployment (for AI assistants and beginners)

You can hand this whole page to an AI assistant: pick command A–D with the decision steps below, replace the model paths with the actual directory, then adjust with the parameter table and the symptom table. Preparation: put the 4 main-model shards, the shared-Q4_K_M draft and the profile file in `<model dir>` (see "Model download"); `ulimit -l unlimited` for the user running the server; do not share these GPUs with other programs.

### Parameter table

The capability / speed / VRAM columns describe what happens when you change (or remove) the parameter from its recommended value; "measured" means measured on the standard test machine, "expected" means not measured.

| Parameter | What it does | Capability | Speed | VRAM | Recommended | When to change |
|---|---|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Which GPUs to use | The GPU count and VRAM per GPU decide whether it fits (see decision steps) | Expected: fewer GPUs put more experts in host memory and decode slower | — | All GPUs | When using only some GPUs |
| `numactl --membind=<CPU nodes>` | Restricts host memory to the NUMA nodes that have CPUs | — | — | Without it on the AC922, page cache from reading the model may land in GPU memory nodes, which CUDA cannot reclaim; shows up as out-of-memory | AC922: `0,8`; x86: omit | Always on machines like the AC922 that online GPU memory into system memory |
| `-m` | Main model, pointing at the first shard | Only UD-Q4_K_XL was tested in this version | — | — | UD-Q4_K_XL | Do not change |
| `-ngl 999`, `--split-mode tensor`, `-fa on`, `-ctk q8_0 -ctv q8_0`; no `--tensor-split` | All layers on GPU (experts that do not fit go to host memory automatically and are still computed on the GPU); routed experts distributed by expert, dense parts tensor-parallel; Flash Attention; 8-bit KV cache; even split | `--n-cpu-moe` / `-ot` are not needed; `layer` and other split modes and `q4_0` KV were not tested in this version | — | Without `-ctk/-ctv` the KV cache is F16 and the expert slots shrink accordingly | As written | Do not change |
| `-c` | Context length (model native 262144) | Prompt plus output of one request is at most this many tokens | Smaller: the freed VRAM goes to expert slots, fewer experts in host memory, faster decode (measured: at `-c 32768` only 881 MiB of experts remain in host memory) | 1437 MiB of context per GPU at 262144, less when smaller | 262144 | Lower it for faster decode when long context is not needed |
| `-b 2048 -ub 2048` | Logical / physical batch: tokens fed to the model per step during prefill | — | Larger is slower: measured (without MTP) `-ub 4096` gives −27% prefill on ~200K tokens and `8192` gives −43% (bigger compute buffers squeeze the expert slots) | 691 MiB compute buffer per GPU at 2048, more when larger | 2048 | Do not change |
| `-md` (`--model-draft`) | MTP draft file | Must be used with the same main model; the draft only proposes, the target model decides the output | Measured: shared-Q8_0 and shared-Q4_K_M give the same decode speed and acceptance; ~200K-token prefill about 2% slower with Q8_0 | Draft 316–330 MiB per GPU; shared-Q8_0 about 150 MiB more per GPU | shared-Q4_K_M | Do not change |
| `--spec-type draft-mtp` | Speculative decoding with the model's MTP layer | The draft only proposes; the target model decides the output | Faster decode (draft acceptance on this page: short prompt 87%, ~16K tokens 63%, ~200K tokens 61%); when on, the draft also processes the prompt | Draft, draft compute buffer and draft context take about 0.4–0.45 GB per GPU in total; expert slots shrink accordingly | On | Only when VRAM is really short: drop `-md` and all `--spec-…` parameters |
| `--spec-draft-n-max` | Maximum drafted tokens per round (default 3) | — | Measured: 3 is fastest; drafting more costs more in verification than it gains | Larger means a larger verification batch | 3 | Do not change (the official vLLM recipe also uses 3) |
| `--spec-draft-ubatch` | Physical batch size of the draft context (inherits `-ub` by default) | — | — | 93 MiB draft compute buffer per GPU at 256; 170 MiB when inheriting `-ub 2048` | 256 | Do not change |
| `--spec-draft-sampling` | `greedy`: the draft takes the most likely token (default); `probabilistic`: drafts by probability with rejection sampling by the target | With either method, output follows the target model's sampling with the request's parameters | Only `greedy` was tested on this page | — | `greedy` | Do not change |
| `--spec-draft-window` | The MTP draft layer attends only to the most recent N positions (default 32768, 0 = all) | — | Larger or 0: the draft gets slower at long context | Larger or 0: the draft KV grows | Omit (32768) | Do not change |
| `--expert-profile` | Expert routing profile: decides the VRAM-resident experts per layer and the experts each GPU gets (see "Expert routing profile") | Changes placement only, not the math | Slower without it: measured, the profile made ~16K-token decode 12% faster and cold ~200K-token prefill 8% faster | No change | A profile collected from content close to the actual use | Re-collect when the model or the main use case changes: run the server with `--dump-expert-routing <file>` on real load, exit normally, then `cat <file>.dev* > profile.txt` |
| `-fitt` (`--fit-target`) | Runtime VRAM headroom left on each GPU at startup (800 MiB by default for this model); the rest goes to expert slots | — | Larger: fewer expert slots, more experts in host memory, slower decode | Larger runtime headroom | Omit (800) | Raise it if VRAM runs out after running for a long time |
| `--cache-ram` | Host-memory prompt cache limit (MiB, default 8192, 0 = off) | Conversations used in turn can restore their prefix from host memory instead of recomputing it | Faster first token on a hit | No VRAM; uses host memory | Omit (8192) | Lower it when host memory is tight (the 6 × 16 GB configuration already pins about 42 GiB) |
| `--jinja`, `--chat-template-kwargs` | Use the model's own chat template; set server defaults for template parameters | Thinking is on by default, `'{"enable_thinking":false}'` turns it off; thinking effort is the request parameter `reasoning_effort` (`xhigh` default, `medium`, `low`) | Turning thinking off or lowering the effort generates fewer tokens and finishes sooner | — | `--jinja`; thinking by use case | Turn thinking off when not needed |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--repeat-penalty` | Server default sampling parameters (request parameters take precedence) | Official: thinking mode 1.0 / 0.95 / 20 / 0 / 0 / 1.0; non-thinking mode 0.7 / 0.8 / 20 / 0 / 1.5 / 1.0 (see "Official sampling recommendations") | Draft acceptance drops above temperature 0; measured at temperature 0.7, decode is 12%–17% lower | — | Official values | Pick the set for thinking / non-thinking mode |
| `--host`, `--port`, `--api-key-file`, `--metrics` | Listen address, port, API key file, Prometheus metrics | Default `127.0.0.1:8080`, local access only; `--host 0.0.0.0` opens it to other machines | — | — | Default for local use; `0.0.0.0` plus `--api-key-file` for external use | When other machines need access |
| `-np`, `--mmproj`, `--video-ffmpeg-dir`, `--rope-scaling yarn` and so on | Number of slots; vision module; YaRN context extension | For the large MoE, this version only tested a single slot, text input and the native 262144 context | — | — | `-np 1`, omit the others | Use 27B for images / video |

### Decision steps

```text
Step 1: how many V100s, and how much VRAM each?
  6 × 16 GB      → command A (measured on the AC922); on x86 drop the numactl part (not tested)
  4 × 32 GB      → command B (x86); on the AC922 add numactl --membind=<CPU nodes>
  other          → not fully tested; start from command A, change CUDA_VISIBLE_DEVICES, then check host memory in step 3
Step 2: image / video input, several slots, or context beyond 262144?
  yes            → not tested in this version; use 27B for images / video (see the 27B page)
  no             → -np 1, -c 262144; with short contexts only, a smaller -c gives more VRAM to expert slots and faster decode
Step 3: enough host memory? (pinned memory = the two tables, 27466 MiB + experts that do not fit in VRAM)
  6 × 16 GB      → about 42 GiB pinned, plus headroom for the system, page cache and --cache-ram
  4 × 32 GB      → all experts resident in VRAM, about 27466 MiB pinned
  other          → experts total about 73477 MiB; whatever does not fit in VRAM goes to host memory; if that does not fit, the server does not start
Step 4: IBM AC922 (GPU memory onlined into system memory)?
  yes            → prefix numactl --membind=<CPU nodes> (usually 0,8); with Docker use command C (-e LLAMA_NUMA_MEMBIND=0,8)
  no             → no numactl and no LLAMA_NUMA_MEMBIND; with Docker use command D
Step 5: do you have a profile file?
  yes            → --expert-profile <profile> (the Release includes flash-next-profile.txt)
  no             → start without --expert-profile first (slower), add --dump-expert-routing <file> and run real load,
                   exit normally, cat <file>.dev* > profile.txt, then restart with the profile
```

### Symptoms and fixes

| Symptom | Check first | Fix |
|---|---|---|
| Out of VRAM at startup (log shows `out of memory` or a failed VRAM allocation) | AC922: whether page cache has landed in GPU memory nodes (see the last row); other processes on the GPUs | Start with `numactl --membind=0,8`; dedicate the GPUs; lower `-c`; keep `--spec-draft-ubatch 256` and `-ub 2048`; if still short, drop MTP |
| Out of VRAM after running for a while | — | Raise `-fitt` (default 800) |
| Pinning fails or host memory runs out at startup | `ulimit -l`; free host memory | `ulimit -l unlimited`; `--ulimit memlock=-1` in Docker; 6 × 16 GB needs about 42 GiB of free pinnable memory plus system headroom; lower `--cache-ram` |
| Slow decode | Whether `--expert-profile` is set and the profile comes from the same model and a similar use; temperature; other programs on the GPUs | Add a profile or re-collect it on the actual load; keep MTP and `--spec-draft-n-max 3`; decode is naturally 12%–17% lower above temperature 0; lower `-c` when long context is not needed |
| Slow prefill | Whether it is the first request after the server started; whether `-ub` was raised | The first request of a server is naturally a bit slower (measured at ~16K tokens: about 2000–2030 tok/s for the first request of a fresh server, about 2200 tok/s for later requests); keep `-ub` at 2048; use a profile |
| Low draft acceptance | Temperature; content type; draft file | Acceptance is higher at temperature 0 or low temperatures; the draft must be the shared file for the same main model; keep `--spec-draft-n-max` at 3 |
| Image / video request errors | — | This version of the large MoE only tested text input; use 27B for images / video |
| VRAM mysteriously occupied on the AC922, out of VRAM after loading the model | In `numactl --hardware`, the nodes without CPUs are GPU memory; check whether their free memory shrank | Start with `numactl --membind=0,8`; wrap downloads and copies of large files in it too; clear page cache that already landed there as root with `sync; echo 1 > /proc/sys/vm/drop_caches` |
