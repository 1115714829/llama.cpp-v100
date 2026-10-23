# Qwen3.8-Flash-Next + MTP

[简体中文](moe-2.0.0.md) | **English** | [← Home README](../../README.en.md) | [AI deployment guide](ai-deploy.en.md) | [27B dense model](27b.en.md) | [Qwen3.8-Flash-Next](moe-2.0.0.en.md) | [Benchmark history](benchmarks.en.md) | [Changelog](../../CHANGELOG.en.md)

This page explains how to run Qwen3.8-Flash-Next on V100: supported configurations, model download, official sampling recommendations, start commands, the expert routing profile, host memory and VRAM, build and Docker, and known limitations. If you have not chosen a configuration yet, decide with the [AI deployment guide](ai-deploy.en.md) first.

> **Supported configurations** (UD-Q4_K_XL main model + MTP draft, context 262144): 6 × V100-16GB is the standard configuration; 2 × V100-16GB is available since 2.0.1.1 and needs about 63 GiB of host memory; 4 × V100-32GB (x86) can run it. Other GPU counts and quantizations will follow.

---

## Overview

- **Model**: for its structure and capabilities see the official model card [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (native context 262144, with a vision encoder); this project only tested text input.
- **Multiple GPUs and host memory**: runs on multiple GPUs with `--split-mode tensor`; whatever does not fit in VRAM is automatically placed in host memory and is still computed on the GPU, with no need for `--n-cpu-moe` / `-ot`. The fewer the GPUs, the more host memory is needed (see "[Host memory, VRAM and startup](#host-memory-vram-and-startup)").
- **Expert routing profile**: `--expert-profile` keeps frequently used experts in VRAM first, which is faster (see "[Expert routing profile](#expert-routing-profile)").
- **MTP speculative decoding**: `--spec-type draft-mtp` drafts with the model's own MTP layer to raise decode speed; the output is decided by the main model.

---

## Measured results

Current release 2.0.1.1, IBM AC922 standard test machine, context 262144, real code, a fresh server for each prompt size:

| Configuration | ~200K tokens: prefill / decode (tok/s) | ~16K tokens: prefill / decode (tok/s) | Methodology |
|---|---|---|---|
| 6 × V100-16GB (command A, with the draft switched to shared-Q8_0) | 3402 / 122.5 | 2805 / 140.5 | temperature 0, thinking off |
| 2 × V100-16GB, on different CPU sockets (command C) | 1300 / 43.1 | 1173 / 38.0 | thinking mode + the model card's recommended sampling parameters |

- The two rows use different methodologies (temperature 0, thinking off vs. thinking mode + temperature 1.0), so their speeds cannot be compared directly.
- 6 GPUs: no regression versus 2.0.1 in a same-session A/B comparison, identical output at temperature 0, the same VRAM per GPU. 2 GPUs is a configuration added in 2.0.1.1, so there are no numbers from the previous release to compare with.
- Test conditions, the comparison with two GPUs on the same socket, VRAM, and all data from previous releases are in [Benchmark history](benchmarks.en.md#2011).

---

## Model download

**Users in mainland China should download from ModelScope.** All files are in the same unsloth repository:

| Purpose | File | ModelScope | Hugging Face |
|---|---|---|---|
| Target model | the 4 shards `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` in the `UD-Q4_K_XL/` directory (~111 GB total) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| MTP draft (for 6-GPU command A) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (~1.9 GB) | same as above | same as above |
| MTP draft (for 2-GPU command C) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (~2.8 GB) | same as above | same as above |

- Put the 4 shards in one directory and point `-m` at the first shard (`-00001-of-00004`) at startup. The expert routing profile `flash-next-profile.txt` comes with the Release (see "[Expert routing profile](#expert-routing-profile)").
- An MTP draft with `shared` in its file name must be used with the same main model. The two drafts give similar decode speed and acceptance rate, and shared-Q4_K_M uses less VRAM; unsloth recommends shared-Q8_0. The measured comparison of the two is in [Benchmark history](benchmarks.en.md#200).
- `scripts/v100-get-models.sh` only downloads the 27B files; download Qwen3.8-Flash-Next with direct links (resumable; on the AC922, wrap it in `numactl --membind=0,8` so page cache does not land in GPU memory nodes):

```bash
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
# Hugging Face: base=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O models/$f $base/$f
done
# shared-Q8_0 draft for 2-GPU command C:
wget -c -O models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf $base/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

---

## Official sampling recommendations

The official Qwen model card ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Best Practices section; the unsloth GGUF model card says the same) gives two sets of recommended values. Append them to the start command to make them server defaults; sampling parameters in a request take precedence:

| Mode | Official values | llama-server parameters |
|---|---|---|
| Thinking mode (model default) | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` | `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0` |
| Non-thinking (Instruct) mode | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 --chat-template-kwargs '{"enable_thinking":false}'` |

- The model card notes that `presence_penalty` can be adjusted between 0 and 2 to reduce endless repetition; high values may occasionally cause language mixing and slightly lower quality.
- Thinking switch: to turn thinking off, pass `"chat_template_kwargs": {"enable_thinking": false}` in the request, or set it as a server default with `--chat-template-kwargs`; thinking effort is the request parameter `reasoning_effort`, which the model card lists as `xhigh` (default), `medium` and `low`.
- The model card gives no MTP recommendation. The official [vLLM deployment recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) drafts 3 tokens per round (same as this page); unsloth's [MTP notes](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) suggest 2 per round; this project has not tested 2.

---

## Recommended start commands

Commands A–C run the programs built from source directly; for Docker, see commands D and E in the "[Build and Docker](#build-and-docker)" section. What each parameter does and how to choose it is in [AI deployment guide · Parameters](ai-deploy.en.md#4-parameter-reference).

### A. 6 × 16 GB GPUs (AC922), context 262144, with MTP

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
- Without a profile yet, drop the `--expert-profile` line and it still starts (usually slower); without MTP, drop the `--model-draft` and `--spec-…` lines; to set the default sampling as officially recommended, append one set of parameters from the table in the previous section.

### B. 4 × 32 GB GPUs (x86)

Command A without `numactl --membind=0,8` and with `CUDA_VISIBLE_DEVICES=0,1,2,3`; everything else stays the same. Needs about 27 GiB of host memory.

### C. 2 × 16 GB GPUs (AC922), two GPUs on different CPU sockets

Available since 2.0.1.1.

```bash
CUDA_VISIBLE_DEVICES=0,3 numactl --membind=0,8 ./build/bin/llama-server \
  -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --jinja \
  --expert-profile ./models/flash-next-profile.txt \
  --model-draft ./models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ubatch 256 --spec-draft-sampling greedy \
  --host 127.0.0.1 --port 8080 --metrics
```

- **Choosing GPUs**: use two GPUs on different CPU sockets. On the AC922 each CPU is connected to 3 GPUs (usually GPUs 0–2 belong to one CPU and 3–5 to the other), hence `0,3`; on other machines, check each GPU's NUMA node with `nvidia-smi topo -m` and pick two GPUs on two different nodes. Two GPUs on the same socket (for example `0,1`) also work, but the host memory headroom is very small, no other programs can be run, and the time per round is also slightly longer.
- **Model files**: when running on 2 GPUs, put the model files on a local SSD / NVMe; on a network drive or a mechanical hard disk it is noticeably slower.
- **Host memory**: about 63 GiB is needed (context 262144; the smaller `-c`, the less is needed), plus headroom for the system; the user running the server needs `ulimit -l unlimited`.
- **Draft**: the 2-GPU measurements on this page use shared-Q8_0; the shared-Q4_K_M used by command A takes less VRAM and has not been measured on 2 GPUs.
- **VRAM**: nearly full VRAM on every GPU is normal; if VRAM is short at startup, lower `-c` first.
- The 2-GPU measurements on this page use thinking mode and the model card's recommended sampling parameters (append `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0` to the command, or pass them in the request).

**Other GPU counts and VRAM sizes** have not been fully tested. The less VRAM in total, the more host memory is needed and the slower decode gets; if host memory is not enough, the server does not start. Just change `CUDA_VISIBLE_DEVICES`; if VRAM is short, lower `-c` first.

---

## MTP draft window

Available since 2.0.0. `--spec-draft-window N` (environment variable `LLAMA_ARG_SPEC_DRAFT_WINDOW`): the MTP draft only attends to the most recent N positions. The default is `32768`, `0` means all positions, and a negative value is rejected as an invalid argument at startup.

- The draft only proposes; the output is decided by the main model after verification, so the window does not affect whether the output is correct; while the prompt does not exceed the window, the output is token-for-token identical to a run without the window.
- The smaller the window, the faster the draft and the less VRAM it uses at long context; but the draft cannot see content outside the window, so the acceptance rate may drop slightly at long context.
- Usually there is no need to change it; to let the draft see all positions, write `--spec-draft-window 0`. This parameter only applies to MTP speculative decoding; DFlash2 on 27B does not use it.

```bash
llama-server ... --spec-type draft-mtp -md <draft file> --spec-draft-window 0      # the draft sees all positions
llama-server ... --spec-type draft-mtp -md <draft file> --spec-draft-window 8192   # example value: smaller window
```

---

## --fit-mmproj

New in 2.0.1. `--fit-mmproj on|off` (environment variable `LLAMA_ARG_FIT_MMPROJ`, default `on`): whether to leave VRAM for the multimodal projector (mmproj) when `--fit` plans VRAM automatically; with `off`, you have to leave it yourself with `--fit-target`. Under tensor parallelism (`--split-mode tensor`, all commands on this page) this parameter has no effect.

---

## Expert routing profile

`--expert-profile <file>` tells the program which experts are used most in each layer: frequently used ones stay in VRAM first and the work each GPU gets is also more balanced, so running with a profile is faster.

- **Affects placement only, not the math**: with a profile, the order of the multi-GPU sum changes, so output may differ very slightly from runs without a profile while being numerically equivalent; the same profile gives the same result on every start. With a profile, startup takes a little longer.
- **Collect per model and use case**: a profile records "layer number, expert number", so a new model needs a new profile; it does not depend on the GPU count. Keep the calibration text close to the main use case (for coding, collect with code).
- The file is plain text, one `layer expert count [launches]` per line, generated with the methods below; there is no need to write it by hand.

Getting a profile:

1. **Use the one that comes with the Release directly:** `flash-next-profile.txt` (collected from real requests); download it and pass it to `--expert-profile`.
2. **Generate it from a calibration text** (build `llama-perplexity` first):

   ```bash
   # -f calibration text, -o output profile path, -c context length for sampling; arguments after -- go to llama-perplexity as is
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 bash scripts/v100-expert-profile.sh \
       -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
       -f calibration.txt -o ./models/flash-next-profile.txt -c 4096 \
       -- -sm tensor -ngl 999 -ctk q8_0 -ctv q8_0
   ```

   The script merges the statistics of all GPUs, checks whether the profile is complete (exiting with an error if not), and finally prints the distribution of the statistics, which tells you whether the calibration text is representative enough; `--help` shows the full usage.
3. **Generate it from real load**: start `llama-server` with `--dump-expert-routing <file>`, use it normally for a while and then exit normally (killing it midway loses the statistics); each GPU writes one `<file>.dev<GPU number>`; merge them with `cat <file>.dev* > profile.txt` and pass the result to `--expert-profile`.

---

## Host memory, VRAM and startup

- **Host memory**: the host memory needed varies with the GPU count and VRAM: about 42 GiB for 6 × 16 GB, about 63 GiB for 2 × 16 GB (context 262144), about 27 GiB for 4 × 32 GB; plus headroom for the system, page cache and `--cache-ram`. If host memory is not enough, the server does not start.
- **Pinned memory limit**: the user running the server needs a sufficient locked-memory limit (`ulimit -l`, `unlimited` recommended; add `--ulimit memlock=-1` in Docker), otherwise the server does not start.
- **Machines with multiple CPU sockets**: on the AC922, start with `numactl --membind=0,8`; with 2 GPUs, use two GPUs on different sockets (see command C).
- **VRAM**: after startup, per-GPU VRAM usage close to the available VRAM is normal (the extra VRAM is used for speed) and does not mean VRAM is short; do not share these GPUs with other programs. If VRAM runs out during a long run, raise `-fitt` (default 800 MiB). When VRAM is very tight, you can lower `--spec-draft-ubatch` or run without MTP.
- **Startup time**: all shards have to be read and host memory prepared, which takes a few minutes (measurements in [Benchmark history](benchmarks.en.md#200)).

---

## Build and Docker

Build as on the [27B page](27b.en.md#build); the profile script also needs `llama-perplexity`: `cmake --build build -j --target llama-server llama-quantize llama-perplexity`.

Usage of the Docker image `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1` is on the [27B page, Docker section](27b.en.md#docker); Qwen3.8-Flash-Next has not been measured under Docker. `--ulimit memlock=-1` is required and the host needs enough host memory; put the profile file in the mounted model directory. The image has no `llama-perplexity`, so making a profile from a calibration text has to happen in a source build; `--dump-expert-routing` works inside the container.

- **D. Docker, 6 × V100-16GB, AC922 (podman)**: prepare podman and CDI as in command C on the 27B page, put the arguments of this page's command A from `-m` onward after the image name, change the model and profile paths to `/models/...`, and drop `--host 127.0.0.1`.
- **E. Docker, 4 × V100-32GB, x86**: the same, following command D on the 27B page (`--gpus all`, no `LLAMA_NUMA_MEMBIND`).
- 2 GPUs (command C) has not been measured under Docker.

---

## Known limitations

- Only text input (the official model has a vision encoder), a single slot (`-np 1`) and the native 262144 context were tested.
- Combinations other than 6 × 16 GB, 2 × 16 GB and 4 × 32 GB have not been fully tested; Docker has not been measured.
- If host memory is not enough, the server does not start.

---

## Quick deployment (for AI assistants and beginners)

Moved to the [AI deployment guide](ai-deploy.en.md): checking the hardware, choosing a configuration, each parameter's effect on capability / speed / VRAM, and the common problems with their fixes are all on that page.
