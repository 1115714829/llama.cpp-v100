# AI deployment guide (for AI assistants and beginners)

[简体中文](ai-deploy.md) | **English** | [← Home README](../../README.en.md) | [27B detail page](27b.en.md) | [Qwen3.8-Flash-Next detail page](moe-2.0.0.en.md) | [Benchmark history](benchmarks.en.md) | [Changelog](../../CHANGELOG.en.md)

This page is for AI assistants and first-time users: follow steps 1–3 to check the hardware and pick a launch command, adjust it with the parameter reference in step 4 and the common problems in step 5, and finally confirm with step 6 that the server works. The launch commands themselves are on the two detail pages and can be copied directly; just replace the model paths with the actual directory.

**Reading order**: this page → "Model download" and the launch commands on the detail page of the chosen model ([27B](27b.en.md) / [Qwen3.8-Flash-Next](moe-2.0.0.en.md)) → [Benchmark history](benchmarks.en.md) when you need speed references.

---

## 1. Check the hardware

First find out the following on the machine you are deploying to:

| What to find out | How to check | What it decides |
|---|---|---|
| How many V100s, and how much VRAM each | `nvidia-smi --query-gpu=index,name,memory.total --format=csv` | Which model can run and which command to use (step 2) |
| Which CPU socket (NUMA node) each GPU belongs to | `nvidia-smi topo -m`, the CPU Affinity / NUMA Affinity columns | Which two GPUs to use for Qwen3.8-Flash-Next on 2 GPUs (step 3) |
| Total and free host memory | `free -g`; on multi-socket machines also the free memory of each node in `numactl --hardware` | Whether Qwen3.8-Flash-Next can start (steps 2 and 3) |
| Whether it is a machine like the IBM AC922 that onlines GPU memory into system memory | `numactl --hardware`: besides the nodes with CPUs, there are several nodes without CPUs, each about the size of one GPU's VRAM | Whether to add `numactl --membind` (step 3) |
| Pinned memory limit | `ulimit -l` (Qwen3.8-Flash-Next requires `unlimited`) | Whether Qwen3.8-Flash-Next can start |
| CPU architecture | `uname -m` (`x86_64` or `ppc64le`) | Whether to use the prebuilt binaries, the Docker image, or build it yourself |
| NVIDIA driver version | Driver Version at the top right of the `nvidia-smi` output | The Docker image needs 550 or newer |

---

## 2. Choose a model and configuration

The two models differ in capability (see their official model cards); this page only covers whether and how they run.

### Decision steps (quick reference)

```text
Step 1: do you need images / PDF / video, concurrency (-np greater than 1), or a context beyond 262144?
  yes     → use 27B (Qwen3.8-Flash-Next was only tested with text input, a single slot and 262144)
  no      → either model works; check the GPU count in the two tables below
Step 2: pick the command by GPU count and VRAM (the two tables below); for Qwen3.8-Flash-Next also check host memory
Step 3: is it an IBM AC922 (GPU memory onlined into system memory)?
  yes     → prefix the command with numactl --membind=<nodes with CPUs> (usually 0,8);
            with Docker use -e LLAMA_NUMA_MEMBIND=0,8 (command C on the 27B page, command D on the Qwen3.8-Flash-Next page)
  no      → no numactl and no LLAMA_NUMA_MEMBIND (Docker: command D on the 27B page, command E on the Qwen3.8-Flash-Next page)
Step 4 (Qwen3.8-Flash-Next): do you have an expert routing profile?
  yes     → --expert-profile <profile> (the Release includes flash-next-profile.txt)
  no      → start without --expert-profile first (slower), add --dump-expert-routing <file> and run real load for a while,
            exit normally, cat <file>.dev* > profile.txt, then restart with the profile
Step 5: VRAM tight, startup fails?
  27B     → try in order: smaller -c → DFlash2 Q4_K_M draft → Q4 configuration (-m UD-Q4_K_M, -ctk q4_0 -ctv q4_0)
  Qwen3.8-Flash-Next  → try in order: smaller -c → keep --spec-draft-ubatch 256 → drop MTP
Step 6: do you want the web UI (chat in a browser)?
  yes     → add nothing: the web UI ships with the source and is embedded into the program at build time, with no network access and no Node.js needed;
            once the server is up, open http://<host>:<port> in a browser (for access from other machines add --host 0.0.0.0)
  no      → add --no-ui to the launch command to keep only the OpenAI-compatible API
```

### 27B (Qwen3.8-27B + DFlash2), 16 GB V100s

| GPUs | Configuration | Maximum context (measured on 1.x) | Command |
|---|---|---|---|
| 6 | Q8 | 524288 (with YaRN) | [27B command A](27b.en.md#a-6-gpus-context-262144) |
| 5 | Q8 | 524288 (with YaRN) | 27B command A, changed to 5 GPUs |
| 4 | Q8; or Q4 | Q8 262144; Q4 524288 | [27B command B](27b.en.md#b-4-gpus-context-262144) |
| 3 | Q8; or Q4 | Q8 131072; Q4 524288 | 27B command B, changed to 3 GPUs |
| 2 | Q4 only | 262144 (with the DFlash2 Q4_K_M draft; up to 131072 with the F16 draft) | 27B command B, changed to 2 GPUs with the Q4 configuration |

- "Q8" = main model Q8_0, KV cache q8_0; "Q4" = main model UD-Q4_K_M, KV cache q4_0 (`-ctk q4_0 -ctv q4_0`). When changing the GPU count, change both `CUDA_VISIBLE_DEVICES` and `--tensor-split` (one 1 per GPU).
- 27B on 32 GB GPUs has not been tested; start from the command for the same GPU count.
- For a context beyond 262144, concurrency, multimodal and the Q4 configuration, see [27B detail page · Variations on A / B](27b.en.md#variations-on-a--b).

### Qwen3.8-Flash-Next (UD-Q4_K_XL + MTP), context 262144

| GPUs and VRAM | Host memory needed (approx.) | Command |
|---|---|---|
| 6 × 16 GB | 42 GiB | [Qwen3.8-Flash-Next command A](moe-2.0.0.en.md#a-6--16-gb-gpus-ac922-context-262144-with-mtp) |
| 2 × 16 GB | 63 GiB; the two GPUs should be on different CPU sockets (see step 3) | [Qwen3.8-Flash-Next command C](moe-2.0.0.en.md#c-2--16-gb-gpus-ac922-two-gpus-on-different-cpu-sockets) |
| 4 × 32 GB (x86) | 27 GiB | [Qwen3.8-Flash-Next command B](moe-2.0.0.en.md#b-4--32-gb-gpus-x86) |
| Other combinations | The less VRAM, the more is needed | Not fully tested; start from command A and change `CUDA_VISIBLE_DEVICES` |

- Besides the host memory in the table, leave headroom for the system, page cache and `--cache-ram`; the user running the server needs `ulimit -l unlimited` (`--ulimit memlock=-1` with Docker). If host memory is not enough, the server does not start.
- With a smaller context (`-c`), somewhat less VRAM and host memory are needed.

---

## 3. Host memory, CPU sockets and platform

- **Machines like the AC922** (GPU memory onlined into system memory): prefix the command with `numactl --membind=<nodes with CPUs>` (usually `0,8` on the AC922); otherwise page cache from reading the model may occupy VRAM, which shows up as out-of-memory; wrap downloads and copies of large files in it too. With Docker, use `-e LLAMA_NUMA_MEMBIND=0,8` instead. Plain x86 servers do not need it.
- **Qwen3.8-Flash-Next on 2 GPUs**: pick two GPUs on different CPU sockets (on the AC922, GPUs 0–2 and 3–5 usually belong to the two CPUs, e.g. `0,3`; on other machines choose from the `nvidia-smi topo -m` result of step 1). Two GPUs on the same socket also work, but the host memory headroom is very small, no other programs can run, and each round is slightly slower; machines with more memory per CPU node have more headroom (not measured).
- **Qwen3.8-Flash-Next speed and host memory**: what does not fit in VRAM lives in host memory, so host memory bandwidth and capacity and the link between GPU and host (NVLink 2.0 on the AC922, usually PCIe 3.0 on x86) all affect speed: it is faster with all memory channels populated and enough capacity.
- **27B**: host memory is mainly used for the prompt cache (`--cache-ram`) and checkpoints (`--ctx-checkpoints`); see step 4.
- **Docker**: one image tag covers x86_64 and ppc64le (AC922 only), so it works on both kinds of machines; x86 uses the NVIDIA Container Toolkit with `--gpus all`, the AC922 uses podman with CDI. The image includes the web UI as well. Full commands in [27B detail page · Docker](27b.en.md#docker).

- **Qwen3.8-Flash-Next on 2 GPUs**: keep the model files on a local SSD / NVMe; on a network drive or a hard disk it is markedly slower.

---

## 4. Parameter reference

"What changing it does" describes the direction of the effect of changing a parameter from its recommended value (or removing it); the measured numbers are in [Benchmark history](benchmarks.en.md).

### Parameters for both models

| Parameter | What it does | What changing it does | Recommended | When to change |
|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Which GPUs to use | Fewer GPUs means more on each GPU: 27B can run a shorter context; Qwen3.8-Flash-Next needs more host memory and decodes slower | Per step 2 | When changing the GPU count |
| `--split-mode tensor`, `--tensor-split 1,…,1` | Multi-GPU tensor parallelism; even split across GPUs (one 1 per GPU) | Other split modes (such as `layer`) have not been tested with the configurations on this page | As in the detail-page commands (the 6-GPU Qwen3.8-Flash-Next command omits `--tensor-split`, which splits evenly by default) | Change together with `CUDA_VISIBLE_DEVICES` when changing the GPU count |
| `-ngl 999` | All layers on GPU | — | As written | Do not change |
| `-c` | Context length: the prompt plus output of one request is at most this many tokens | Larger takes more VRAM; for Qwen3.8-Flash-Next, the VRAM freed by a smaller value is used for speed, so decode is faster | 262144 | Lower it when long context is not needed; for 27B, add YaRN to go longer (see the table below) |
| `-b 2048 -ub 2048` | Tokens fed to the model per batch during prefill | Larger uses more VRAM; for Qwen3.8-Flash-Next prefill gets slower instead; for 27B prefill gets faster and decode on real content slower | 2048 | Do not change |
| `-fa on` | Flash Attention | Turning it off is markedly slower | `on` | Do not change |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache | Without it the KV cache is stored as F16 and its VRAM doubles; `q4_0` halves it (used by the 27B Q4 configuration, not tested on Qwen3.8-Flash-Next) | `q8_0`; `q4_0` for the 27B Q4 configuration | When switching to the 27B Q4 configuration |
| `--model-draft` (`-md`), `--spec-type`, `--spec-draft-n-max` | Speculative decoding: draft model (DFlash2 for 27B, the MTP draft for Qwen3.8-Flash-Next), type (`draft-dflash` / `draft-mtp`), maximum drafted tokens per round (7 for 27B, 3 for Qwen3.8-Flash-Next) | Without speculative decoding, decode is markedly slower and a little less VRAM is used; the draft only proposes, the main model decides the output | As in the detail-page commands | Remove only when VRAM is really short |
| `--spec-draft-ubatch` | Physical batch size of the draft (inherits `-ub` by default) | Smaller saves VRAM | 256 for Qwen3.8-Flash-Next; omit for 27B | Do not change |
| `--spec-draft-sampling` | `greedy` (default): the draft takes the most likely token; `probabilistic`: drafts by probability, with rejection sampling by the main model | With either method, the output follows the main model's sampling with the request's parameters | `greedy` | Do not change |
| `--jinja`, `--chat-template-kwargs`, `reasoning_effort` (request parameter) | Chat template; thinking switch and thinking effort | Thinking is on by default, `'{"enable_thinking":false}'` turns it off; `reasoning_effort` sets the thinking effort (27B: `low` / `medium` / `high` / `xhigh` (default), `none` turns thinking off; the Qwen3.8-Flash-Next model card lists `xhigh` (default) / `medium` / `low`); the less thinking, the sooner the answer finishes | `--jinja` for Qwen3.8-Flash-Next; thinking by use case | Turn thinking off when not needed |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--repeat-penalty` | Server default sampling parameters (parameters in a request take precedence) | The higher the temperature, the lower the draft acceptance rate and the slower the decode; on 27B, `top_k` above 64 or DRY and other samplers make decode slower | The model card's recommended values (see the detail pages) | Pick the set for thinking / non-thinking mode |
| `--cache-ram`, `--ctx-checkpoints` | Host-memory prompt cache limit (MiB, default 8192, 0 = off); checkpoints kept per slot (default 32) | When several conversations take turns, the prefix can be restored from host memory, giving a faster first token; larger values use more host memory: host memory ≈ base use + `--cache-ram` + slots × `--ctx-checkpoints` × checkpoint size, and going over gets the process killed by the system | Default | Lower them when host memory is tight; Qwen3.8-Flash-Next already uses a lot of host memory, so do not raise them |
| `--host`, `--port`, `--api-key-file`, `--metrics` | Listen address, port, API key file, Prometheus metrics | Default `127.0.0.1:8080`, local access only; `--host 0.0.0.0` opens it to other machines | Default for local use; `0.0.0.0` plus `--api-key-file` for external use | When other machines need access |
| `--ui` / `--no-ui` | Whether to serve the web UI at `http://<host>:<port>` (served by default; the old spellings `--webui` / `--no-webui` still work) | When off, only the API remains and a browser shows no UI | Default | When you only need the API, or the port is open to others but you do not want to offer the web page |
| `numactl --membind`, `LLAMA_NUMA_MEMBIND` (Docker) | Restricts host memory to the NUMA nodes that have CPUs | Without it, VRAM may run out on the AC922 | AC922: `0,8`; x86: omit | See step 3 |
| `NCCL_P2P_LEVEL` (environment variable) | Set to `SYS` automatically by the program on ppc64le | — | Do not set | The program does not override a value you set yourself |

### Build options (when building yourself)

The full build commands are in [27B detail page · Build](27b.en.md#build) (the same for Qwen3.8-Flash-Next). The source already includes the web UI (`tools/ui/dist`), so a clone builds offline: the whole build needs no access to any website and no Node.js. The two options below only concern the web UI and usually need not be set:

| Option | What it does | Default | When to change |
|---|---|---|---|
| `-DLLAMA_BUILD_UI=ON` | Rebuilds the web UI from its source (`tools/ui`) | `OFF` | Only needed if you changed the UI source: delete `tools/ui/dist` first and install Node.js 22.12 or newer; the build downloads dependencies from the npm registry. In mainland China, first set `export npm_config_registry=https://registry.npmmirror.com` (Node.js can also be downloaded from npmmirror); elsewhere the default registry is fine. The UI build tools fail on ppc64le (AC922), so build it on x86_64 or macOS and copy the generated `tools/ui/dist` over |
| `-DLLAMA_USE_PREBUILT_UI` | When the source has no `tools/ui/dist`, downloads ready-made UI files from Hugging Face (needs access to sites outside mainland China) | `ON` | This project's source includes the UI files, so no download is triggered. To make sure the build never goes online, set it to `OFF` (if the UI files are missing, the resulting server has no web UI) |

### 27B-only parameters

| Parameter | What it does | What changing it does | Recommended | When to change |
|---|---|---|---|---|
| `-m` | Main model: Q8_0 (~29 GB) or UD-Q4_K_M (~16.5 GB, Q4 configuration) | Q4 saves VRAM (2 GPUs require Q4; 3 and 4 GPUs reach 524288 with Q4) and decodes slower than Q8 | Q8_0 | Switch to the Q4 configuration (together with `-ctk q4_0 -ctv q4_0`) for few GPUs or a longer context |
| Quantization of `--model-draft` | DFlash2 draft: F16, Q8_0, Q4_K_M | On short tasks F16 has a clearly higher acceptance rate, at long context the three are similar; Q8_0 / Q4_K_M use less VRAM | F16 | Q4_K_M when VRAM is tight; required for 2-GPU Q4 at 262144 |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144`, `--override-kv qwen35.context_length=int:524288` | Extends the context to 524288 with YaRN and allows a single slot beyond the model's training length | The YaRN setting applies to all lengths; output quality above 262144 was not tested; the KV cache takes correspondingly more VRAM | Add when needed | To go beyond 262144 |
| `-np`, `--prefill-pace` | Number of slots (context per slot = `-c` ÷ slots); with several slots, the maximum share of time that prompt processing of other requests may take while a request is decoding (default 30; 0 or 100 = no limit) | With several slots, per-slot decode is slower than with one; a larger `--prefill-pace` gives later requests a faster first token and slows down requests that are decoding | `-np 1` | When several clients use the server at the same time; a single agent that needs the full long context still uses `-np 1` |
| `--mmproj` | Vision module: images, PDFs (the client renders each page to an image), video | Split across all GPUs used by the main model by default, taking some VRAM | Add when needed | For images / video |
| `--no-mmproj-offload`, `-mmdev <device>` | Vision module on the CPU; or only on one given GPU (device names from `--list-devices`; `-mmdev none` equals CPU) | On the CPU it saves GPU VRAM, but images and video are much slower | Omit | When there is not enough VRAM for the vision module |
| `--video-ffmpeg-dir` | Directory of `ffmpeg` and `ffprobe` (searched in PATH by default) | Needed for video | Omit when ffmpeg is in PATH | When video requests report that ffmpeg cannot be found; `/opt/tools/bin` in Docker |

### Qwen3.8-Flash-Next-only parameters

| Parameter | What it does | What changing it does | Recommended | When to change |
|---|---|---|---|---|
| `-m` | Main model, pointing at the first shard | This project only tested UD-Q4_K_XL | UD-Q4_K_XL | Do not change |
| Quantization of `--model-draft` | MTP draft: shared-Q4_K_M or shared-Q8_0, must be used with the same main model | Both give similar decode speed and acceptance rate; shared-Q4_K_M uses less VRAM | shared-Q4_K_M on 6 GPUs; the 2-GPU measurements used shared-Q8_0 | Do not change |
| `--expert-profile` | Expert routing profile: keeps the most used experts in VRAM first and balances the load across GPUs | Slower without a profile | The `flash-next-profile.txt` that comes with the Release, or a profile collected from content close to the actual use | Re-collect when the model or the main use case changes (see [Expert routing profile](moe-2.0.0.en.md#expert-routing-profile)) |
| `--dump-expert-routing <file>` | Writes this run's routing statistics when the server exits normally, used to generate a profile | — | Omit | When collecting your own profile |
| `-fitt` (`--fit-target`) | Runtime VRAM headroom left on each GPU at startup (default 800 MiB) | Larger: less VRAM left for the model, slower decode | Omit | Raise it if VRAM runs out after running for a long time |
| `--spec-draft-window` | The MTP draft attends only to the most recent N positions (default 32768, 0 = all) | Larger or 0: the draft gets slower and uses more VRAM at long context | Omit | Do not change |
| `--fit-mmproj` | Whether `--fit` reserves VRAM for the multimodal projector when planning VRAM (default `on`) | No effect under tensor parallelism (every command on the detail pages) | Omit | Do not change |
| `ulimit -l` (system setting) | Pinned memory limit | The server does not start if it is too low | `unlimited`; `--ulimit memlock=-1` with Docker | Set it before deploying |
| `-np`, `--mmproj`, `--rope-scaling yarn` and so on | Concurrency, vision module, YaRN | Qwen3.8-Flash-Next was only tested with a single slot, text input and the native 262144 context | `-np 1`, omit the others | Use 27B when you need these features |

---

## 5. Common problems

| Symptom | Applies to | Check first | Fix |
|---|---|---|---|
| Out of VRAM at startup (log shows `out of memory` or a failed VRAM allocation) | Both models | AC922: whether page cache has landed in GPU memory nodes (see the last row); other processes on the GPUs; whether the GPU count and context exceed the range of step 2 | Start with `numactl --membind=0,8`; dedicate the GPUs; lower `-c`; keep `-ub` at 2048. 27B: Q4_K_M draft → Q4 configuration; if VRAM is short for the vision module, use `--no-mmproj-offload` or `-mmdev`. Qwen3.8-Flash-Next: keep `--spec-draft-ubatch 256`; if still short, drop MTP |
| Out of VRAM after running for a while | Qwen3.8-Flash-Next | — | Raise `-fitt` (default 800) |
| Pinning fails or host memory runs out at startup | Qwen3.8-Flash-Next | `ulimit -l`; free host memory (per node on multi-socket machines) | `ulimit -l unlimited`; `--ulimit memlock=-1` with Docker; keep enough free memory for the host memory requirement in step 2; lower `--cache-ram`; on 2 GPUs, switch to two GPUs on different sockets |
| Slow decode | Both models | Temperature and sampling settings; whether several slots are in use; whether Qwen3.8-Flash-Next has a profile; whether other programs are using the GPUs | Acceptance is naturally lower above temperature 0; 27B: keep `top_k` at most 64 and do not use DRY or other samplers; with several slots, per-slot decode is naturally slower than with one, and a smaller `--prefill-pace` makes requests that are decoding faster. Qwen3.8-Flash-Next: add a profile or re-collect it on the actual load; keep MTP and `--spec-draft-n-max 3`; lower `-c` when long context is not needed |
| Slow prefill | Both models | Whether it is the first request after the server started; whether `-ub` was changed | For Qwen3.8-Flash-Next the first request after the server starts is naturally a bit slower; keep `-ub` at 2048; use a profile with Qwen3.8-Flash-Next |
| Low draft acceptance rate | Both models | Temperature; content type; draft file | Acceptance is higher at temperature 0 or low temperatures; for 27B the F16 draft accepts more on short tasks; the Qwen3.8-Flash-Next draft must be the shared file for the same main model; long context naturally yields fewer tokens per round |
| Slow time to first token for later requests with several slots | 27B | — | Prompt processing is queued in order, so later slots wait for earlier prompts to be processed; a larger `--prefill-pace` gives later requests a faster first token |
| Image / video request errors | Both models | Whether 27B is used; whether `--mmproj` is set; whether ffmpeg can be found for video; whether PDFs were converted to images first | Qwen3.8-Flash-Next was only tested with text input, so use 27B for images / video; add `--mmproj`; add `--video-ffmpeg-dir` for video; the client renders each PDF page to an image before sending; with 27B at `-c 1048576 -np 2` there is not enough VRAM to process images, use `--no-mmproj-offload` |
| Log shows `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | 27B | — | This is a warning; the server starts with the parameters in the command; it can be ignored |
| VRAM mysteriously occupied on the AC922, out of VRAM after loading the model | Both models | In `numactl --hardware`, the nodes without CPUs are GPU memory; check whether their free memory shrank | Start with `numactl --membind=0,8`; wrap downloads and copies of large files in it too; clear page cache that has already landed there by running `sync; echo 1 > /proc/sys/vm/drop_caches` as root |

---

## 6. Check after startup

Qwen3.8-Flash-Next takes a few minutes to start (all shards have to be read); until `/health` returns normally, the server is still loading. The server provides an OpenAI-compatible API:

```bash
curl -s http://127.0.0.1:8080/health          # returns {"status":"ok"} once ready
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Open `http://<host>:8080` in a browser to see the web UI (not available if the server was started with `--no-ui`). If you use `--api-key-file`, add `-H "Authorization: Bearer <key>"` to requests, and enter the same key in the web UI's settings. When comparing speeds with [Benchmark history](benchmarks.en.md), make sure the test conditions (temperature, thinking on or off, prompt length) are the same.
