# Qwen3.8-Flash-Next + MTP

**简体中文** | [English](moe-2.0.0.en.md) | [← 主页 README](../../README.md) | [AI 部署指引](ai-deploy.md) | [27B 稠密模型](27b.md) | [Qwen3.8-Flash-Next](moe-2.0.0.md) | [历次实测数据](benchmarks.md) | [更新日志](../../CHANGELOG.md)

本页说明怎样在 V100 上运行 Qwen3.8-Flash-Next：支持的配置、模型下载、官方推荐采样参数、启动命令、专家路由画像、主机内存与显存、编译与 Docker、已知限制。还没选好配置时，先按 [AI 部署指引](ai-deploy.md) 判断。

> **支持的配置**（主模型 UD-Q4_K_XL + MTP 草稿，上下文 262144）：6 × V100-16GB 是标准配置；2 × V100-16GB 自 2.0.1.1 起可用，需要约 63 GiB 主机内存；4 × V100-32GB（x86）可以运行。其他卡数与量化后续陆续兼容。

---

## 概要

- **模型**：结构与能力见官方模型卡 [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)（原生上下文 262144，带视觉编码器）；本项目只测了文字输入。
- **多卡与主机内存**：用 `--split-mode tensor` 多卡运行；显存放不下的部分自动放在主机内存、仍由 GPU 计算，不需要 `--n-cpu-moe` / `-ot`。卡越少，需要的主机内存越多（见“[主机内存、显存与启动](#主机内存显存与启动)”）。
- **专家路由画像**：`--expert-profile` 让常用的专家优先留在显存，速度更快（见“[专家路由画像](#专家路由画像)”）。
- **MTP 投机解码**：`--spec-type draft-mtp` 用模型自带的 MTP 层起草，提高吐字速度；输出由主模型决定。

---

## 实测数据

当前版本 2.0.1.1，IBM AC922 标准测试机，上下文 262144，真实代码，每档新起服务：

| 配置 | 约 20 万 token：预填充 / 吐字（tok/s） | 约 1.6 万 token：预填充 / 吐字（tok/s） | 口径 |
|---|---|---|---|
| 6 × V100-16GB（命令 A，草稿换成 shared-Q8_0） | 3402 / 122.5 | 2805 / 140.5 | 温度 0、不开思考 |
| 2 × V100-16GB，分属不同 CPU 插槽（命令 C） | 1300 / 43.1 | 1173 / 38.0 | 思考模式 + 模型卡官方采样参数 |

- 两行口径不同（温度 0、不开思考 对 思考模式 + temperature 1.0），速度不能直接比较。
- 6 卡与 2.0.1 同场对比无退化，温度 0 输出相同，每卡显存相同。2 卡是 2.0.1.1 新增的配置，没有上一版的对照数字。
- 测试条件、同一插槽两张卡的对照、显存与历次版本的全部数据见[历次实测数据](benchmarks.md#2011)。

---

## 模型下载

**国内用户建议从魔搭（ModelScope）下载。** 文件都在 unsloth 的同一个仓库里：

| 用途 | 文件 | 魔搭 ModelScope | Hugging Face |
|---|---|---|---|
| 目标模型 | `UD-Q4_K_XL/` 目录下的 4 个分片 `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf`（合计约 111 GB） | [unsloth/Qwen3.8-Flash-Next-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| MTP 草稿（6 卡命令 A 用） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`（约 1.9 GB） | 同上 | 同上 |
| MTP 草稿（2 卡命令 C 用） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`（约 2.8 GB） | 同上 | 同上 |

- 4 个分片放在同一目录，启动时 `-m` 指向第一个分片（`-00001-of-00004`）。专家路由画像 `flash-next-profile.txt` 随 Release 附带（见“[专家路由画像](#专家路由画像)”）。
- 文件名带 `shared` 的 MTP 草稿要和同一个主模型一起用。两种草稿的吐字与接受率相近，shared-Q4_K_M 占的显存更小；unsloth 推荐 shared-Q8_0。两者的实测对比见[历次实测数据](benchmarks.md#200)。
- `scripts/v100-get-models.sh` 只下载 27B 的文件；Qwen3.8-Flash-Next 用直链下载（支持断点续传；AC922 上建议用 `numactl --membind=0,8` 包起来，避免页缓存落进 GPU 显存的节点）：

```bash
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
# Hugging Face：base=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O models/$f $base/$f
done
# 2 卡命令 C 用的 shared-Q8_0 草稿：
wget -c -O models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf $base/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

---

## 官方推荐采样参数

Qwen 官方模型卡（[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的 Best Practices 一节；unsloth 的 GGUF 模型卡写的相同）给出两套推荐值。加在启动命令末尾即设为服务默认，请求里带的采样参数优先：

| 模式 | 官方取值 | 对应的 llama-server 参数 |
|---|---|---|
| 思考模式（模型默认） | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` | `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1.0` |
| 非思考（Instruct）模式 | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 --chat-template-kwargs '{"enable_thinking":false}'` |

- 模型卡说明 `presence_penalty` 可在 0～2 之间调整以减少无休止的重复，取值偏高时偶尔会出现语言混杂、效果略有下降。
- 思考开关：关闭思考在请求里传 `"chat_template_kwargs": {"enable_thinking": false}`，或用 `--chat-template-kwargs` 设为服务默认；思考强度用请求参数 `reasoning_effort`，模型卡注明支持 `xhigh`（默认）、`medium`、`low`。
- 模型卡没有给 MTP 的推荐设置。官方的 [vLLM 部署配方](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) 每轮起草 3 个 token（与本页相同）；unsloth 的 [MTP 说明](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) 建议每轮 2 个，本项目没有测过 2。

---

## 推荐启动命令

命令 A～C 用源码编译出的程序直接运行；Docker 用法见“[编译与 Docker](#编译与-docker)”一节的命令 D、E。各参数做什么、怎么选见 [AI 部署指引 · 参数说明](ai-deploy.md#4-参数说明)。

### A. 6 卡 16 GB（AC922），上下文 262144，带 MTP

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

- `numactl --membind=0,8` 把主机内存限定在有 CPU 的 NUMA 节点（AC922 把显存并入系统内存，不限定时页缓存可能占用显存），节点号用 `numactl --hardware` 查看；普通 x86 服务器去掉这一段。
- 还没有画像时去掉 `--expert-profile` 一行也能启动（通常更慢）；不用 MTP 时去掉 `--model-draft` 与 `--spec-…` 两行；按官方建议设默认采样时在末尾加上一节表中的一组参数。

### B. 4 卡 32 GB（x86）

命令 A 去掉 `numactl --membind=0,8`，改 `CUDA_VISIBLE_DEVICES=0,1,2,3`，其余不变。需要约 27 GiB 主机内存。

### C. 2 卡 16 GB（AC922），分属不同 CPU 插槽的两张卡

2.0.1.1 起提供。

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

- **选卡**：用分属不同 CPU 插槽的两张卡。AC922 每颗 CPU 连 3 张卡（通常 0～2 号属一颗、3～5 号属另一颗），所以写 `0,3`；其他机器用 `nvidia-smi topo -m` 看每张卡的 NUMA 节点，选分属两个节点的两张。同一插槽的两张卡（例如 `0,1`）也能跑，但主机内存余量很小，不能再运行别的程序，每轮也稍慢。
- **模型文件**：2 卡运行时请把模型文件放在本机的 SSD / NVMe 上；放在网络盘或机械硬盘上会明显变慢。
- **主机内存**：约需 63 GiB（上下文 262144；`-c` 越小需要的越少），另留系统余量；运行服务的用户要 `ulimit -l unlimited`。
- **草稿**：本页 2 卡实测用的是 shared-Q8_0；命令 A 用的 shared-Q4_K_M 占显存更小，2 卡上没有实测。
- **显存**：每卡显存几乎用满是正常的；启动时显存不足先调小 `-c`。
- 本页 2 卡实测按思考模式与模型卡官方采样参数（在命令末尾加 `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`，或在请求里带）。

**其他卡数与显存**没有做完整测试。显存总量越少，需要的主机内存越多，吐字也越慢；主机内存不够时无法启动。改 `CUDA_VISIBLE_DEVICES` 即可，显存不够时先调小 `-c`。

---

## MTP 草稿窗口

自 2.0.0 起提供。`--spec-draft-window N`（环境变量 `LLAMA_ARG_SPEC_DRAFT_WINDOW`）：MTP 草稿只看最近 N 个位置。默认 `32768`，`0` 表示看全部位置，负数在启动时报参数错误。

- 草稿只提议，输出由主模型验证后决定，窗口不影响输出是否正确；提示词不超过窗口时，输出与不设窗口逐字相同。
- 窗口越小，长上下文时草稿越快、占的显存越少；但窗口以外的内容草稿看不到，长上下文时接受率可能略降。
- 一般不用改；要让草稿看全部位置写 `--spec-draft-window 0`。本参数只作用于 MTP 投机解码，27B 的 DFlash2 不使用它。

```bash
llama-server ... --spec-type draft-mtp -md <草稿文件> --spec-draft-window 0      # 草稿看全部位置
llama-server ... --spec-type draft-mtp -md <草稿文件> --spec-draft-window 8192   # 示例值：缩小窗口
```

---

## --fit-mmproj

2.0.1 新增。`--fit-mmproj on|off`（环境变量 `LLAMA_ARG_FIT_MMPROJ`，默认 `on`）：用 `--fit` 自动规划显存时，是否给多模态投影器（mmproj）留出显存；设为 `off` 时需要自己用 `--fit-target` 留出。张量并行（`--split-mode tensor`，本页所有命令）下本参数不起作用。

---

## 专家路由画像

`--expert-profile <文件>` 告诉程序每层哪些专家最常用：常用的优先留在显存，各卡分到的计算量也更均衡，所以带画像更快。

- **只影响放置，不改变计算公式**：带画像时多卡求和的顺序会变，输出可能与不带画像时有极细微的差别，数值上等价；同一份画像每次启动的结果相同。带画像时启动多花一些时间。
- **按模型与用途采集**：画像按“层号 专家号”记录，换模型要重新采；与卡数无关。校准文本尽量贴近主要用途（写代码就用代码采）。
- 文件是纯文本，每行 `layer expert count [launches]`，由下面的方法生成，不需要手写。

获取画像：

1. **直接用 Release 附带的** `flash-next-profile.txt`（用真实请求采集），下载后传给 `--expert-profile`。
2. **用校准文本生成**（需要先编译 `llama-perplexity`）：

   ```bash
   # -f 校准文本，-o 画像输出路径，-c 采样时的上下文长度；-- 之后的参数原样传给 llama-perplexity
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 bash scripts/v100-expert-profile.sh \
       -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
       -f calibration.txt -o ./models/flash-next-profile.txt -c 4096 \
       -- -sm tensor -ngl 999 -ctk q8_0 -ctv q8_0
   ```

   脚本合并各卡的统计、校验画像是否完整（不完整就报错退出），最后打印统计分布，用来判断校准文本够不够有代表性；`--help` 查看完整用法。
3. **用真实负载生成**：`llama-server` 加 `--dump-expert-routing <文件>`，照常使用一段时间后正常退出（中途强制结束会丢掉统计），每张卡写出一份 `<文件>.dev<卡号>`，`cat <文件>.dev* > 画像.txt` 合并后交给 `--expert-profile`。

---

## 主机内存、显存与启动

- **主机内存**：需要的主机内存随卡数与显存变化——6 × 16 GB 约 42 GiB，2 × 16 GB 约 63 GiB（上下文 262144），4 × 32 GB 约 27 GiB；另要留出系统、页缓存与 `--cache-ram` 的余量。主机内存不够时无法启动。
- **锁页内存上限**：运行服务的用户需要足够的锁页内存上限（`ulimit -l`，建议 `unlimited`；Docker 加 `--ulimit memlock=-1`），否则无法启动。
- **多 CPU 插槽的机器**：AC922 上要用 `numactl --membind=0,8` 启动；2 卡时用分属不同插槽的两张卡（见命令 C）。
- **显存**：启动后每卡显存占用接近可用显存是正常的（多出来的显存用来提速），不表示显存不够；这些 GPU 不要再与其他程序共用。长时间运行中出现显存不足时调大 `-fitt`（默认 800 MiB）。显存很紧时可以调小 `--spec-draft-ubatch`，或不开 MTP。
- **启动时间**：要读入全部分片并准备主机内存，需要几分钟（实测见[历次实测数据](benchmarks.md#200)）。

---

## 编译与 Docker

编译方法与 [27B 页](27b.md#编译) 相同；生成画像的脚本还要用到 `llama-perplexity`：`cmake --build build -j --target llama-server llama-quantize llama-perplexity`。

Docker 镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1` 的用法见 [27B 页的 Docker 一节](27b.md#docker)，Qwen3.8-Flash-Next 在 Docker 下没有实测。必须加 `--ulimit memlock=-1`，宿主机要有足够的主机内存；画像文件放进挂载的模型目录。镜像里没有 `llama-perplexity`，用校准文本生成画像要在源码编译的环境里做，`--dump-expert-routing` 可以在容器里用。

- **D. Docker，6 × V100-16GB，AC922（podman）**：按 27B 页命令 C 准备 podman 与 CDI，镜像名之后换成本页命令 A 从 `-m` 开始的参数，模型与画像路径改成 `/models/...`，去掉 `--host 127.0.0.1`。
- **E. Docker，4 × V100-32GB，x86**：同上，按 27B 页命令 D（`--gpus all`，不设 `LLAMA_NUMA_MEMBIND`）。
- 2 卡（命令 C）在 Docker 下没有实测。

---

## 已知限制

- 只测了文字输入（官方模型带视觉编码器）、单路（`-np 1`）与原生 262144 上下文。
- 6 × 16 GB、2 × 16 GB 与 4 × 32 GB 以外的组合没有完整测试；Docker 下没有实测。
- 主机内存不够时无法启动。

---

## 快速部署（给 AI 助手与新手）

已移到 [AI 部署指引](ai-deploy.md)：查硬件、选配置、每个参数对能力 / 速度 / 显存的影响和常见问题对照都在那一页。
