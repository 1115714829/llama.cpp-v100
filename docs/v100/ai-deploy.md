# AI 部署指引（给 AI 助手与新手）

**简体中文** | [English](ai-deploy.en.md) | [← 主页 README](../../README.md) | [27B 详情页](27b.md) | [Qwen3.8-Flash-Next 详情页](moe-2.0.0.md) | [历次实测数据](benchmarks.md) | [更新日志](../../CHANGELOG.md)

本页写给 AI 助手和第一次部署的用户：按第 1～3 步查硬件、选出启动命令，再用第 4 步的参数说明和第 5 步的常见问题对照调整，最后按第 6 步确认服务正常。启动命令本身在两张详情页里，可以直接复制，把模型路径换成实际目录即可。

**阅读顺序**：本页 → 选定模型的详情页（[27B](27b.md) / [Qwen3.8-Flash-Next](moe-2.0.0.md)）里的“模型下载”和启动命令 → 需要速度参考时看[历次实测数据](benchmarks.md)。

---

## 1. 查硬件

先在要部署的机器上查清这几件事：

| 要知道的 | 怎么查 | 用来决定 |
|---|---|---|
| 有几张 V100、每张显存多大 | `nvidia-smi --query-gpu=index,name,memory.total --format=csv` | 能跑哪个模型、用哪条命令（第 2 步） |
| 每张 GPU 属于哪个 CPU 插槽（NUMA 节点） | `nvidia-smi topo -m`，看 CPU Affinity / NUMA Affinity 两列 | Qwen3.8-Flash-Next 用 2 张卡时选哪两张（第 3 步） |
| 主机内存总量与空闲 | `free -g`；多插槽的机器再看 `numactl --hardware` 里每个节点的 free | Qwen3.8-Flash-Next 能不能启动（第 2、3 步） |
| 是不是 IBM AC922 这类把显存并入系统内存的机器 | `numactl --hardware`：除了有 CPU 的节点，还有若干没有 CPU、大小约等于一张卡显存的节点 | 要不要加 `numactl --membind`（第 3 步） |
| 锁页内存上限 | `ulimit -l`（Qwen3.8-Flash-Next 要求 `unlimited`） | Qwen3.8-Flash-Next 能不能启动 |
| CPU 架构 | `uname -m`（`x86_64` 或 `ppc64le`） | 用预编译程序、Docker 镜像还是自己编译 |
| NVIDIA 驱动版本 | `nvidia-smi` 输出右上角的 Driver Version | Docker 镜像需要 550 或更新 |

---

## 2. 选模型与配置

两个模型的能力不同（看各自的官方模型卡），本页只管能不能跑、怎么跑。

### 判断步骤（速查）

```text
第 1 步：要处理图片 / PDF / 视频、要多路并发（-np 大于 1）、或要超过 262144 的上下文吗？
  要      → 用 27B（Qwen3.8-Flash-Next 只测了文字输入、单路、262144）
  不要    → 两个模型都可以，按下面两张表看卡数
第 2 步：按卡数与显存选命令（下面两张表）；Qwen3.8-Flash-Next 同时核对主机内存
第 3 步：是 IBM AC922（GPU 显存并入系统内存）吗？
  是      → 命令前加 numactl --membind=<有 CPU 的节点>（一般 0,8）；
            Docker 用 -e LLAMA_NUMA_MEMBIND=0,8（27B 页命令 C、Qwen3.8-Flash-Next 页命令 D）
  否      → 不加 numactl，也不设 LLAMA_NUMA_MEMBIND（Docker：27B 页命令 D、Qwen3.8-Flash-Next 页命令 E）
第 4 步（Qwen3.8-Flash-Next）：有专家路由画像吗？
  有      → --expert-profile <画像>（Release 附带 flash-next-profile.txt）
  没有    → 先去掉 --expert-profile 启动（更慢），加 --dump-expert-routing <文件> 跑一段真实负载，
            正常退出后 cat <文件>.dev* > 画像.txt，再带上画像重启
第 5 步：显存紧张、启动失败？
  27B     → 依次试：-c 调小 → 草稿换 DFlash2 Q4_K_M → 换 Q4 配置（-m UD-Q4_K_M，-ctk q4_0 -ctv q4_0）
  Qwen3.8-Flash-Next  → 依次试：-c 调小 → 保持 --spec-draft-ubatch 256 → 去掉 MTP
第 6 步：要网页界面（在浏览器里对话）吗？
  要      → 什么都不用加：网页界面随源码自带，编译时直接嵌进程序，不需要联网，也不需要 Node.js；
            服务起来后浏览器打开 http://<主机>:<端口>（别的机器访问要加 --host 0.0.0.0）
  不要    → 启动命令加 --no-ui，只保留 OpenAI 兼容接口
```

### 27B（Qwen3.8-27B + DFlash2），16 GB 的 V100

| 卡数 | 配置 | 最高上下文（1.x 版本实测） | 命令 |
|---|---|---|---|
| 6 | Q8 | 524288（加 YaRN） | [27B 命令 A](27b.md#a-6-卡上下文-262144) |
| 5 | Q8 | 524288（加 YaRN） | 27B 命令 A，改成 5 张卡 |
| 4 | Q8；或 Q4 | Q8 262144；Q4 524288 | [27B 命令 B](27b.md#b-4-卡上下文-262144) |
| 3 | Q8；或 Q4 | Q8 131072；Q4 524288 | 27B 命令 B，改成 3 张卡 |
| 2 | 只能用 Q4 | 262144（草稿用 DFlash2 Q4_K_M；F16 草稿最高 131072） | 27B 命令 B，改成 2 张卡并用 Q4 配置 |

- “Q8”= 主模型 Q8_0、KV 缓存 q8_0；“Q4”= 主模型 UD-Q4_K_M、KV 缓存 q4_0（`-ctk q4_0 -ctv q4_0`）。改卡数要同时改 `CUDA_VISIBLE_DEVICES` 与 `--tensor-split`（几张卡写几个 1）。
- 32 GB 的卡跑 27B 没有测过，从同样卡数的命令起步。
- 超过 262144 的上下文、多路并发、多模态、Q4 配置的改法见 [27B 详情页 · 在 A / B 的基础上改](27b.md#在-a--b-的基础上改)。

### Qwen3.8-Flash-Next（UD-Q4_K_XL + MTP），上下文 262144

| 卡与显存 | 需要的主机内存（约） | 命令 |
|---|---|---|
| 6 × 16 GB | 42 GiB | [Qwen3.8-Flash-Next 命令 A](moe-2.0.0.md#a-6-卡-16-gbac922上下文-262144带-mtp) |
| 2 × 16 GB | 63 GiB；两张卡要分属不同 CPU 插槽（见第 3 步） | [Qwen3.8-Flash-Next 命令 C](moe-2.0.0.md#c-2-卡-16-gbac922分属不同-cpu-插槽的两张卡) |
| 4 × 32 GB（x86） | 27 GiB | [Qwen3.8-Flash-Next 命令 B](moe-2.0.0.md#b-4-卡-32-gbx86) |
| 其他组合 | 显存越少，需要越多 | 没有完整测试；从命令 A 起步改 `CUDA_VISIBLE_DEVICES` |

- 表中的主机内存之外，还要留出系统、页缓存与 `--cache-ram` 的余量；运行服务的用户要 `ulimit -l unlimited`（Docker 加 `--ulimit memlock=-1`）。主机内存不够时无法启动。
- 上下文（`-c`）调小时，需要的显存与主机内存都会少一些。

---

## 3. 主机内存、CPU 插槽与平台

- **AC922 这类机器**（GPU 显存并入系统内存）：命令前加 `numactl --membind=<有 CPU 的节点>`（AC922 一般是 `0,8`），否则读模型产生的页缓存可能占用显存，表现为显存不足；下载、复制大文件时也用它包起来。Docker 改用 `-e LLAMA_NUMA_MEMBIND=0,8`。普通 x86 服务器不加。
- **Qwen3.8-Flash-Next 用 2 张卡**：选分属不同 CPU 插槽的两张（AC922 上通常 0～2 号与 3～5 号分属两颗 CPU，例如 `0,3`；其他机器按第 1 步 `nvidia-smi topo -m` 的结果选）。同一插槽的两张卡也能跑，但主机内存余量很小，不能再运行别的程序，每轮也稍慢；每个 CPU 节点内存更大的机器余量会多一些（没有实测）。
- **Qwen3.8-Flash-Next 的速度与主机内存**：显存放不下的部分在主机内存里，主机内存的带宽、容量以及 GPU 与主机之间的链路（AC922 为 NVLink 2.0，x86 一般为 PCIe 3.0）都会影响速度：内存通道插满、容量充足时更快。
- **27B**：主机内存主要用于提示词缓存（`--cache-ram`）与检查点（`--ctx-checkpoints`），见第 4 步。
- **Docker**：同一个镜像标签含 x86_64 与 ppc64le（仅 AC922），两种机器都能用；x86 用 NVIDIA Container Toolkit 与 `--gpus all`，AC922 用 podman 与 CDI。镜像里同样自带网页界面。完整命令见 [27B 详情页 · Docker](27b.md#docker)。

- **Qwen3.8-Flash-Next 用 2 张卡时**：模型文件放在本机的 SSD / NVMe 上；放在网络盘或机械硬盘上会明显变慢。

---

## 4. 参数说明

“改了会怎样”写的是把参数从推荐值改掉（或去掉）的影响方向；具体的实测数字见[历次实测数据](benchmarks.md)。

### 两个模型都用的参数

| 参数 | 做什么 | 改了会怎样 | 推荐值 | 什么时候改 |
|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 选用哪几张 GPU | 卡越少，每卡要放的越多：27B 能开的上下文变短；Qwen3.8-Flash-Next 需要的主机内存变多、吐字变慢 | 按第 2 步 | 换卡数时 |
| `--split-mode tensor`、`--tensor-split 1,…,1` | 多卡张量并行；各卡均分（几张卡写几个 1） | 其他切分方式（如 `layer`）没有按本页的配置测过 | 照详情页命令（Qwen3.8-Flash-Next 6 卡命令不写 `--tensor-split`，默认均分） | 换卡数时与 `CUDA_VISIBLE_DEVICES` 一起改 |
| `-ngl 999` | 所有层放 GPU | — | 照写 | 不改 |
| `-c` | 上下文长度：单个请求的提示加输出最多这么多 token | 越大占显存越多；Qwen3.8-Flash-Next 调小时省下的显存用来提速，吐字更快 | 262144 | 用不到长上下文时可以调小；27B 要更长时加 YaRN（见下表） |
| `-b 2048 -ub 2048` | 预填充时每批送进模型的 token 数 | 调大后显存占用增加；Qwen3.8-Flash-Next 预填充反而变慢；27B 预填充变快、真实内容的吐字变慢 | 2048 | 不改 |
| `-fa on` | Flash Attention | 关掉会明显变慢 | `on` | 不改 |
| `-ctk q8_0 -ctv q8_0` | KV 缓存用 8 位 | 不写时按 F16 存，KV 显存翻倍；`q4_0` 时减半（27B Q4 配置用，Qwen3.8-Flash-Next 没有测） | `q8_0`；27B Q4 配置用 `q4_0` | 换 27B Q4 配置时 |
| `--model-draft`（`-md`）、`--spec-type`、`--spec-draft-n-max` | 投机解码：草稿模型（27B 用 DFlash2，Qwen3.8-Flash-Next 用 MTP 草稿）、类型（`draft-dflash` / `draft-mtp`）、每轮最多起草几个（27B 写 7，Qwen3.8-Flash-Next 写 3） | 去掉投机解码时吐字明显变慢，少占一些显存；草稿只提议，输出由主模型决定 | 照详情页命令 | 显存实在不够时才去掉 |
| `--spec-draft-ubatch` | 草稿的物理批大小（默认继承 `-ub`） | 调小省显存 | Qwen3.8-Flash-Next 写 256；27B 不写 | 不改 |
| `--spec-draft-sampling` | `greedy`（默认）：草稿取概率最大的 token；`probabilistic`：按概率起草，主模型做拒绝采样 | 两种方式的输出都服从主模型按请求参数的采样 | `greedy` | 不改 |
| `--jinja`、`--chat-template-kwargs`、`reasoning_effort`（请求参数） | 聊天模板；思考开关与思考强度 | 思考默认开启，`'{"enable_thinking":false}'` 关闭；`reasoning_effort` 调思考强度（27B：`low` / `medium` / `high` / `xhigh`（默认），`none` 关闭思考；Qwen3.8-Flash-Next 模型卡注明 `xhigh`（默认）/ `medium` / `low`）；思考越少，回答越快结束 | Qwen3.8-Flash-Next 加 `--jinja`；思考按用途 | 不需要思考时关闭 |
| `--temp`、`--top-p`、`--top-k`、`--min-p`、`--presence-penalty`、`--repeat-penalty` | 服务端默认采样参数（请求里带的优先） | 温度越高，草稿接受率越低，吐字越慢；27B 上 `top_k` 大于 64 或用 DRY 等其他采样器时吐字变慢 | 模型卡推荐值（见详情页） | 按思考 / 非思考模式选一组 |
| `--cache-ram`、`--ctx-checkpoints` | 主机内存里的提示词缓存上限（MiB，默认 8192，0 为关闭）；每路保留的检查点个数（默认 32） | 多个对话轮流使用时可以从主机内存恢复前缀，首字更快；调大占更多主机内存：主机内存约为 基础占用 + `--cache-ram` + 路数 × `--ctx-checkpoints` × 每个检查点的大小，超出会被系统结束进程 | 默认 | 主机内存紧张时调小；Qwen3.8-Flash-Next 本身已占用较多主机内存，不要调大 |
| `--host`、`--port`、`--api-key-file`、`--metrics` | 监听地址、端口、API 密钥文件、Prometheus 指标 | 默认 `127.0.0.1:8080`，只有本机能访问；`--host 0.0.0.0` 对外开放 | 本机用默认；对外用 `0.0.0.0` 并加 `--api-key-file` | 要给其他机器用时 |
| `--ui` / `--no-ui` | 是否在 `http://<主机>:<端口>` 提供网页界面（默认提供；旧写法 `--webui` / `--no-webui` 仍可用） | 关掉后只剩 API，浏览器打开没有界面 | 默认 | 只需要 API、或端口对外开放但不想提供网页时 |
| `numactl --membind`、`LLAMA_NUMA_MEMBIND`（Docker） | 把主机内存限定在有 CPU 的 NUMA 节点 | AC922 上不加时可能显存不足 | AC922：`0,8`；x86：不加 | 见第 3 步 |
| `NCCL_P2P_LEVEL`（环境变量） | ppc64le 上程序自动设为 `SYS` | — | 不设 | 已自行设置时程序不覆盖 |

### 编译参数（自己编译时）

完整编译命令见 [27B 详情页 · 编译](27b.md#编译)（Qwen3.8-Flash-Next 相同）。源码已自带网页界面（`tools/ui/dist`），克隆下来即可离线编译，整个编译过程不需要访问任何网站，也不需要 Node.js。下面两个参数只和网页界面有关，一般不用写：

| 参数 | 做什么 | 默认 | 什么时候改 |
|---|---|---|---|
| `-DLLAMA_BUILD_UI=ON` | 从界面源码（`tools/ui`）重新构建网页界面 | `OFF` | 只有改了界面源码才需要：先删除 `tools/ui/dist`，装 Node.js 22.12 或更新的版本，编译时会从 npm 仓库下载依赖。国内先设 `export npm_config_registry=https://registry.npmmirror.com`（Node.js 也可以从 npmmirror 下载）；国外用默认仓库即可。ppc64le（AC922）上界面的构建工具会失败，需要在 x86_64 或 macOS 上构建，再把生成的 `tools/ui/dist` 拷过去 |
| `-DLLAMA_USE_PREBUILT_UI` | 源码里没有 `tools/ui/dist` 时，从 Hugging Face 下载现成的界面文件（需要能访问国外网站） | `ON` | 本项目源码自带界面文件，不会触发下载。想确保编译绝不联网就设为 `OFF`（缺界面文件时编出的服务没有网页界面） |

### 27B 专用参数

| 参数 | 做什么 | 改了会怎样 | 推荐值 | 什么时候改 |
|---|---|---|---|---|
| `-m` | 主模型：Q8_0（约 29 GB）或 UD-Q4_K_M（约 16.5 GB，Q4 配置） | Q4 省显存（2 卡只能用 Q4；3、4 卡用 Q4 可以开到 524288），吐字比 Q8 慢 | Q8_0 | 卡少或要更长上下文时换 Q4 配置（同时 `-ctk q4_0 -ctv q4_0`） |
| `--model-draft` 的量化 | DFlash2 草稿：F16、Q8_0、Q4_K_M | 短任务 F16 的接受率明显更高，长上下文三者相近；Q8_0 / Q4_K_M 占显存更少 | F16 | 显存紧时换 Q4_K_M；2 卡 Q4 开 262144 必须换 |
| `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144`、`--override-kv qwen35.context_length=int:524288` | 用 YaRN 把上下文扩到 524288，并允许单路超过模型训练长度 | YaRN 设置对所有长度都生效；没有测 262144 以上的输出质量；KV 占的显存相应变大 | 需要时加 | 要超过 262144 时 |
| `-np`、`--prefill-pace` | 并发路数（每路上下文 = `-c` ÷ 路数）；多路时，有请求在吐字，其他请求的提示词处理最多占用的时间百分比（默认 30，0 或 100 不限流） | 多路时每路吐字比单路慢；`--prefill-pace` 调大，后来的请求首字更快、正在吐字的请求更慢 | `-np 1` | 有多个客户端同时使用时；单个 Agent 要开满长上下文时仍用 `-np 1` |
| `--mmproj` | 视觉模块：处理图片、PDF（客户端把每页渲染成图片）、视频 | 默认分到主模型使用的全部 GPU，占一些显存 | 需要时加 | 要处理图片 / 视频时 |
| `--no-mmproj-offload`、`-mmdev <设备>` | 视觉模块放 CPU；或只放在指定的一张 GPU 上（设备名用 `--list-devices` 查看，`-mmdev none` 等同放 CPU） | 放 CPU 省 GPU 显存，但处理图片、视频慢很多 | 不写 | 显存不够放视觉模块时 |
| `--video-ffmpeg-dir` | `ffmpeg`、`ffprobe` 所在目录（默认在 PATH 里找） | 处理视频需要 | ffmpeg 在 PATH 里时不写 | 视频请求报找不到 ffmpeg 时；Docker 里用 `/opt/tools/bin` |

### Qwen3.8-Flash-Next 专用参数

| 参数 | 做什么 | 改了会怎样 | 推荐值 | 什么时候改 |
|---|---|---|---|---|
| `-m` | 主模型，指向第一个分片 | 本项目只测了 UD-Q4_K_XL | UD-Q4_K_XL | 不改 |
| `--model-draft` 的量化 | MTP 草稿：shared-Q4_K_M 或 shared-Q8_0，必须与同一个主模型配用 | 两者吐字与接受率相近，shared-Q4_K_M 占显存更小 | 6 卡用 shared-Q4_K_M；2 卡实测用的是 shared-Q8_0 | 不改 |
| `--expert-profile` | 专家路由画像：让常用的专家优先留在显存、各卡负载更均衡 | 去掉画像会更慢 | Release 附带的 `flash-next-profile.txt`，或用与实际用途相近的内容采的画像 | 换模型或主要用途变化时重采（见[专家路由画像](moe-2.0.0.md#专家路由画像)） |
| `--dump-expert-routing <文件>` | 服务正常退出时写出这次运行的路由统计，用来生成画像 | — | 不写 | 要自己采画像时 |
| `-fitt`（`--fit-target`） | 启动时给每张卡留的运行期显存余量（默认 800 MiB） | 调大：留给模型的显存变少，吐字变慢 | 不写 | 长时间运行后出现显存不足时调大 |
| `--spec-draft-window` | MTP 草稿只看最近 N 个位置（默认 32768，0 = 全部） | 调大或设 0：长上下文时草稿变慢、占显存更多 | 不写 | 不改 |
| `--fit-mmproj` | 用 `--fit` 规划显存时是否给多模态投影器留出显存（默认 `on`） | 张量并行（详情页所有命令）下不起作用 | 不写 | 不改 |
| `ulimit -l`（系统设置） | 锁页内存上限 | 不够时无法启动 | `unlimited`；Docker 加 `--ulimit memlock=-1` | 部署前设好 |
| `-np`、`--mmproj`、`--rope-scaling yarn` 等 | 并发、视觉模块、YaRN | Qwen3.8-Flash-Next 只测了单路、文字输入与原生 262144 上下文 | `-np 1`，其余不加 | 需要这些功能时改用 27B |

---

## 5. 常见问题

| 症状 | 适用 | 先查 | 处理 |
|---|---|---|---|
| 启动时显存不足（日志有 `out of memory` 或显存分配失败） | 两个模型 | AC922：页缓存是否落进了 GPU 显存节点（见最后一行）；GPU 上有没有其他进程；卡数与上下文是否超出第 2 步的范围 | 用 `numactl --membind=0,8` 启动；GPU 独占；`-c` 调小；`-ub` 保持 2048。27B：草稿换 Q4_K_M → 换 Q4 配置；视觉模块显存不够时用 `--no-mmproj-offload` 或 `-mmdev`。Qwen3.8-Flash-Next：保持 `--spec-draft-ubatch 256`，仍不够再去掉 MTP |
| 运行一段时间后显存不足 | Qwen3.8-Flash-Next | — | 调大 `-fitt`（默认 800） |
| 启动时锁页失败或主机内存不足 | Qwen3.8-Flash-Next | `ulimit -l`；空闲主机内存（多插槽的机器看各节点） | `ulimit -l unlimited`；Docker 加 `--ulimit memlock=-1`；按第 2 步的主机内存需求留足空闲内存；调小 `--cache-ram`；2 卡时换成分属不同插槽的两张卡 |
| 吐字慢 | 两个模型 | 温度与采样设置；是否多路；Qwen3.8-Flash-Next 有没有带画像；GPU 是否被其他程序占用 | 温度大于 0 时接受率本来就低一些；27B：`top_k` 不超过 64、不用 DRY 等其他采样器；多路时每路本来就慢于单路，`--prefill-pace` 调小可让正在吐字的请求更快。Qwen3.8-Flash-Next：加画像或按实际负载重采；保持 MTP 与 `--spec-draft-n-max 3`；用不到长上下文时调小 `-c` |
| 预填充慢 | 两个模型 | 是不是服务启动后的第一个请求；`-ub` 是否改过 | Qwen3.8-Flash-Next 服务启动后的第一个请求本来就慢一些；`-ub` 保持 2048；Qwen3.8-Flash-Next 带上画像 |
| 草稿接受率低 | 两个模型 | 温度；内容类型；草稿文件 | 温度 0 或低温时接受率更高；27B 短任务用 F16 草稿接受率更高；Qwen3.8-Flash-Next 草稿必须是同一主模型的 shared 文件；长上下文时每轮产出本来就少一些 |
| 多路时后来的请求首字慢 | 27B | — | 提示词处理按顺序排队，后面几路要等前面的处理完；`--prefill-pace` 调大，后来的请求首字更快 |
| 图片 / 视频请求报错 | 两个模型 | 用的是不是 27B；`--mmproj` 是否加了；视频能否找到 ffmpeg；PDF 是否先转成了图片 | Qwen3.8-Flash-Next 只测了文字输入，图片 / 视频用 27B；加 `--mmproj`；视频加 `--video-ffmpeg-dir`；PDF 由客户端把每页渲染成图片再发；27B `-c 1048576 -np 2` 时处理图片显存不够，改用 `--no-mmproj-offload` |
| 日志里有 `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | 27B | — | 这是警告，按命令里的参数启动，可以忽略 |
| AC922 上显存被莫名占用、加载模型后显存不足 | 两个模型 | `numactl --hardware` 里没有 CPU 的节点就是 GPU 显存，看它们的 free 是否变小 | 用 `numactl --membind=0,8` 启动；下载、复制大文件也用它包起来；已经落进去的页缓存用 root 执行 `sync; echo 1 > /proc/sys/vm/drop_caches` 清掉 |

---

## 6. 启动后检查

Qwen3.8-Flash-Next 启动要几分钟（要读入全部分片），`/health` 返回正常之前服务还在加载。服务是 OpenAI 兼容接口：

```bash
curl -s http://127.0.0.1:8080/health          # 就绪后返回 {"status":"ok"}
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

浏览器打开 `http://<主机>:8080` 能看到网页界面（启动时加了 `--no-ui` 则没有）。用了 `--api-key-file` 时，请求加 `-H "Authorization: Bearer <密钥>"`，网页界面里也要在设置中填入同一个密钥。速度与[历次实测数据](benchmarks.md)对照时，注意测试条件（温度、是否开思考、提示词长度）要一致。
