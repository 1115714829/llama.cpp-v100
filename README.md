# llama.cpp-v100

**简体中文** | [English](README.en.md) | [AI 部署指引](docs/v100/ai-deploy.md) | [27B 详情页](docs/v100/27b.md) | [Qwen3.8-Flash-Next 详情页](docs/v100/moe-2.0.0.md) | [历次实测数据](docs/v100/benchmarks.md) | [更新日志](CHANGELOG.md)

面向 NVIDIA V100（SM70）的 llama.cpp 修改版。当前版本 **2.0.1.2**（2026-11-03）。

---

## 前言

本项目的初衷，是让大家用低成本的 V100 畅玩大模型：6 张 V100 跑 27B、上下文做到 1M，从不可能变成了可能。

**特别感谢群友正博、蓝莲、小王的王Ww 的资金赞助**，也感谢提供测试设备、协助测试的群友和网友们，项目能走到今天离不开大家（名单见[致谢](#致谢)）。

项目目前由我个人维护。维持现在的节奏，最稀缺的是两样东西：

- **时间**：项目越来越大，又要同时兼容 27B 与 Qwen3.8-Flash-Next（MoE）两条线，测试与研发时间成倍增加，已经严重挤占我的个人工作时间。发一个版本，需要 3 台机器连续跑约 12 小时，才能走完全部验收、拿到发版数据。
- **token**：开发、排查与测试主要靠 AI 协作完成，token 消耗很大。免费的 token 已经用完，~~Claude 的 20 倍套餐 11 月底到期~~（**Claude 喜提封号，2026-11-02，祝福 Anthropic 早日倒闭**）；测试机器也都是借来的，随时可能归还。

所以后续更新一定会慢下来，希望大家提前有个心理预期。项目不会中断，短期内也不会停更：还有很多想法没有实现，包括一些从底层自己写的实现方案，以及因为时间关系登记下来的待优化、待处理项，都会在后面逐步推进。

**更新会变慢，但版本质量不打折。** 每个正式版发布前都要走完下面的验收，任何一项不过就不发版：

- **与上一正式版同场比较**：新旧版本在同一台机器、同一时段按 A-B-B-A 交替测量，多次取中位数，离散度大时补足样本再判；预填充、吐字、每卡显存峰值、主机内存峰值（进程 RSS，含锁页内存）四项只许持平或更好。
- **按真实使用方式测**：公开数字一律在开思考、模型卡官方采样参数、加载视觉模块的配置下测得，每档新起服务、提示词前缀不命中缓存。
- **数值不变**：改到的算子逐个通过 `test-backend-ops` 全量用例；温度 0 下输出逐字节比对；困惑度（PPL）与 KL 散度（KLD）对照，确认模型输出没有被悄悄改变。
- **两条模型线互不拖累**：动到共享代码的改动，都要回归 27B 张量并行的速度、显存与输出。
- **测量环境干净、可追溯**：测量期间整机独占，不编译、不跑其他任务，全程采样负载与外部 GPU 进程；被测程序按提交号单独构建、编完冻结只读，测量前核对二进制哈希与版本号，并按 PCI 总线号核对进程实际用到的卡。
- **以前能启动的配置照样能启动**：1、2、3、6 卡，以及多路并发、多模态等配置逐一验证。

---

## 简介

基于 llama.cpp（上游 v0.7.0，2026-10-15 的 `620d2225c`）修改，从计算内核到运行时为 V100 重新实现与优化；引用的外部实现见文末致谢，出处与许可见各文件头和 `licenses/` 目录。

**本项目只做底层框架（推理运行环境）的优化，不制作、不修改量化模型**：使用的是模型发布方公开的 GGUF 权重，量化格式与权重原样不动，改动的是计算内核、多卡并行、调度与投机解码的执行链路，并以逐位不变或温度 0 逐字比对为验收标准。模型的智力与能力请参照原模型的模型卡与所选量化档位，本项目只改变它跑得多快、占多少显存。

围绕 V100（SM70）的硬件特性，从计算内核到运行时进行了系统性优化：

- **多卡并行**：2～6 卡高效协同，适配 NVLink 拓扑，卡数不能整除注意力头数时（如 6 卡）同样可用。
- **长上下文**：在超长上下文下保持预填充与吐字速度的稳定，并降低计算缓冲的显存占用。
- **投机解码**：实现 DFlash2 投机解码的完整链路，优化采样与每轮开销，提升吐字速度。
- **计算内核**：优化 SM70 计算内核并适配 Q8_0 量化格式，同时进行算子融合。
- **平台适配**：针对 IBM AC922 的 NVLink 互联与 NUMA 内存架构进行适配。
- **多路并发**：多路请求时吐字优先、限制提示词处理占用的时间，多路的解码批次、投机解码与显存预留都按路数优化。
- **多模态**：支持视觉模块（mmproj）在 GPU 上运行，处理图片、PDF 与视频。
- **Qwen3.8-Flash-Next**：16 GB 显存的 V100 也能跑百亿级参数的 MoE 模型与 26 万上下文，显存放不下的部分自动利用主机内存；MTP 投机解码与稀疏注意力针对 V100 优化。

---

## 支持的模型与现状

### Qwen3.8-27B（稠密模型）+ DFlash2

- **卡数**：1～6 张 V100 都能运行。2～6 卡有完整实测（Q8_0 与 Q4 两套配置，2 卡用 Q4）；1 卡需换用更低比特的量化模型。6 卡以上在代码上同样可用，但手头没有更多设备，没有做专项优化和实测。
- **上下文与并发**：最高 524288（YaRN）；支持多路并发（2、4、6 卡两路，4 卡四路）以及图片、PDF、视频输入。
- **效果**：目前在 V100 上属于阶段性领先的水平。

| 卡数 | 配置 | 上下文 | 约 1.6 万 token 真实代码<br>预填充 / 吐字（tok/s） |
|---|---|---:|---|
| 6 | Q8_0 | 262144 | 2786 / 152.7 |
| 6 | Q8_0 | 524288 | 2774 / 114.1 |
| 5 | Q8_0 | 524288 | 2612 / 126.1 |
| 4 | Q8_0 | 262144 | 2584 / 131.7 |
| 3 | Q8_0 | 131072 | 2470 / 90.0 |
| 2 | Q4（草稿 Q4_K_M） | 262144 | 1841 / 78.6 |

<sub>6 卡 262144 一行为 2.0.1.1 实测（温度 0、不开思考；约 20 万 token 时预填充 2187、吐字 124.5 tok/s），其余各行为 1.0.6～1.1.1 实测，测试条件见 [27B 实测归档（1.1.1）](docs/v100/27b-results-1.1.1.md)。</sub>

作者正式服务在用的模型（含配套 DFlash2 草稿，魔搭可直接下载）见 [27B 详情页 · 模型下载](docs/v100/27b.md#模型下载)；各卡数的启动命令、并发与多模态的用法见 [27B 详情页](docs/v100/27b.md)。

### Qwen3.8-Flash-Next（MoE）+ MTP

- **卡数**：1 / 2 / 3 / 6 张 V100-16GB 都能启动（上下文 262144，本版实测）；4 × V100-32GB（x86）也可运行。显存放不下的专家放在主机内存里、仍由 GPU 计算，卡越少需要的主机内存越多：6 卡约 42 GiB，2 卡约 75.08 GiB。两卡推荐用同一 CPU 插槽的两张（例如 0、1）。
- **块大小 2048 / 4096 / 8192 是什么**：就是启动参数 `-b` 与 `-ub` 的取值，表示处理提示词（预填充）时每一步一次处理多少个 token。块越大，同一份专家权重一次服务的 token 越多，预填充越快；代价是计算缓冲更大、留给专家缓存的显存变少，吐字略慢、主机内存略高。三档的启动差别只有这两个参数：`-b 2048 -ub 2048`、`-b 4096 -ub 4096`、`-b 8192 -ub 8192`。
- **怎么选**：对话、写代码这类以吐字为主的场景用 2048（也是开 MTP 时的推荐值）；一次读入大量代码或文档、以预填充为主时用 4096 或 8192。8192 块不建议配 MTP。

| 块大小 | 预填充 tok/s | 吐字 tok/s | 每卡显存峰值 MiB | 主机内存峰值 GiB |
|---|---|---|---|---|
| 2048 | `1155.48` | `43.55` | `14824` | `75.08` |
| 4096 | `2151.70` | `41.53` | `14856` | `77.00` |

<sub>两张 V100-SXM2-16GB（同一 CPU 插槽），上下文 262144，约 20 万 token 真实代码，输出 1024 token，开思考、模型卡官方采样参数、加载视觉模块，开 MTP 投机（每轮 3 个草稿）。</sub>

模型下载、各卡数的启动命令、MTP、专家路由画像与主机内存的说明见 [Qwen3.8-Flash-Next 详情页](docs/v100/moe-2.0.0.md)。

---

## AI 部署指引

**给 AI 助手：部署本项目前，先完整读 [docs/v100/ai-deploy.md](docs/v100/ai-deploy.md)，按其中的步骤查硬件、选配置，再到详情页复制启动命令。** 自己部署也按这份指引走。

| 模型 | 支持的配置 | 启动命令 |
|---|---|---|
| Qwen3.8-27B（Q8_0 / Q4）+ DFlash2 投机解码 | 2～6 × V100-16GB，上下文最高 524288；支持多路并发与图片 / PDF / 视频 | [27B 详情页 · 启动参数](docs/v100/27b.md#启动参数) |
| Qwen3.8-Flash-Next（UD-Q4_K_XL）+ MTP 投机解码 | 6 × V100-16GB、2 × V100-16GB（需要约 63 GiB 主机内存）、4 × V100-32GB；上下文 262144，只测了文字输入与单路 | [Qwen3.8-Flash-Next 详情页 · 推荐启动命令](docs/v100/moe-2.0.0.md#推荐启动命令) |

指引的内容：[1. 查硬件](docs/v100/ai-deploy.md#1-查硬件)（卡数、显存、主机内存、GPU 属于哪个 CPU 插槽）→ [2. 选模型与配置](docs/v100/ai-deploy.md#2-选模型与配置) → [3. 主机内存、CPU 插槽与平台](docs/v100/ai-deploy.md#3-主机内存cpu-插槽与平台) → [4. 参数说明](docs/v100/ai-deploy.md#4-参数说明) → [5. 常见问题](docs/v100/ai-deploy.md#5-常见问题) → [6. 启动后检查](docs/v100/ai-deploy.md#6-启动后检查)。

---

## 本版更新（2.0.1.2）

- **同等条件下，预填充比 Strata 快约 10 %**：同一台机器、同样两张卡（同一插槽）、同样的上下文与请求，双方都开投机解码、都加载视觉模块，开思考、模型卡官方采样参数；约 20 万 token 真实代码的预填充在 2048 档快 10.0 %、4096 档快 13.1 %（见下方同场对比表）。
- **两卡同一 CPU 插槽（卡 0、1）优化**：Qwen3.8-Flash-Next 在同一插槽的两张 V100-16GB 上，块大小 2048 与 4096 均可运行，上下文 262144；约 20 万 token 真实代码（提示 209 075 token、输出 1024 token）的**预填充为 1155.48 / 2151.70 tok/s，吐字 22.941 / 24.075 ms/token**。口径：**两卡同一插槽 + MTP-3 投机 + 视觉模块 + 开思考 + 模型卡官方采样参数**（temperature 1.0、top_p 0.95、top_k 20、min_p 0、presence_penalty 0、repetition_penalty 1.0）。8192 块不建议配 MTP，未列。
- **主机内存按两个 CPU 节点分开放置**：专家补集在两个节点上对半放置；两卡同一插槽、上下文 262144 时，主机内存峰值（进程 RSS，含锁页）为 75.08 / 77.00 GiB（块 2048 / 4096）。
- **吐字更快**：专家换入改为自适应换入（换入窗口带屏障）。
- **预填充更快**：专家补集经拷贝引擎以流式环搬运、与计算重叠，并修正了解码暂存的对齐问题。
- **计算内核**：优化 SM70 计算内核并适配 Q8_0 量化格式，同时进行算子融合；数值口径见[更新日志](CHANGELOG.md)。
- **1 卡、2 卡、3 卡、6 卡都可运行**：Qwen3.8-Flash-Next 在 1 / 2 / 3 / 6 张 V100-16GB 上均可启动，上下文 262144（显存不足时可降低上下文或改用更低的量化格式）。实测见[历次实测数据](docs/v100/benchmarks.md#2012)。
- **27B 多路并发**：27B 在 2、4、6 张 V100-16GB 上都支持两路并发（`-np 2`，Q8 与 Q4 配置，2 卡用 Q4），4 卡支持四路并发（`-np 4`，每路 65536），6 卡可两路各开 262144 上下文（`-c 524288 -np 2`，YaRN）并同时处理图片与视频。各配置的实测见 [27B 实测归档（1.1.1）](docs/v100/27b-results-1.1.1.md)。
- **多模态与思考模式使用说明**：两项能力都可用，但要用对方式——
  - **视觉模块要单独下载官方件**：`mmproj-F16.gguf`（27B：927 607 488 B，在 `Qwen3.8-27B-GGUF` 仓库；Flash-Next：904 004 000 B，在 `Qwen3.8-Flash-Next-GGUF` 仓库；两个文件不是同一个，别混用）。[Hugging Face](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) ｜ [魔搭 ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) 与 [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) ｜ [魔搭 ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF)；启动时加 `--mmproj <该文件>`，视觉模块默认放 GPU，多卡时按层切到各卡。不带它时带图请求会返回 `image input is not supported - hint: if this is unexpected, you may need to provide the mmproj`。
  - **思考内容在 `reasoning_content` 字段**：请求体加 `"chat_template_kwargs": {"enable_thinking": true}` 即开思考（`false` 关）；默认格式把思考从正文里抽出来放进 `message.reasoning_content`，只读 `message.content` 的客户端看不到它（想留在正文里并同时保留字段用 `--reasoning-format deepseek-legacy`）；**`--reasoning-format none` 则把思考并入正文、不再分栏**。服务端另有 `--reasoning on/off`、`--reasoning-effort`、`--reasoning-budget`、`--reasoning-format`；网页界面会单独渲染思考段。
  - **小图建议加 `--image-min-tokens 1024`**：这一族模型每张图至少要这么多图像 token，不设时小图只产生约 280 个图像 token，识别会变差（服务启动时也会打印提示）。这只是建议值，默认值不变。
  - **不要用 `-fit off`**：这一族大 MoE 模型的专家并行模式下，自动参数规划同时就是专家的放置规划器，关掉它模型直接加载失败；用默认的自动规划即可。
  - **27B 线（`-sm tensor`）与视觉模块**：张量并行下视觉模块不在自动规划内；显存紧张时可用 `--fit-target` 预留余量，或把视觉模块放到 CPU。
- **两卡投机推荐**：两卡（同一插槽 0、1）推荐配 **MTP 投机（每轮 3 个草稿）+ 2048 块**（`-b 2048 -ub 2048 --spec-draft-n-max 3`，草稿 `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`）：约 20 万 token 真实代码下吐字每 token 22.941 ms，草稿接受率 0.4536；**不推荐在 8192 块上配投机**；MTP 需显式开启，默认值不变。 MTP 投机目前请配 `-np 1` 使用。
- **显存**：两卡每卡显存峰值 14824 / 14856 MiB（块 2048 / 4096）；27B（4 卡）每卡显存峰值 10933 MiB。
- **27B 零退化结论**：4 卡（Q8_0 + DFlash2、块 2048、16K 真实代码）对 2.0.1.1 零退化 —— 预填充 +0.70 %，吐字持平，每卡显存峰值 10933 MiB（与对照逐值相同），主机内存峰值 12.49 GiB（对照 12.48 GiB，差在测量散布内）。
- **合并上游 llama.cpp v0.7.0**（2026-10-15 的 `620d2225c`）：本项目的 V100 专项优化随之迁移到新的上游版本。
- Docker 镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.1.2`。
- 命令行参数与默认值与 2.0.1.1 相同。

**与 Strata 同场对比（两卡同一插槽，双方都开投机解码、都加载视觉模块）**

| 块大小 | 预填充 tok/s<br>Strata / 本项目 | 吐字 tok/s<br>Strata / 本项目 | 每卡显存峰值 MiB<br>Strata / 本项目 | 主机内存峰值 GiB<br>Strata / 本项目 |
|---|---|---|---|---|
| 2048 | `1050.80 / 1155.48` | `45.65 / 43.55` | `15101 / 14824` | `91.11 / 75.08` |
| 4096 | `1902.00 / 2151.70` | `48.30 / 41.53` | `15089 / 14856` | `91.22 / 77.00` |

<sub>同一台 IBM AC922、两张 V100-SXM2-16GB（同一 CPU 插槽），同一时段交替测量；上下文 262144，约 20 万 token 真实代码（提示 209 075 token；对方引擎 209 102 token），输出 1024 token，开思考、模型卡官方采样参数；本项目开 MTP 投机（每轮 3 个草稿），Strata 开其自身的投机解码，双方都加载视觉模块；每格为 8 次测量的中位数。吐字 tok/s 由每 token 耗时换算；主机内存为进程 RSS（含锁页内存）。</sub>

8192 块不建议配 MTP，未列。

各版本的改动见 [CHANGELOG.md](CHANGELOG.md)。

---

## 性能一览

| 配置（上下文 262144） | 约 20 万 token 真实代码<br>预填充 / 吐字（tok/s） | 约 1.6 万 token 真实代码<br>预填充 / 吐字（tok/s） | 口径 |
|---|---|---|---|
| 27B Q8_0 + DFlash2，6 × V100-16GB | 2187 / 124.5 | 2786 / 152.7 | 温度 0、不开思考 |
| Qwen3.8-Flash-Next + MTP，6 × V100-16GB | 3402 / 122.5 | 2805 / 140.5 | 温度 0、不开思考 |
| Qwen3.8-Flash-Next + MTP，2 × V100-16GB | 1300 / 43.1 | 1173 / 38.0 | 开思考 + 模型卡采样参数 |
| Qwen3.8-Flash-Next + MTP，2 × V100-16GB（同一插槽，块 2048，本版推荐配置） | `1155.48` / `43.55` | `1084.58` / `42.70` | 开思考 + 模型卡官方采样参数 + 带视觉模块 + **MTP-3（每轮 3 个草稿）** |

<sub>测试数据：2026-11-03 实测，2.0.1.2 发布构建，IBM AC922 + Tesla V100-SXM2-16GB；Qwen3.8-Flash-Next 为 UD-Q4_K_XL + MTP 草稿、带专家路由画像，27B 为 Q8_0 + DFlash2 F16；真实代码，每档新起服务、前缀不命中缓存。6 卡两行为温度 0、不开思考（与 2.0.1.1 同场交替测量，本版无变化时沿用上次测得的数字）；**本版新增的 2 卡一行是推荐配置（同一插槽两张卡 + MTP-3 投机 + 块 2048 + 带视觉模块 + 开思考 + 模型卡官方采样参数）**，与 6 卡两行及表中原有 2 卡行的口径都不同，组间速度不可直接比较。全部测试条件、与上一版的同场对比及各版本数据见 [历次实测数据](docs/v100/benchmarks.md#2012)。</sub>

---

## 快速开始

- **预编译程序**：[Releases](https://github.com/1115714829/llama.cpp-v100/releases) 页面提供 x86_64 Linux 版本。
- **源码编译**：见 [27B 详情页 · 编译](docs/v100/27b.md#编译)。
- **启动命令**：[27B 详情页 · 启动参数](docs/v100/27b.md#启动参数)、[Qwen3.8-Flash-Next 详情页 · 推荐启动命令](docs/v100/moe-2.0.0.md#推荐启动命令)；不知道选哪条时看 [AI 部署指引](docs/v100/ai-deploy.md)。

### Docker 安装

镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.1.2`（也可用 `:latest`），同一个标签含 x86_64 与 IBM Power AC922（ppc64le），`docker pull` 按机器架构自动选择。宿主机需要 NVIDIA 驱动 550 或更新。

1. 拉取镜像：`docker pull ghcr.io/1115714829/llama.cpp-v100:2.0.1.2`
2. 准备 GPU 访问（只需一次）：x86 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，运行时用 `--gpus all`；AC922 用 podman（`sudo dnf install -y podman podman-docker`），在源码目录执行 `sudo bash .devops/v100-ac922-cdi.sh` 生成 GPU 描述，运行时用 `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8`。
3. 启动（以 27B、x86、4 卡为例；`<模型目录>` 写绝对路径）：

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <模型目录>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.1.2 \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --port 8080
```

服务起来后访问 `http://<主机>:8080`（OpenAI 兼容接口 `/v1/chat/completions`）。AC922 6 卡、Qwen3.8-Flash-Next 的 Docker 命令，以及不能访问 ghcr.io 时用 Releases 页面的镜像文件导入的方法，见 [27B 详情页 · Docker](docs/v100/27b.md#docker) 与[Qwen3.8-Flash-Next 详情页 · 编译与 Docker](docs/v100/moe-2.0.0.md#编译与-docker)。

---

## 测试平台

- **标准测试机**（本项目的主要数据都在这台机器上测）：IBM Power AC922，2 × POWER9，96 GB 内存（两个 CPU 节点各 48 GB），6 × Tesla V100-SXM2-16GB，CUDA 12.4。每颗 CPU 连 3 张 GPU，同一颗 CPU 下的 GPU 之间 NVLink 直连；GPU 与 CPU 之间是 NVLink 2.0。
- **参考主机**（x86）：4 × Tesla V100-SXM2-32GB（GPU 之间 NVLink 全互联，GPU 与主机之间 PCIe 3.0，136 GB 内存，1 个 NUMA 节点，CUDA 12.8），用于 Qwen3.8-Flash-Next 的 4 卡参考数据；2 × Tesla V100-SXM2-16GB（NVLink，PCIe 3.0，CUDA 12.8），用于编译与正确性测试。

同一套代码在 AC922 与 x86 上运行，之后会按架构分别优化。部署时的差别：

- AC922 把 GPU 显存并入系统内存（每张 GPU 一个 NUMA 节点），读大文件产生的页缓存可能占用显存，所以要用 `numactl --membind=<CPU 节点>` 启动（例如 `0,8`），x86 不需要；AC922 上每张卡可用的显存也比 x86 上同型号略少。
- Qwen3.8-Flash-Next 显存放不下的部分在主机内存里，速度受主机内存的带宽与容量、以及 GPU 与主机之间链路的影响（AC922 为 NVLink 2.0，6 卡配置每张 GPU 2 条、理论单向 50 GB/s；x86 一般为 PCIe 3.0 x16，理论单向约 16 GB/s）：内存通道插满、容量充足时更快。

---

## 版本

- 当前版本 **2.0.1.2**（2026-11-03）。各版本的改动见 [CHANGELOG.md](CHANGELOG.md)，各版本的实测数据见[历次实测数据](docs/v100/benchmarks.md)。
- 1.1.1 时期 27B 在 2～6 卡、Q4 配置、多路并发、多模态等配置的实测表与启动参数见 [27B 实测归档（1.1.1）](docs/v100/27b-results-1.1.1.md)；更早版本的说明见对应的 `v100-x.y.z` 标签。

---

## 交流群

| 微信 | QQ 群 |
|---|---|
| <img src="media/v100-wechat.jpg" width="220" alt="微信二维码"> | <img src="media/v100-qq-group.jpg" width="220" alt="QQ 群二维码"> |
| 扫码加作者微信，由作者拉你进微信群（微信群二维码会过期，所以先加好友） | 扫码直接加入 QQ 群 |

---

## 许可

本项目自 2.0.0 起采用 GPL-3.0 许可（全文见 `LICENSE`）；1.1.1 及更早版本为 MIT。上游 llama.cpp 的 MIT 许可文本保留在 `licenses/LICENSE-llama.cpp`，移植的第三方代码保留各自原许可，出处与许可见各文件头与 `licenses/README.md`。

---

## 致谢

**特别感谢群友正博、蓝莲、小王的王Ww 的资金赞助。**

**感谢群友 [umacro](https://github.com/umacro)、看透<!-- TODO: GitHub 链接待补 -->、[天语星落](https://github.com/ativx928)、[lsaac](https://github.com/isaac-bj)。** 没有他们提供的测试机设备与 token，本项目不会这么快实现。

参与开发：仓库作者（发起项目，提供硬件与测试环境，确定目标与取舍）；外部贡献者 [天语星落（ATIVX928）](https://github.com/ativx928)（[PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1) 双卡 Q4 支持、[PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2) Q2_K～Q6_K 小批量内核扩展）、[umacro](https://github.com/umacro)（[PR #3](https://github.com/1115714829/llama.cpp-v100/pull/3) 多模态 FFmpeg 诊断）、[wangxso](https://github.com/wangxso)（[PR #9](https://github.com/1115714829/llama.cpp-v100/pull/9) 按设备 SM 数确定分组验证注意力的 KV 切分数）、[Poqianhu7895](https://github.com/Poqianhu7895)（[PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4)，并报告和定位了 [#8](https://github.com/1115714829/llama.cpp-v100/pull/8)、[#13](https://github.com/1115714829/llama.cpp-v100/issues/13) 两处并发问题）。参考的项目与使用的 AI 助手见[文末致谢](#致谢项目与-ai-助手)。

### 致谢（项目与 AI 助手）

- **参考与借鉴的项目**（移植代码的出处与许可见各文件头与 `licenses/`）：[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)（本项目的基础框架）；[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（V100 工程实践的参考，小批量量化矩阵乘、分组验证注意力、D256 预填充注意力、推式 all-reduce 等 SM70 内核移植自它）；[Strata](https://github.com/Niko1221/Strata) 及其 AC922 分支 [Strata-AC922](https://github.com/eelgaev/Strata-AC922)（Qwen3.8-Flash-Next 的专家融合矩阵乘、QSA 选择与注意力、超连接融合、n-gram 表读取等移植自它们，专家缓存、预填充环与 MTP 草稿窗口参考其设计）；[SGLang-V100](https://github.com/haohervchb/sglang-V100)（推式 all-reduce 与块缩放适配的来源）；[v100-skinny](https://github.com/dnv2003/v100-skinny)（QPN8 权重布局）；[flash-attention-v100（ai-bond）](https://github.com/ai-bond/flash-attention-v100) 与 [flash-attention-v100（zhinianqin）](https://github.com/zhinianqin/flash-attention-v100)（Volta 注意力内核）；[CUTLASS / CuTe](https://github.com/NVIDIA/cutlass)（注意力内核的支撑层）；[TileLang](https://github.com/tile-ai/tilelang) 与 FlashQLA（GDN 分块预填充内核）；[ninfer-v100](https://github.com/dollarwong/ninfer-v100)（Q4 小批量内核的设计参考）；[unsloth 的 llama.cpp 分支](https://github.com/unslothai/llama.cpp)（MTP 草稿借用目标模型张量）；[vLLM](https://github.com/vllm-project/vllm)（投机解码等设计的参考）；[z-lab](https://huggingface.co/z-lab)（DFlash2 投机解码方法与草稿模型）。
- **AI 助手**：开发过程由 Claude、DeepSeek、小米 MiMo、MiniMax 协作完成。
