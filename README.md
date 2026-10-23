# llama.cpp-v100

**简体中文** | [English](README.en.md) | [AI 部署指引](docs/v100/ai-deploy.md) | [27B 详情页](docs/v100/27b.md) | [Qwen3.8-Flash-Next 详情页](docs/v100/moe-2.0.0.md) | [历次实测数据](docs/v100/benchmarks.md) | [更新日志](CHANGELOG.md)

面向 NVIDIA V100（SM70）的 llama.cpp 修改版。当前版本 **2.0.1.1**（2026-10-23）。

---

## 简介

基于 llama.cpp（上游 v0.6.0，2026-10-05 的 `d81235049`）修改，从计算内核到运行时为 V100 重新实现与优化；引用的外部实现见文末致谢，出处与许可见各文件头和 `licenses/` 目录。

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

## AI 部署指引

**给 AI 助手：部署本项目前，先完整读 [docs/v100/ai-deploy.md](docs/v100/ai-deploy.md)，按其中的步骤查硬件、选配置，再到详情页复制启动命令。** 自己部署也按这份指引走。

| 模型 | 支持的配置 | 启动命令 |
|---|---|---|
| Qwen3.8-27B（Q8_0 / Q4）+ DFlash2 投机解码 | 2～6 × V100-16GB，上下文最高 524288；支持多路并发与图片 / PDF / 视频 | [27B 详情页 · 启动参数](docs/v100/27b.md#启动参数) |
| Qwen3.8-Flash-Next（UD-Q4_K_XL）+ MTP 投机解码 | 6 × V100-16GB、2 × V100-16GB（需要约 63 GiB 主机内存）、4 × V100-32GB；上下文 262144，只测了文字输入与单路 | [Qwen3.8-Flash-Next 详情页 · 推荐启动命令](docs/v100/moe-2.0.0.md#推荐启动命令) |

指引的内容：[1. 查硬件](docs/v100/ai-deploy.md#1-查硬件)（卡数、显存、主机内存、GPU 属于哪个 CPU 插槽）→ [2. 选模型与配置](docs/v100/ai-deploy.md#2-选模型与配置) → [3. 主机内存、CPU 插槽与平台](docs/v100/ai-deploy.md#3-主机内存cpu-插槽与平台) → [4. 参数说明](docs/v100/ai-deploy.md#4-参数说明) → [5. 常见问题](docs/v100/ai-deploy.md#5-常见问题) → [6. 启动后检查](docs/v100/ai-deploy.md#6-启动后检查)。

---

## 本版更新（2.0.1.1）

- Qwen3.8-Flash-Next 支持 2 张 V100-16GB，上下文 262144，带 MTP 投机解码。
- 2 卡需要约 63 GiB 主机内存；推荐用分属不同 CPU 插槽的两张卡，同一插槽的两张卡也能跑，但主机内存余量很小。
- 6 卡 Qwen3.8-Flash-Next 与 27B 与 2.0.1 同场对比：速度没有退化，温度 0 输出相同，每卡显存相同。
- 网页界面随源码自带：克隆下来即可离线编译，编译过程不需要联网。
- Docker 镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1`。

各版本的改动见 [CHANGELOG.md](CHANGELOG.md)。

---

## 性能一览

| 配置（上下文 262144） | 约 20 万 token 真实代码<br>预填充 / 吐字（tok/s） | 约 1.6 万 token 真实代码<br>预填充 / 吐字（tok/s） | 口径 |
|---|---|---|---|
| 27B Q8_0 + DFlash2，6 × V100-16GB | 2187 / 124.5 | 2786 / 152.7 | 温度 0、不开思考 |
| Qwen3.8-Flash-Next + MTP，6 × V100-16GB | 3402 / 122.5 | 2805 / 140.5 | 温度 0、不开思考 |
| Qwen3.8-Flash-Next + MTP，2 × V100-16GB | 1300 / 43.1 | 1173 / 38.0 | 开思考 + 模型卡采样参数 |

<sub>测试数据：2026-10-23，2.0.1.1 发布构建，IBM AC922 + Tesla V100-SXM2-16GB（2 卡为同一台机器上分属不同 CPU 插槽的两张），真实代码，每档新起服务；Qwen3.8-Flash-Next 为 UD-Q4_K_XL + shared-Q8_0 MTP 草稿、带专家路由画像，27B 为 Q8_0 + DFlash2 F16。6 卡两行是温度 0、不开思考，2 卡一行是开思考 + 模型卡官方采样参数（temperature 1.0、top_p 0.95、top_k 20），口径不同，两组速度不能直接比较。6 卡两行为与 2.0.1 同场交替测量中 2.0.1.1 各轮的平均；2 卡开思考时草稿接受率随采样波动，吐字 tok/s 跟着变。完整条件、与上一版的同场对比和历次版本的数据见[历次实测数据](docs/v100/benchmarks.md#2011)。</sub>

---

## 快速开始

- **预编译程序**：[Releases](https://github.com/1115714829/llama.cpp-v100/releases) 页面提供 x86_64 Linux 版本。
- **源码编译**：见 [27B 详情页 · 编译](docs/v100/27b.md#编译)。
- **启动命令**：[27B 详情页 · 启动参数](docs/v100/27b.md#启动参数)、[Qwen3.8-Flash-Next 详情页 · 推荐启动命令](docs/v100/moe-2.0.0.md#推荐启动命令)；不知道选哪条时看 [AI 部署指引](docs/v100/ai-deploy.md)。

### Docker 安装

镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.1.1`（也可用 `:latest`），同一个标签含 x86_64 与 IBM Power AC922（ppc64le），`docker pull` 按机器架构自动选择。宿主机需要 NVIDIA 驱动 550 或更新。

1. 拉取镜像：`docker pull ghcr.io/1115714829/llama.cpp-v100:2.0.1.1`
2. 准备 GPU 访问（只需一次）：x86 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，运行时用 `--gpus all`；AC922 用 podman（`sudo dnf install -y podman podman-docker`），在源码目录执行 `sudo bash .devops/v100-ac922-cdi.sh` 生成 GPU 描述，运行时用 `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8`。
3. 启动（以 27B、x86、4 卡为例；`<模型目录>` 写绝对路径）：

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <模型目录>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.1.1 \
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

- 当前版本 **2.0.1.1**（2026-10-23）。各版本的改动见 [CHANGELOG.md](CHANGELOG.md)，各版本的实测数据见[历次实测数据](docs/v100/benchmarks.md)。
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

**感谢群友 [umacro](https://github.com/umacro)、看透<!-- TODO: GitHub 链接待补 -->、[天语星落](https://github.com/ativx928)、[lsaac](https://github.com/isaac-bj)。** 没有他们提供的测试机设备与 token，本项目不会这么快实现。

参与开发：仓库作者（发起项目，提供硬件与测试环境，确定目标与取舍）；外部贡献者 [天语星落（ATIVX928）](https://github.com/ativx928)（[PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1)、[PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2)）、[Poqianhu7895](https://github.com/Poqianhu7895)（[PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4)）。参考的项目与使用的 AI 助手见[文末致谢](#致谢项目与-ai-助手)。

### 致谢（项目与 AI 助手）

- **参考与借鉴的项目**：[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)（本项目的基础框架）；[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（V100 工程实践的参考，早期部分 SM70 内核移植自它，出处与许可见各文件头与 `licenses/`）；[Strata](https://github.com/Niko1221/Strata) 及其 AC922 分支 [Strata-AC922](https://github.com/eelgaev/Strata-AC922)（Qwen3.8-Flash-Next 的专家融合矩阵乘、QSA 选择与注意力等内核移植自它们，专家缓存与预填充环参考其设计，出处与许可见各文件头与 `licenses/`）；[vLLM](https://github.com/vllm-project/vllm)（投机解码等设计的参考）。
- **AI 助手**：开发过程由 Claude、DeepSeek、小米 MiMo、MiniMax 协作完成。
