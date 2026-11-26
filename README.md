# llama.cpp-v100

**简体中文** | [English](README.en.md)

面向 NVIDIA V100（SM70）的 llama.cpp 优化版。当前版本 **2.0.2**。

---

> [!IMPORTANT]
> # 全部公布跑分均为严格零精度损耗
>
> **本项目的每一个公开性能数字，都在「零精度损耗」口径下取得。**
>
> - 不靠降低量化、有损低精度通信或近似计算把分数换上去；
> - 所有被改动的算子逐个通过全量算子测试；
> - 温度 0 下输出逐字节比对，并做困惑度（PPL）与 KL 散度（KLD）对照，确认模型输出没有被悄悄改变；
> - 投机解码按草稿接受率与输出一致性双重验证。
>
> 速度是真的，输出还是同一个模型。

> [!NOTE]
> # 双模型线定位：稠密 27B 走 DFlash2，大 MoE 走 MTP，两条线相对上一正式版（2.0.1.2）均零回退
>
> - **Qwen3.8-27B（稠密）+ DFlash2**：沿用既有 DFlash2 链路，相对上一正式版 **2.0.1.2**，预填充与吐字均落在测量噪声带内，温度 0 输出逐字节命中既有锚点 —— **相对 2.0.1.2 严格零回退**。
> - **Qwen3.8-Flash-Next（大 MoE）+ MTP**：专攻 MTP 投机解码，**不需要设置任何环境变量**，代码默认值即为满血状态；相对上一正式版 **2.0.1.2** 全面提速且零精度损耗，下表预填充与吐字全部是开箱即用的实测值。
> - 两条线共用同一套运行时，任何一条线的改动都必须回归另一条线的速度、显存与输出。

---

## 前言

本项目的初衷，是让大家用低成本的 V100 畅玩大模型：6 张 V100 跑 27B、上下文做到 1M，从不可能变成了可能。

**特别感谢群友正博、蓝莲、小王的王Ww 的资金赞助**，也感谢提供测试设备、协助测试的群友和网友们，项目能走到今天离不开大家（名单见[致谢](#致谢)）。

项目目前由我个人维护。维持现在的节奏，最稀缺的是两样东西：

- **时间**：项目越来越大，又要同时兼容 27B 与 Qwen3.8-Flash-Next（MoE）两条线，测试与研发时间成倍增加，已经严重挤占我的个人工作时间。发一个版本，需要多台机器连续跑十几个小时，才能走完全部验收、拿到发版数据。
- **token**：开发、排查与测试主要靠 AI 协作完成，token 消耗很大。免费的 token 已经用完，~~Claude 的 20 倍套餐 11 月底到期~~（**Claude 喜提封号，2026-11-02，祝福 Anthropic 早日倒闭**）。**现在主力是美国大豆包Gemini,感谢Google 美国大豆包🐂🍺**

所以后续更新一定会慢下来，希望大家提前有个心理预期。项目不会中断，短期内也不会停更：还有很多想法没有实现，包括一些从底层自己写的实现方案，以及因为时间关系登记下来的待优化、待处理项，都会在后面逐步推进。

**更新会变慢，但版本质量不打折。** 每个正式版发布前都要走完严苛验收（独占机器、零环境变量、全量算子测试、温度 0 逐字节与 PPL/KLD 零精度损耗核验、双模型线回归），任何一项不过就不发版。

---

## ✨ 2.0.2 核心特性总览

> [!TIP]
> ### 核心亮点
> **⚡ 万级 200K 超长预填充**（6 卡 `9,372～9,401` · 8 卡 **`11,240 tok/s`**） ｜ **🎯 100% 严格零精度损耗**（全算子 + T=0 逐字节 + PPL/KLD） ｜ **🧠 最低 32 GB 内存跑 111 GB 大 MoE**（1～8 卡通吃、不降量化不缩上下文、宿主机安全防护） ｜ **🚀 零环境变量开箱即满血**（硬件自动适配 + 内置离线中文监控 WebUI）

> 本项目只做底层推理运行环境的深度定制，**不制作也不修改量化模型**：公开 GGUF 权重原样使用，改的是面向 NVIDIA V100（SM70）的底层算子、多卡协同、三层放置与投机执行链路。

| 核心特性 | 实测范围 / 支持规格 | 说明 |
|---|---|---|
| **⚡ 万级 200K 长上下文预填充** | 纯预填充 **9,372～11,240 tok/s**；开 MTP-3 仍达 **7,412～8,471 tok/s** | 20 万 token 长代码库数秒读完；开启 `MTP-3` 后代码编辑吐字达 **119～167 tok/s**，在低成本 V100 上即可流畅处理长上下文与代码编辑 |
| **🧠 最低 32 GB 内存跑百 GB 大 MoE** | **≥ 32 GB 主机内存**即可启动 **111 GB 大 MoE**（含单张 16 GB 卡与 1～8 卡） | **统一三层自适应放置（显存 → 内存 → 磁盘）**：不靠砍量化、降精度或缩上下文凑合；内存大时满血常驻、内存小时自动分层流式计算，并自动预留系统安全余量、不把 swap 算作可用内存、避免 OOM 或拖垮宿主机 |
| **🎯 严格零精度损耗 & 双线零回退** | 全量算子回归 + T=0 逐字节一致 + PPL/KLD 双验证 | 不使用有损低精度通信或近似计算换分；稠密 27B（DFlash2）与大 MoE（MTP）双线相对 `2.0.1.2` 严格零回退 |
| **🚀 零环境变量开箱即满血** | **0 个调优环境变量**，推荐命令直达公布跑分 | 启动自动识别卡数、显存、NVLink/PCIe 拓扑与可用内存，自动规划最优放置与块大小；高级参数仍支持显式锁定 |
| **🌌 超长上下文 & 多路吐字优先** | 大 MoE 原生 **256K**；稠密 27B YaRN 最高 **512K～1M（1,048,576）** | 可容纳整仓代码与长程 Agent 记忆；`-np N` 多路并发内置**吐字优先限流调度（`--prefill-pace`）**，后来的长请求不会阻塞正在吐字的会话 |
| **🔥 双投机解码引擎分线专精** | 大 MoE 走 **MTP**（+ 可选有界后缀查找）；稠密 27B 走 **DFlash2** | 按模型结构量身选型，针对真实生产采样惩罚与中英文代码/长文场景专项提速，带来 **2～3×** 无损吐字加速 |
| **🖥️ 跨架构 · 全卡数 · 16G/32G 通吃** | **x86_64 + IBM POWER9 AC922（ppc64le）**；**1/2/3/4/5/6/8 卡**；**16G / 32G** | 一套源码通吃主流双架构与所有常见 V100 卡数（兼容 3/5/6 等非整除注意力头数拓扑），同步提供双架构 Docker / Podman 镜像与 x86 预编译包 |
| **🧩 紧跟上游现代低比特生态（SM70 硬件原生承载）** | **FP8 / NVFP4 / MXFP4 / Q8_0 / Q4_K** 及上游现代架构 | 同步上游 llama.cpp 现代权重格式与新模型支持，以 Volta（SM70）真实硬件原生算力承载低比特计算（非慢速 CPU 模拟），新模型下载即跑 |
| **🖼️ 全 GPU 多模态推理** | 图片 / 小图增强 / PDF（转图） / 视频抽帧（`ffmpeg`） | 视觉编码器（`mmproj`）直接在 GPU 加速运行，完整提供 OpenAI 兼容 `/v1/chat/completions` 多模态接口 |
| **🌐 全新内置离线中文监控 WebUI + 清爽日志 + AI 部署指令** | 随源码自带 **70 件前端资源**，零联网零 Node.js 编译内嵌 | 彻底替换官方界面：**默认简体中文 + 日间/夜间双主题一键切换**，内置同源实时状态监控（**实时预填充/吐字速度与 60 秒曲线、单条消息吸收/生成明细、6 档中文思考等级、逐张 GPU 显存/利用率/温度/实时功耗 W**）；默认日志干净不刷屏；附带可直接粘贴给 AI 助手的标准部署规范与魔搭直链 |

---

## 性能矩阵

> [!TIP]
> ### 🚀 200K 超长上下文极限预填充实力（无投机解码、零环境变量、严格零精度损耗）
>
> | **IBM AC922 · 6 × 16 GB** *(16 GB 权威定论)* | **x86 · 6 × 32 GB** | **x86 · 8 × 32 GB** *(实破 1.1 万)* | **x86 · 4 × 32 GB** |
> |:---:|:---:|:---:|:---:|
> | **`9,372 tok/s`** | **`9,401 tok/s`** | **`11,240 tok/s`** | **`6,664 tok/s`** |
>
> *(6 卡 200K 纯预填充已逼近 10,000 tok/s，8 卡实破 11,240 tok/s；下方完整对照表展示实际日常开启投机解码后的预填充与吐字全平台表现。)*

### 表 1｜Qwen3.8-Flash-Next（MoE）+ MTP-3 投机解码：全平台对比矩阵（200K 上下文）

> 统一口径：262144 上下文、4096 块、单路、零环境变量默认配置、每格多次取中位；**预填充与吐字同报**。

| 平台类别 | 卡数 × 显存 | 测试平台（客观硬件环境） | 200K 预填充 tok/s | 200K 吐字 tok/s | 代码编辑 / 新写代码 / 自然写作 吐字 (tok/s) |
|---|---|---|---:|---:|---|
| **16 GB 权威定论** | **2 × 16 GB** | IBM POWER9 AC922 双路 96 GB；同插槽 NVLink2、GPU–CPU NVLink2；CUDA 12.4（4096 块） | **2528.49** | **36.41** | 49.49 / 51.75 / 39.24 *(2048 块)* |
| **16 GB 权威定论** | **4 × 16 GB** | IBM POWER9 AC922 双路 96 GB；双插槽跨岛 NVLink2；CUDA 12.4 | **4755.04** | **69.54** | 92.74 / 90.03 / 65.84 |
| **16 GB 权威定论** | **6 × 16 GB** | **IBM POWER9 AC922 双路 96 GB；双插槽 6 卡 NVLink2；CUDA 12.4** | **7411.83** | **60.57** | **118.78 / 111.12 / 64.47** |
| x86 16 GB 参考 | **2 × 16 GB** | x86_64 96 核 126 GB；2 × V100-SXM2-16GB 300W 同岛 NVLink2；PCIe 3.0；CUDA 12.8 | **681.76** | **22.02** | 20.40 / 20.31 / 20.24 *(2048 块)* |
| x86 16 GB 参考 | **4 × 16 GB** | x86_64 96 核 126 GB；4 × V100-SXM2-16GB 300W 同岛 NVLink2；PCIe 3.0；CUDA 12.8 | **1947.81** | **55.19** | 71.52 / 69.62 / 55.11 |
| x86 16 GB 参考 | **6 × 16 GB** | x86_64 96 核 126 GB；6 × V100-SXM2-16GB 300W；PCIe 3.0；CUDA 12.8 | **4158.75** | **65.54** | 104.43（有界查找 **122.36**） / 98.22 / 57.21 |
| x86 32 GB 参考 | **2 × 32 GB** | x86_64 104 核 252 GB；2 × V100-SXM2-32GB 同岛 NVLink2；PCIe 3.0；CUDA 12.8 | **2318.28** | **65.20** | 77.20 / 84.49 / 62.56 *(2048 块)* |
| x86 32 GB 参考 | **4 × 32 GB** | x86_64 44 核 136 GB；4 × V100-SXM2-32GB 全互联 NVLink2；PCIe 3.0；CUDA 12.8 | **5771.30** | **66.57** | 126.55（有界查找 **139.16**） / **116.82** / **66.18** |
| x86 32 GB 参考 | **6 × 32 GB** | x86_64 104 核 252 GB；8 × V100-SXM2-32GB 双 4 卡 NVLink2 岛（取 6 卡）；PCIe 3.0；CUDA 12.8 | **7597.04** | **89.53** | 130.40（有界查找 **166.69**） / **120.22** / **67.27** |
| x86 32 GB 参考 | **8 × 32 GB** | x86_64 104 核 252 GB；8 × V100-SXM2-32GB 双 4 卡 NVLink2 岛全机；PCIe 3.0；CUDA 12.8 | **8470.63** | **87.43** | 125.49（有界查找 **159.12**） / 115.95 / 65.81 |

**怎么看懂表里两组吐字和预填充的差异？**

- **为什么「200K 吐字」是 60～90 tok/s，而后面的「代码编辑 / 新写代码」能跑到 110～167 tok/s？**
  1. **场景可预测度与草稿命中率不同**：在「代码编辑」（基于已有代码片段做修改重构）和「新写代码」场景下，上下文结构规律极强，MTP-3 草稿接受率高达 **87%～99.8%**（配合有界后缀查找每轮可一次连吐 4～7 个 token），吐字直接飙上 **110～167 tok/s**；而在「自然写作」（开启真实生产采样：`temperature=0.6, top_p=0.95, presence_penalty=1.0`）和 200K 超长文续写下，文本发散度高、草稿接受率约 **42%～67%**，每轮平均吐 2～3 个 token，速度相应落在 **60～90 tok/s**。
  2. **上下文长度开销不同**：200K 吐字在每生成一个新词时，都要在 20 万 token 级超长历史里做稀疏注意力检索与缓存访问；而日常中短提示词（数百到数千 token）的注意力负担轻得多，因此吐字更快。
- **为什么开 `MTP-3` 后 200K 预填充是 7,412 / 7,597 / 8,471，而不是顶部的 9,372 / 9,401 / 11,240？**
  - 顶部是关闭投机时纯 48 层主模型的极限预填充速度；开启 `MTP-3` 后，末卡上的草稿层也必须把 200K 上下文同步预填充一遍以建立状态，从而换取解码吐字 **2～3 倍**的大幅提升。
- **16 GB 权威定论标准机与跨平台差异**：IBM POWER9 AC922（6 × V100-SXM2-16GB，96 GB）是本项目的 16 GB 权威定论标准机；x86 16 GB 与 32 GB 各行清晰反映了 CPU–GPU 互联带宽（AC922 GPU–CPU NVLink2 vs x86 PCIe 3.0）与显存容量（16 GB 混合专家池 vs 32 GB 更大常驻槽）的真实表现。
- **可复现性与零精度损耗互证**：AC922 6 × 16 GB 独立复现窗得 7384.79 / 61.43 tok/s（互差 ≤ 1.42%，输出逐字节相同）；x86 16 GB 2/4/6 卡在 200K 下的 T=0 贪心输出**跨卡数逐字节完全相同**；AC922 2 × 16 GB 在 2048 块下 200K 为 1375.39 / 24.70 tok/s，x86 2 × 16 GB / 2 × 32 GB 在 2048 块下分别为 432.14 / 14.05 与 1273.16 / 61.99 tok/s。
- **中文场景实测（x86 6 × 32 GB，MTP-3 默认 / 有界查找）**：中文问答 **89.40 / 90.24 tok/s**、中文写作 **63.88 / 63.38 tok/s**、中文代码 **96.17 / 91.79 tok/s**；8 × 32 GB 预填充最强（8470.63，有界查找 8478.28），6 × 32 GB 单路吐字略优（89.53 vs 87.43）。

### 表 2｜Qwen3.8-27B（稠密）+ DFlash2 投机解码（零回退验证门）

| 卡数 × 显存 | 测试平台（客观硬件环境） | 测试口径 | 预填充 tok/s | 吐字 | 零回退验证结果 |
|---|---|---|---:|---:|---|
| **2 × 16 GB** | IBM POWER9 AC922 双路 96 GB（同插槽 NVLink2） | 上下文 32768，提示 2048 → 128 token，温度 0、关思考 | **2116.48** | **129.9 tok/s**（7.70 ms/tok） | **4/4 冷启动逐字节命中既有锚点，草稿接受 101/210，预填充 +0.19%、吐字 −0.84%（带内零回退）** |

> 27B 稠密线与历史版本保持完全一致的张量并行 + DFlash2 路径与多卡配置（2～6 卡、YaRN 524288、多路并发与视觉多模态），详尽历史多卡配置与用法见归档页与启动说明。

---

## 本版 vs 上一正式版 2.0.1.2：自比一览

> 仅与本项目自己的上一正式版比较，绝不与外部项目混口径对比；口径不一致的格子只并列展示、不硬算虚假百分比。

| 配置（IBM POWER9 AC922，16 GB 权威定论机） | 上一正式版 2.0.1.2 | 本版实测（零环境变量、MTP-3） | 变化说明 |
|---|---|---|---|
| **2 × 16 GB MoE，200K，4096 块** | 2151.70 / 41.53 tok/s（开思考 + 官方采样 + 视觉） | **2528.49 / 36.41 tok/s**（温度 0 关思考、零环境变量） | 4096 块 200K 预填充跃升至 2528+ tok/s（口径略有差异，并列呈现） |
| **4 × 16 GB MoE，200K，4096 块** | *2.0.1.2 未列 4 卡 MoE* | **4755.04 / 69.54 tok/s** | 新增完整 4 卡零环境变量实测 |
| **6 × 16 GB MoE，200K，4096 块** | 3402 / 122.5 tok/s（旧架构口径） | **7411.83 / 60.57 tok/s**（编辑 **118.78** / 代码 **111.12**） | 200K 预填充由 3402 跃升至 **7411.83 tok/s（2.18×）** |
| **27B Q8_0 + DFlash2（零回退门）** | 2.0.1.2 基准锚点 | **2116.48 tok/s / 129.9 tok/s（4/4 逐字节同锚点）** | **严格零回退、零精度损耗** |

---

## 深度定制说明与 AI 一键部署规范

> [!WARNING]
> **本项目是对 NVIDIA V100（SM70）深度定制的专用推理引擎，不是普通上游模板的简单封装。**
>
> - **默认参数即满血**：推荐命令中的默认配置已经过大量真机独占调校，**直接按推荐参数启动即可跑出上表实测性能**，不要自行堆砌网传环境变量或随意删改核心参数。
> - **缺什么依赖就补什么依赖，绝不“阉割编译”**：例如 **NCCL**（多卡通信必需，须同时具备 `lib/libnccl.so.2` 与 `include/nccl.h`）、**CUDA 12.x**、**GCC 12+**、**numactl**（AC922 必装）、**nvidia-persistenced**（AC922 驱动持久化必开）以及 **ffmpeg**（视频多模态）。**缺少依赖必须先完整安装配置好，才能跑出本项目的真实效果**；如果为了省事关掉 NCCL、去掉 SM70 架构参数或删掉投机/画像参数，速度会大幅暴跌，甚至无法启动。

### 一键复制给你的 AI 助手（标准部署规范指令）

如果你打算让自己的 AI 助手（Claude Code / Codex / Cursor / OpenCode / DeepSeek 等）帮你部署，**直接完整复制下面整段代码块发给它**，约束它严格按规范执行、不要擅自更改编译与启动参数：

```text
请严格按照以下硬规矩，帮我在当前服务器部署 llama.cpp-v100。这是一个面向 NVIDIA V100（SM70）深度定制的高性能推理引擎，你必须严格遵守文档建议，禁止任何擅自发挥、魔改参数或降级绕过：

【铁律 1：先摸底硬件，禁止盲猜】
1. 动手前先依次执行只读检查：nvidia-smi、nvidia-smi topo -m、numactl -H、free -h、df -h、nvcc --version、gcc --version、cmake --version，并查找本机是否存在 libnccl.so.2 与 nccl.h。
2. 先把探测到的硬件规格（架构 x86_64/ppc64le、V100 卡数与显存 16G/32G、NVLink 拓扑、主机内存与有内存的 NUMA 节点、空闲磁盘盘符、缺失依赖清单）汇报给我确认，再开始安装与编译。

【铁律 2：缺什么补什么，严禁阉割编译选项】
1. 编译必须严格使用 README 指定的全部核心选项：
   -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=<真实路径> -DNCCL_LIBRARY=<真实路径>
2. 如果缺少 NCCL、GCC 12+、CUDA 12.x 或 numactl，必须老老实实帮我安装配置齐全并验证可用；严禁为了省事私自把 GGML_CUDA_NCCL 改成 OFF、把 CMAKE_CUDA_ARCHITECTURES=70 删掉或改成 native、或关闭 FlashAttention / CUDA Graphs。
3. 本仓库源码已内置离线 WebUI（tools/ui/dist/），编译时会自动离线嵌入 llama-server，严禁执行 npm install / npm build、严禁尝试从外网重下前端，也严禁修改仓库内的任何 C++/CUDA 源码。

【铁律 3：严格照搬默认推荐参数，禁止私自魔改】
1. 启动命令必须严格按本仓库 README 中与我硬件（卡数/显存/模型）完全匹配的推荐命令执行，只需替换真实的模型与画像文件路径。
2. 本项目所有核心优化已默认开启，严禁你自作聪明添加任何多余的 GGML_* / LLAMA_* 环境变量！
3. 严禁擅自更改并行策略（大 MoE Qwen3.8-Flash-Next 必须用 --split-mode layer + --fit on + --expert-profile + MTP 投机参数；稠密 Qwen3.8-27B 必须用 --split-mode tensor + DFlash2 投机参数），严禁私自改小 -c、私自更改 -b/-ub、私自删掉投机解码或 --expert-profile、严禁私自替换为更低比特的量化模型；下载模型请严格使用 README 推荐清单中的对应仓库与文件名（优先魔搭 ModelScope），确保主模型、草稿与视觉模块配套一致。
4. 若本机是 IBM AC922（ppc64le）：必须确认 nvidia-persistenced 已启动（Persistence-M=On），且启动命令必须带 numactl --membind=<仅有真实主机内存的 CPU 节点，如 0,8>，绝对禁止绑到 GPU 显存节点（250-255）或无内存的空节点！普通 x86 服务器则不要加 AC922 的 membind 节点号。

【铁律 4：遇错停下报根因，严禁糊弄绕过】
1. 任何编译报错、依赖缺失、模型校验失败或启动 OOM，必须如实把完整报错原文展示给我并说明根因与正规修法；严禁私自通过关闭功能、降精度、砍上下文或改源码来“假装部署成功”。
2. 启动后必须逐项自检：/health 返回 200、用一条真实请求实测预填充与吐字速度（tok/s）、核对 nvidia-smi 逐卡显存占用与浏览器访问内置 WebUI（http://<IP>:<端口>）正常，默认保持简洁日志（排错时才临时加 -v）。全部验证通过后把最终启动命令、端口与实测速度汇报给我。
```

---

## 开箱即用

**预编译程序**：[Releases](https://github.com/1115714829/llama.cpp-v100/releases)（x86_64 Linux）。
**源码编译**：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR=/path/to/nccl/include -DNCCL_LIBRARY=/path/to/nccl/lib/libnccl.so.2
cmake --build build -j --target llama-server llama-quantize
```

### 内置离线 WebUI

网页界面随源码自带（`tools/ui/dist/`），编译时直接嵌入 `llama-server`：克隆下来即可完全离线编译，不需要联网下载界面资源，也不需要 Node.js（只有改了界面源码才需要重新构建界面）。服务起来后打开 `http://<主机>:8080` 就能用网页对话，OpenAI 兼容接口在 `/v1/chat/completions`。默认发布版日志简洁不刷屏，排查问题时加 `-v`（或 `-lv 5`）打开详细日志。

### Qwen3.8-Flash-Next（MoE）+ MTP

下面命令**不需要设置任何环境变量**，默认值即为满血。AC922 上保留 `numactl --membind=<节点>` 一段（把主机内存限定在有 CPU 的 NUMA 节点），普通 x86 服务器去掉这一段。

**6 × 16 GB（AC922）**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./build/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 999 \
  --split-mode layer -c 262144 --fit on --fit-ctx 262144 -np 1 \
  -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 --jinja --reasoning off \
  --expert-profile /models/flash-next-profile.txt \
  --model-draft /models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-ubatch 256 --spec-draft-window 32768 \
  --host 127.0.0.1 --port 8080 --metrics
```

**4 × 32 GB（x86）**：去掉 `numactl` 一段，改 `CUDA_VISIBLE_DEVICES=0,1,2,3`，其余不变。

**8 × 32 GB（x86）**：同上，改 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

**2 × 16 GB**：改 `CUDA_VISIBLE_DEVICES=0,1`（建议同一 CPU 插槽的两张卡），其余不变；需要的主机内存更多。

- 想要吐字更快、以对话为主：把 `-b 4096 -ub 4096` 改成 `-b 2048 -ub 2048`。
- 不处理图片时可以去掉 `--expert-profile` 一行（通常更慢）；不带 `--model-draft` 与 `--spec-…` 几行即关闭投机解码。
- 每卡显存接近用满是正常的（多出来的显存用来提速）；显存不够先调小 `-c`。

### Qwen3.8-27B（稠密）+ DFlash2

**4 × 16 GB**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --host 127.0.0.1 --port 8080 --metrics
```

改 `CUDA_VISIBLE_DEVICES` 与 `--tensor-split`（几张卡写几个 `1`）即可适配 2 / 6 卡；2 卡请改用 Q4 配置（主模型换成 `Qwen3.8-27B-UD-Q4_K_M.gguf`，`-ctk q4_0 -ctv q4_0`，草稿换 `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`）。

### 推荐模型与下载地址

> **国内用户强烈建议优先从魔搭（ModelScope）直链下载**（支持 `wget -c` 断点续传；IBM AC922 上建议用 `numactl --membind=0,8` 包裹下载命令，防止页缓存挤占 GPU 显存节点）。本项目只做推理引擎，不制作也不修改权重，请严格按下面清单下载配套模型与草稿，切勿混用不同模型的草稿或视觉模块。

#### 1. Qwen3.8-Flash-Next（大 MoE）+ MTP 推荐清单

文件全部在 unsloth 同一个仓库里（[魔搭 ModelScope](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) ｜ [Hugging Face](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)）：

| 用途 | 精确文件名与路径 | 大小 | 说明 |
|---|---|---|---|
| 目标模型（4 分片） | `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf` | 合计约 111 GB | 4 个分片放同一目录，启动 `-m` 指第 1 个分片 |
| MTP 草稿（显存更省，推荐 16 GB） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 约 1.9 GB | **必须选带 `shared` 字样的草稿**与主模型配套 |
| MTP 草稿（32 GB / 高精度备选） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 约 2.8 GB | unsloth 官方推荐，与 Q4_K_M 接受率相近 |
| 专家路由画像 | `flash-next-profile.txt` | 约 370 KB | 随本仓库 [Releases](https://github.com/1115714829/llama.cpp-v100/releases) 附带，用于热专家优先驻留显存 |

```bash
# 一键从魔搭断点续传下载 Qwen3.8-Flash-Next（AC922 请在 wget 前加 numactl --membind=0,8）
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O "models/$f" "$base/$f"
done
# 如需 Q8_0 草稿（可选）：
# wget -c -O models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf "$base/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
```

#### 2. Qwen3.8-27B（稠密）+ DFlash2 推荐清单

- **★ 作者正式服务同款推荐（强烈推荐 · 魔搭三件套即下即用，无需转换草稿）**
  仓库：[魔搭 ModelScope（推荐）](https://modelscope.cn/models/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF) ｜ [Hugging Face](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF)（全部在仓库 `Q8_0/` 目录下，**配套 Q8_0 DFlash2 草稿开箱直用，免去 HF 草稿失效或手动转 F16 的麻烦**）：

| 用途 | 精确文件名（仓库 `Q8_0/` 目录下） | 大小 | 说明 |
|---|---|---|---|
| 目标模型 | `Qwen3.8-27B-EfficientThink-SimPO-Q8_0.gguf` | 约 28.6 GB | 作者自用主力 27B（SFT + SimPO 微调） |
| DFlash2 草稿 | `dflash2-qwen38-27b-Q8_0.gguf` | 约 2.1 GB | 配套 Q8_0 草稿，**下载即用无需转换** |
| 视觉模块（可选） | `mmproj-Qwen3.8-27B-Q8_0.gguf` | 约 0.6 GB | 图片 / PDF / 视频多模态输入用 |

```bash
# 一键从魔搭断点续传下载作者同款 27B + DFlash2 Q8_0 三件套
base=https://modelscope.cn/models/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/resolve/master/Q8_0
mkdir -p models/27b
for f in Qwen3.8-27B-EfficientThink-SimPO-Q8_0.gguf dflash2-qwen38-27b-Q8_0.gguf mmproj-Qwen3.8-27B-Q8_0.gguf; do
  wget -c -O "models/27b/$f" "$base/$f"
done
```

- **标准原版 27B（备选 / 2 卡 Q4 配置）**
  - 主模型与视觉模块：[魔搭 ModelScope `unsloth/Qwen3.8-27B-GGUF`](https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF) ｜ [HF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)——4～6 卡用 `Qwen3.8-27B-Q8_0.gguf`（约 29 GB），2 卡 16 GB 用 `Qwen3.8-27B-UD-Q4_K_M.gguf`（约 16.5 GB），视觉模块 `mmproj-F16.gguf`（约 0.9 GB）；
  - DFlash2 草稿：优先使用上面魔搭 `Merkyor` 仓库配套的 `dflash2-qwen38-27b-Q8_0.gguf`（或查魔搭 [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://modelscope.cn/models/z-lab/Qwen3.8-27B-DFlash2-GGUF) 的 `Q8_0` / `Q4_K_M`；注：部分海外镜像旧链接可能不可用，若下到 BF16 请先用 `./build/bin/llama-quantize <输入BF16> <输出F16> F16` 转为 F16，因 V100 无原生 BF16 硬件加速）。

### 更多用法

- **27B 超长上下文（512K / 1M YaRN）**：
  - **524288（512K）**：`-c 524288 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 --override-kv qwen35.context_length=int:524288`；
  - **1048576（1M，6 × 16 GB 可跑）**：`-c 1048576 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 --override-kv qwen35.context_length=int:1048576`（YaRN 设置对所有长度生效）。
- **多路并发**：`-np N`，每路上下文 = `-c` ÷ N；单个 Agent 要开满长上下文时仍用 `-np 1`。
- **多模态**：加 `--mmproj /models/mmproj-F16.gguf`（27B 与 Flash-Next 的 mmproj 不是同一个文件，别混用），处理视频再加 `--video-ffmpeg-dir <目录>`；请求走 OpenAI 兼容的 `/v1/chat/completions`。
- **思考模式**：请求体加 `"chat_template_kwargs": {"enable_thinking": true}` 开启、`false` 关闭；思考内容默认在 `message.reasoning_content` 字段，只读 `message.content` 的客户端看不到。
- **小图**：建议加 `--image-min-tokens 1024`，否则小图产生的图像 token 太少、识别变差。

### Docker

镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.2`（也可用 `:latest`），同一标签含 x86_64 与 IBM Power AC922（ppc64le），`docker pull` 按机器架构自动选择。宿主机需要 NVIDIA 驱动 550 或更新。

```bash
docker run -d --name llama-v100 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8080:8080 -v <模型目录>:/models \
  ghcr.io/1115714829/llama.cpp-v100:2.0.2 \
  -m /models/Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1 \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft /models/Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --port 8080
```

AC922 上用 podman（`sudo dnf install -y podman podman-docker`），在源码目录执行 `sudo bash .devops/v100-ac922-cdi.sh` 生成 GPU 描述，运行时加 `--device nvidia.com/gpu=all -e LLAMA_NUMA_MEMBIND=0,8`。服务起来后访问 `http://<主机>:8080`（OpenAI 兼容接口 `/v1/chat/completions`）。

---

## 硬件自动适配、32 GB 低内存支持与三层放置

- **零配置默认**：启动时自动检测卡数、显存、拓扑与主机可用内存，按当前机器条件自动规划最优放置与块大小；用户显式传参永远优先覆盖默认。
- **统一三层放置（最低支持 32 GB 内存跑 111 GB 大 MoE）**：模型与专家按统一规则沿「**显存 → 主机内存 → 磁盘**」自适应下沉——够就常驻显存，放不下进主机内存，内存仍不够则自动流式落盘，全程始终由 GPU 计算。**无需降量化、无需缩上下文**：大内存机器自动满血常驻，哪怕是仅 **32 GB 主机内存**（或只有 1～2 张 16 GB V100）的小配置机器也能稳健拉起完整百 GB 级大 MoE。
- **宿主机安全硬防护**：内置统一可用内存账本与系统安全预留（不把 swap 算作可用内存），装不下时提前明确报错并给出建议，绝不硬塞导致宿主机 OOM 假死或把系统盘写爆。
- **块大小怎么选**：`-b` / `-ub` 表示预填充每一步处理的 token 数。块越大预填充越快、吐字略慢、主机内存占用略高；日常对话写代码推荐 2048，一次吞吐整仓代码或超长文档推荐 4096。

---

## 测试平台与定论边界

- **16 GB 权威定论标准机**：IBM POWER9 AC922 双路 96 GB 内存，6 × Tesla V100-SXM2-16GB；岛内 NVLink2、GPU–CPU NVLink2；CUDA 12.4。本项目的 16 GB 定论数据一律出自该平台。
- **x86 16 GB 参考环境**：126 GB 内存，7 × V100-16GB 300W；PCIe 3.0；CUDA 12.8。
- **x86 4 × 32 GB 参考环境**：V100-SXM2-32GB 全互联 NVLink2，PVE 硬件直通，136 GB 内存；PCIe 3.0；CUDA 12.8。
- **x86 8 × 32 GB 参考环境**：双 4 卡 NVLink2 岛，252 GB 内存；PCIe 3.0；CUDA 12.8。

同一套代码在 POWER9 与 x86 上运行。除特别标注「权威定论」的行以外，其余平台的数字均为多平台参考实测，用于展示跨平台表现。

---

## 交流群

| 微信 | QQ 群 |
|---|---|
| <img src="media/v100-wechat.jpg" width="220" alt="微信二维码"> | <img src="media/v100-qq-group.jpg" width="220" alt="QQ 群二维码"> |
| 扫码加作者微信，由作者拉你进微信群（微信群二维码会过期，所以先加好友） | 扫码直接加入 QQ 群 |

---

## 许可

本项目自 2.0.0 起采用 GPL-3.0 许可（全文见 `LICENSE`）；1.1.1 及更早版本为 MIT。上游 llama.cpp 的 MIT 许可文本保留在 `licenses/LICENSE-llama.cpp`，移植的第三方代码保留各自原许可，出处与许可见各文件头与 `licenses/README.md`。

历史版本说明与历次实测数据完整归档于 [docs/v100/archive/](docs/v100/archive/README.md)。

---

## 致谢

**特别感谢群友正博、蓝莲、小王的王Ww 的资金赞助。**

**感谢群友 [umacro](https://github.com/umacro)、看透、[天语星落](https://github.com/ativx928)、[Isaac](https://github.com/isaac-bj)、愚者生（GLM-5.2 提供）、并夕夕（设备提供）、协调员（设备提供）、东海龙、孙阳、Kingsman、TangMonk5859。** 没有他们提供的测试机设备、算力与 token，本项目不会这么快实现。

**参与开发**：仓库作者（发起项目，提供硬件与测试环境，确定目标与取舍）；外部贡献者 [天语星落（ATIVX928）](https://github.com/ativx928)（[PR #1](https://github.com/1115714829/llama.cpp-v100/pull/1) 双卡 Q4 支持、[PR #2](https://github.com/1115714829/llama.cpp-v100/pull/2) Q2_K～Q6_K 小批量内核扩展）、[umacro](https://github.com/umacro)（[PR #3](https://github.com/1115714829/llama.cpp-v100/pull/3) 多模态 FFmpeg 诊断）、[wangxso](https://github.com/wangxso)（[PR #9](https://github.com/1115714829/llama.cpp-v100/pull/9) 按设备 SM 数确定分组验证注意力的 KV 切分数）、[Poqianhu7895](https://github.com/Poqianhu7895)（[PR #4](https://github.com/1115714829/llama.cpp-v100/pull/4)，并报告和定位了 [#8](https://github.com/1115714829/llama.cpp-v100/pull/8)、[#13](https://github.com/1115714829/llama.cpp-v100/issues/13) 两处并发问题）。

### 致谢（项目与 AI 助手）

- **参考与借鉴的项目**（移植代码的出处与许可见各文件头与 `licenses/`）：[llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)（本项目的基础框架）；[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（V100 工程实践的参考）；[Strata](https://github.com/Niko1221/Strata) 及其 AC922 分支 [Strata-AC922](https://github.com/eelgaev/Strata-AC922)（专家缓存、预填充环与 MTP 草稿窗口参考其设计）；[SGLang-V100](https://github.com/haohervchb/sglang-V100)（推式 all-reduce 与块缩放适配的来源）；[v100-skinny](https://github.com/dnv2003/v100-skinny)（QPN8 权重布局）；[flash-attention-v100（ai-bond）](https://github.com/ai-bond/flash-attention-v100) 与 [flash-attention-v100（zhinianqin）](https://github.com/zhinianqin/flash-attention-v100)（Volta 注意力内核）；[CUTLASS / CuTe](https://github.com/NVIDIA/cutlass)（注意力内核的支撑层）；[TileLang](https://github.com/tile-ai/tilelang) 与 FlashQLA（GDN 分块预填充内核）；[ninfer-v100](https://github.com/dollarwong/ninfer-v100)（Q4 小批量内核的设计参考）；[unsloth 的 llama.cpp 分支](https://github.com/unslothai/llama.cpp)（MTP 草稿借用目标模型张量）；[vLLM](https://github.com/vllm-project/vllm)（投机解码等设计的参考）；[z-lab](https://huggingface.co/z-lab)（DFlash2 投机解码方法与草稿模型）。
- **AI 助手**：开发过程由 Gemini 4 Neon、Claude、DeepSeek、GLM、美团 LongCat、小米 MiMo、MiniMax 协作完成。
