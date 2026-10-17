# Qwen3.8-Flash-Next（大 MoE）+ MTP

**简体中文** | [English](moe-2.0.0.en.md) | [← 主页 README](../../README.md) | [27B 稠密模型](27b.md) | [大 MoE](moe-2.0.0.md) | [更新日志](../../CHANGELOG.md)

2.0.0 新增大 MoE 模型 Qwen3.8-Flash-Next 的 V100 专项支持。本页包括：支持的配置与实现概要、实测数据、模型下载、官方推荐采样参数、启动命令、专家路由画像、主机内存与显存、编译与 Docker、已知限制；最后一节“[快速部署（给 AI 助手与新手）](#快速部署给-ai-助手与新手)”给出参数影响表、判断步骤与常见问题对照。

> **支持的配置**：标准配置是 6 × V100-16GB + UD-Q4_K_XL 主模型 + shared-Q4_K_M MTP 草稿、上下文 262144（下文实测都是它）；4 × V100-32GB（x86）可以运行，专家全部常驻显存；其他卡数与量化后续陆续兼容。

---

## 模型与实现概要

官方模型卡（[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)）给出的结构：语言模型 125B 参数、每个 token 激活约 6B；另有 51B 参数的 n-gram 嵌入表（在第 2 层，用二元与三元组查表）和 4B 参数的 MTP 层（1 层）；共 48 层，每 4 层为 3 层 Gated DeltaNet（线性注意力）加 1 层 Qwen Sparse Attention（QSA，稀疏注意力）；每层 512 个专家，每个 token 选 10 个路由专家，另有 1 个共享专家；原生上下文 262144。官方模型带视觉编码器，本项目只测了文字输入。本项目在 V100 上的做法：

- **多卡与显存**：`--split-mode tensor` 下多卡协同运行；启动时按每张卡剩余的显存自动决定放在显存里的部分，放不下的部分放在主机内存、仍由 GPU 计算，不需要 `--n-cpu-moe` / `-ot`。专家路由画像（`--expert-profile`）能让常用的部分优先留在显存。
- **稀疏注意力与 MTP 投机解码**：都有 V100 专用实现，262144 上下文可在 16 GB 的卡上运行；MTP 草稿窗口与草稿物理批可单独设置（`--spec-draft-window`、`--spec-draft-ubatch`）。
---

## 实测数据

### 2.0.1 实测数据（与 2.0.0 同场交替）

测试条件：IBM AC922（2 × POWER9，主机内存 96 GB）+ 6 × Tesla V100-SXM2-16GB；Qwen3.8-Flash-Next UD-Q4_K_XL + MTP 草稿（Q4_K_M，每轮 3 个），6 卡，上下文 262144。提示词为约 20 万 token 真实代码、约 1.6 万 token 真实代码、短提示三档，对应输出 1024 / 768 / 256 token；**温度 0、不开思考**。2.0.1 与 2.0.0 在同一时段交替测量（同一台机器、同一批提示词），每个用例都新起服务、机器独占。表中 2.0.1 一列包含路由权重 BF16 存储与显存峰值修复；2.0.0 一列是 2.0.0 发布线构建的同场测量，与下文“2.0.0 发布时实测”的结果一致。

<sub>测试数据：2026-10-17，2.0.1 发布构建对 2.0.0 发布线构建，同一台机器、同一时段交替测量；表中 2.0.0 一列即该次同场测得。</sub>

| 用例 | 2.0.0 | 2.0.1 | 变化 |
|---|---|---|---|
| 约 20 万 token 预填充（tok/s） | 2344 | 3464 | +48% |
| 约 1.6 万 token 预填充（tok/s） | 2053 | 2823 | +38% |
| 约 20 万 token 吐字（tok/s） | 109.9 | 118.5 | +8% |
| 约 20 万 token 吐字每轮耗时（ms） | 25.60 | 23.04 | −10% |
| 约 1.6 万 token 吐字（tok/s） | 117.9 | 139.6 | +18% |
| 约 1.6 万 token 吐字每轮耗时（ms） | 24.59 | 21.92 | −11% |
| 短提示吐字（tok/s） | 135.5 | 151.0 | +11% |
| 短提示吐字每轮耗时（ms） | 26.62 | 23.89 | −10% |

约 20 万 token 预填充与本项目此前实测的最好水平（3459 tok/s）持平。多提示词接受数另测（24 条提示词，Q4_K_M 草稿）：每轮接受 token 数在温度 0 下 2.752 → 2.734、温度 0.6 下 2.592 → 2.579，都在测量噪声内，同批提示每轮耗时 24.4 → 21.3 ms（温度 0）、26.1 → 23.2 ms（温度 0.6）。

每卡显存峰值（各卡中的最高值，MiB）：

| 用例 | 2.0.0 | 2.0.1 |
|---|---|---|
| 大 MoE 约 20 万 token | 14838 | 14804 |
| 大 MoE 约 1.6 万 token | 14838 | 14804 |
| 大 MoE 短提示 | 14655 | 14624 |

### 2.0.0 发布时实测

标准测试机：2026-10-11，IBM Power AC922（2 × POWER9，96 GB 内存）+ 6 × Tesla V100-SXM2-16GB，CUDA 12.4，发布构建 `4d50148cc`；UD-Q4_K_XL + MTP shared-Q4_K_M（每轮起草 3 个）、带专家路由画像，用下文命令 A；非流式、温度 0、不开思考，每个请求都新起服务（前缀不命中缓存），共两轮（表中“a / b”为两轮各自的结果，两轮的输出逐字相同）。真实代码 = 一个开源 C++ 项目的源码加一句概括要求。

| 请求 | 提示 token | 输出 token | 预填充（tok/s） | 吐字（tok/s） | 草稿接受（接受 / 起草） | 请求后每卡显存 |
|---|---:|---:|---:|---:|---:|---:|
| 短提示（一句话的编程题） | 33 | 256 | — | 134.1 / 134.4 | 185 / 213（87%） | 14648～14652 MiB |
| 真实代码，约 1.6 万 token | 16030 | 768 | 2022.2 / 2069.1 | 117.4 / 116.3 | 503 / 795（63%） | 14810～14835 MiB |
| 真实代码，约 20 万 token | 209030 | 686（上限 1024，自然结束） | 2334.8 / 2340.8 | 107.9 / 109.7 | 442 / 729（61%） | 14793～14838 MiB |

- **稳定性**（2026-10-12，发布构建，同一命令）：3 次冷启动（第一个请求即约 1.6 万 token）全部正常，预填充 2055.4～2060.9 tok/s；20 分钟 41 个混合请求（短提示到约 20 万 token，温度 0 与 0.7 交替）0 失败，单卡显存峰值 14974 MiB。从启动到可以服务约 3 分钟。
- **温度大于 0**：同一服务交替发温度 0 与温度 0.7 请求的长跑实测，温度 0.7 时草稿接受率从 0.58～0.66 降到 0.54～0.58，吐字低 12%～17%。
- **与 27B 同机对比**：同一台机器、同一构建、同一份约 20 万 token 真实代码，大 MoE 的预填充比 27B（Q8_0 + DFlash2）高约 13%，吐字约为 27B 的 87%。
- **参考主机：x86，4 × V100-SXM2-32GB**（2026-10-11，GPU 之间 NVLink 全互联、GPU 与主机之间 PCIe 3.0，136 GB 内存，1 个 NUMA 节点，CUDA 12.8；发布前的较早构建 `638de68da`，仅作参考；命令 B，同一服务依次发三个请求，温度 0）：短提示吐字 95.0 tok/s；约 1.6 万 token（输出 768）预填充 2239、吐字 88.2 tok/s；约 20 万 token（开头约 1.4 万 token 命中前一请求的缓存，实际处理 195048 token，533 token 时自然结束）预填充 2434、吐字 66.4 tok/s。专家全部常驻显存，每卡显存峰值 24916～24970 MiB。

---

## 模型下载

**国内用户建议从魔搭（ModelScope）下载。** 文件都在 unsloth 的同一个仓库里：

| 用途 | 文件 | 魔搭 ModelScope | Hugging Face |
|---|---|---|---|
| 目标模型 | `UD-Q4_K_XL/` 目录下的 4 个分片 `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf`（合计约 111 GB） | [unsloth/Qwen3.8-Flash-Next-GGUF](https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| MTP 草稿（推荐） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`（约 1.9 GB） | 同上 | 同上 |
| MTP 草稿（可选） | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`（约 2.8 GB） | 同上 | 同上 |

- 4 个分片放在同一目录，启动时 `-m` 指向第一个分片（`-00001-of-00004`）。专家路由画像 `flash-next-profile.txt` 随 Release 附带（见“专家路由画像”）。
- 文件名带 `shared` 的 MTP 草稿要和同一个主模型一起用。unsloth 推荐 shared-Q8_0；本项目用更小的 shared-Q4_K_M：实测 shared-Q8_0 的吐字与接受率相同，每卡多占约 150 MiB 显存，约 20 万 token 预填充低约 2%。
- `scripts/v100-get-models.sh` 只下载 27B 的文件；大 MoE 用直链下载（支持断点续传；AC922 上建议用 `numactl --membind=0,8` 包起来，避免页缓存落进 GPU 显存的节点）：

```bash
base=https://modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/master
# Hugging Face：base=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main
mkdir -p models/UD-Q4_K_XL models/MTP
for f in UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
         MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; do
  wget -c -O models/$f $base/$f
done
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

### A. 6 卡 16 GB（AC922），上下文 262144，带 MTP（本页实测所用）

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
- 还没有画像时去掉 `--expert-profile` 一行也能启动（专家按默认方式放置，通常更慢）；不用 MTP 时去掉 `--model-draft` 与 `--spec-…` 两行；按官方建议设默认采样时在末尾加上一节表中的一组参数。各参数的作用与影响见最后一节的参数表。

### B. 4 卡 32 GB（x86）

命令 A 去掉 `numactl --membind=0,8`，改 `CUDA_VISIBLE_DEVICES=0,1,2,3`，其余不变。4 张 32 GB 的卡放得下全部专家，主机内存里只有 n-gram 嵌入表与词嵌入表（仍要锁页，见“主机内存、显存与启动”）；画像这时只影响专家在各卡之间的分配。

**其他卡数与显存**没有做完整测试。启动时按每张卡的显存自动规划：显存总量越少，放在主机内存里的专家越多，需要的主机内存越多，吐字也越慢；主机内存放不下时无法启动。改 `CUDA_VISIBLE_DEVICES` 即可，显存不够时先调小 `-c`。

---

## MTP 草稿窗口

自 2.0.0 起提供。

MTP 投机解码（`--spec-type draft-mtp`）的草稿层只看最近的若干个位置，而不是主上下文的全部位置；窗口大小由 `--spec-draft-window N` 指定，默认 32768，设为 0 表示看全部位置。草稿只提出候选，最终输出的 token 由主模型验证后决定，因此草稿的可见范围可以比主上下文小。

| | |
|---|---|
| 参数 | `--spec-draft-window N`（环境变量 `LLAMA_ARG_SPEC_DRAFT_WINDOW`） |
| 默认值 | `32768`；`0` = 草稿层看全部位置 |
| 负数 | 启动时报参数错误并退出 |

参数默认开启，理由如下：

- 草稿的计算量与缓存都随窗口大小增长。不限制窗口时，草稿每一步的计算量和缓存占用都随主上下文增长，上下文开到几十万 token 时吐字速度下降，且草稿缓存占用显存。
- 限制窗口后，草稿每一步的计算量和缓存占用都有固定上界，与主上下文长度无关。
- 提示词长度不超过窗口时，全部位置都在草稿的可见范围内，输出与不设此参数时逐字相同；因此默认值对常见用法没有行为差异，不需要额外配置。
- 本项目的典型配置是把上下文开到数十万 token，默认开启可以让默认配置即为可用配置。

对显存与吐字的影响方向：

- **显存**：草稿上下文的缓存按窗口大小加一批的余量分配，主上下文越长、窗口越小，省下的越多。主模型自身的 KV 缓存不受本参数影响。窗口设为 0 时，草稿缓存按主上下文长度分配。
- **吐字**：草稿每一步只对窗口内的位置做计算，上下文越长，效果越明显。上下文不超过窗口时没有差别。
- **预填充**：草稿只需补齐窗口内的位置，窗口外的前缀不再逐位置补算。
- **接受率**：窗口以内的位置对草稿完全可见；超过窗口的更早位置草稿看不到，长上下文下接受率可能下降，接受率下降会部分抵消吐字上的收益。接受率与吐字的具体数字见上文“实测数据”。

关闭（草稿层看全部位置）与调整窗口，都在启动命令中追加本参数即可：

```bash
llama-server ... --spec-type draft-mtp -md <草稿文件> --spec-draft-window 0      # 关闭：草稿层看全部位置
llama-server ... --spec-type draft-mtp -md <草稿文件> --spec-draft-window 8192   # 示例值：缩小窗口，草稿缓存更小
```

本参数只作用于 MTP 投机解码的草稿上下文。使用独立草稿模型的投机解码（例如 27B 的 DFlash2）不使用本参数，启动命令中带上它也不会改变行为。

---

## --fit-mmproj

2.0.1 新增。

`--fit` 是否把多模态投影器（mmproj）的估算显存计入每卡余量（环境变量 `LLAMA_ARG_FIT_MMPROJ`）。默认 `on`：给定 `--fit` 时按现行行为把投影器的估算显存计入每卡留出的余量；设为 `off` 时不计入，投影器所需的显存需自行用 `--fit-target` 留出。张量并行（`--split-mode tensor`）下 fit 本身不运行，本参数在该配置下没有作用。

---

## 专家路由画像

MoE 模型的专家权重加起来常常比显存还大，只有一部分专家能常驻显存。默认的放置不看热度，每层分到的显存槽一样多，但真实请求里各个专家被选中的次数差得很远；多卡时各卡的计算量也会不均。`--expert-profile 文件` 用一份纯文本画像解决这两件事，每行三个或四个数 `layer expert count [launches]`：

- `count`：该层该专家被路由到的次数。启动时按它重新安排每层的专家在各卡之间的分配，使各卡分到的计算量接近。
- `launches`（可选）：路由到该专家的矩阵乘发射次数，一次发射不论带多少 token 都只算一次。吐字时每次发射用到放在主机内存里的专家都要先搬进显存，所以有这一列时每层按 `launches` 从大到小（相同时看 `count`）选常驻显存的专家；文件里只要有一行带第 4 列，整份都按它选。只有 3 列时按 `count` 选；没选进显存槽的专家放主机内存。

Release 附带的画像 `flash-next-profile.txt` 用真实请求采集，下载后直接传给 `--expert-profile`。自己生成有两种做法：

1. **用校准文本**（需要先编译 `llama-perplexity`）：

   ```bash
   # -f 校准文本，-o 画像输出路径，-c 采样时的上下文长度；-- 之后的参数原样传给 llama-perplexity
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 bash scripts/v100-expert-profile.sh \
       -m ./models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
       -f calibration.txt -o ./models/flash-next-profile.txt -c 4096 \
       -- -sm tensor -ngl 999 -ctk q8_0 -ctv q8_0
   ```

   脚本把各卡写出的次数合并成一份画像，补齐从未被路由到的专家（记 0），按 `layer expert` 排序并校验行数等于“层数 × 每层专家数”，不符就报错退出；最后打印路由总次数、被路由到的专家数和每层的分布，用来判断校准文本够不够有代表性。`--help` 查看完整用法。
2. **用真实负载**：`llama-server` 加 `--dump-expert-routing 文件`，照常使用一段时间后正常退出，每张卡写出一份 `文件.dev<卡号>`（4 列），`cat 文件.dev* > 画像.txt` 合并后交给 `--expert-profile`。中途强制结束会丢掉这一轮的统计。

- **画像只决定专家放在哪里，不改变计算公式。** 带画像时多卡求和的顺序随分卡变化，输出与不带画像时可能有极细微的差别，数值上等价；同一份画像每次启动的结果相同。带画像时启动要按画像重排专家，多花一些时间。
- **效果**：实测（只有 `count` 列的画像、不带 MTP）带画像时约 1.6 万 token 的吐字快 12%、约 20 万 token 的冷预填充快 8%，吐字时的专家命中率从 82.6% 升到约 98%。
- **按模型与用途采集。** 画像按“层号 专家号”记录，换模型要重新采；画像与卡数无关。代码语料采的画像对写代码最有利（实测预填充：代码文本快 6.8%，英文文档快 3.6%），校准文本尽量贴近主要用途。

---

## 主机内存、显存与启动

- **主机内存**：n-gram 嵌入表与词嵌入表（27466 MiB）和显存放不下的专家都放在锁页主机内存；6 × 16 GB 标准配置共锁页约 42 GiB，另要留出系统、页缓存与 `--cache-ram` 的余量。卡数越少、显存越小，需要的主机内存越多；放不下时无法启动（本版不支持从硬盘读专家）。
- **锁页内存上限**：运行服务的用户需要足够的锁页内存上限（`ulimit -l`，建议 `unlimited`；Docker 用 `--ulimit memlock=-1`），否则无法启动。
- **NUMA**：主机内存里的专家按 GPU 所在的 CPU 节点放置，锁页内存由所在节点的线程分配和填充，放置由程序在启动时实测各 GPU 的读取速度后自动决定；AC922 上仍然要用 `numactl --membind=0,8`。
- **显存**：放下模型其余部分、KV 缓存与计算缓冲之后剩下的显存都用作专家槽，所以每卡显存占用接近可用显存，这不表示显存不够；这些 GPU 不要再与其他程序共用。长时间运行中出现显存不足时调大 `-fitt`（默认 800 MiB）。MTP 草稿也占每张卡的一部分显存，显存很紧时可以不开 MTP，或调小 `--spec-draft-ubatch`。
- **启动时间**：要读入全部分片、把 n-gram 嵌入表与主机内存里的专家拷进锁页内存，带画像时还要重排专家，标准配置约 3 分钟。

---

## 编译与 Docker

编译方法与 [27B 页](27b.md#编译) 相同；生成画像的脚本还要用到 `llama-perplexity`：`cmake --build build -j --target llama-server llama-quantize llama-perplexity`。

Docker 镜像 `ghcr.io/1115714829/llama.cpp-v100:2.0.0` 的用法见 [27B 页的 Docker 一节](27b.md#docker)，大 MoE 在 Docker 下没有实测。必须加 `--ulimit memlock=-1`，宿主机要有足够的主机内存；画像文件放进挂载的模型目录。镜像里没有 `llama-perplexity`，用校准文本生成画像要在源码编译的环境里做，`--dump-expert-routing` 可以在容器里用。

- **C. Docker，6 × V100-16GB，AC922（podman）**：按 27B 页命令 C 准备 podman 与 CDI，镜像名之后换成命令 A 从 `-m` 开始的参数，模型与画像路径改成 `/models/...`，去掉 `--host 127.0.0.1`。
- **D. Docker，4 × V100-32GB，x86**：同上，按 27B 页命令 D（`--gpus all`，不设 `LLAMA_NUMA_MEMBIND`）。

---

## 已知限制

- 只测了文字输入（官方模型带视觉编码器）、单路（`-np 1`）与原生 262144 上下文。
- 6 卡 16 GB 与 4 卡 32 GB 以外的组合没有完整测试；Docker 下没有实测。
- 本版专家只放在显存与主机内存两级，主机内存放不下时无法启动（不支持从硬盘读专家）。
- n-gram 查表由 GPU 直接读主机内存，没有与官方设计中第 1 层的计算重叠做异步预取。

---

## 快速部署（给 AI 助手与新手）

本页可以整页交给 AI 助手：按下面的判断步骤选命令 A～D，把模型路径换成实际目录，再按参数表与症状对照调整。准备：`<模型目录>` 下放 4 个主模型分片、shared-Q4_K_M 草稿与画像文件（见“模型下载”）；运行服务的用户 `ulimit -l unlimited`；这些 GPU 不与其他程序共用。

### 参数表

“能力 / 速度 / 显存”三栏写的是把参数从推荐值改掉（或去掉）会怎样；“实测”为在标准测试机上的测量，“预期”为没有实测的推断。

| 参数 | 作用 | 能力 | 速度 | 显存 | 推荐值 | 什么时候改 |
|---|---|---|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 选用哪几张 GPU | 卡数与每卡显存决定放不放得下（见判断步骤） | 预期：卡越少，放在主机内存里的专家越多，吐字越慢 | — | 全部卡 | 只用其中几张卡时 |
| `numactl --membind=<CPU 节点>` | 把主机内存限定在有 CPU 的 NUMA 节点 | — | — | AC922 上不加时，读模型产生的页缓存可能落进 GPU 显存节点，CUDA 回收不了，表现为显存不足 | AC922：`0,8`；x86：不加 | 在 AC922 这类显存并入系统内存的机器上一定要加 |
| `-m` | 主模型，指向第一个分片 | 本版只测了 UD-Q4_K_XL | — | — | UD-Q4_K_XL | 不改 |
| `-ngl 999`、`--split-mode tensor`、`-fa on`、`-ctk q8_0 -ctv q8_0`；`--tensor-split` 不写 | 所有层放 GPU（显存放不下的专家自动放主机内存，仍由 GPU 计算）；路由专家按专家分到各卡，稠密部分按张量并行；Flash Attention；KV 缓存 8 位；各卡均分 | 不需要 `--n-cpu-moe` / `-ot`；`layer` 等其他切分方式与 `q4_0` KV 本版没有测 | — | 不写 `-ctk/-ctv` 时 KV 按 F16 存，专家槽相应变小 | 照写 | 不改 |
| `-c` | 上下文长度（模型原生 262144） | 单个请求的提示加输出最多这么多 token | 调小：省下的显存给专家槽，主机内存里的专家变少，吐字变快（实测 `-c 32768` 时主机内存里的专家只剩 881 MiB） | 262144 时每卡上下文 1437 MiB，调小则变小 | 262144 | 用不到长上下文、想要更快的吐字时调小 |
| `-b 2048 -ub 2048` | 逻辑批 / 物理批：预填充时每次送进模型的 token 数 | — | 调大反而更慢：实测（不带 MTP）`-ub 4096` 时约 20 万 token 预填充 −27%，`8192` 时 −43%（计算缓冲变大，挤掉专家槽） | 2048 时每卡计算缓冲 691 MiB，越大越多 | 2048 | 不改 |
| `-md`（`--model-draft`） | MTP 草稿文件，| 必须与同一个主模型配用；草稿只提议，输出由目标模型决定 | 实测 shared-Q8_0 与 shared-Q4_K_M 吐字、接受率相同，约 20 万 token 预填充低约 2% | 草稿每卡 316～330 MiB；shared-Q8_0 每卡多约 150 MiB | shared-Q4_K_M | 不改 |
| `--spec-type draft-mtp` | 用模型的 MTP 层做投机解码 | 草稿只提议，输出由目标模型决定 | 提高吐字（本页草稿接受率：短提示 87%、约 1.6 万 token 63%、约 20 万 token 61%）；开着时草稿也要处理提示词 | 草稿、草稿计算缓冲与草稿上下文合计每卡约 0.4～0.45 GB，专家槽相应变小 | 开 | 显存实在不够时去掉 `-md` 与全部 `--spec-…` 参数 |
| `--spec-draft-n-max` | 每轮最多起草几个 token（默认 3） | — | 实测 3 最快，起草更多时验证的代价大于收益 | 越大验证批越大 | 3 | 不改（官方 vLLM 配方也用 3） |
| `--spec-draft-ubatch` | 草稿上下文的物理批大小（默认继承 `-ub`） | — | — | 256 时每卡草稿计算缓冲 93 MiB；继承 `-ub 2048` 时 170 MiB | 256 | 不改 |
| `--spec-draft-sampling` | `greedy`：草稿取概率最大的 token（默认）；`probabilistic`：按概率起草，目标模型做拒绝采样 | 两种方式的输出都服从目标模型按请求参数的采样 | 本页只测了 `greedy` | — | `greedy` | 不改 |
| `--spec-draft-window` | MTP 草稿层只看最近 N 个位置（默认 32768，0 = 全部） | — | 调大或设 0：长上下文时草稿变慢 | 调大或设 0：草稿的 KV 变大 | 不写（32768） | 不改 |
| `--expert-profile` | 专家路由画像：决定每层常驻显存的专家与各卡的专家分配（见“专家路由画像”） | 只改放置，不改计算公式 | 去掉画像更慢：实测带画像时约 1.6 万 token 吐字快 12%、约 20 万 token 冷预填充快 8% | 不变 | 用与实际用途相近的内容采的画像 | 换模型或主要用途变化时重采：服务加 `--dump-expert-routing <文件>` 跑一段真实负载后正常退出，`cat <文件>.dev* > 画像.txt` |
| `-fitt`（`--fit-target`） | 启动时给每张卡留的运行期显存余量，其余显存给专家槽 | — | 调大：专家槽变小，主机内存里的专家变多，吐字变慢 | 运行期余量变大 | 不写（800） | 长时间运行后出现显存不足时调大 |
| `--cache-ram` | 主机内存里的提示词缓存上限（MiB，默认 8192，0 为关闭） | 多个对话轮流使用时可以从主机内存恢复前缀，不必整段重算 | 命中时首字更快 | 不占显存，占主机内存 | 不写（8192） | 主机内存紧张时调小（6 × 16 GB 配置已锁页约 42 GiB） |
| `--jinja`、`--chat-template-kwargs` | 使用模型自带的聊天模板；设置模板参数的服务默认值 | 思考默认开启，`'{"enable_thinking":false}'` 关闭；思考强度用请求参数 `reasoning_effort`（`xhigh` 默认、`medium`、`low`） | 关闭思考或降低思考强度，生成的 token 变少，回答更快结束 | — | `--jinja`；思考按用途 | 不需要思考时关闭 |
| `--temp`、`--top-p`、`--top-k`、`--min-p`、`--presence-penalty`、`--repeat-penalty` | 服务端默认采样参数（请求里带的优先） | 官方推荐：思考模式 1.0 / 0.95 / 20 / 0 / 0 / 1.0；非思考模式 0.7 / 0.8 / 20 / 0 / 1.5 / 1.0（见“官方推荐采样参数”） | 温度大于 0 时草稿接受率下降，实测温度 0.7 吐字低 12%～17% | — | 官方推荐值 | 按思考 / 非思考模式选一组 |
| `--host`、`--port`、`--api-key-file`、`--metrics` | 监听地址、端口、API 密钥文件、Prometheus 指标 | 默认 `127.0.0.1:8080`，只有本机能访问；`--host 0.0.0.0` 对外开放 | — | — | 本机用默认；对外用 `0.0.0.0` 并加 `--api-key-file` | 要给其他机器用时 |
| `-np`、`--mmproj`、`--video-ffmpeg-dir`、`--rope-scaling yarn` 等 | 并发路数；视觉模块；YaRN 扩展上下文 | 大 MoE 本版只测了单路、文字输入与原生 262144 上下文 | — | — | `-np 1`，其余不加 | 要处理图片 / 视频时改用 27B |

### 判断步骤

```text
第 1 步：有几张 V100，每张多少显存？
  6 × 16 GB      → 命令 A（AC922 实测）；x86 去掉 numactl 那一段（没有测过）
  4 × 32 GB      → 命令 B（x86）；AC922 上加 numactl --membind=<CPU 节点>
  其他组合       → 没有完整测试；从命令 A 起步改 CUDA_VISIBLE_DEVICES，再按第 3 步核对主机内存
第 2 步：要图片 / 视频输入、多路并发或超过 262144 的上下文吗？
  要             → 本版没有测；图片 / 视频改用 27B（见 27B 页）
  不要           → -np 1，-c 262144；只用短上下文时可以调小 -c，省下的显存给专家槽，吐字更快
第 3 步：主机内存够吗？（锁页内存 = 两张表 27466 MiB + 显存放不下的专家）
  6 × 16 GB      → 锁页约 42 GiB，再留出系统、页缓存与 --cache-ram 的余量
  4 × 32 GB      → 专家全部常驻显存，锁页约 27466 MiB
  其他组合       → 专家合计约 73477 MiB，显存放不下的部分都要放主机内存；放不下就无法启动
第 4 步：是 IBM AC922（GPU 显存并入系统内存）吗？
  是             → 命令前加 numactl --membind=<CPU 节点>（一般 0,8）；用 Docker 时用命令 C（-e LLAMA_NUMA_MEMBIND=0,8）
  否             → 不加 numactl，也不设 LLAMA_NUMA_MEMBIND；用 Docker 时用命令 D
第 5 步：有画像文件吗？
  有             → --expert-profile <画像>（Release 附带 flash-next-profile.txt）
  没有           → 先去掉 --expert-profile 启动（更慢），加 --dump-expert-routing <文件> 跑一段真实负载，
                   正常退出后 cat <文件>.dev* > 画像.txt，再带上画像重启
```

### 症状与处理

| 症状 | 先查 | 处理 |
|---|---|---|
| 启动时显存不足（日志有 `out of memory` 或显存分配失败） | AC922：页缓存是否落进了 GPU 显存节点（见最后一行）；GPU 上有没有其他进程 | 用 `numactl --membind=0,8` 启动；GPU 独占；`-c` 调小；保持 `--spec-draft-ubatch 256` 与 `-ub 2048`；仍不够再去掉 MTP |
| 运行一段时间后显存不足 | — | 调大 `-fitt`（默认 800） |
| 启动时锁页失败或主机内存不足 | `ulimit -l`；空闲主机内存 | `ulimit -l unlimited`；Docker 加 `--ulimit memlock=-1`；6 × 16 GB 要有约 42 GiB 可锁页的空闲内存再加系统余量；调小 `--cache-ram` |
| 吐字慢 | 有没有带 `--expert-profile`，画像是否来自同一模型与相近用途；温度；GPU 是否被其他程序占用 | 加画像，或按实际负载重采；保持 MTP 与 `--spec-draft-n-max 3`；温度大于 0 时吐字本来就低 12%～17%；用不到长上下文时调小 `-c` |
| 预填充慢 | 是不是服务启动后的第一个请求；`-ub` 是否调大了 | 服务的第一个请求本来就慢一些（实测约 1.6 万 token：新起服务的第一个请求约 2000～2030 tok/s，之后的请求约 2200 tok/s）；`-ub` 保持 2048；带上画像 |
| 草稿接受率低 | 温度；内容类型；草稿文件 | 温度 0 或低温时接受率更高；草稿必须是同一主模型的 shared 文件；`--spec-draft-n-max` 保持 3 |
| 图片 / 视频请求报错 | — | 大 MoE 本版只测了文字输入，图片 / 视频改用 27B |
| AC922 上显存被莫名占用、加载模型后显存不足 | `numactl --hardware` 里没有 CPU 的节点就是 GPU 显存，看它们的 free 是否变小 | 用 `numactl --membind=0,8` 启动；下载、复制大文件也用它包起来；已经落进去的页缓存用 root 执行 `sync; echo 1 > /proc/sys/vm/drop_caches` 清掉 |
