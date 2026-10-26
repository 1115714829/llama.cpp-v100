# DFlash2 投机解码多卡并发槽位崩溃：GGML_ASSERT(ggml_is_contiguous)

**现象**：6 卡 `--split-mode tensor` + `--model-draft`（DFlash2）+ `-np 4` 时，多槽位**并发**运行即触发 `ggml-backend-meta.cpp:4728` 的 `GGML_ASSERT(ggml_is_contiguous(tensor))` 失败，进程 abort。systemd 下表现为崩溃循环（每 ~16s 一次 exit status=1）。单请求完全正常。2.0.1.1 与 2.0.1 两个构建均复现。

## 环境

- 硬件：Lenovo AC922（POWER9 ppc64le，双 socket 176 核，221GiB RAM），6× Tesla V100-SXM2-16GB
  - NUMA 0：GPU 0,1,2；NUMA 8：GPU 3,4,5；NVLink 仅同 socket 内
- 驱动 / CUDA：550.54.15 / CUDA 12.4
- 版本：Llama.cpp-V100 **2.0.1.1**（2.0.1 构建同样 abort）
- 主模型：Qwen3.8-27B-Q8_0.gguf（~28G，64 层含 nextn/MTP 层）
- 草稿：Qwen3.8-27B-DFlash2-F16.gguf（~3.6G），`--spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 0`
  - 日志参数：block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true

## 复现命令（6 卡，524K 池，4×131K 槽位）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 numactl --membind=0,8 ./llama-server \
  -m Qwen3.8-27B-Q8_0.gguf -ngl 999 \
  --split-mode tensor --tensor-split 1,1,1,1,1,1 \
  -c 524288 -np 4 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --model-draft Qwen3.8-27B-DFlash2-F16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 0 \
  --host 0.0.0.0 --port 8081 --metrics
```

启动后发 **4 个并发** chat 请求（每槽位一个）。

## 崩溃过程

前 2 个槽位正常完成（draft acceptance = 0.714 / 0.560，mean len = 6.00 / 4.92），随后某槽位的下一次 pre_decode abort：

```
ggml/src/ggml-backend-meta.cpp:4728: GGML_ASSERT(ggml_is_contiguous(tensor)) failed
```

backtrace 关键帧（自下而上）：

```
server_queue::start_loop
server_context_impl::update_slots
server_context_impl::pre_decode
common_speculative_draft
common_speculative_impl_draft_dflash::draft
llama_process
llama_context::decode
llama_context::process_ubatch
llama_context::graph_compute_impl
ggml_backend_sched_graph_compute_async
ggml_backend_sched_compute_splits
ggml_backend_tensor_set_async          <- 失败点
```

失败源码（`ggml_backend_meta_set_tensor_async`，ggml-backend-meta.cpp:4726-4728）：

```c
static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));   // <- 4728 失败
```

## 配置矩阵（同机实测，2026-10-25）

| 配置 | spec 草稿 | 并发 | 结果 |
|---|---|---|---|
| 6 卡 `-np 4` | DFlash2 | 1 请求 | ✅ 正常（acceptance 0.571） |
| 6 卡 `-np 4` | DFlash2 | **4 并发** | ❌ **GGML_ASSERT abort（稳定复现）** |
| 6 卡 `-np 4` | 无 | 多任务长跑 | ✅ 正常 |
| 6 卡 `-np 8` | 无 | 至多 8 槽 | ✅ 正常（Merkyor Q8 主模型，曾为现役档） |
| 4 卡（GPU 1,2,4,5）`-np 2` | DFlash2 | ≤2 并发 | ✅ 正常（已服役数天，~142 tok/s） |
| 6 卡 `-np 6` | DFlash2 | — | 未跑成（测试被 :8080 端口占用拦截，无数据） |
| 6 卡 `-np 4` | MTP（nextn） | — | ❌ 另一故障：nextn h sink 张量缺失（与本 bug 独立） |

## 推断

DFlash draft 路径（`common_speculative_impl_draft_dflash::draft`）在多槽位并发处理时，向 meta backend 的 `set_tensor_async` 传入**非连续张量**（strided view）。疑似 per-slot draft 状态在 `SPLIT_MODE_TENSOR` 下的竞态/别名问题；单槽流量不触发，4 卡 `-np 2`（≤2 并发）也未触发。

## 伴随观察

1. 启动即有警告：`common_fit_params: failed to fit params to free device memory: llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` —— 文案写 abort 但进程继续（fit_params 对 tensor split 未实现）。
2. 主模型含 nextn（MTP）层而草稿用 DFlash2 时，加载阶段报 11 个 `blk.64.nextn.*` / `blk.64.attn_*` / `blk.64.ffn_*` unused tensor 被忽略 —— 无害但噪音大。

## Workaround

`-np > 2` 时去掉 `--model-draft`（DFlash2）。代价：6 卡高并发档失去投机解码加速（本环境实测 acceptance 0.56-0.71、mean len 5-6，是有效加速）。

## 建议修复方向

1. dflash draft 路径在 `tensor_set` 前对目标张量做 `ggml_contiguous`，或排查该 strided view 的来源（哪个 per-slot 缓冲）。
2. 至少将 4728 的 assert 降级为错误返回（`ggml_status`），避免服务端崩溃循环。

## 附：原始日志

- 单请求正常：`np4_dflash_assert_repro.log`（1.55 min 加载 → 1 请求 61 tok 完成 → 干净退出）
- 4 并发崩溃：`np4_dflash_concurrent.log`（完整 backtrace）
- 服务态崩溃循环：`v100_6gpu_np4.log`（高 task id 下多次触发同一 assert）
