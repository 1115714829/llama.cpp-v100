# X99 部署 Strata(Qwen3.8-Flash-Next 125B MoE)— 已退役(2026-10-26)

上游 [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.41 在 X99(2×V100-16G,121 G RAM)上的部署记录。
2026-10-26 随 27B 部署上线退役(`disable` + `stop`),文件保留,可一条命令复启。

## 组件

| 项 | 位置 / 值 |
|---|---|
| 仓库 | `/home/dxc/strata/strata-repo`(origin `Niko1221/Strata`,HEAD `fb58e0d` = v0.1.41) |
| venv | `/home/dxc/strata/python` |
| 引擎 | 本地构建 0.1.41(`engine/BUILD.json` `source=local`);V100 属上游 OLDER_GPUS 实验路径(CUDA 12 引擎 + `-DSTRATA_EXPERIMENTAL_SM60=ON`) |
| 模型 | `/mnt/strata/`(独立 458 G NVMe,`/dev/disk/by-label/strata`,fstab 挂载):`IQ3_XXS` 71 G、`IQ3_S` 78 G、`SWIFT_IQ3_XXS` 等 |
| 配置 | `strata-repo/strata-iq3_s_8080.json`(service 实际引用它,按 service 描述为 IQ3_XXS 256K no-thinking;同目录另有 `strata-iq3_xxs*.json` 等) |

## 服务 `strata-flash-256k.service`(system 级,User=dxc,已 disable)

```ini
[Unit]
Description=Strata Qwen3.8-Flash-Next (IQ3_XXS) 256K context, no-thinking, on 2080 Ti (GPU0), port 8080
After=network.target

[Service]
Type=simple
User=dxc
# 引擎是 47GB experts 锁进内存 + 14GB GPU 专家缓存，需要 unlimited memlock
LimitMEMLOCK=infinity
Environment=CUDA_VISIBLE_DEVICES=0
WorkingDirectory=/home/dxc/strata/strata-repo
ExecStart=/home/dxc/strata/python/bin/python3 /home/dxc/strata/strata-repo/serve/server.py --engine strata --config /home/dxc/strata/strata-repo/strata-iq3_s_8080.json --port 8080
# 加载 71GB 模型 + 锁内存约 6-7 分钟，放宽启动超时
TimeoutStartSec=600
TimeoutStopSec=60
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

(Description 里的 "2080 Ti" 是换卡前写的,现已过期;当前 GPU0/1 为 V100-SXM2-16G NVLink。)

## 操作

```sh
# 复启(冷加载 6-7 min;引擎 READY 上限 900 s,启动窗口内要等)
sudo systemctl start strata-flash-256k
# 停止
sudo systemctl disable --now strata-flash-256k
# API
curl http://127.0.0.1:8080/v1/models   # 主端口 8080,次端口 8082
```

## 坑

1. **`LimitMEMLOCK=infinity` 必须**:experts page-lock 进 RAM,默认 memlock 上限会导致启动失败
2. **停止时清理有 traceback**(`serve/server.py:57`,journal 可见):unit 转 inactive 后,`192.168.8.96:8080/8082` 仍由 `~/.hermes/scripts/lan_forward_8080_8082.py` 监听转发——8082 腿现在转 27B(有效),8080 腿为死转发(可清理该脚本的 8080 部分)
3. **单卡 16 G 只能吃 IQ3_XXS 档**;IQ3_S(78 G)需 2 卡或更大显存;`CUDA_VISIBLE_DEVICES` 与 config 的 `gpu`/`--gpus` 联动
4. **V100 = 上游实验路径**:官方支持列表是 RTX 20/30/40/50 + 部分 AMD;V100/P40 见上游 `docs/OLDER_GPUS.md`(CUDA 12 引擎、`mma.m8n8k4` prompt attention、FP16 张量核 BF16 投影、`NVIDIA_V100.md` 专篇)
5. **内存门槛**:IQ3_XXS 锁 ~47 G experts + 14 G GPU 专家缓存,64 G RAM 起;启动时系统会明显变卡 1-3 分钟(正常)
