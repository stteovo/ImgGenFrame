# Scaling 记录（exp_100m vs exp_300m）

> 第 17 步。只改变 model scale / compute，保持 architecture、data pipeline、objective 不变。

## 缩放曲线（RTX 5070 Ti 16GB，bf16，batch=4）

| 指标 | exp_100m | exp_300m |
| --- | --- | --- |
| 配置 | dim=768, depth=8, heads=6 | dim=1024, depth=18, heads=8 |
| 参数量 | 95.66M | 301.47M |
| FLOPs 前向（256², 264 token） | 7.73e10 | 2.50e11 |
| forward 峰值 VRAM（256²） | 0.235 GB | 0.623 GB |
| forward+backward 峰值 VRAM（256²） | 0.98 GB | 2.545 GB |
| 吞吐（256²） | 9.38 it/s | 6.89 it/s |

## 显存随分辨率（300M，batch 按显存）

| 分辨率 | token 数 | 峰值 VRAM | 吞吐 |
| --- | --- | --- | --- |
| 256²（batch4） | 264 | 2.545 GB | 6.89 it/s |
| 512²（batch4） | 1032 | 9.076 GB | 2.68 it/s |
| 1024²（batch1） | 4104 | 13.233 GB | 0.98 it/s |

**结论**：300M 前向/反向在 16GB 内完成（至 1024²）；4096 token 时逼近上限（13.2GB），
更高分辨率/更大 batch 需 gradient checkpointing / 8bit optim / offload。

## Scale 升级出口条件

- M1（100M）：✅ 结构+训练循环验证、全单测通过、过拟合收敛（loss 2.34→0.25、重建 MSE 0.085）。
- M2（300M）：✅ 本步显存/吞吐实测通过 → 升级条件满足；下一步 300M 训练（256² 数据 v0）。
