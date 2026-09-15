# 实验记录 — Z-Image-Tiny 过拟合（第一实验）

- **实验 ID**：EXP-TRAIN-001（20260915-tiny-overfit）
- **git commit**：cfdf72a（`[feat] 组建最小可运行 Z-Image pipeline`）
- **配置**：`configs/tiny/{model.yaml, training.yaml, data.yaml}`（冻结）

## 动机与假设

[EXPERIMENT] 验证 100M 缩比 S3-DiT 能否过拟合极小数据集（8~32 张），证明训练循环、Flow Matching 目标、checkpoint 链路正确，再考虑扩大数据。

## 配置要点

| 项 | 值 | 来源 |
| --- | --- | --- |
| 模型 | 95.66M（dim=768, depth=8, heads=6, ffn=2048） | [IMPLEMENTATION] 缩比 |
| 分辨率 | 256²（latent 32×32 → 256 token） | [PAPER §4.3] low-res |
| 目标 | Flow Matching（v=x1−x0, MSE） | [PAPER §4.3] |
| 精度 | BF16（autocast）+ FP32 主权重 | [OFFICIAL-CODE] 推荐 bf16 |
| 优化器 | AdamW, lr=1e-4, wd=0.01, β=(0.9,0.999) | [ASSUMPTION] SD3/Flux 惯例 |
| t 采样 | logit-normal（σ=1.0） | [PAPER §4.3] + [ASSUMPTION] σ |
| 数据 | 16 张确定性合成 latent（N(0,1)）+ mock 文本特征 | [IMPLEMENTATION] |
| seed | 42（模型）+ 0（数据） | — |

## 结果

- **loss 曲线**：first=2.338 → last=0.245（≈9.5× 下降，收敛）
- **过拟合重建**：16 步 Euler 从噪声恢复训练 latent，MSE=0.085（latent std≈1，重建质量好）
- **吞吐**：21.2 it/s（RTX 5070 Ti）
- **峰值显存**：1.63 GB（远低于 16GB，为后续 300M/1B 留足余量）
- **冒烟**：1/10/100 step 均通过（loss 有限且递减：2.34 → 2.12 → 1.68）

## 已知问题 / 下一步

- 当前为合成 latent 数据（无真实图像/文本），下一步接真实数据 + 真实 Qwen3-4B 文本特征（第 16 步闭环）。
- CFG dropout、EMA 未启用（[ASSUMPTION]，open_questions A4/A6）。
- checkpoint 续训已由单测覆盖（save/load/resume 一致）。
