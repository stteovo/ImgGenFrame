# Learning Report — 训练循环（Flow Matching 端到端首次训练）

## 1. 做了什么

实现 `src/zimage/training/`（`FlowMatchingTrainer`、`SyntheticLatentDataset`、配置 dataclass+YAML 加载），`configs/tiny/{model,training,data}.yaml`，`scripts/train_tiny.py`。在 RTX 5070 Ti 上完成 **EXP-TRAIN-001**：100M 模型过拟合 16 张合成 latent，loss 2.34→0.25，16 步 Euler 重建 MSE=0.085，峰值显存 1.63GB，21 it/s。

## 2. 为什么需要

前 9 步只验证了「模型能前向/反向」，但从未验证「能否真的学会」。训练循环是把 FM 目标、t 采样、优化器、checkpoint 串起来的最小闭环；过拟合极小数据集是训练链路正确性的**金标准冒烟**——若连 16 张都记不住，说明目标/梯度/数据流有 bug，扩大数据毫无意义。

## 3. 数学原理（训练步）

```
x1（干净 latent）               t ~ logit-normal（σ=1）
x0 ~ N(0,I)                     x_t = (1−t)·x0 + t·x1
v_target = x1 − x0              v_pred = model(x_t, t, text)
L = ||v_pred − v_target||²      AdamW(θ) ← ∇L
```

初始 loss 量级：合成 latent x1~N(0,1)、x0~N(0,1) → E[||v||²]=2，故首步 loss≈2.3（含 text 条件微扰），与实测 2.338 吻合——这是「loss 初值量级」的解析校验。

## 4. Z-Image 中如何使用

- low-res 预训练：256²、T2I、FM 目标、logit-normal t、动态 shift [PAPER §4.3]。
- 冻结 VAE/text encoder 用 DP、DiT 用 FSDP2、全层梯度 checkpointing [PAPER §4.2]。
- 优化器/lr/EMA 等论文**未给**（[UNKNOWN]，open_questions A1–A8），我们取 SD3/Flux 惯例并显式标 [ASSUMPTION]。

## 5. 当前项目如何实现

- `FlowMatchingTrainer`：`train_step`（采 t → 插值 → 前向 → MSE → clip → step）；checkpoint 含 model/optimizer/step/全部 RNG 状态/配置，支持续训。
- BF16：CUDA 下 autocast + FP32 主权重（AdamW 状态仍 FP32）。
- 配置全部 YAML 冻结（`configs/tiny/`），实验引用后不可改。
- 数据：合成 latent（`SyntheticLatentDataset`，确定性 seed），真实数据第 16 步接入。

[ASSUMPTION] 记录：AdamW(1e-4, wd=0.01, β=(0.9,0.999))、grad_clip=1.0、logit-normal σ=1.0、无 EMA、无 CFG dropout、batch=4。

## 6. 其他可能实现

- 真实图像 + 真实 VAE encode + 真实 Qwen3-4B 文本特征（第 16 步）。
- 学习率调度（warmup + cosine）：论文未给，后续消融。
- EMA：SD3/Flux 常用，但论文未提 [UNKNOWN]，先不用，采样质量对照后再定。
- 动态 shift：多分辨率训练才需要（本步单分辨率 256²，未启用）。

## 7. Trade-offs

- BF16 autocast：显存/速度↑，但数值精度略降（本步无 NaN）。
- 合成 latent 过拟合：快、确定性、验证链路；但无真实统计结构，不能外推到真实生成质量。
- 无 EMA：省显存/时间，但最终采样质量可能略逊。

## 8. 如何验证

- 单测：train_step 有限、checkpoint save/load/resume、同种子首步 loss 一致、单图 200 步 loss 显著下降（`tests/training/test_trainer.py`）。
- 冒烟：1/10/100 step 通过（loss 2.34→2.12→1.68）。
- 过拟合：2000 步 loss→0.245；16 步 Euler 重建 MSE=0.085。
- 记录：EXP-TRAIN-001（record.md / environment.txt / dataset.md / metrics JSON / checkpoint）。

## 9. Failure Modes

- 直接抄 SD3/Flux 的 lr/EMA 不标 [ASSUMPTION]。
- BF16 模型 + FP32 输入在 eval（无 autocast）时 dtype 报错——已修（reconstruct 显式 cast 输入）。
- 过拟合失败就扩大数据（应先 debug 链路）。
- checkpoint 只存 model 不存 optimizer/RNG → 无法真正续训。

## 10. 当前仍然未知的问题

- 真实数据 + 真实文本编码器下的收敛行为（第 16 步）。
- 优化器/lr/EMA/CFG dropout 的最终取值（消融计划，open_questions A1–A8）。
- 动态 shift、序列长度感知组 batch（多分辨率训练时，第 22 步）。
