---
name: flow-matching
description: Implementation and review of Z-Image's flow matching training objective and sampling (paper §4.3 eq.1: x_t = t·x1 + (1−t)·x0, velocity target v = x1 − x0, MSE; logit-normal timestep sampling; Flux-style dynamic time shifting with official shift constants). Use when implementing the FM loss, noise scheduling, t sampling, or Euler samplers.
---

# Flow Matching（训练目标与采样）

## Purpose
复现 Z-Image 的 flow matching 目标：线性插值路径 + 速度场回归 + MSE 损失，以及配套的 logit-normal t 采样与动态时间偏移（多分辨率 SNR 补偿 [PAPER §4.3]）。

## Scope
- 目标函数：`x_t = t·x1 + (1−t)·x0`（x0=噪声，x1=数据 [PAPER §4.3]）、`v = x1 − x0`、MSE。
- t 采样：logit-normal（沿 SD3 [PAPER §4.3]）。
- 动态 shift：`BASE_IMAGE_SEQ_LEN=256、MAX_IMAGE_SEQ_LEN=4096、BASE_SHIFT=0.5、MAX_SHIFT=1.15` [OFFICIAL-CODE]；注意推理调度器默认 `shift=3.0、use_dynamic_shifting=False` [OFFICIAL-CODE]——训练/推理路径差异需 Stage 1 对照 diffusers 确认。
- 采样器：Euler ODE、CFG 加权（CFG 强度与负向提示逻辑归 `conditioning`）。
- 不包含：VAE 编解码（`vae-tokenization`）、模型结构（`s3-dit`）。

## Inputs
- paper-facts.md §3（训练目标事实）、§7（UNKNOWN：CFG dropout 概率、优化器等）
- diffusers 中 Z-Image 使用的调度器实现（`FlowMatchEulerDiscreteScheduler` 等）与官方 pipeline 调用
- 现有 `src/zimage/scheduling/`、`src/zimage/training/` 实现（先检查再动手）

## Outputs
- FM 损失/t 采样/shift 函数/采样器代码 + pytest
- t 分布直方图、shift 随序列长度变化曲线的验证图（入实验记录）
- 与 diffusers 调度器对固定输入输出数值对齐报告

## Rules
1. 改代码前先检查现有实现与测试。
2. **方向约定**：本项目沿用论文约定 x0=噪声、x1=数据 [PAPER §4.3]；与 SD3/Flux 常见写法的差异必须在代码注释外、记录中明示，禁止混用两套约定。
3. 预测类型（velocity）与 loss 权重（论文未给权重函数 → 默认恒 1 [IMPLEMENTATION]）必须显式记录。
4. logit-normal 参数（论文未给具体 σ 值 → 查 SD3 惯例并标 [ASSUMPTION]）不得拍脑袋定。
5. 动态 shift 公式以 diffusers/官方代码为准复刻，先对齐再谈"更优变体"。

## Verification
- 数学测试：x_t 端点（t=0/1）、v 目标闭式解、t 采样分布直方图与 logit-normal 理论曲线对比。
- 对齐测试：shift(t, seq_len) 与 diffusers 实现逐值一致；Euler 采样步进与 diffusers 一致。
- 训练冒烟：loss 初始值接近速度场方差的理论量级；收敛后 toy 数据可辨识。

## Learning
- 练习：解释为什么动态 shift 在多分辨率训练中是必要的（不同分辨率 SNR 不一致 [PAPER §4.3]）。
- 练习：logit-normal 为什么把训练集中在中间 t。
- 记录：训练与推理 shift 默认值的差异及影响面。

## Common failure modes
- x0/x1 方向与论文相反（v 预测变 eps 预测而采样器没同步改）。
- 直接套用 Flux 的 shift 常数而忽略 Z-Image 的 BASE/MAX 常数差异。
- 训练用动态 shift、采样用固定 shift=3.0 且不记录（[OFFICIAL-CODE] 默认如此，必须显式决策）。
- 忽略论文未给的 CFG dropout 概率，不做显式 [ASSUMPTION] 标注。
