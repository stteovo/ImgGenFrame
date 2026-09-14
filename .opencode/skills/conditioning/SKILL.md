---
name: conditioning
description: Implementation and review of Z-Image's conditioning path — timestep embedding (FREQUENCY_EMBEDDING_SIZE=256, MAX_PERIOD=10000, T_SCALE=1000 [OFFICIAL-CODE]), low-rank adaLN scale/gate modulation [PAPER §4.1], classifier-free guidance (guidance 3.0–5.0 for Z-Image, 0 for Turbo, negative prompts, cfg_normalization [OFFICIAL-CODE]). Use when implementing timestep/caption conditioning, adaLN gating, or CFG logic.
---

# Conditioning（条件注入与引导）

## Purpose
复现 S3-DiT 的条件路径：把 t、text caption（Qwen3-4B 特征 2560 维 [OFFICIAL-CODE]）、编辑任务的参考图条件转换为调制信号（scale/gate），以及推理侧的 CFG 组合逻辑。

## Scope
- 时间嵌入：频率嵌入相关常量 `FREQUENCY_EMBEDDING_SIZE=256、MAX_PERIOD=10000、T_SCALE=1000` [OFFICIAL-CODE]（精确用法 Stage 1 读码）。
- 低秩 adaLN：共享层无关 down-proj（256 维 [OFFICIAL-CODE]）+ 每层 up-proj → 调制归一化输入/输出 [PAPER §4.1]。
- 文本条件注入：text embeddings 作为条件进入调制路径（精确机制以官方代码为准）。
- CFG：训练期 dropout（概率论文未给 → [UNKNOWN]，我们的取值标 [ASSUMPTION]）；推理期 guidance（Z-Image 建议 3.0–5.0、负向提示、`cfg_normalization` 开关 [OFFICIAL-CODE README]）；Turbo 无 CFG（guidance=0 [OFFICIAL-CODE]）。
- 不包含：RoPE（`rope-3d`）、backbone 结构（`s3-dit`）、文本编码器加载（其冻结与接入见 `s3-dit`/`vae-tokenization` 同级的 encoders 约定）。

## Inputs
- paper-facts.md §2.2（调制/嵌入事实）、§8（推理参数）
- 官方 `src/config/model.py` 常量、diffusers `ZImagePipeline` 的 CFG/`cfg_normalization` 实现
- 现有 `src/zimage/` conditioning 相关代码（先检查再动手）

## Outputs
- 时间嵌入 + adaLN 调制模块 + pytest
- CFG 训练/推理逻辑 + 对齐测试
- CFG dropout 概率等 [ASSUMPTION] 项的决定记录

## Rules
1. 改代码前先检查现有实现与测试。
2. adaLN 的共享 down-proj 是**层无关共享**、up-proj 是**每层独立** [PAPER §4.1]——实现必须体现这一参数结构，验证参数量。
3. scale/gate 的初始化方式论文未给 → 标 [IMPLEMENTATION]，参考官方代码（若有）或记录假设。
4. `cfg_normalization` 行为以 diffusers 实现为准复刻，不得自行定义。
5. 训练与推理的 CFG 语义差异（训练 dropout、推理双 forward 加权）必须区分并测试。

## Verification
- 单元测试：共享 down-proj 参数确实跨层共享（参数指针/计数断言）；每层 up-proj 独立。
- 对齐测试：与 diffusers pipeline 同输入同种子下，CFG 组合后的隐变量逐值一致。
- 消融准备：cfg_normalization 开/关、不同 guidance 强度的采样对比图入实验。

## Learning
- 练习：解释低秩分解 adaLN 的参数量节省与"层间共享、层内独立"的设计动机 [PAPER §4.1]。
- 练习：CFG 为什么是"额外计算一次无条件前向"（~100 NFE = 50 步×2 [PAPER §4.5] 的由来）。
- 记录：Qwen3-4B 冻结特征 2560 维如何进入 256 维共享调制空间。

## Common failure modes
- 把 down-proj 做成每层独立（参数翻倍、结构失真）。
- 训练时忘记 CFG dropout 或概率拍脑袋且不记录。
- 推理把 guidance 用到 Turbo（应为 0 [OFFICIAL-CODE]）。
- T_SCALE=1000 与 shift 语义混淆，时间嵌入口径不一致。
