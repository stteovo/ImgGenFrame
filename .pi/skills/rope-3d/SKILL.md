---
name: rope-3d
description: "Implementation and review of the 3D Unified RoPE used by Z-Image (paper §4.1: image tokens on spatial axes, text tokens on temporal axis; official constants ROPE_THETA=256, axes dims (32,48,48), axes lens (1536,512,512)). Use when implementing RoPE, position encoding, or editing-task temporal offsets."
---

# 3D Unified RoPE

## Purpose
复现 Z-Image 的 3D 统一 RoPE：对文本/图像/语义 token 拼接成的单流序列提供位置编码——图像 token 沿空间维 (h,w) 展开，文本 token 沿时间维递增 [PAPER §4.1]；编辑任务中参考图与目标图共享空间坐标、时间维错开 1 单位 [PAPER §4.1]。

## Scope
- RoPE 数学核心与 3D 轴分配；官方常数：theta=256.0、axes dims=(32,48,48)（和=128=head_dim [OFFICIAL-CODE]）、axes lens=(1536,512,512) [OFFICIAL-CODE]（语义在 Stage 1 读码确认）。
- 单流混合序列的位置表构造（文本/图像 token 的坐标生成规则——以官方代码为准）。
- 编辑模式：参考/目标图的空间对齐 + 时间偏移。
- 不涉及主干结构（归 `s3-dit`）、不涉及 t 条件注入（归 `conditioning`）。

## Inputs
- paper-facts.md §2.2（RoPE 事实与 UNKNOWN 项）
- 官方代码中 RoPE 实现与调用点（`src/zimage/`、`src/utils/`）及 diffusers 实现
- 现有 `src/zimage/` 中 RoPE/位置编码代码（先检查再动手）

## Outputs
- RoPE 模块代码 + pytest 测试（数学性质、轴分配、偏移逻辑）
- 与官方/diffusers 的数值对齐报告
- 3D 轴分配规则的记录（含 [INFERENCE] 推导链，如 axes_lens 与最大分辨率的对应）

## Rules
1. 改代码前先检查现有实现与测试。
2. 轴分配（哪条轴给空间、哪条给时间）必须与官方代码一致，不得按"直觉"设计。
3. 维度和必须等于 head_dim；缩比模型若改 head_dim 需同步调整轴维分配并记录推导 [IMPLEMENTATION]。
4. theta、轴长等常数禁止"顺眼就改"；任何改动要过差异表流程。
5. 编辑任务的 temporal offset 语义（"错开 1 个单位间隔" [PAPER]）在实现前先读官方编辑推理代码确认精确值。

## Verification
- 数学性质测试：相对位置不变性、旋转矩阵正交性、长序列外推行为。
- 数值对齐：与 diffusers/官方实现对固定位置表输出逐值对比（bf16 容差）。
- 构造测试：文本 token 位置沿时间轴递增、图像 token 沿 (h,w) 展开，用断言验证坐标表。

## Learning
- 练习：解释"为什么文本沿时间维而非单独 RoPE 轴"（单流统一序列建模 [PAPER §4.1]）。
- 练习：推导 128 维 head 上 (32,48,48) 的切分含义与 2D RoPE 的关系。
- 记录：编辑任务为什么需要时间维错位（区分参考图/目标图上下文）。

## Common failure modes
- 轴分配颠倒（时间给图像、空间给文本）。
- 与 head_dim 不一致（如用了 120 维的"32 heads"组合）。
- 忘记编辑任务的时间偏移，导致参考图与目标图位置混淆。
- 只看输出"能训练"就通过，未做逐值对齐。
