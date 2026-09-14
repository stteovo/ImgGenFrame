---
name: s3-dit
description: Implementation and review of the S3-DiT single-stream diffusion transformer backbone for Z-Image (paper §4.1, Table 2; official config src/config/model.py). Use when implementing or modifying the DiT model, its blocks, norms (QK-Norm/Sandwich-Norm/RMSNorm), adaLN modulation, or patchify layers.
---

# S3-DiT（单流 DiT backbone）

## Purpose
复现 Z-Image 的 S3-DiT 主干：文本/视觉语义/VAE token 序列级拼接的单流 MM-DiT，实现与官方 6B 架构的数值级对齐（缩比梯子 100M→300M→1B→3B→6B）。

## Scope
- 单流 backbone 结构：modality-specific processors（每模态 2 blocks [PAPER §4.1]，对应代码 `N_REFINER_LAYERS=2` [OFFICIAL-CODE]，精确对应关系读码确认）、主块堆叠、patchify。
- 归一化族：RMSNorm（eps=1e-5 [OFFICIAL-CODE]）、QK-Norm、Sandwich-Norm [PAPER §4.1]。
- 低秩 adaLN（共享 down-proj + 每层 up-proj，embed dim=256 [OFFICIAL-CODE]）——调制逻辑与 `conditioning` skill 协作。
- 官方常数：dim=3840、30 层、FFN=10240、in_channels=16、patch=(2,2)、`SEQ_MULTI_OF=32` [OFFICIAL-CODE]；heads 争议（论文 32 vs 代码 30）以 HF config 裁决。

## Inputs
- `docs/reproduction/paper-facts.md`（§2 架构事实、§6 差异表）
- 官方代码：`Tongyi-MAI/Z-Image/src/config/model.py`、`src/zimage/` 模型实现、diffusers `ZImageTransformer2DModel`
- 现有实现（`src/zimage/models/`）与 `tests/`（先检查再动手）

## Outputs
- 模型模块代码 + pytest 测试
- 该级规模配置（layers/dim/heads 推导过程记录，标注 [IMPLEMENTATION]）
- 与官方/diffusers 的模块级数值对齐报告

## Rules
1. **改代码前先检查现有实现与测试**：`src/zimage/` 中已有模块不得重写，只做有证据驱动的修改。
2. 测试先行（遵循 `module-development` skill）；数学模块必须先数值对齐。
3. 缩比时保持结构比例（dim:FFN≈1:2.67、head_dim=128），缩放推导写入实验记录，不得随意改结构。
4. 论文未规定的工程选择必须标 `[IMPLEMENTATION]`（如 bias 使用、初始化方式）。
5. 归一化顺序、残差位置、调制位置以官方代码为准，而非"常见 DiT 写法"。

## Verification
- 单元测试：shape 一致性（含 `SEQ_MULTI_OF=32` 约束）、各 norm 的数学性质、低秩 adaLN 与全秩版本参数对比。
- 对齐测试：随机/官方权重下，逐层与 diffusers 实现对数值（bf16 容差内）。
- 冒烟：100M 级前向+反向在 16GB VRAM 内完成，显存记录入实验。

## Learning
- 练习：解释"为什么单流比双流参数效率高"（密集跨模态交互、参数复用 [PAPER §1/§4.1]）。
- 练习：Sandwich-Norm 与普通 pre-norm 的信号幅度差异。
- 记录：6B 与 100M 的参数占比分布（attn/FFN/调制），理解缩放决策。

## Common failure modes
- heads 直接抄论文的 32 而不查代码/HF 的 30（head_dim 错 → RoPE 维度错）。
- 用 SD3/Flux 的"常见"归一化顺序替代官方实现。
- 把 Qwen3-4B 的 2560 维特征直接进 3840 维主干而不经 processor 对齐 [PAPER §4.1]。
- 缩比时改 head_dim 或 FFN 比例，破坏与 6B 的可比性。
