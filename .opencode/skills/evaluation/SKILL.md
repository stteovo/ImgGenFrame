---
name: evaluation
description: Evaluation methodology for Z-Image reproduction models — metric selection for small-scale T2I models (FID/CLIPScore-style at mini scale), fixed-prompt fixed-seed sample galleries, reference to paper §5 benchmarks (CVTG-2K, LongText-Bench, OneIG, GenEval, DPG-Bench, TIIF, PRISM-Bench) as full-scale targets. Use when adding metrics, building eval sets, or judging an ablation's outcome.
---

# Evaluation（评测方法论）

## Purpose
为缩比模型建立可信、可比较的评测体系：标准指标 + 固定 prompt/种子画廊 + 逐步接近论文 §5 的全尺寸基准。防止"看着不错"式的伪结论。

## Scope
- 小规模指标：FID/CLIPScore 类在 100M–1B 上的适用方式（数据集、样本数、实现选择标 [IMPLEMENTATION]）。
- 固定评测集：固定 prompt 集（双语）+ 固定种子 → 采样画廊，作为阶段间与消融间的人类可查对照。
- 论文 §5 基准（CVTG-2K/LongText-Bench/OneIG/GenEval/DPG-Bench/TIIF/PRISM-Bench [PAPER §5.2]）：仅在模型能力接近时引入；小模型跑大基准无意义，需在计划中论证。
- 文字渲染与指令遵循的自建 mini 评测（呼应论文强调的双语文字能力 [PAPER §5.3.2]）。
- 不包含：实验设计（`experiment`）、数据管线（`data-infrastructure`）。

## Inputs
- paper-facts.md §5 基准清单与推理参数（guidance 3–5、负向提示、cfg_normalization [OFFICIAL-CODE]）
- 待评测模型 + 其训练实验记录（数据版本、配置）
- 现有 `scripts/` 评测工具（先检查再动手）

## Outputs
- 指标实现 + 单元测试（指标本身有已知参考值验证）
- 固定评测集定义文件（prompts + 种子，冻结版本）
- 评测报告（入对应实验目录 `metrics/`）

## Rules
1. 改代码前先检查现有评测工具，指标实现不得重复造。
2. 指标实现先过正确性测试（如构造数据验证 FID 计算与参考实现一致）。
3. 评测必须声明前提：样本数、种子、使用的模型/数据版本；不同前提的数字禁止直接比较。
4. 画廊必须固定 prompt + 固定种子，跨实验可比。
5. 论文基准的正确推理参数（Turbo 无 CFG / Z-Image 有 CFG [OFFICIAL-CODE]）必须照做，否则结果与论文不可比。

## Verification
- 指标单测：构造/退化场景（相同分布 → FID≈0 等）数值符合预期。
- 画廊一致性：同模型同种子重跑，输出逐像素一致（或量化误差可解释）。
- 评测复现：同一实验目录重跑评测，指标一致。

## Learning
- 练习：解释"为什么小模型跑 CVTG-2K 没意义"（基准难度与模型能力阶段匹配）。
- 练习：固定种子画廊 vs 随机采样评测各自的证据强度。
- 记录：我们评测体系与论文 §5 的差距与演进计划。

## Common failure modes
- 跨实验比较指标却无视种子/样本数/数据版本差异。
- 用单一 FID 给消融下结论，不配画廊人工确认。
- 给 Turbo 设 guidance>0 或给 Z-Image 设 guidance=0 后与论文数字对照 [OFFICIAL-CODE]。
- 指标脚本"能跑"但从未与参考实现对过数。
