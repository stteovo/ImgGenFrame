---
name: reproduction-audit
description: Faithfulness audit of the Z-Image reproduction — compares our src/zimage implementation against the paper, official code, diffusers implementation, and HF checkpoints, and maintains the discrepancy table in paper-facts.md §6 and statuses in reproduction_matrix.md. Use when reviewing implementations, resolving paper/code conflicts, or marking components verified.
---

# Reproduction Audit

## Purpose
裁决"我们的实现是否忠实于 Z-Image"，维护三方（论文/官方代码/我们）差异表与组件状态矩阵；防止"能跑"被误认为"正确"。

## Scope
- 审计对象：`src/zimage/` 各模块、`tests/`、实验记录中的实现声明。
- 审计基准：paper-facts.md、官方 repo（仅推理代码）、diffusers `ZImagePipeline`/`ZImageTransformer2DModel`、HF `Tongyi-MAI/Z-Image(-Turbo)` config.json。
- 只审计与记录，不顺手改实现（改动走对应模块 skill）。

## Inputs
- 待审计的实现代码 + 其声称依据（哪个标签、哪条事实）
- `docs/reproduction/paper-facts.md`、`reproduction_matrix.md`
- 官方代码与 diffusers 源码（必要时 clone/安装到只读位置）

## Outputs
- 审计报告：逐项结论 = 一致 / 不一致（附双方证据）/ 无法裁决（[UNKNOWN]）
- paper-facts.md §6 差异表更新、reproduction_matrix.md 状态更新
- 数值对齐证据（如权重加载后的 max-err 统计），无证据不得标记"已验证"

## Rules
1. **改代码前先看现有实现**：审计前完整阅读对应模块在 `src/zimage/` 的现有代码与 tests；不重复造。
2. **"已验证"需要数值证据**：架构忠实度的金标准 = 加载官方权重，输出与 diffusers 对齐（Stage 6 全量；Stage 2 起做模块级对齐）。
3. **默认不信任**：训练超参（optimizer/lr/EMA）论文未给出 → 一律 `[UNKNOWN]`，我们的选择标 `[IMPLEMENTATION]`/`[ASSUMPTION]`。
4. **静默差异优先排查**：dtype（bf16 vs fp32）、eps、归一化顺序、常数（theta、shift）等不会报错但会破坏对齐。
5. 差异裁决顺序：HF config.json > 官方代码 > 论文文字 > 我们实现。

## Verification
- 每条"一致/不一致"结论必须能回溯到两处可指认证据。
- 数值对齐：固定种子下与 diffusers pipeline 逐层/逐值对比，记录 max-err 与误差分布。
- 审计本身可复现：记录审计所用官方代码 commit、diffusers 版本。

## Learning
- 练习：解释"为什么 6.15B 权重加载对齐全套验证比训练收敛曲线更有说服力"。
- 记录：每次审计发现的新差异写入 paper-facts §6，并注明发现过程（读码/对撞实验）。

## Common failure modes
- 只看输出图像"像不像"，不做数值对齐。
- 把实验跑通当成实现正确。
- 忽略 dtype/设备/归一化 eps 等静默差异。
- 对已冻结的 configs/ 或事实库私下改动而不走差异表流程。
