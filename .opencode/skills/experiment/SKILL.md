---
name: experiment
description: Design and execution of controlled experiments and ablations in the Z-Image reproduction (hypothesis-driven, single-variable, baseline-referenced), with mandatory recording per the experiment-protocol skill. Use when designing or launching a training run, ablation, probing, or benchmark — the top-level experiment workflow.
---

# Experiment（受控实验与消融）

## Purpose
保证每个实验（训练/消融/探针/基准）都：有带标签的假设、有 baseline、只变一个变量、可复现、有完整记录。这是本项目科研结论可信度的来源。

## Scope
- 实验设计：假设（带证据标签）、变量清单、baseline 选择、出口判据。
- 执行约束：种子、数据版本、环境、config 冻结。
- 记录契约：委托 `experiment-protocol` skill 的目录与字段规范。
- 不包含：指标定义（`evaluation`）、训练器实现（`training`）、论文事实（`paper-analysis`）。

## Inputs
- roadmap（当前阶段与出口条件）、reproduction_matrix（当前组件状态）
- 既往 `experiments/` 记录（先查，避免重复实验）
- 待验证假设与候选 baseline

## Outputs
- 实验计划（假设+标签、变量、判据、预计算力）
- 完整实验目录（`experiments/<YYYYMMDD>-<slug>/`，按 experiment-protocol 契约）
- 结论（标 [EXPERIMENT]，附记录路径）与下一步建议

## Rules
1. 改代码前先检查现有实现与既往实验；**设计实验前必须读 roadmap 与矩阵**，确认该实验属于当前阶段。
2. 一次只变一个变量；对照组必须有共同 baseline 记录。
3. 假设必须带标签：`[PAPER]` 依据的声明 vs `[ASSUMPTION]` 的猜测要分开写。
4. 种子/数据/环境不完整 = 实验无效，宁可不跑。
5. 失败实验也要记录（负结果是知识），不得只留成功案例。
6. 结论写成 `[EXPERIMENT]` 需附可指认的实验目录。

## Verification
- 计划评审：变量唯一性、baseline 存在、判据可度量。
- 复现性：同一实验目录按 record.md 重跑，关键指标一致（至少 loss 曲线趋势一致）。
- 记录完整性：按 experiment-protocol 的七项清单逐项核对。

## Learning
- 练习：为一个论文未给的超参（如 CFG dropout）设计"最小消融方案"。
- 练习：区分"exploratory run（探索）"与"controlled ablation（受控）"的记录标准差异。
- 记录：每次实验后写 3 行"学到了什么 + 下一步"，形成实验链。

## Common failure modes
- 同时改数据、超参、结构，事后无法归因。
- 无 baseline 或 baseline 不在同一数据/种子/环境上。
- 只记录成功的实验，掩盖失败路径。
- 结论写"验证了论文 X"但实际只跑了 toy 实验（证据强度不符）。
