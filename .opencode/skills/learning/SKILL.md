---
name: learning
description: Mentor-mode knowledge transfer for the Z-Image reproduction project — after every task, explains the key knowledge points behind the completed work (concept, motivation, math, connection to Z-Image, verification) per the seven mentor questions in AGENTS.md. Use after completing any module, experiment, or reading task, or when the user asks "explain what we just did".
---

# Learning（导师模式：知识讲解）

## Purpose
把每个完成的模块/实验转化为用户（学习者）的知识增量。每次任务结束后主动输出结构化讲解，而不是只交付代码。

## Scope
- 讲解对象：任何已完成的模块、实验结论、论文精读成果。
- 讲解依据：只能来自已核实事实（paper-facts.md、实验记录），禁止编造。
- 不包含：新代码编写；讲解中发现的 `[UNKNOWN]` 只记录，不顺手实现。

## Inputs
- 刚完成任务的产物（代码/实验记录/笔记）
- paper-facts.md、reproduction_matrix.md、roadmap（对照当前阶段）
- 导师七问（AGENTS.md）

## Outputs
- 结构化解说：概念是什么 → 为什么这么设计 → 数学原理 → 与 Z-Image 的对应点 → 我们在哪里（文件/实验路径）→ 怎么验证的 → 还可以怎么验证
- 建议写入 `docs/` 或实验记录的学习笔记（用户确认后）
- 给用户的 1–3 个自测问题（可选）

## Rules
1. 改代码前先检查现有实现——讲解若涉及代码位置必须真实指认（文件+行号级别），不得虚构。
2. **完成即讲解**：模块/实验收尾后主动执行本 skill，不等待用户要求。
3. 讲解语言：概念-动机-数学三层递进；数学必须可查证（公式、出处）。
4. 事实纪律：引用事实带标签；自己的理解与论文事实分开表述。
5. 控制篇幅：讲解聚焦"关键知识点"，不做流水账复盘。
6. 发现用户可能误解的常见陷阱（参考对应模块 skill 的 failure modes）要主动点出。

## Verification
- 用户确认：每个知识点讲解后留一个检查问题，用户能答出即视为理解。
- 自检：讲解中每个断言都能回溯到事实库或实验记录；否则重写。
- 覆盖检查：七问（AGENTS.md）中与本模块相关的条目都覆盖到。

## Learning（本 skill 自指）
- 这是唯一"服务用户学习"的 skill；讲解质量本身也要迭代——用户标记"没懂"的部分要换讲法并记录。
- 记录：建立一个"讲解日志"（可在实验记录或 docs 笔记中），追踪哪些概念用户已掌握。

## Common failure modes
- 只交付代码不讲原理，用户"看起来进展了"但没学到东西。
- 讲解掺入未核实内容（把 SD3/Flux 常识当 Z-Image 事实讲）。
- 用术语堆砌代替三层递进讲解。
- 讲解过长失去重点，或过短只剩结论。
