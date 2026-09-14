---
name: distillation
description: "Study and (mini-scale) reproduction of Z-Image's few-step distillation stack — Decoupled DMD (arXiv 2511.22677: CFG-Augmentation as engine, Distribution Matching as regularizer, decoupled renoising schedules) and DMDR (arXiv 2511.13649: DMD meets RL), plus RLHF DPO/GRPO stages (paper §4.6). Use when studying these papers or attempting mini distillation/RL experiments."
---

# Distillation（少步蒸馏与 RL 后训练）

## Purpose
理解并（在 roadmap Stage 9 允许时）小规模复现 Z-Image 的蒸馏与 RL 管线：Decoupled-DMD、DMDR、DPO/GRPO [PAPER §4.5–4.6]。本模块以**研究为主、实验为辅**，算力门槛明确。

## Scope
- Decoupled-DMD（独立论文 2511.22677）：CFG Augmentation（引擎）与 Distribution Matching（正则器）解耦，各自独立 renoise schedule [PAPER §4.5.1]。
- DMDR（独立论文 2511.13649）：RL 目标 + DM 正则，防 reward hacking [PAPER §4.5.2]。
- RLHF 两阶段：DPO 用客观维度偏好对（VLM 生成 + 人验 + 课程）[PAPER §4.6.2]；GRPO 复合优势在线优化 [PAPER §4.6.3]。
- 关键数值事实：Turbo 8 NFE、无 CFG [OFFICIAL-CODE]；SFT 模型 ~100 NFE 带 CFG [PAPER §4.5]。
- 不包含：训练课程（`training`）、评测（`evaluation`）、reward model 训练（官方内部，仅观测 [PAPER §4.6.1]）。

## Inputs
- paper-facts.md §4 后训练事实；arXiv 2511.22677 / 2511.13649 全文
- 现有蒸馏/RL 相关代码与笔记（先检查再动手）
- roadmap Stage 9 的立项条件

## Outputs
- 论文精读笔记（带证据标签，入 `docs/`）
- 若立项：mini 蒸馏实验计划 + 实验记录；不立项则只出笔记与可行性评估

## Rules
1. 改代码前先检查现有实现与笔记；**读两篇独立论文是动手前提**（细节在独立论文而非主报告 [PAPER §4.5]）。
2. 未经 roadmap Stage 9 立项不得写蒸馏训练代码（防范围蔓延）。
3. 概念严谨：CA 与 DM 的职责区分、renoise schedule 的解耦必须按 2511.22677 的表述复刻，不凭直觉。
4. 主报告对 DMD/DMDR 只有定性描述，一切定量细节以独立论文为准并标出处。
5. mini 复现的所有缩放与替代（如公开 reward model 替代）标 [IMPLEMENTATION]。

## Verification
- 笔记审查：每个机制（CA/DM/decoupled schedule）能独立复述其作用与来源。
- 若实验：以 SFT 模型为 teacher baseline，记录 8-NFE 学生与 teacher 的质量/指标对照。
- 数值核对：8 NFE、无 CFG 的设置与官方推理参数一致 [OFFICIAL-CODE]。

## Learning
- 练习：解释"为什么 CA 是引擎、DM 是正则器"（各自失效的实验现象 [PAPER §4.5.1 引 2511.22677]）。
- 练习：解释 DMDR 如何用 DM 项防 RL 的 reward hacking [PAPER §4.5.2]。
- 记录：蒸馏/RL 与全量训练在本机算力上的可行边界分析。

## Common failure modes
- 只读主报告就写 DMD 实现（关键细节全在 2511.22677）。
- 把普通 DMD 当 Decoupled-DMD（丢失 renoise schedule 解耦）。
- 未立项就"顺便"写蒸馏代码，违反一次一阶段。
- 用公开 reward model 替代却不标注，暗示与官方 reward model 等价。
