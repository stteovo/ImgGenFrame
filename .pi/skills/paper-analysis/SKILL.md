---
name: paper-analysis
description: Methodical reading and fact extraction of Z-Image papers (arXiv 2511.22699, Decoupled-DMD 2511.22677, DMDR 2511.13649) and related work, with evidence tags and mandatory source citations. Use when reading a paper, extracting claims, updating docs/reproduction/paper-facts.md, or answering "what does the paper say" questions.
---

# Paper Analysis

## Purpose
把论文声明转化为带证据标签、可追溯出处的事实条目，维护 `docs/reproduction/paper-facts.md` 作为唯一事实库，防止记忆、猜测、二手摘要污染复现决策。

## Scope
- 目标论文：Z-Image（arXiv:2511.22699，HTML: https://arxiv.org/html/2511.22699v1）、Decoupled-DMD（2511.22677）、DMDR（2511.13649）、以及后续 Stage 8 调研论文。
- 只做事实提取与对照，不做实现、不改业务代码。
- 与官方代码/HF 权重冲突时必须生成"差异记录"，交由 `reproduction-audit` skill 复核。

## Inputs
- 论文（arXiv URL / PDF / HTML）
- `docs/reproduction/paper-facts.md` 当前事实库（先读，避免重复收录）
- 官方代码（github.com/Tongyi-MAI/Z-Image，如 `src/config/model.py`）用于交叉验证

## Outputs
- 带标签的事实条目：`[PAPER]`（注明章节/表号）、`[OFFICIAL-CODE]`（注明文件）、`[OFFICIAL-CKPT]`、`[INFERENCE]`（附推理链）、`[UNKNOWN]` 清单
- 更新后的 `paper-facts.md`（含 §6 差异表、§7 UNKNOWN 清单）
- 无法裁决的冲突 → 明确列出，不自行裁决

## Rules
1. **先读事实库**：提取前先读 paper-facts.md，新条目必须注明是否与现有条目冲突。
2. **引用必须可定位**：论文引用到章节/表号，代码引用到文件与常量名；禁止"论文提到过"这类无出处表述。
3. **禁止编造**：信息缺失 → `[UNKNOWN]` 并列入 §7；不得从模型记忆或类似工作补全 Z-Image 未说明的细节。
4. **区分声明层级**：官方 README/博客是宣传声明，不升级为论文事实；三者（论文/代码/权重）冲突时记录为差异。
5. **标签纪律**：`[ASSUMPTION]` 必须写明假设内容与验证计划；`[INFERENCE]` 必须写出推理链。
6. 已知差异示例（paper-facts §6）：heads 32 vs 30；VAE 4ch 遗留常量 vs Flux 16ch——引用现状，不擅自定论。

## Verification
- 抽查：每条新事实能指出原文/代码的精确位置（章节号或文件+行号范围）。
- 交叉验证：架构类声明与官方 `src/config/model.py`、HF `config.json` 对撞，冲突进差异表。
- 完整性：新增 UNKNOWN 项必须同时给出"计划在哪个 Stage 调查"。

## Learning
- 练习：解释"为什么 Table 2 的 32 heads 与代码 30 heads 是复现陷阱"（head_dim=128 与 RoPE 轴维和 128 自洽 [OFFICIAL-CODE]；裁决依据是 HF config.json）。
- 记录：一次分析产出的推理链（如 RoPE 轴长度 512 与 2048px 的关系推导）写入学习笔记，标 `[INFERENCE]`。

## Common failure modes
- 凭记忆引用论文数字而不打开原文 → 数字出错（如 arXiv ID、heads 数）。
- 把官方 README 表述当作论文事实。
- 用其他工作（SD3/Flux）的做法脑补 Z-Image 缺失细节。
- 摘要式转述丢失前提条件（如"Turbo 8 步"漏掉"无 CFG"前提）。
