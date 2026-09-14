---
name: data-infrastructure
description: Mini-scale reproduction of Z-Image's four-module data infrastructure (paper §2: Data Profiling Engine, Cross-modal Vector Engine, World Knowledge Topological Graph, Active Curation Engine) for the reproduction codebase. Use when building the data engine, filtering rules, curation loops, or the Z-Image-Turbo data pipeline for small-scale training.
---

# Data Infrastructure（数据基础设施 mini 复现）

## Purpose
以公开数据与开源工具实现论文 §2 四模块的**思想验证版**（reproduction_matrix B 层 B1–B4），服务于缩比训练，而非重造阿里的内部数据系统。

## Scope
- Profiling Engine mini：分辨率/文件大小、pHash、压缩比、信息熵（边界方差、JPEG 重编码 BPP）、启发式质量过滤 [PAPER §2.1]；自训模型项（美学/AIGC/VLM tag/CN-CLIP 对齐）用公开替代并标 [IMPLEMENTATION]。
- Vector Engine mini：k-NN 建图 + 社区检测去重（k-NN 替代 range_search、Leiden [PAPER §2.2]），10⁵–10⁶ 规模（官方为 rapidsai 8×H800、1B 条目 [PAPER §2.2]，仅作参照，不做性能对标）。
- Knowledge Graph mini：小规模概念树 + 加权采样（见 `knowledge-graph` skill）。
- Active Curation mini：用小模型失败案例检索补数据 [PAPER §2.4 思想]。
- 不包含：caption 生成（`caption-pipeline`）、去重细节实现（`semantic-dedup`）、图构建（`knowledge-graph`）。

## Inputs
- paper-facts.md §5（数据基础设施事实）
- 可用公开数据集与开源工具清单（进入该阶段时调研记录）
- 现有 `src/zimage/data/` 实现（先检查再动手）

## Outputs
- 各模块 mini 实现 + 测试
- 数据版本管理方案（hash/版本号，配合 `experiment` skill 的 dataset.md）
- 过滤/采样行为的对照评估记录

## Rules
1. 改代码前先检查现有实现。
2. **范围克制**：只做 mini 版；官方指标（如 8h/1B 条目吞吐 [PAPER §2.2]）不构成我们任何 KPI。
3. 一切公开替代品（美学模型、VLM tagger 等）必须标 [IMPLEMENTATION]，不得暗示与官方内部模型等价。
4. 数据一经训练引用即冻结版本（configs 与 dataset.md 双重记录）。
5. 过滤规则要有可解释性：每条规则的动机引用论文，参数（阈值等）标 [IMPLEMENTATION]。

## Verification
- 人工抽查：随机 1K 样本，检查过滤行为与规则意图一致，记录查准/查全。
- 数值测试：熵、压缩比等特征的边界用例（纯色图、纯噪声图）符合预期。
- 流水线测试：端到端从原始数据到可训练样本的确定性（同版本数据两次跑结果一致）。

## Learning
- 练习：解释"为什么论文把数据基建放在架构之前讲"（信息增益率决定训练效率上限 [PAPER §2]）。
- 练习：pHash 去重与语义去重的分工（低层指纹 vs 嵌入近邻）。
- 记录：每个 mini 模块与官方方案的差距清单（规模、模型、算力）。

## Common failure modes
- 把官方 8×H800 的工程规模当成我们的实现目标，过度设计。
- 用规则引擎复刻"自训模型"的效果却不标注差异。
- 数据没版本化，实验无法追溯。
- 过滤太狠把稀有概念（长尾）杀光，与论文的"概念广度"目标相悖。
