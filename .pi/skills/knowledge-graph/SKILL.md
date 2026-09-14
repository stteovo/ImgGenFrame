---
name: knowledge-graph
description: "Mini reproduction of Z-Image's World Knowledge Topological Graph for concept-balanced sampling (paper §2.3: Wikipedia entities, PageRank pruning, visual generatability filtering, hierarchical tag augmentation, BM25 + hierarchy-weighted sampling). Use when building the concept graph, concept coverage analysis, or semantic-level data balancing."
---

# Knowledge Graph（世界知识拓扑图 mini）

## Purpose
复现论文 §2.3 的概念组织与**语义级均衡采样**：以概念图为骨架，对训练数据做概念层面的加权采样，保证长尾概念覆盖（与 SFT 期 tagged resampling [PAPER §4.4] 衔接）。

## Scope
- 图构建 mini：Wikipedia 子图（实体+超链接）→ PageRank 剪枝 → 可视化生成性过滤（VLM 判定概念能否生成）[PAPER §2.3]。
- 层级化扩充 mini：从 caption tags 的嵌入做层级聚类/树构建（引 Vo et al. 2024 [PAPER §2.3]；我们规模用简化层级 [IMPLEMENTATION]）。
- 采样权重：tag → 图节点映射，结合 BM25 分数与父子层级关系计算采样权重 [PAPER §2.3]。
- 不包含：完整 Wikipedia 全量图与内部数据扩充（超出范围）；嵌入索引（`semantic-dedup`）。

## Inputs
- paper-facts.md §5（Topological Graph 事实）
- 训练 caption 的 tag 集（来自 `caption-pipeline` 输出）
- 现有图/采样相关代码（先检查再动手）

## Outputs
- 概念图构建与层级化脚本 + 加权采样器 + 测试
- 概念覆盖分析报告（训练前后分布对比）
- 权重计算与超参决策记录

## Rules
1. 改代码前先检查现有实现。
2. 权重公式忠实于论文组合：**BM25 + 层级关系** [PAPER §2.3]；简化处标 [IMPLEMENTATION] 并给出理由。
3. 手动 up-weight 高频用户概念 [PAPER §2.3] 在我们的版本中需标注来源（公开 prompt 统计或 [ASSUMPTION]）。
4. 图与采样权重随数据版本冻结。
5. 目标不是图本身，是采样分布：任何实现都要落到"采样器输出分布可度量"。

## Verification
- 单元测试：权重计算满足层级语义（子概念过采→父概念受控）；BM25 得分对权重影响方向正确。
- 分布度量：采样前后概念频率分布（如长尾分位数覆盖率）对照，记录指标。
- 下游验证：概念均衡采样对 100M/300M 模型长尾生成能力的 ablation。

## Learning
- 练习：解释"为什么采样权重同时考虑 BM25 与层级"（文本匹配 + 语义广度控制 [PAPER §2.3]）。
- 练习：PageRank 剪枝 + 可视化生成性过滤各自过滤掉什么（孤立概念 vs 抽象不可见概念）。
- 记录：我们的图规模/层级深度与官方方案的差距。

## Common failure modes
- 建了图却不接采样，变成纯学术展示。
- 用拍脑袋的类别均衡代替论文的 BM25+层级加权。
- 忽略"可视化生成性过滤"，把不可视觉化的概念留在图中浪费权重。
- 权重不冻结，训练中数据分布漂移导致实验不可比。
