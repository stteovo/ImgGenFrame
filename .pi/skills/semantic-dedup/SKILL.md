---
name: semantic-dedup
description: "Graph-based semantic deduplication following Z-Image's Cross-modal Vector Engine (paper §2.2: reformulation of SD3 dedup as k-NN proximity graph + community detection, replacing range_search with k-NN, Leiden algorithm; official scale 100-NN, 1B items on 8xH800 as reference only). Use when implementing embedding-based dedup, k-NN indexing, or cluster-based dataset pruning at mini scale."
---

# Semantic Dedup（语义去重）

## Purpose
以 k-NN 建图 + 社区检测实现语义去重（官方 Cross-modal Vector Engine 的 mini 版 [PAPER §2.2]），去掉训练集冗余、提升单位算力的信息增益。

## Scope
- 嵌入提取：CLIP 类公开模型（官方用 CN-CLIP [PAPER §2.1]，我们用公开替代并标 [IMPLEMENTATION]）。
- k-NN 索引与近邻图构建（官方 k=100 [PAPER §2.2] 作参照；我们按数据规模重定 k 并记录）。
- 社区检测去重：Leiden [PAPER §2.2 引 Traag 2019]；社区内按策略采样。
- pHash 等低层去重归 `data-infrastructure` 的 profiling；本 skill 专注嵌入级。
- 官方 rapidsai/8×H800 性能数字仅作背景，不做对标。

## Inputs
- paper-facts.md §5（Vector Engine 事实）
- 待去重数据（版本化）与嵌入缓存
- 现有去重相关代码（先检查再动手）

## Outputs
- 去重管线（嵌入→索引→建图→社区检测→采样策略）+ 测试
- 去重率/近重复召回的评估报告（入实验记录）
- k、阈值、采样策略等超参的决策记录（标 [IMPLEMENTATION]）

## Rules
1. 改代码前先检查现有实现。
2. 核心结构忠于论文：**k-NN 替代 range_search、图 + 社区检测** [PAPER §2.2]；不得退化为纯阈值去重且声称等价。
3. 嵌入模型版本与去重结果绑定（模型升级 = 新数据版本）。
4. 保守策略：宁可漏删不可错删长尾概念（与论文"概念广度"目标一致）。
5. 各参数（k、相似度阈值、社区内保留比例）显式记录，进 dataset.md。

## Verification
- 召回测试：构造已知近重复对（缩放/裁剪/重压/轻微改字），验证被聚到同社区。
- 误删检查：人工抽查被删样本，确保无长尾概念误伤。
- 可复现：同版本数据两次运行得到相同划分（固定随机源）。
- 规模测试：10⁵–10⁶ 级在单机可承受时间内完成（记录耗时）。

## Learning
- 练习：解释"为什么 k-NN 建图 + 社区检测能替代 pairwise 去重"（计算复杂度与近似等价性 [PAPER §2.2]）。
- 练习：社区（簇）内保留策略对数据多样性的影响。
- 记录：我们规模下 k 的取值与官方 100 的差异理由。

## Common failure modes
- 用全局阈值两两比对替代图+社区检测，违背论文方法却不自知。
- 阈值太激进，长尾概念被当重复删除。
- 嵌入模型更换后不复算去重，版本错乱。
- 把"8 小时/1B 条目"当成本机性能目标。
