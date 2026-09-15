# Data Infrastructure Mapping（现有实现 ↔ Z-Image 论文 ↔ 目标实现）

> 第 12 步产出：inventory + gap analysis。原则：不重造轮子，先摸清现有资产与缺口。
> 依据：paper-facts.md §5（数据基础设施）、`data-infrastructure` skill。

## 0. Inventory 结论（2026-09-15）

| 资产 | 现状 |
| --- | --- |
| 数据基础设施代码（`src/zimage/data/`） | **无**（第 12 步前不存在；`training/dataset.py` 是合成 latent，非数据基建） |
| ConceptParsing | **无** |
| Rapidata processing | **无**（仅 HF 缓存 `datasets--Rapidata--human-style-preferences-images`） |
| kNN | **无**（依赖 `scikit-learn 1.9.1` 可用） |
| Louvain / Leiden | **无**（依赖 `python-louvain 0.16` 可用） |
| Concept taxonomy | **无** |
| 本地数据集缓存 | flickr30k、conceptual_captions、Rapidata human-style-preferences、madao33 new-title-chinese |

**结论**：四模块（Profiling / Vector / Knowledge Graph / Active Curation）在仓库中均无现有实现——gap 是**整体空白**，需要从骨架开始搭建；可复用的只有依赖（sklearn、python-louvain）与本地数据集缓存。

## 1. 语义组织：不是"最近邻去重"

Z-Image 的语义组织链路必须按如下理解（[PAPER §2.2]），**不得简化为 nearest-neighbor dedup**：

```
embedding（跨模态向量）
  → kNN graph（k 近邻建图，替代 SD3 的 range_search）
  → community detection / modularity（Leiden，社区发现）
  → semantic clusters（语义簇）
  → deduplication（簇内冗余分析，标记而非删除）
  → balancing（概念均衡采样，见 knowledge-graph）
```

与"NN dedup"的区别：NN 只看局部成对相似；社区检测看**全局图结构**（簇的模块度/密度），能区分"语义相似的类目簇"与"真正的重复冗余"。

## 2. 逐模块 mapping

| Z-Image 论文 [PAPER §2] | 现有实现 | 目标实现（mini） | Gap / 待办 |
| --- | --- | --- | --- |
| **Data Profiling Engine**（§2.1）：分辨率/文件大小、pHash、压缩伪影、质量模型、信息熵、美学、AIGC 检测、VLM tag+NSFW、CN-CLIP 对齐、多级 caption | 无 | 规则启发式：分辨率、文件大小、aHash（pHash 替代）、压缩比、信息熵、边界方差；自训模型项用公开替代 [IMPLEMENTATION] | 第 12 步搭骨架+规则；质量/美学/VLM tag 后续接公开模型 |
| **Cross-modal Vector Engine**（§2.2）：kNN + Leiden 社区检测去重、跨模态检索 | 无（sklearn/python-louvain 可用） | kNN 建图 + Leiden（`semantic-dedup` skill，第 14 步） | 第 14 步实现 |
| **World Knowledge Topological Graph**（§2.3）：Wikipedia 实体图 + PageRank + 层级化 + BM25 采样权重 | 无 | mini 概念树 + 加权采样（`knowledge-graph` skill，第 15 步） | 第 15 步实现 |
| **Active Curation Engine**（§2.4）：小模型诊断 + human-in-the-loop | 无 | 小模型失败案例检索 mini（后步） | 第 16 步闭环后 |

## 3. 语义组织 mapping（§2.2 专项）

| 阶段 | Z-Image [PAPER §2.2] | 目标实现 | 状态 |
| --- | --- | --- | --- |
| embedding | 跨模态向量（CLIP 类） | 公开 embedding 模型（CN-CLIP 类，标 [IMPLEMENTATION]） | 待接（第 14 步） |
| kNN graph | k=100，rapidsai 全 GPU（8×H800 1B 条目 8h，仅参照） | sklearn kNN（10⁵–10⁶ 级，不做性能对标） | 待实现 |
| community detection | Leiden（Traag 2019） | `python-louvain`（Leiden/Louvain） | 待实现 |
| semantic clusters | 社区即语义簇 | 同左 | 待实现 |
| dedup / balancing | 簇内冗余 + 概念均衡 | `candidate_duplicate=true` 标记（不删原始） | 待实现 |

## 4. Gap 优先级

| 优先级 | 项 | 依赖 |
| --- | --- | --- |
| P0 | Profiling 规则骨架（本步） | 无 |
| P0 | data_mapping 落地（本步） | 无 |
| P1 | embedding + kNN + 社区检测（第 14 步） | 公开 embedding 模型 |
| P1 | 概念图 + 采样权重（第 15 步） | 概念 taxonomy |
| P2 | Active Curation（第 16 步后） | 小模型训练产物 |

## 5. 数据版本纪律

- 数据一经训练引用即冻结（`configs/` + `dataset.md` 双重记录）。
- 过滤阈值等参数标 [IMPLEMENTATION]，动机引用论文条目。
- 过滤不得杀光长尾概念（与论文"概念广度"目标相悖）。
