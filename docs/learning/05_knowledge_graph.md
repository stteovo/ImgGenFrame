# Learning Report — Knowledge Graph（概念图与采样权重）

## 1. 做了什么

实现 `src/zimage/data/knowledge_graph.py`：`Concept`/`ConceptGraph`（parent/child/synonym/related/visualizability/rarity）、BM25 IDF 稀有度、数据集频率、`compute_sampling_weights`（visualizability × rarity^α × hierarchy）、`concept_report`（逐概念记录 entity/hierarchy/frequency/rarity/visualizability/sampling_weight）。8 项测试通过。

## 2. 为什么 KG 不只是 ontology，而是采样引擎

普通 ontology（概念分类树）是**静态知识组织**；Z-Image 的 World Knowledge Topological Graph 是**采样决策的依据** [PAPER §2.3, §4.4]：

```
caption concepts → 图节点 → BM25 稀有度 + 层级 + 可视化过滤
                              ↓
                      sampling_weight
                              ↓
              tagged resampling（mini-batch 动态加权）
                              ↓
                长尾概念不被遗忘（防灾难性遗忘）
```

所以图本身不是终点，"采样分布可度量"才是终点。没有接采样器的概念图只是学术展示。

## 3. 为什么需要它

训练数据天然长尾：常见概念（dog/cat）海量、稀有概念（tibetan_mastiff）稀疏。若均匀采样，模型会**灾难性遗忘长尾**；若拍脑袋"类别均衡"，会破坏分布。Z-Image 用**图 + BM25 稀有度**给出可解释的采样权重：稀有概念高权重 → 语义级均衡，既保长尾又保分布结构。

## 4. 核心公式（BM25 口径可解释）

```
IDF(c) = log((N − df + 0.5)/(df + 0.5) + 1)     # BM25 IDF 变体
rarity(c) = IDF(c)                                # 稀有 → 高 IDF
hierarchy(c) = 1 + ln(1 + n_children(c))          # 父节点受控（子越多越中心）
sampling_weight(c) = visualizability(c) · rarity(c)^α · hierarchy(c)
```

- **Corpus** = 全部 caption；**Document** = 单条 caption 的 tag 集合；**Tokenizer** = 空白/逗号分词。
- **visualizability**：VLM 判定"概念能否被可视化生成"，不可见概念（抽象情绪）权重归零 [PAPER §2.3 可视化生成性过滤]。
- **rarity**：BM25 IDF，稀有概念权重高。

## 5. 当前实现要点与差距

| 论文 [PAPER §2.3] | 本实现 | 差距 |
| --- | --- | --- |
| Wikipedia 实体图 + PageRank 剪枝 | 手动小图（animal→dog→golden_retriever） | ⚠️ 全量 Wikipedia/PageRank 未做 |
| VLM 可视化生成性过滤 | `visualizability` 字段（手动/占位） | ⚠️ 未接真实 VLM |
| 层级化扩充（Vo et al. 2024） | 手动 parent/child | ⚠️ 未做嵌入层级聚类 |
| BM25 + 层级采样权重 | ✅ 已实现（可解释） | ✅ |

## 6. 验证

- 稀有概念（tibetan_mastiff）权重大于高频（dog）；不可视觉化 → 0；父节点层级因子高于叶子；report 含全部要求字段。

## 7. Failure Modes

- 建图不接采样（纯展示）。
- 拍脑袋类别均衡替代 BM25+层级。
- 忽略可视化过滤，把抽象概念留在图里浪费权重。
- 权重不随数据版本冻结，分布漂移。

## 8. 下一步

- 接真实 VLM 可视化过滤 + 真实概念 taxonomy。
- 概念覆盖分析（采样前后长尾分布对比）。
- 与第 16 步 Data→Training 闭环、第 22 步 SFT tagged resampling 衔接。
