使用 knowledge-graph skill。

实现一个最小 Z-Image-style visual concept knowledge graph。

不要一开始构建完整 Wikipedia KG。

先支持：

concept
parent
child
synonym
related
visualizability
rarity

输入：

caption concepts

输出：

concept hierarchy

要求：

支持：

BM25 rarity score
+
hierarchy
+
dataset frequency

生成：

sampling_weight

目标：

用于后续 long-tail concept balancing。

创建：

docs/learning/05_knowledge_graph.md

重点解释：

为什么 Z-Image 的 KG 不只是一个 ontology，
而是最终参与 data sampling / curation。