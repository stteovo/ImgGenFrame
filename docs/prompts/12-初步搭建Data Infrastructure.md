现在开始 data-infrastructure skill。

请检查当前项目以及我已有的 Z-Image DataInfrastructure。

目标：

不要重新发明已有的数据处理代码。

首先：

1. inventory existing code
2. inventory existing dataset schema
3. inventory existing ConceptParsing
4. inventory existing Rapidata processing
5. inventory existing KNN
6. inventory existing Louvain
7. inventory existing concept taxonomy

然后建立：

docs/reproduction/data_mapping.md

把：

existing implementation
↔
Z-Image paper
↔
target implementation

逐项对应。

注意：

Z-Image semantic organization 必须理解为：

embedding
→ kNN graph
→ community detection / modularity
→ semantic clusters
→ deduplication / balancing

不要把它简化成 nearest-neighbor dedup。

当前阶段：

只做 mapping 和 gap analysis。

不要重写已有代码。