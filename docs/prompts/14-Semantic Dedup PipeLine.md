使用 semantic-dedup skill。

实现 Z-Image style semantic dedup pipeline。

Pipeline：

image
→ embedding
→ kNN
→ graph
→ community detection
→ cluster statistics
→ duplicate candidate
→ representative selection

不要直接删除数据。

第一阶段只生成：

cluster_id
similarity
community_id
duplicate_score
representative_score

输出：

dedup_candidates.parquet

然后生成：

docs/reports/dedup_report.md

包括：

- number of images
- number of clusters
- cluster size distribution
- suspected duplicates
- long-tail concepts
- data distribution

要求：

支持小数据集 CPU version。

之后再考虑 GPU acceleration。