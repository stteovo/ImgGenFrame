# Semantic Dedup 报告（合成数据 demo）

> 第 14 步产出。当前为**合成 embedding demo**（真实 CLIP 嵌入未接入，见下）。
> 方法：kNN proximity graph（min_similarity 剪枝）+ Louvain 社区检测 → candidate 标记（不删除）。

## 1. 数据概况

| 项 | 值 |
| --- | --- |
| 图片数 | 750（500 唯一 + 50 组近重复 × 5） |
| 嵌入维度 | 64（合成，正态随机） |
| k（近邻数） | 10 |
| min_similarity | 0.7 [IMPLEMENTATION] |
| 社区检测 | Louvain（python-louvain，[IMPLEMENTATION] 替代 Leiden） |

## 2. 聚类结果

| 项 | 值 |
| --- | --- |
| 簇数 | 550 |
| 疑似重复 | 200 |
| 重复率 | 26.7%（= 50 组 × 4 非代表 / 750） |

## 3. 簇规模分布

```
规模  数量
  1   500   （唯一图 → 单例簇）
  5    50   （每组近重复 → 一簇）
```

分布符合预期：50 个近重复组被完整识别（每组 5 张聚为一簇），500 张唯一图各自成为单例簇，无过度聚类、无长尾误伤。

## 4. 疑似重复

- 每个近重复组中 1 张为 representative（`representative_score` 最高），其余 4 张标记 `candidate_duplicate=true`。
- **未删除任何数据**：仅标记，输出 `dedup_candidates.parquet`（`experiments/dedup_demo_candidates.parquet`），字段含 index / image_id / community_id / similarity / duplicate_score / representative_score / candidate_duplicate。

## 5. 长尾概念

合成随机数据无真实语义概念，长尾分析**不适用**。真实数据下的长尾概念分析依赖第 15 步知识图谱（concept → community 映射），届时补充：被标记为重复的簇中若含稀有概念，需人工复核（保守策略：宁可漏删不可错删长尾）。

## 6. 数据分布与后续

- 当前 pipeline 为 CPU 单机小数据版（sklearn kNN + python-louvain），750 张秒级完成。
- 真实数据接入后需：① 真实 CLIP 类嵌入（标 [IMPLEMENTATION]）；② k/min_similarity 依据数据规模重定并记录；③ 嵌入模型版本与去重结果绑定（模型升级 = 新数据版本）。

## 7. 与论文的对应与差距

| 论文 [PAPER §2.2] | 本实现 | 差距 |
| --- | --- | --- |
| k-NN 替代 range_search | sklearn NearestNeighbors kNN | ✅ 同法 |
| Leiden 社区检测（Traag 2019） | python-louvain（Louvain） | ⚠️ Louvain 替代 Leiden（leidenalg 未装） |
| rapidsai 全 GPU、k=100、8×H800 1B 条目 8h | CPU、k=10（demo） | 仅参照，不对标 |
