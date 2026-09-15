# Changelog — Z-Image 复现（按步骤）

> 每步一条：commit + 一句话总结。详情见 `git log` 与 `docs/learning/`、`docs/reproduction/`。

## 2026-09-15
- `b0c20fb` [feat] 第 17 步：测试 300M 模型（缩放基准，scale 升级出口满足）
- `ac25e59` [feat] 建立 Agent 长期状态文件与上下文预算（STATE/TASK/CHANGELOG + compaction）
- `8086dd2` [feat] 第 16 步：Data→Training 闭环（DatasetManifest 版本化 + checkpoint 记录 dataset_version）
- `f42bddf` [feat] 第 15 步：Knowledge Graph（概念层级 + BM25 采样权重）
- `870fee5` [feat] 第 14 步：Semantic Dedup（kNN graph + Louvain，candidate_duplicate 标记）
- `444bcbf` [feat] 第 13 步：Caption Pipeline（多级 caption + observed/inferred + parquet）
- `13b2313` [feat] 第 12 步：Data Infrastructure 骨架 + profiling 规则
- `d34d420` [feat] 第 11 步：里程碑一导师回顾学习报告
- `bc0481e` [feat] 第 10 步：训练 100M Z-Image-Tiny（EXP-TRAIN-001 过拟合）
- `cfdf72a` [feat] 第 9 步：最小可运行 pipeline（smoke_test）
- `ad82f5b` [feat] 第 8 步：Conditioning 抽象 + 低秩 adaLN
- `fc42dc9` [feat] 第 7 步：VAE Tokenization（Flux VAE + patchify）
- `9819c54` [feat] 第 6 步：Flow Matching 原语 + Euler 采样器
- `836aeab` [feat] 第 5 步：3D Unified RoPE
- `2bef47b` [feat] 第 4 步：最小可配置 S3-DiT
