# Task List — Z-Image 复现（docs/prompts/ 第 4–29 步）

> 状态与 Pi goal 任务树同步。每步完成 = 实现/研究 + 验证 + 一次 git 提交 + 学习报告 + 文档更新。
> 标记：`[x]` 完成，`[ ]` 待办。

## Phase 1 — 核心架构与最小 pipeline（已完成）
- [x] 第 4 步：最小可配置 S3-DiT（src/zimage/models/ + 测试先行）
- [x] 第 5 步：3D Unified RoPE（temporal/height/width 独立模块）
- [x] 第 6 步：Flow Matching（velocity 目标 + 时间采样）
- [x] 第 7 步：VAE Tokenization（Flux VAE 解耦 + patchify）
- [x] 第 8 步：Conditioning（timestep 嵌入 + 低秩 adaLN + CFG 逻辑）
- [x] 第 9 步：最小可运行 pipeline（scripts/smoke_test.py）

## Phase 2 — 首次训练与数据基建（已完成）
- [x] 第 10 步：训练 100M Z-Image-Tiny（EXP-TRAIN-001）
- [x] 第 11 步：导师回顾学习报告
- [x] 第 12 步：Data Infrastructure 四模块骨架 + profiling 规则
- [x] 第 13 步：Caption Pipeline（多级 caption + span-grounded 抽取）
- [x] 第 14 步：Semantic Dedup（kNN graph + Louvain 社区检测）
- [x] 第 15 步：Knowledge Graph（概念层级 + BM25 采样权重）
- [x] 第 16 步：Data → Training 闭环（版本化 dataset 可追溯）

## Phase 3 — 缩放与分布式（下一步开始）
- [ ] 第 17 步：测试 300M 模型（Z-Image-Small）
- [ ] 第 18 步：测试 1B 模型
- [ ] 第 19 步：FSDP2 分布式训练（单 GPU 正确后再上）

## Phase 4 — 实验框架与评测
- [ ] 第 20 步：统一实验框架
- [ ] 第 21 步：统一 image generation evaluation framework
- [ ] 第 22 步：真正复现 Z-Image Training Recipe（分阶段课程）
- [ ] 第 23 步：2026 SOTA branch（受控研究，与 Track A 隔离）

## Phase 5 — 2026 SOTA 实验
- [ ] 第 24 步：Muon 实验
- [ ] 第 25 步：Image-only Pretraining
- [ ] 第 26 步：Distillation（mini Decoupled-DMD）
- [ ] 第 27 步：DMDR（DMD meets RL）
- [ ] 第 28 步：RL GRPO
- [ ] 第 29 步：创建科研审计 Agent（reproduction audit）
