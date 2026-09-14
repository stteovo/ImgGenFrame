# Z-Image Reproduction Roadmap

> 原则：一次只推进一个阶段；每阶段有明确出口条件（exit criteria）；未满足出口条件不得进入下一阶段。
> 每阶段产出的模块必须"先测试后训练"（见 `.opencode/skills/module-development/SKILL.md`）。
> 实验记录规范见 `.opencode/skills/experiment-protocol/SKILL.md`。

## Stage 0 — 基础设施（当前阶段）

- [x] 检查 repository（uv 项目、Python 3.13、torch 2.14+cu130、diffusers 0.40 含 ZImagePipeline）
- [x] AGENTS.md（规范、标签系统、硬规则）
- [x] `.opencode/skills/`（paper-facts / experiment-protocol / module-development）
- [x] `docs/reproduction/`（paper-facts、reproduction_matrix、roadmap）
- [x] 确认论文与官方代码来源并建立事实库
- [ ] **待决**：在本目录初始化独立 git 仓库（父目录 `~/learn` 已是 git repo，嵌套仓库才能做 per-experiment commit 记录）
- [ ] 待决：加入 pytest 等开发依赖（Stage 1 一并做）

## Stage 1 — 官方实现精读 + 本机基线（研究，不写模型代码）

**目标**：在写任何代码之前，把官方推理代码和 diffusers 实现读透，把 paper-facts 中所有 "待确认" 项清零。

任务：
1. 精读 `Tongyi-MAI/Z-Image` 的 `src/zimage/`、`src/utils/`、`inference.py`（模型前向、RoPE、调制、加载逻辑）。
2. 精读 diffusers `ZImagePipeline` + `ZImageTransformer2DModel`（PR #12703/#12715）。
3. 下载 Z-Image-Turbo checkpoint，在 5070 Ti 上跑通官方推理（8 步），记录：VRAM 峰值、延迟、固定种子输出（作为后续对比基线）。
4. 下载 Z-Image（SFT 版）配置，核对 HF `config.json` 与 paper-facts §2.2 / §6 差异表（heads=30 vs 32、VAE 常量、调度器 shift）。
5. 探针脚本（只读不训练）：Qwen3-4B 输出特征统计、Flux VAE latent 统计、RoPE 常量的实际用法。

出口条件：paper-facts §6 差异表全部裁决；§7 UNKNOWN 至少解决与架构相关项；基线图 + 探针数据入 `experiments/`。

## Stage 2 — M1：100M S3-DiT + 单元测试（写代码从这一阶段开始）

任务：
1. 搭 `src/zimage/` 骨架 + pytest 基建。
2. 按依赖序实现（每个先写测试）：RMSNorm/QK-Norm/Sandwich-Norm → 3D RoPE → 低秩 adaLN → modality processor → 单流 block → 完整 S3-DiT（可配 100M）。
3. 实现时间/条件嵌入与 patchify。
4. 数学模块与 diffusers 官方实现对数值对齐（用官方小权重或随机权重对齐前向）。
5. M1 级配置设计与推导记录。

出口条件：全部单测通过；100M 前向/反向在 16GB 内；模块与官方实现对齐报告；实验记录模板首次实战。

## Stage 3 — Flow Matching 训练循环 + 玩具数据验证

任务：
1. 从零实现 FM 训练：x_t 插值、v 目标、MSE、logit-normal t 采样。
2. 从零实现采样器：Euler ODE、CFG、负向提示。
3. 玩具/小型数据（如 MNIST/合成形状）端到端训练 100M。
4. 实验记录全流程首次完整演练。

出口条件：loss 收敛、采样出可辨认图像；t 采样分布、shift 行为有直方图级验证；CFG 开/关行为符合预期。

## Stage 4 — M2（300M）+ 数据管线 v0 + 256² 真实数据

任务：
1. 数据管线 v0：下载/筛选小型公开 T2I 数据集（候选：CC12M 子集 / OIT / 自选，进入时评估）+ B5 caption 管线。
2. B1/B2 mini：profiling 规则 + kNN-社区检测去重。
3. 低分辨率预训练阶段复现（256²、logit-normal、动态 shift）。
4. 训练 CFG dropout、EMA（若采用需标记 [ASSUMPTION]，论文未说明）。

出口条件：300M 在 256² 出像；去重与过滤行为有人工评估记录；完整课程记录（数据版本、配比）。

## Stage 5 — M3（1B）+ 多分辨率 + 基建 mini 全链路

任务：B3/B4（图谱采样 + active curation mini）；任意分辨率训练（B6）+ 序列长度感知组 batch/动态 batch（B7）；显存工程（grad ckpt / 8bit optim / offload 组合实测）。

出口条件：任意长宽比出像；组 batch 相比朴素 batching 的吞吐对照实验完成；1B 训练稳定 ≥N 天。

## Stage 6 — 6B 架构实例化 + 官方权重对齐（关键里程碑）

任务：
1. 以原尺寸配置实例化我们的 S3-DiT。
2. 加载官方 Z-Image / Z-Image-Turbo 权重，与 diffusers pipeline 做固定种子逐像素对齐。
3. 本机推理优化实验（offload、量化可选）。
4. 官方 checkpoint 结构分析（写进报告）。

出口条件：对齐报告（最大误差、误差分布）通过预设阈值；此项是架构忠实复现的最终裁决。

## Stage 7 — M4（3B）+ SFT 三技术 + controlled ablations

ablation 候选（进入时定稿，全部在 100M/300M 上做）：
- Sandwich-Norm vs 普通 pre-norm
- 低秩分解 adaLN vs 全秩 adaLN（参数量/质量）
- logit-normal vs uniform t 采样
- 单流 vs（参考实现）双流参数效率对照
- caption 粒度（长/短/混合）对文字渲染与对齐的影响
- 论文未说明但实践关键项：CFG dropout 概率、EMA、lr 调度（标记 [EXPERIMENT]）

## Stage 8 — 2026 年新方法研究 + recipe 改进

系统调研 2026 年文生图方法（架构、训练目标、数据、蒸馏、RL），写成 survey 文档；挑选 2–3 项在我们的 M1/M2 上做对照改进实验（保持受控：只改一个变量）。

## Stage 9（可选）— mini 蒸馏

在最佳小模型上 mini 复现 Decoupled-DMD（读透 arXiv:2511.22677 后再立项）。

## 里程碑依赖图

```
S0 → S1 → S2 → S3 → S4 → S5 → S7 ─┐
                └─────────────── S6（权重对齐可提前至 S2 后并行做模块级对齐）
S8 依赖 S4+；S9 依赖 S7
```

> S6 的"全量权重对齐"不依赖训练完成，只依赖 S2 的 6B 配置实例化——它是检验复现正确性的最硬指标，可与其余训练阶段并行推进。
