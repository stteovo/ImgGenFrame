# Project State — Z-Image Research

> 精简状态文件（<3000 tokens）。代码/测试/文档是 source of truth，本文件只存"续作必需的状态"。

## Current Goal
按 `docs/prompts/` 文件名首数字顺序连续执行第 4–29 步（26 步）。每步：依据提示词文件 + 对应 skill + `docs/reproduction/` 完成实现/研究 → 运行验证 → 用 `[feat|fix] 主题\n\n<body>` 格式做一次 git 提交 → 学习报告 + 文档更新。不逐步停下等确认。

## Current Phase
**第 17 步：测试 300M 模型（Z-Image-Small）**。已完成 13/26 步（第 4–16 步）。

## Completed
- 第 4–16 步全部完成：S3-DiT、3D RoPE、Flow Matching、VAE Tokenization、Conditioning、最小 pipeline、100M 训练、导师回顾、Data Infra 骨架、Caption Pipeline、Semantic Dedup、Knowledge Graph、Data→Training 闭环。
- 184 项 pytest 通过；EXP-TRAIN-001（100M 过拟合 16 张合成 latent：loss 2.34→0.25、16 步 Euler 重建 MSE 0.085、峰值 VRAM 1.63GB、21 it/s）。

## In Progress
无（第 16 步已提交 `8086dd2`）。

## Known Issues
- 训练超参（AdamW 1e-4 / logit-normal σ=1.0 / 无 EMA / 无 CFG dropout）均 [ASSUMPTION]，待消融（open_questions A1–A8）。
- 真实图像数据 + 真实 Qwen3-4B 文本编码器未接入（当前合成 latent + mock 文本）。
- 社区检测用 Louvain 替代 Leiden（leidenalg 未装，标 [IMPLEMENTATION]）。
- ruff/mypy 未启用（pypi TLS 抖动，网络时好时坏）。
- 环境变量含泄露 API key（DASHSCOPE_API_KET），切勿写入任何文件/日志。

## Important Decisions
- 缩比约束：head_dim=128=sum(rope_axes_dims)、ffn_dim=int(dim/3*8)（config 校验硬编码）。
- patchify token 序 = 官方 `[pH,pW,C]`（roundtrip 回归测试锁死，此前曾错为 [C,pH,pW]）。
- Flow Matching 方向约定（论文）：x_t=(1−t)x0+t·x1、v=x1−x0、x0=噪声 x1=数据。
- Flux VAE 16ch / scaling 0.3611 / shift 0.1159（非遗留 4ch/0.18215）。
- 语义去重 = kNN graph + 社区检测，仅标记 candidate_duplicate（权重归零），不删除数据。
- 数据集版本化：checkpoint 记录 dataset_version + manifest_hash。

## Tests
- 全量：`pytest tests/ -q` → 184 passed（models/diffusion/tokenization/conditioning/pipeline/training/data/captioning）。
- 冒烟：`scripts/smoke_test.py --device cuda`；`scripts/train_tiny.py --smoke --device cuda`。
- 测试失败先跑 `pytest <target_file> -q`，再跑相关文件，最后全量。

## Important Files
- 事实库：`docs/reproduction/paper-facts.md`、`reproduction_matrix.md`、`open_questions.md`、`roadmap.md`、`data_mapping.md`、`milestone_01_s3dit.md`
- 源码根：`src/zimage/`（models/ diffusion/ tokenization/ conditioning/ training/ data/ captioning/ pipeline.py）
- 配置：`configs/tiny/{model,training,data}.yaml`
- 实验：`experiments/20260915-tiny-overfit/`（record/environment/dataset/metrics）
- 报告/学习：`docs/reports/dedup_report.md`、`docs/learning/*.md`

## Next Step
第 17 步：读 `docs/prompts/17-*.md` + 相关 skill（distributed-training / evaluation），测试 300M 前向/反向显存与吞吐，产出显存预算表，验证 scale 升级出口条件。
