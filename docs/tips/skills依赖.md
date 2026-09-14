# skills之间的依赖关系
1. 事实层    paper-analysis ──► paper-facts.md ◄── reproduction-audit（审计裁决）
          （已有 z-image-paper-facts 负责快速查证）

2. 模型层    s3-dit ──► 引用 rope-3d、conditioning
          conditioning ──► 依赖 s3-dit 的调制位点
          vae-tokenization、flow-matching ──► 给 training 提供 latent 路径与目标

3. 数据层    data-infrastructure ──► 编排 caption-pipeline / semantic-dedup / knowledge-graph
          knowledge-graph ──► 输出权重给 training 的 SFT resampling

4. 训练层    training ──► 消费模型层+数据层；distributed-training 支撑其显存方案
          experiment ──► 统领所有执行类 skill（记录契约→experiment-protocol）
          evaluation ──► 为 experiment 提供判据

5. 后训练    distillation ──► 消费 flow-matching/training/evaluation，受 roadmap Stage 9 立项门控

6. 横切层    learning 覆盖所有 skill；module-development（已有）是模型层共用的测试先行流程

---
此外，所有 skill 均满足要求：含 Purpose/Scope/Inputs/Outputs/Rules/Verification/Learning/Common failure modes 八节；Rules 均含"改代码前先检查现有实现"；事实均引用论文章节或官方代码常量（如 src/config/model.py）；未知项显式标 [UNKNOWN]。