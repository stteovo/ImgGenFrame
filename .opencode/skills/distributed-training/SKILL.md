---
name: distributed-training
description: Memory and parallelism engineering for Z-Image training on our hardware — official strategy per paper §4.2 (FSDP2 for DiT, DP for frozen VAE/text encoder, gradient checkpointing on all DiT layers, torch.compile) adapted to a single 16GB RTX 5070 Ti. Use when adding gradient checkpointing, offloading, mixed precision, batching memory math, or deciding parallelism strategy.
---

# Distributed Training（显存与并行工程）

## Purpose
把论文 §4.2 的训练效率策略（FSDP2 + DP + grad ckpt + torch.compile）转化为单卡 16GB 环境下的可执行工程方案；官方配置是事实参照，我们的配置是 [IMPLEMENTATION]。

## Scope
- 官方事实：DiT 用 FSDP2；冻结的 VAE/TE 用 DP；所有 DiT 层 gradient checkpointing；torch.compile；序列长度感知组 batch + 动态 batch size [PAPER §4.2]。
- 本机适配（[IMPLEMENTATION]）：单卡 5070 Ti 16GB 上 FSDP2 无收益 → 用 grad ckpt + 8bit 优化器/CPU offload 等替代组合；torch.compile 保留。
- 显存预算：模型梯子各级（100M→3B）的权重/梯度/优化器状态/激活显存估算与实际测量对照。
- 不包含：训练目标与课程设计（`training`）、数据管线（`data-infrastructure`）。

## Inputs
- paper-facts.md §4（训练系统事实）
- 目标模型级配置与实测显存数据
- 现有训练脚本中的显存优化代码（先检查再动手）

## Outputs
- 显存/吞吐测量脚本 + 预算表（估算 vs 实测）
- 训练启动配置（dtype、grad ckpt、optimizer offload 等的组合决策记录）
- 每级模型的可行配置结论（如"M3 1B 需要 X 组合"）

## Rules
1. 改代码前先检查现有实现；**不要在单卡上机械复刻 FSDP2**——先论证收益再引入。
2. 所有优化必须与 baseline 做数值一致性检查（同种子 loss 曲线一致），防止优化破坏正确性。
3. dtype 策略（bf16 训练、混合精度细节）显式记录；对齐测试的容差前提由此决定。
4. 官方 8×H800 相关吞吐数字仅作参照，不设为本机目标。
5. 每级模型的实际显存峰值必须实测并记录（供下一级规划用）。

## Verification
- 数值一致性：开启/关闭 grad ckpt、不同 dtype 下，同种子训练 N 步 loss 一致。
- 显存对照：预算表估算与实际峰值对比，误差过大要找出原因（如激活估算错误）。
- 吞吐基准：不同优化组合的 it/s 对照实验入记录。
- 稳定性：长时训练无 OOM/NaN（观察窗口按阶段出口条件定）。

## Learning
- 练习：解释"为什么冻结组件用 DP、大模型用 FSDP2"（内存占用差异决定分片策略 [PAPER §4.2]）。
- 练习：手算 1B 模型 Adam 状态 ≈ 参数量×多少字节，说明 16GB 为何不够。
- 记录：官方配置与本机配置的映射表（哪条照搬、哪条替代、为什么）。

## Common failure modes
- 单卡硬上 FSDP2 浪费显存与性能。
- 开启优化后不复核数值一致性，静默改变训练语义。
- 显存预算只算权重不算激活/优化器状态，规划失真。
- 把 torch.compile 的首次编译开销误判为性能回归。
