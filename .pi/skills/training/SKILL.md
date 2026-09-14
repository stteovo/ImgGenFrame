---
name: training
description: Z-Image training loop implementation — flow matching pretraining curriculum (256² low-res then omni multi-task [PAPER §4.3]), SFT techniques (distribution narrowing, tagged resampling, model merging [PAPER §4.4]), seq-length-aware batching and dynamic batch sizing [PAPER §4.2]. Use when implementing the trainer, training loops, batching, checkpointing, or curriculum logic.
---

# Training（训练循环与课程）

## Purpose
把 Z-Image 的训练课程（low-res → omni → SFT）在缩比模型上跑通：训练循环、序列长度感知组 batch、动态 batch size、SFT 三技术（narrowing / balancing / merging）[PAPER §4.2–4.4]。

## Scope
- 训练循环：FM 目标（`flow-matching`）+ 模型（`s3-dit`）+ 冻结组件（Qwen3-4B/Flux VAE，DP 放置 [PAPER §4.2]）。
- 课程：256² 单阶段 low-res [PAPER §4.3]；omni 的任意分辨率映射（函数形式 [UNKNOWN] → 我们的设计标 [IMPLEMENTATION]）、T2I+I2I 联合、多级 caption 混用。
- SFT 三技术：高质量数据收窄分布、基于图+BM25 的动态重采样 [PAPER §4.4]、多变体线性权重合并 θ=Σαiθi [PAPER §4.4]。
- 系统：序列长度感知组 batch + 动态 batch size [PAPER §4.2]；checkpoint 与续训。
- 论文未给（→ 全部显式标注）：optimizer、lr 调度、batch size、EMA、warmup [UNKNOWN→ASSUMPTION/EXPERIMENT]。

## Inputs
- paper-facts.md §3/§4（目标与课程事实）、§7（UNKNOWN 超参清单）
- 各模块 skill 产出（flow-matching/s3-dit/conditioning/data-infrastructure）
- 现有 `src/zimage/training/` 实现与 `experiments/` 既往记录（先检查再动手）

## Outputs
- 训练器代码 + 测试（含续训/种子复现测试）
- 训练配置（冻结于 `configs/`）+ 完整实验记录（`experiment` skill 契约）
- 每个 [ASSUMPTION] 超参的 ablation 计划

## Rules
1. 改代码前先检查现有实现与既往实验记录，避免重复试验。
2. **单一变量**：超参逐个确定，禁止一次性改多个未确定超参。
3. 所有 [UNKNOWN] 超参的取值必须写 [ASSUMPTION] 并计划消融验证；结论标 [EXPERIMENT] 并附记录路径。
4. 种子/数据版本/环境三者必须可复现（`experiment` skill 契约）。
5. 组 batch 策略先做吞吐对照实验再作为默认开启。

## Verification
- 复现测试：固定种子两次训练 loss 曲线一致；checkpoint 续训与连续训练一致。
- 冒烟测试：训练一步 forward/backward 数值有限、无 NaN；100M 级数步收敛趋势正常。
- 吞吐对照：序列长度感知组 batch vs 朴素 batch 的 padding 率与 it/s 对照（入实验）。
- 课程出口：每阶段出口条件见 `docs/reproduction/roadmap.md`，未达标不进下一级。

## Learning
- 练习：解释"为什么序列长度感知组 batch 在多分辨率训练中是必须的" [PAPER §4.2]。
- 练习：model merging 的线性插值为什么能中和单变体偏差 [PAPER §4.4]。
- 记录：我们选定 lr/EMA 等的理由链（来自哪些文献惯例 + 我们的消融）。

## Common failure modes
- 直接抄 SD3/Flux 的 lr/EMA 而不标 [ASSUMPTION]，把惯例当论文事实。
- 动态 batch size 只实现"放大"不实现"缩小"，长序列 OOM。
- SFT 数据不均衡导致长尾灾难性遗忘（论文用 tagged resampling 解决 [PAPER §4.4]，遗漏该步 = 方法性错误）。
- 训练脚本跑通即认为"复现了 recipe"，忽略课程结构与数据配比。
