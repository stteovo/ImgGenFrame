# Milestone 01 — S3-DiT 最小可运行 pipeline（第 4–9 步总结）

> 状态：完成。对应 `docs/prompts/` 第 4–9 步；commit 2bef47b → 本步。

## 1. 实现了什么

从零实现并串联了 Z-Image 的核心前向链路（T2I 基础模式，不训练）：

```
text feats [B,T,2560]（mock 文本编码器）   latent [B,16,H,W]（mock Flux VAE latent）
        │                                            │
        └────────────── ConditioningAdapter ──────────┘
                        position_ids / modality_ids
                                │
                         3D Unified RoPE
                                │
                     S3-DiT（单流 backbone）
                                │
                    velocity latent [B,16,H,W]
```

| 模块 | 文件 | 关键内容 |
| --- | --- | --- |
| S3-DiT | `src/zimage/models/` | 单流 backbone、Sandwich-Norm、QK-Norm、低秩 adaLN（scale/gate） |
| 3D RoPE | `src/zimage/models/rope3d.py` | 三轴复指数表 + 文本/图像位置生成，与 diffusers 逐值对齐 |
| Flow Matching | `src/zimage/diffusion/` | 插值/velocity/MSE、logit-normal t、动态 shift、Euler 采样器 |
| VAE Tokenization | `src/zimage/tokenization/` | Flux VAE 解耦、scale/unscale、patchify/unpatchify（[pH,pW,C] 序） |
| Conditioning | `src/zimage/conditioning/` | `ConditioningOutput`、Encoder 协议、ConditioningAdapter |
| Pipeline | `src/zimage/pipeline.py` | 组合以上模块的最小前向管线 |

- **测试**：140 项 pytest 通过（形状/数值/对齐/梯度/组合/BF16/变分辨率/变序列）。
- **冒烟**：`scripts/smoke_test.py` 输出参数量 95,661,120（≈95.7M，M1 缩比梯子）、activation shape、forward 20.5ms、peak VRAM 0.43GB（CUDA bf16）、backward 梯度有限。
- **数值对齐**：RoPE、timestep 嵌入、shift、patchify 与 diffusers [OFFICIAL-CODE] 逐值一致；真实 Flux VAE（Tongyi-MAI/Z-Image vae 子目录）集成验证 16ch/0.3611/0.1159。

## 2. 没实现什么

- **训练**：优化器 / 数据加载 / checkpoint / 训练循环（第 10 步）。
- **CFG**：推理引导与训练 dropout（第 10 步训练循环接入时实现）。
- **真实文本编码器**：Qwen3-4B 未接入（当前为 mock 特征，第 9 步后接入真实编码器属可选）。
- **编辑/Omni 模式**：SigLIP 语义条件、参考图/目标图双时间条件（编辑阶段）。
- **batch 内变长 padding**：当前 batch 内等长（padding+组 batch 属训练阶段，open_questions B10）。
- **性能优化**：FlashAttention / torch.compile / 梯度 checkpointing（Stage 后续）。

## 3. 与论文的对应关系

| 论文主张 [PAPER] | 本实现 | 一致性 |
| --- | --- | --- |
| 单流 MM-DiT（序列级拼接，无双流） | `S3DiT` 统一 token 流 | ✅ |
| 每模态 2 block processor | `noise_refiner`/`context_refiner` 各 2 blocks | ✅ |
| Sandwich-Norm + QK-Norm（RMSNorm） | 块内 4 norm + QK RMSNorm | ✅ |
| 低秩 adaLN（共享 down-proj + 每层 up-proj） | 共享 `t_embedder` + 每层 `Linear(256→4d)` | ✅ |
| 3D RoPE（图像空间轴、文本时间轴） | `rope3d.py` 三轴 | ✅（diffusers 对齐） |
| Flow Matching（v=x1−x0，MSE） | `diffusion/` | ✅ |
| logit-normal t + 动态 shift | `timestep_sampling.py` | ✅（σ=1.0 标 [ASSUMPTION]） |
| Flux VAE 冻结、16ch、8×、patch 2×2 | `tokenization/` | ✅（真实 VAE 集成） |

已知裁决差异（paper-facts §6）：heads=30（非论文 32）、Flux VAE 16ch（非遗留 4ch）、训练动态 shift vs 推理固定 shift。

## 4. 下一步

1. **第 10 步**：训练 100M Z-Image-Tiny（玩具数据 + Flow Matching loss + checkpoint），端到端首次训练。
2. 接入真实 Qwen3-4B 文本编码器（若算力允许）。
3. CFG 训练 dropout + 推理引导。
4. Stage 6：加载官方 6B 权重做逐值对齐（架构忠实复现的最终裁决）。
