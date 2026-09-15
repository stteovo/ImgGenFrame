# 1B Z-Image 训练计划（M3 缩比梯子）

> 第 18 步。只规划 + 实测，不训练。关键参数不凭经验估算，无法确认处标 [UNKNOWN]。
> 依据：paper-facts.md §4.2（训练系统）、reproduction_matrix PRE-02、distributed-training skill。

## 1. Model Config

```yaml
hidden_size: 1792
depth: 21
num_heads: 14        # head_dim = 1792/14 = 128（守恒）
ffn_dim: 4778        # int(1792/3*8)
# RoPE/VAE/patch/refiner 与 100M/300M 完全一致（同架构，只改 scale）
```
- 参数量：**1,011,654,464（≈1.01B）**（实测）
- FLOPs（前向，256² 264 token）：**8.60e11**（实测）

## 2. Expected VRAM（显存预算表，RTX 5070 Ti 16GB）

| 组成 | 公式 | 字节 | 来源 |
| --- | --- | --- | --- |
| 权重 bf16 | 1.01B × 2 | 2.02 GB | 解析 |
| 梯度 bf16 | 1.01B × 2 | 2.02 GB | 解析 |
| 激活（256² batch1） | — | ≈1.8 GB | 实测（fwd+bwd 峰值 3.85GB 减梯度 2.02GB） |
| AdamW fp32（m+v+master） | 1.01B × 12 | 12.14 GB | 解析 |
| **合计（无优化）** | | **≈17.98 GB → OOM** | |
| 8bit AdamW（m/v 量化） | 1.01B × 6 | 6.06 GB | 解析 |
| **合计（8bit optim）** | | **≈11.9 GB ✓** | |
| grad checkpointing | 激活 → ≈0.3GB/层重算 | ≈10.4 GB ✓ | 估算 |

- **实测**：fwd 峰值 1.94GB；fwd+bwd 峰值 3.85GB（@256² batch1，不含 optimizer 状态）。
- 结论：1B **无法**用 fp32 AdamW 在 16GB 训练（18GB）；必须 **8bit 优化器 + grad ckpt**（≈10.4GB），
  或 CPU offload（GPU 5.8GB，慢）。

## 3. GPU Topology

- 单卡 RTX 5070 Ti 16GB（WSL2）。无多卡，FSDP2 分片无收益 [IMPLEMENTATION]。
- 官方 8×H800 数字仅参照，不作为本机目标。

## 4. FSDP2 Strategy

- 官方 [PAPER §4.2]：DiT 用 FSDP2，冻结 VAE/TE 用 DP。
- 本机：单卡 → FSDP2 `fully_shard` 可用（torch 2.14 已确认），但分片到单卡无收益；
  **不机械复刻 FSDP2**，改用 grad ckpt + 8bit optim / offload [IMPLEMENTATION]。
- 多卡可用时再引入 FSDP2（第 19 步）。

## 5. Activation Checkpointing

- 官方：所有 DiT 层 gradient checkpointing [PAPER §4.2]。
- 本机：对 21 个主块 + refiner 块开 `torch.utils.checkpoint`，激活 O(1) 显存、重算换时间。
- 需验证数值一致性（同种子 loss 曲线一致）。

## 6. BF16

- 权重/激活 bf16（autocast），主权重 fp32 [OFFICIAL-CODE 推荐 bf16]。
- 8bit optim 时 master 权重 fp32 保留，m/v 量化。

## 7. Optimizer

- 论文**未给**优化器/lr [UNKNOWN]（open_questions A1–A8）。
- 计划 [ASSUMPTION]：AdamW（沿用 100M/300M 的 lr=1e-4、wd=0.01，需消融）。
- 显存：8bit（bitsandbytes，**当前未安装**，需加依赖；网络时好时坏）或 CPU offload（accelerate）。

## 8. Batch Strategy

- 论文：序列长度感知组 batch + 动态 batch size [PAPER §4.2]。
- 本机：256² 单分辨率阶段用固定 batch（grad ckpt + 8bit 后 batch≈2–4）；多分辨率再引入组 batch（第 22 步）。

## 9. Checkpoint Strategy

- 复用 `FlowMatchingTrainer`：model/optimizer/step/全部 RNG/dataset_version/manifest_hash。
- 每 N 步保存 + 保留最佳 loss；续训一致性已有单测。

## 10. Estimated Compute

- 单步 FLOPs = 3 × 8.60e11 ≈ 2.58e12（fwd+bwd）。
- 训练总 FLOPs = 单步 × steps × batch —— **steps/data 规模 [UNKNOWN]**（论文未给 1B 级配方）。

## 11. Dataset Size

- [UNKNOWN] 论文未给各级数据规模；reproduction_matrix 附录 B 建议 M3 ≈ 10⁵–10⁶ 级（[ASSUMPTION]）。
- 真实数据（flickr30k/conceptual_captions）+ 真实 Qwen3-4B 文本特征需先接入。

## 12. Training Duration

- 实测吞吐：2.63 it/s（256² batch1，无优化）；grad ckpt 后预计 ~1.5–2 it/s。
- 总时长 = steps / it/s —— 依赖 [UNKNOWN] 的 steps/data，进入训练时定。

## 13. Failure Recovery

- OOM：降 batch / 开 grad ckpt / 8bit optim / CPU offload（逐级）。
- NaN：grad clip + bf16 数值检查 + 降 lr（沿用 100M 的 Debug Protocol）。
- checkpoint 续训 + 固定种子保证可复现。

## 官方策略核对 + 工具链

| 项 | 官方 [PAPER §4.2 / §1] | 本机 | 状态 |
| --- | --- | --- | --- |
| FSDP2 | DiT 用 FSDP2 | 单卡不引入 | ✅ 可用（fully_shard 已确认） |
| 冻结组件 DP | VAE/TE 用 DP | 单卡无分片 | ✅ |
| grad ckpt | 所有 DiT 层 | 计划开启 | 待实现 |
| torch.compile | 训练用 | 保留（首次编译开销非回归） | 待引入 |
| FlashAttention | §1 脚注：推理 FA3 | SDPA flash 后端可用；当前 attention 用手写 matmul，需切 SDPA | ⚠️ 待改 |
| TorchTitan | 未提及（Meta 参考实现） | 仅参考 | ℹ️ |

## UNKNOWN 汇总

优化器/lr/warmup/EMA/CFG dropout（A1–A8）、1B 级数据规模与训练 steps、是否训练用 FlashAttention、官方 batch 策略精确参数。
