# Learning Report — 里程碑一导师回顾（S3-DiT → 100M 训练）

> 对应 docs/prompts/ 第 4–10 步。不超过 1500 字，聚焦关键知识点。

## 1. What

从零实现并串联了 Z-Image 的完整前向+训练链路：S3-DiT 单流主干、3D Unified RoPE、Flow Matching 目标、Flux VAE tokenization、conditioning 抽象、100M 端到端过拟合训练（EXP-TRAIN-001）。共 140+ 项 pytest、diffusers 逐值对齐、真实 VAE 集成、1/10/100 步冒烟。

## 2. Why

Z-Image 的核心卖点是**单流参数效率**：文本/图像/语义 token 拼成一条序列进同一个 transformer，逐层密集跨模态交互 + 参数复用，比双流 MMDiT 更省参更高效。复现它的难点不在"能不能跑通"，而在每处细节与官方数值一致（否则 6B 权重无法对齐）。

## 3. Math

- **Flow Matching**：`x_t=(1−t)x0+t·x1`（x0=噪声、x1=数据）、`v=x1−x0`、`L=||v_pred−v||²`；t~logit-normal（σ=1）。
- **RoPE**：`freqs_m=θ^(−2m/d)`，`angle[i,m]=i·freqs_m`，复指数 `exp(j·angle)` 旋转 Q/K；三轴 (32,48,48) 和=128=head_dim。
- **低秩 adaLN**：共享 down-proj 到 256，每层 up-proj 到 4·d；`gate=tanh(·)`、`scale=1+·`，逐子层调制。
- **RMSNorm**：`x/rms(x)·γ`（不中心化）；SwiGLU：`silu(w1x)·w3x`。

## 4. Z-Image

- **单流**：每模态 2 个 lightweight block 先对齐，再统一主干；拼接顺序 T2I 为 `[x, cap]`。
- **3D RoPE**：图像沿空间 (h,w)、文本沿时间 t——1D 无法表达空间、2D 无法区分文本/编辑时间轴。
- **Flux VAE**：16ch、8×、scaling 0.3611；patch 2×2 → 256² 图 = 256 token（=BASE_IMAGE_SEQ_LEN）。
- **低秩 adaLN**：论文 §4.1 显式设计，非随意。

## 5. Implementation

`src/zimage/`：`models/`（s3dit/attention/mlp/normalization/embeddings/rope3d）、`diffusion/`（flow_matching/timestep_sampling/scheduler）、`tokenization/`、`conditioning/`、`training/`、`pipeline.py`。配置冻结于 `configs/tiny/`，实验记录于 `experiments/20260915-tiny-overfit/`。

## 6. Alternatives

双流 MMDiT（SD3/Flux）；1D/2D RoPE；DDPM ε-prediction；SD 系 4ch VAE（0.18215）；全秩 adaLN；把 Qwen 写死进主干。我们全部按 Z-Image 官方取向，其余仅作对照记录。

## 7. Trade-offs

单流序列更长（O((T_txt+T_img)²)），需序列长度感知组 batch 缓解（后步）；BF16 省显存但需 FP32 主权重；合成 latent 过拟合快但无真实统计结构；冻结 VAE 省算力但上限被重建质量封死。

## 8. Verification

RoPE/时间嵌入/shift/patchify 与 diffusers 逐值一致；真实 Flux VAE（Tongyi-MAI/Z-Image）16ch/0.3611/0.1159 确认、重建 PSNR>15dB；低秩 adaLN 共享/每层结构断言 + scale/gate 逐子层数值对齐；过拟合 loss 2.34→0.25（初值≈E[||v||²]=2 解析吻合）、16 步 Euler 重建 MSE=0.085；checkpoint 续训 + 同种子复现单测。

## 9. Failure Modes

heads 抄论文 32（实为 30）；patchify token 序 `[C,pH,pW]` vs 官方 `[pH,pW,C]`（已用 roundtrip 测试锁死）；RoPE 频率设备与模型不一致；BF16 eval 无 autocast 时 dtype 报错；套用 0.18215/4ch 遗留常量；x0/x1 方向与采样器不配套。

## 10. Questions

下一步思考：① 真实数据 + 真实 Qwen3-4B 文本特征接入后收敛行为如何？② CFG dropout / EMA 是否必要（论文未给，需消融）？③ 动态 shift 在多分辨率下是否显著？④ 从 100M 扩到 300M 时显存/吞吐如何规划？——这些分别对应第 12–22 步的数据基建、训练 recipe 与缩放。
