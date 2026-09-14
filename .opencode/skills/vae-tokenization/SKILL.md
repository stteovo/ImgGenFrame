---
name: vae-tokenization
description: Handling of the Flux VAE for Z-Image latent tokenization — frozen Flux VAE [PAPER §4.1], 16 latent channels, 8x downsample, 2x2 patchify [OFFICIAL-CODE transformer in_channels=16, patch=(2,2)]; probing latent statistics and resolving the legacy 4-channel constants discrepancy (paper-facts §6). Use when working with VAE encode/decode, latent preprocessing, or patchify.
---

# VAE Tokenization（Flux VAE 潜空间处理）

## Purpose
正确接入冻结的 Flux VAE [PAPER §4.1]，把图像转成 16 通道潜变量并 patchify 成 S3-DiT 输入 token；澄清官方 config 中遗留常量的语义，保证预处理与官方 pipeline 一致。

## Scope
- Flux VAE 的加载/冻结/使用（diffusers `AutoencoderKL`）；latent 16 通道、空间下采样 8× [OFFICIAL-CODE]。
- patchify：(2,2) patch → token [OFFICIAL-CODE]；`SEQ_MULTI_OF=32` 填充约束。
- 潜变量统计探针（均值/方差/量级）与缩放因子的正确用法。
- 已知差异裁决：config 中存在 `DEFAULT_VAE_LATENT_CHANNELS=4、SCALING_FACTOR=0.18215` 等 SD 风格常量 [OFFICIAL-CODE]，与 Flux 16ch 不符 → 疑似遗留死代码，Stage 1 读官方 loader 确认（paper-facts §6 #2）。
- 不包含：VAE 训练（frozen，永不训练 [PAPER §4.2]）、RoPE（`rope-3d`）。

## Inputs
- paper-facts.md §2.3（VAE 事实）、§6 #2 差异
- 官方 `src/utils/loader.py`、diffusers `ZImagePipeline` 的 VAE 预处理路径
- 现有 `src/zimage/vae/` 实现（先检查再动手）

## Outputs
- VAE 包装模块 + 探针脚本 + pytest（含对齐测试）
- 潜变量统计报告（入 `experiments/`）
- 对"4ch 遗留常量"的裁决结论与证据

## Rules
1. 改代码前先检查现有实现。
2. VAE 必须冻结：训练中不传梯度、不加到 optimizer；VAE 用 DP 而非 FSDP [PAPER §4.2]。
3. 预处理（归一化范围、缩放因子、round/floor 选择）必须与官方 pipeline 逐值一致——以 diffusers 代码为准。
4. 不得复用 0.18215（SD 缩放）等常量，除非裁决证明官方在用（[OFFICIAL-CODE] 证据）。
5. 任意分辨率映射到潜空间的整除关系（8× 下采样 + 2× patch）要显式处理与记录。

## Verification
- 重建测试：encode→decode 的 PSNR/SSIM 达标；与官方 pipeline 对同一图像得到相同 latent（bf16 容差）。
- 统计探针：latent 通道均值/方差记录入实验，作为后续 shift 设计的依据。
- patchify 形状测试：H/8/2、W/8/2 整除性与 `SEQ_MULTI_OF=32` 填充。

## Learning
- 练习：解释"为什么 Z-Image 用 Flux VAE 而不是自己训 VAE"（成熟重建质量 + 省算力 [PAPER §4.1] 的解读，标注 [INFERENCE]）。
- 练习：16ch latent vs 4ch（SD 系）对 DiT 输入通道数的影响。
- 记录：latent 统计与 shift 常数的关系（BASE_IMAGE_SEQ_LEN=256 = 256² 图的 token 数推导）。

## Common failure modes
- 套用 SD 的 0.18215 缩放与 4 通道假设（遗留常量陷阱）。
- 预处理差一步（如未除以官方 scaling），latent 分布偏移导致训练发散。
- patchify 前忘记对齐到 16 的倍数分辨率，静默产生填充 token。
- 把 8× 下采样写成 32×（混淆 Flux/SD VAE 的压缩比）。
