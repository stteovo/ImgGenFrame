# Learning Report — VAE Tokenization（图像 → latent → token）

## 1. 做了什么

实现 `src/zimage/tokenization/`：`latent_utils`（Flux VAE 缩放常量与 scale/unscale + 统计探针）、`patchifier`（patchify/unpatchify 可逆变换）、`image_tokenizer`（image↔latent↔token 全链路）。fake latent 单测 + 真实 Flux VAE 集成测试（`Tongyi-MAI/Z-Image` vae 子目录）全部通过（116 项测试）。

修复了一个此前未暴露的 bug：patchify 的 token 维度顺序必须为官方 `[pH, pW, C]`（先 2×2 像素、后 16 通道），而非直觉的 `[C, pH, pW]`——前者是 Stage 6 权重对齐的前提，且保证 patchify/unpatchify 互为逆变换。

## 2. 为什么需要

扩散模型如果直接在 RGB 像素空间工作，每一步前向都要处理 3×H×W 的完整像素（256² 就是 196K 元素），自注意力更是 O((H·W)²)。这算力上不可行，且像素空间充满低信息量的高频冗余。

VAE 把图像压缩到低维潜空间：Flux VAE 把 3 通道、H×W 的像素压成 16 通道、H/8×W/8 的 latent（空间 8× 下采样，信息量压缩 3×64/16 = 12×）。DiT 在这个紧凑、语义化的 latent 上训练/推理，算力下降两个数量级，而重建质量几乎无损。

## 3. 数学/数据流

```
image [B, 3, H, W]                 ([-1, 1] 归一化)
  → VAE encode → raw latent [B, 16, H/8, W/8]
  → scale: scaled = (raw − shift_factor) · scaling_factor      （0.1159 / 0.3611）
  → patchify(2,2) → tokens [B, (H/16)(W/16), 64]              （64 = 2×2×16）
  → S3-DiT
  → unpatchify → scaled latent [B, 16, H/8, W/8]
  → unscale: raw = scaled / scaling_factor + shift_factor
  → VAE decode → image [B, 3, H, W]
```

- VAE 是一个自编码器：encoder 把图映射为 latent 高斯分布的参数，decoder 从 latent 重建图。Flux VAE 用 KL 正则 + patch 判别器训练（与 SD 系 VAE 同族）。
- `scaling_factor/shift_factor` 是训练 VAE 时对 latent 做的归一化，官方 pipeline 在喂给 DiT 前必须做同样的 scale，否则 latent 分布偏移会导致训练/推理不一致。
- patchify 把空间相邻的 2×2 latent 像素拼成一个 64 维 token，与 ViT 的 patch embedding 同思路，进一步把序列长度压到 H/16×W/16（256² → 256 token）。

## 4. 为什么 diffusion transformer 不直接在 RGB pixel space 工作

1. **算力**：注意力 O(L²)。256² 像素 L≈196K，不可行；latent 空间 L≈256（token），可行。
2. **信息冗余**：像素空间高频冗余大，模型浪费容量建模无关纹理；latent 已去除大部分冗余，模型专注语义。
3. **压缩即正则**：低维 latent 是强先验，生成更稳定。
4. **两阶段分工**：VAE 负责「像素↔语义压缩」（感知质量），DiT 负责「语义分布建模」（生成），各司其职、均可冻结/复用成熟组件。

## 5. Z-Image 中如何使用

- Flux VAE，**冻结**（[PAPER §4.2]：VAE 与 text encoder 都不训练），16 通道、8× 下采样 [OFFICIAL-CKPT]。
- patch=(2,2) [OFFICIAL-CODE]，token 维度顺序 `[pH, pW, C]`（`_patchify_image`）。
- scaling 0.3611 / shift 0.1159；官方 repo 遗留 4ch / 0.18215 为 SD 死代码，禁用（paper-facts §6 #2）。
- 256² 图 → 16×16=256 token，正是动态 shift 的 `BASE_IMAGE_SEQ_LEN=256` 的由来。

## 6. 本项目实现要点

- `ImageTokenizer` 与 VAE 解耦：`vae=None` 时只做 latent↔token（fake latent 测试）；真实 VAE 才做 image↔latent。
- 分辨率无关：任意 H/W（只需被 16 = 8×2 整除），空间元数据 `grid=(H_t, W_t)` 显式保留。
- encode 默认 `latent_dist.mode()`（确定性，供重建/对齐测试）；训练期如需重参数化采样可选 `deterministic=False` [IMPLEMENTATION]。
- 统计探针实测（渐变图）：latent mean≈−0.19、std≈1.47、范围≈[−5.2, 5.6]。

## 7. Trade-offs

- 8× 压缩：算力↓，但文字/细线等高频细节有损（需 DiT 后期重建，Z-Image 靠强 caption + SFT 补文字渲染）。
- 冻结 VAE：省算力、稳分布；但上限被 VAE 重建质量封死（可换更强 VAE，属 Track B）。
- patch 2×2：序列更短；但每个 token 信息更杂（4 像素×16 通道），需 DiT 足够容量。

## 8. 如何验证

- 形状：多分辨率（256/512/512×768/768×512/non-square）token 数正确。
- 可逆：patchify→unpatchify `torch.equal`；tokenize→detokenize 一致。
- 缩放：scale/unscale 闭式往返；真实 latent 往返一致。
- 集成：真实 VAE encode→decode 重建 PSNR>15 dB（渐变图）；VAE config 与 paper-facts 一致（16ch/0.3611/0.1159）。

## 9. Failure Modes

- 套用 SD 的 4ch / 0.18215（遗留常量陷阱）。
- 忘记 scale（喂原始 latent 给 DiT）导致分布偏移。
- patchify 前不对齐 16 的倍数分辨率，静默产生错位。
- token 维度顺序与官方不一致（[C,pH,pW] vs [pH,pW,C]），权重对齐必失败——已用 roundtrip 测试锁死。

## 10. 当前仍然未知的问题

- 训练期 VAE encode 用 mode 还是 sample（论文未说明；本实现默认 mode，训练数据预处理阶段再定）[ASSUMPTION]。
- 文本截断长度（与 Qwen3 用法，属 text encoder，第 8/9 步接入时确认）。
