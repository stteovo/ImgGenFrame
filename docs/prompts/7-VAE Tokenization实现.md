使用 vae-tokenization skill。

现在实现 Z-Image image tokenization pipeline。

目标：

image
→ Flux VAE
→ latent
→ patchify
→ image tokens
→ unpatchify
→ latent
→ VAE decode

创建：

src/zimage/tokenization/

├── image_tokenizer.py
├── patchifier.py
└── latent_utils.py

要求：

1. resolution independent
2. 支持 arbitrary H/W
3. 保持 spatial metadata
4. patchify/unpatchify 可逆
5. 明确 latent shape
6. 明确 token shape

首先写：

tests/tokenization/

测试：

- 256x256
- 512x512
- 512x768
- 768x512
- non-square
- patchify → unpatchify

使用真实 Flux VAE 前，先用 fake latent 做测试。

然后再做真实 VAE integration test。

记录：

docs/learning/03_vae_tokenization.md

解释：

image → latent → token

为什么 diffusion transformer 通常不直接在 RGB pixel space 工作。