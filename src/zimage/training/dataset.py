"""训练数据：第一实验的合成 latent 数据集（确定性，用于过拟合验证）。

真实数据（flickr30k / conceptual_captions 等）在第 16 步 Data→Training 闭环接入。
此处的「图像」是 Flux VAE scaled latent，训练直接在 latent 空间做 FM（无需 VAE 往返）。
"""

from __future__ import annotations

import torch


class SyntheticLatentDataset:
    """N 张确定性 latent「图像」+ 固定文本特征。

    latent [N, C, S, S] ~ N(0,1)（固定种子）；text [N, T, cap_feat_dim] ~ N(0,1)（固定种子）。
    用于验证 100M 模型能否过拟合极小数据集。
    """

    def __init__(
        self,
        num_images: int,
        latent_size: int = 32,
        latent_channels: int = 16,
        text_len: int = 8,
        cap_feat_dim: int = 2560,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.num_images = num_images
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.text_len = text_len
        self.cap_feat_dim = cap_feat_dim
        self.device = device

        g = torch.Generator(device=device).manual_seed(seed)
        self.latents = torch.randn(
            num_images, latent_channels, latent_size, latent_size, generator=g, device=device
        )
        self.texts = torch.randn(num_images, text_len, cap_feat_dim, generator=g, device=device)

    def __len__(self) -> int:
        return self.num_images

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.num_images, (batch_size,))
        return self.latents[idx], self.texts[idx]
