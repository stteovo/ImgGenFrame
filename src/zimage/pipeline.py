"""最小可运行 Z-Image 前向 pipeline。

数据流（T2I 基础模式，不训练）：
  text feats [B, T, cap_feat_dim]（冻结文本编码器，mock 或 Qwen3）
  latent [B, 16, H, W]（Flux VAE scaled latent，mock 或真实 VAE）
    → ConditioningAdapter → position_ids / modality_ids
    → RopeEmbedder → image_freqs / caption_freqs
    → S3-DiT → velocity latent [B, 16, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conditioning.conditioning import (
    ConditioningAdapter,
    ConditioningOutput,
    EncodedImage,
    EncodedText,
)
from .models.config import S3DiTConfig
from .models.rope3d import RopeEmbedder
from .models.s3dit import S3DiT


class ZImagePipeline(nn.Module):
    """组合 conditioning 抽象 + 3D RoPE + S3-DiT 的最小前向管线。"""

    def __init__(self, config: S3DiTConfig):
        super().__init__()
        self.config = config
        self.model = S3DiT(config)
        self.adapter = ConditioningAdapter(
            cap_feat_dim=config.cap_feat_dim,
            latent_channels=config.in_channels,
            patch_size=config.patch_size,
            f_patch_size=config.f_patch_size,
        )
        self.rope = RopeEmbedder(
            theta=config.rope_theta,
            axes_dims=config.rope_axes_dims,
            axes_lens=config.rope_axes_lens,
        )

    def build_conditioning(
        self,
        latent: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> ConditioningOutput:
        B = text_feats.shape[0]
        text = EncodedText(
            features=text_feats,
            mask=torch.ones(text_feats.shape[:2], dtype=torch.bool, device=text_feats.device),
        )
        image = EncodedImage(latent=latent)
        return self.adapter(text=text, image=image)

    def _freqs(self, cond: ConditioningOutput):
        """从 position_ids 计算 image / caption 频率（3D RoPE 接线）。"""
        pos = cond.position_ids  # [1, T_total, 3]
        n_img = cond.num_image_tokens
        image_freqs = self.rope.embed_batched(pos[:, :n_img]) if n_img > 0 else None
        caption_freqs = (
            self.rope.embed_batched(pos[:, n_img:]) if cond.num_text_tokens > 0 else None
        )
        return image_freqs, caption_freqs

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        """latent [B,16,H,W] + t [B] + text_feats [B,T,cap_feat_dim] → velocity [B,16,H,W]。"""
        cond = self.build_conditioning(latent, text_feats)
        image_freqs, caption_freqs = self._freqs(cond)
        return self.model(
            latent,
            t,
            text_feats,
            image_freqs_cis=image_freqs,
            caption_freqs_cis=caption_freqs,
        )
