"""图像 tokenization 管线：image ↔ Flux VAE latent ↔ image tokens。

数据流（每步 shape 显式）：
  image [B, 3, H, W]
    → VAE encode → raw latent [B, 16, H/8, W/8]
    → scale → scaled latent [B, 16, H/8, W/8]
    → patchify(2,2) → image tokens [B, (H/16)·(W/16), 64]
    → (S3-DiT)
    → unpatchify → scaled latent [B, 16, H/8, W/8]
    → unscale → raw latent [B, 16, H/8, W/8]
    → VAE decode → image [B, 3, H, W]
约束：H、W 必须被 16 整除（8× 下采样 × 2 patch）。VAE 冻结 [PAPER §4.2]。
"""

from __future__ import annotations

from typing import Tuple

import torch

from .latent_utils import (
    FLUX_DOWNSAMPLE,
    FLUX_LATENT_CHANNELS,
    FLUX_SCALING_FACTOR,
    FLUX_SHIFT_FACTOR,
    scale_latents,
    unscale_latents,
)
from .patchifier import patchify, unpatchify


class ImageTokenizer:
    def __init__(
        self,
        vae=None,
        scaling_factor: float = FLUX_SCALING_FACTOR,
        shift_factor: float = FLUX_SHIFT_FACTOR,
        patch_size: int = 2,
        latent_channels: int = FLUX_LATENT_CHANNELS,
        downsample: int = FLUX_DOWNSAMPLE,
    ):
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.downsample = downsample
        # H、W 必须被 downsample * patch_size 整除
        self.alignment = downsample * patch_size

    # ------------------------------------------------------------------
    # 分辨率
    # ------------------------------------------------------------------
    def check_resolution(self, h: int, w: int) -> None:
        if h % self.alignment != 0 or w % self.alignment != 0:
            raise ValueError(
                f"图像高宽必须被 {self.alignment} 整除（{self.downsample}× 下采样 × "
                f"{self.patch_size} patch），实际 ({h}, {w})"
            )

    # ------------------------------------------------------------------
    # image ↔ latent
    # ------------------------------------------------------------------
    def encode(self, image: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """image [B,3,H,W] → scaled latent [B,16,H/8,W/8]（VAE 冻结，no_grad）。"""
        if self.vae is None:
            raise RuntimeError("未设置 VAE，仅支持 latent 级操作")
        self.check_resolution(image.shape[-2], image.shape[-1])
        with torch.no_grad():
            dist = self.vae.encode(image).latent_dist
            raw = dist.mode() if deterministic else dist.sample()
        return scale_latents(raw, self.scaling_factor, self.shift_factor)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """scaled latent [B,16,h,w] → image [B,3,h·8,w·8]。"""
        if self.vae is None:
            raise RuntimeError("未设置 VAE，仅支持 latent 级操作")
        raw = unscale_latents(latent, self.scaling_factor, self.shift_factor)
        with torch.no_grad():
            return self.vae.decode(raw).sample

    # ------------------------------------------------------------------
    # latent ↔ tokens
    # ------------------------------------------------------------------
    def tokenize(self, latent: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """scaled latent [B,C,h,w] → (tokens [B,N,C·p²], grid (h/p, w/p))。"""
        return patchify(latent, self.patch_size)

    def detokenize(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """tokens → scaled latent [B,C,h,w]。"""
        return unpatchify(tokens, grid, self.patch_size, self.latent_channels)

    # ------------------------------------------------------------------
    # 端到端
    # ------------------------------------------------------------------
    def image_to_tokens(self, image: torch.Tensor, deterministic: bool = True):
        latent = self.encode(image, deterministic=deterministic)
        return self.tokenize(latent)

    def tokens_to_image(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        latent = self.detokenize(tokens, grid)
        return self.decode(latent)
