"""latent 工具：Flux VAE 缩放常量、scale/unscale、统计探针。

Flux VAE 约定 [OFFICIAL-CKPT]（HF vae/config.json）：
  latent_channels=16、scaling_factor=0.3611、shift_factor=0.1159、8× 下采样。
与官方 pipeline 一致 [OFFICIAL-CODE]：
  encode: scaled = (raw - shift_factor) * scaling_factor
  decode: raw = scaled / scaling_factor + shift_factor
注意：官方 repo 遗留 4ch / 0.18215 为 SD 风格死代码，严禁使用（paper-facts §6 #2）。
"""

from __future__ import annotations

import torch

# [OFFICIAL-CKPT] Flux VAE 常量
FLUX_LATENT_CHANNELS = 16
FLUX_SCALING_FACTOR = 0.3611
FLUX_SHIFT_FACTOR = 0.1159
FLUX_DOWNSAMPLE = 8

# 官方 repo 遗留死代码常量（SD 风格），仅作对照记录，禁止使用
LEGACY_SD_LATENT_CHANNELS = 4
LEGACY_SD_SCALING_FACTOR = 0.18215


def scale_latents(
    raw: torch.Tensor,
    scaling_factor: float = FLUX_SCALING_FACTOR,
    shift_factor: float = FLUX_SHIFT_FACTOR,
) -> torch.Tensor:
    """encode 侧缩放：(raw - shift) * scale。"""
    return (raw - shift_factor) * scaling_factor


def unscale_latents(
    scaled: torch.Tensor,
    scaling_factor: float = FLUX_SCALING_FACTOR,
    shift_factor: float = FLUX_SHIFT_FACTOR,
) -> torch.Tensor:
    """decode 侧反缩放：scaled / scale + shift。"""
    return scaled / scaling_factor + shift_factor


def latent_statistics(latent: torch.Tensor) -> dict:
    """latent 统计探针：总体与逐通道均值/方差/极值。

    用于记录 latent 分布（作为 shift 设计与初始化判断的依据）。
    """
    x = latent.detach().float()
    return {
        "shape": tuple(x.shape),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "channel_mean": x.mean(dim=(0, 2, 3)).tolist(),
        "channel_std": x.std(dim=(0, 2, 3)).tolist(),
    }
