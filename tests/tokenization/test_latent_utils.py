"""latent 缩放 / 统计探针测试。"""

import torch

from zimage.tokenization.latent_utils import (
    FLUX_LATENT_CHANNELS,
    FLUX_SCALING_FACTOR,
    FLUX_SHIFT_FACTOR,
    latent_statistics,
    scale_latents,
    unscale_latents,
)


def test_constants_match_paper_facts():
    # [OFFICIAL-CKPT] Flux VAE：16ch、0.3611、0.1159（paper-facts §2.3 / §6 #2 裁决）
    assert FLUX_LATENT_CHANNELS == 16
    assert abs(FLUX_SCALING_FACTOR - 0.3611) < 1e-6
    assert abs(FLUX_SHIFT_FACTOR - 0.1159) < 1e-6


def test_scale_unscale_roundtrip():
    raw = torch.randn(2, 16, 32, 32)
    scaled = scale_latents(raw)
    restored = unscale_latents(scaled)
    assert torch.allclose(restored, raw, atol=1e-5)


def test_scale_formula():
    raw = torch.tensor([0.5])
    scaled = scale_latents(raw)
    assert torch.allclose(scaled, (raw - FLUX_SHIFT_FACTOR) * FLUX_SCALING_FACTOR)


def test_unscale_formula():
    scaled = torch.tensor([1.0])
    raw = unscale_latents(scaled)
    assert torch.allclose(raw, scaled / FLUX_SCALING_FACTOR + FLUX_SHIFT_FACTOR)


def test_latent_statistics_fields():
    latent = torch.randn(2, 16, 8, 8)
    stats = latent_statistics(latent)
    assert stats["shape"] == (2, 16, 8, 8)
    assert "mean" in stats and "std" in stats and "min" in stats and "max" in stats
    assert len(stats["channel_mean"]) == 16
    assert len(stats["channel_std"]) == 16


def test_latent_statistics_values():
    latent = torch.full((1, 16, 4, 4), 3.0)
    stats = latent_statistics(latent)
    assert stats["mean"] == 3.0
    assert stats["std"] == 0.0
    assert stats["min"] == 3.0 and stats["max"] == 3.0
