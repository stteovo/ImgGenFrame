"""真实 Flux VAE 集成测试（需网络下载，失败则跳过）。

覆盖：encode→decode 重建 PSNR、tokenize→detokenize 可逆、
scale/unscale 与官方 pipeline 口径一致、latent 统计探针。
"""

import pytest
import torch

from zimage.tokenization.image_tokenizer import ImageTokenizer
from zimage.tokenization.latent_utils import latent_statistics, scale_latents, unscale_latents


@pytest.fixture(scope="module")
def flux_vae():
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKL

    try:
        vae = AutoencoderKL.from_pretrained(
            "Tongyi-MAI/Z-Image",
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.eval()
        return vae
    except Exception as e:  # 网络/下载/依赖问题
        pytest.skip(f"Flux VAE 加载失败: {e}")


def _gradient_image(batch=1, h=256, w=256):
    # [-1,1] 平滑渐变图（VAE 可良好重建），3 通道略有差异
    y = torch.linspace(-1, 1, h).view(1, h, 1)
    x = torch.linspace(-1, 1, w).view(1, 1, w)
    base = (y * 0.6 + x * 0.4).expand(batch, 1, h, w)
    return base.repeat(1, 3, 1, 1)


def test_vae_config_matches_paper_facts(flux_vae):
    # [OFFICIAL-CKPT] latent 16ch、scaling 0.3611、shift 0.1159
    cfg = flux_vae.config
    assert cfg.latent_channels == 16
    assert abs(cfg.scaling_factor - 0.3611) < 1e-6
    assert abs(cfg.shift_factor - 0.1159) < 1e-6


def test_encode_decode_reconstruction(flux_vae):
    tok = ImageTokenizer(vae=flux_vae)
    image = _gradient_image(1, 256, 256)
    latent = tok.encode(image, deterministic=True)
    assert latent.shape == (1, 16, 32, 32)  # 8× 下采样

    recon = tok.decode(latent)
    assert recon.shape == image.shape

    mse = (recon - image).pow(2).mean().item()
    psnr = -10 * torch.tensor(mse).log10().item()
    assert psnr > 15.0, f"重建 PSNR 过低：{psnr:.2f} dB"


def test_tokenize_detokenize_on_real_latent(flux_vae):
    tok = ImageTokenizer(vae=flux_vae)
    image = _gradient_image(1, 512, 512)
    latent = tok.encode(image)
    tokens, grid = tok.tokenize(latent)
    assert tokens.shape == (1, 32 * 32, 64)
    restored = tok.detokenize(tokens, grid)
    assert torch.allclose(restored, latent, atol=1e-5)


def test_scale_unscale_roundtrip_on_real_latent(flux_vae):
    tok = ImageTokenizer(vae=flux_vae)
    latent = tok.encode(_gradient_image(1, 256, 256))
    raw = unscale_latents(latent, tok.scaling_factor, tok.shift_factor)
    scaled_back = scale_latents(raw, tok.scaling_factor, tok.shift_factor)
    assert torch.allclose(scaled_back, latent, atol=1e-5)


def test_latent_statistics_probe(flux_vae):
    tok = ImageTokenizer(vae=flux_vae)
    latent = tok.encode(_gradient_image(2, 256, 256))
    stats = latent_statistics(latent)
    assert stats["shape"] == (2, 16, 32, 32)
    assert len(stats["channel_mean"]) == 16
    print("\n=== Flux VAE latent 统计 ===")
    print(f"  mean={stats['mean']:.4f} std={stats['std']:.4f} "
          f"min={stats['min']:.4f} max={stats['max']:.4f}")
