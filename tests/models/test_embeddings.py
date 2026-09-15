"""嵌入模块（时间嵌入 / patchify / caption 嵌入）测试。"""

import torch

from zimage.models.embeddings import CaptionEmbedder, PatchEmbed, TimestepEmbedder


def test_timestep_embedder_shape_and_finite():
    te = TimestepEmbedder(out_size=256, mid_size=1024, frequency_embedding_size=256)
    t = torch.rand(4) * 1000
    out = te(t)
    assert out.shape == (4, 256)
    assert torch.isfinite(out).all()


def test_timestep_embedding_t0_known_values():
    # t=0：args=0 → cos=1, sin=0；嵌入前一半全 1、后一半全 0
    emb = TimestepEmbedder.timestep_embedding(torch.zeros(1), 256)
    assert emb.shape == (1, 256)
    assert torch.allclose(emb[:, :128], torch.ones(1, 128), atol=1e-6)
    assert torch.allclose(emb[:, 128:], torch.zeros(1, 128), atol=1e-6)


def test_patch_embed_grid_and_shape():
    pe = PatchEmbed(in_channels=16, patch_size=2, f_patch_size=1, dim=128)
    latent = torch.randn(2, 16, 8, 8)
    tokens, grid = pe(latent)
    assert grid == (4, 4)
    assert tokens.shape == (2, 16, 128)


def test_patch_embed_arbitrary_grid():
    pe = PatchEmbed(in_channels=16, patch_size=2, f_patch_size=1, dim=128)
    latent = torch.randn(1, 16, 16, 32)
    tokens, grid = pe(latent)
    assert grid == (8, 16)
    assert tokens.shape == (1, 128, 128)


def test_caption_embedder_shape():
    ce = CaptionEmbedder(cap_feat_dim=2560, dim=128)
    cap = torch.randn(2, 5, 2560)
    assert ce(cap).shape == (2, 5, 128)
