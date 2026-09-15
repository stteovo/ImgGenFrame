"""patchify / unpatchify 测试：多分辨率、可逆、空间元数据。"""

import pytest
import torch

from zimage.tokenization.patchifier import patchify, unpatchify


@pytest.mark.parametrize("h,w", [(8, 8), (16, 16), (16, 24), (24, 16), (32, 32), (10, 10)])
def test_patchify_unpatchify_roundtrip(h, w):
    torch.manual_seed(0)
    latent = torch.randn(2, 16, h, w)
    tokens, grid = patchify(latent, patch_size=2)
    assert grid == (h // 2, w // 2)
    assert tokens.shape == (2, (h // 2) * (w // 2), 16 * 4)
    restored = unpatchify(tokens, grid, patch_size=2)
    assert torch.equal(restored, latent)


def test_patchify_shapes_256():
    # 256×256 图像 → 32×32 latent → 16×16 tokens（每 token 64 维）
    latent = torch.randn(1, 16, 32, 32)
    tokens, grid = patchify(latent)
    assert grid == (16, 16)
    assert tokens.shape == (1, 256, 64)


def test_patchify_shapes_512():
    # 512×512 图像 → 64×64 latent → 32×32 tokens
    latent = torch.randn(1, 16, 64, 64)
    tokens, grid = patchify(latent)
    assert grid == (32, 32)
    assert tokens.shape == (1, 1024, 64)


@pytest.mark.parametrize("h,w", [(32, 48), (48, 32)])
def test_patchify_non_square(h, w):
    # 512×768 / 768×512 对应 latent 64×96 / 96×64
    latent = torch.randn(1, 16, h, w)
    tokens, grid = patchify(latent)
    assert tokens.shape == (1, (h // 2) * (w // 2), 64)


def test_patchify_preserves_spatial_metadata():
    # 空间顺序：w 最快；像素 (h=0, w=2) 落在 patch (ht=0, wt=1) → token 索引 1
    latent = torch.zeros(1, 16, 4, 4)
    latent[:, :, 0, 2] = 1.0  # h=0, w=2
    tokens, grid = patchify(latent, patch_size=2)
    # patch grid (2,2)；索引 1 对应 (ht=0, wt=1) 的 patch
    assert tokens[0, 1].sum() > 0
    assert tokens[0, 0].sum() == 0


def test_unpatchify_invalid_token_count_raises():
    tokens = torch.randn(1, 5, 64)
    with pytest.raises(ValueError):
        unpatchify(tokens, (2, 2), patch_size=2)


def test_patchify_non_divisible_raises():
    latent = torch.randn(1, 16, 7, 8)
    with pytest.raises(ValueError):
        patchify(latent, patch_size=2)
