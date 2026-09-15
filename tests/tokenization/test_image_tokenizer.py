"""ImageTokenizer 测试（fake latent，不加载真实 VAE）。"""

import pytest
import torch

from zimage.tokenization.image_tokenizer import ImageTokenizer


@pytest.fixture
def tok():
    return ImageTokenizer(vae=None)  # latent 级操作，无需真实 VAE


def test_tokenize_detokenize_roundtrip(tok):
    torch.manual_seed(0)
    latent = torch.randn(2, 16, 32, 32)
    tokens, grid = tok.tokenize(latent)
    assert tokens.shape == (2, 256, 64)
    assert grid == (16, 16)
    restored = tok.detokenize(tokens, grid)
    assert torch.equal(restored, latent)


@pytest.mark.parametrize("h,w", [(16, 16), (32, 32), (32, 48), (48, 32)])
def test_tokenize_arbitrary_resolution(tok, h, w):
    latent = torch.randn(1, 16, h, w)
    tokens, grid = tok.tokenize(latent)
    assert tokens.shape == (1, (h // 2) * (w // 2), 64)
    assert tok.detokenize(tokens, grid).shape == latent.shape


def test_resolution_check(tok):
    tok.check_resolution(256, 256)
    tok.check_resolution(512, 768)
    with pytest.raises(ValueError):
        tok.check_resolution(250, 250)  # 250 不能被 16 整除


def test_encode_decode_require_vae(tok):
    image = torch.randn(1, 3, 256, 256)
    with pytest.raises(RuntimeError):
        tok.encode(image)
    with pytest.raises(RuntimeError):
        tok.decode(torch.randn(1, 16, 32, 32))


def test_shape_contract():
    # image [B,3,H,W] → latent [B,16,H/8,W/8] → token [B,(H/16)(W/16),64]
    tok = ImageTokenizer(vae=None)
    h = w = 512
    latent_h = latent_w = h // tok.downsample
    n_tokens = (latent_h // tok.patch_size) * (latent_w // tok.patch_size)
    assert latent_h == 64 and latent_w == 64
    assert n_tokens == 1024
    latent = torch.randn(2, tok.latent_channels, latent_h, latent_w)
    tokens, grid = tok.tokenize(latent)
    assert tokens.shape == (2, n_tokens, 64)
    assert grid == (32, 32)
