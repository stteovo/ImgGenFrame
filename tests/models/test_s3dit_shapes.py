"""S3-DiT 形状 / batch / 序列长度 / 分辨率 / head 配置测试。"""

import pytest
import torch

from zimage.models import S3DiT, S3DiTConfig


def _model(cfg: S3DiTConfig = None) -> S3DiT:
    return S3DiT(cfg or S3DiTConfig.tiny())


def _inputs(cfg, batch=2, h=16, w=16, cap_len=8):
    latent = torch.randn(batch, cfg.in_channels, h, w)
    t = torch.rand(batch)
    cap = torch.randn(batch, cap_len, cfg.cap_feat_dim)
    return latent, t, cap


def test_forward_output_shape():
    cfg = S3DiTConfig.tiny()
    model = _model(cfg)
    latent, t, cap = _inputs(cfg, batch=2)
    out = model(latent, t, cap)
    assert out.shape == (2, cfg.out_channels, 16, 16)


@pytest.mark.parametrize("batch", [1, 3])
def test_batch_size(batch):
    cfg = S3DiTConfig.tiny()
    model = _model(cfg)
    latent, t, cap = _inputs(cfg, batch=batch)
    out = model(latent, t, cap)
    assert out.shape == (batch, cfg.out_channels, 16, 16)


@pytest.mark.parametrize("cap_len", [1, 5, 17])
def test_variable_caption_length(cap_len):
    cfg = S3DiTConfig.tiny()
    model = _model(cfg)
    latent, t, cap = _inputs(cfg, batch=2, cap_len=cap_len)
    out = model(latent, t, cap)
    assert out.shape == (2, cfg.out_channels, 16, 16)


@pytest.mark.parametrize("h,w", [(8, 8), (16, 32), (32, 16)])
def test_arbitrary_image_grid(h, w):
    cfg = S3DiTConfig.tiny()
    model = _model(cfg)
    latent, t, cap = _inputs(cfg, batch=2, h=h, w=w)
    out = model(latent, t, cap)
    assert out.shape == (2, cfg.out_channels, h, w)


def test_unified_modality_ids_boundary():
    # 单流序列边界信息：图像 token 在前（=1），文本 token 在后（=0）
    ids = S3DiT.unified_modality_ids(3, 2)
    assert ids.tolist() == [1, 1, 1, 0, 0]


def test_config_rejects_mismatched_axes_dims():
    with pytest.raises(ValueError):
        S3DiTConfig(hidden_size=768, num_heads=6, rope_axes_dims=(16, 16, 16))


def test_config_rejects_bad_ffn_dim():
    with pytest.raises(ValueError):
        S3DiTConfig(hidden_size=768, num_heads=6, ffn_dim=999)


def test_config_rejects_non_divisible_heads():
    with pytest.raises(ValueError):
        S3DiTConfig(hidden_size=768, num_heads=7, ffn_dim=2048)


@pytest.mark.parametrize("hidden,heads,ffn", [(768, 6, 2048), (1280, 10, 3413), (3840, 30, 10240)])
def test_head_dim_conservation(hidden, heads, ffn):
    # 缩比梯子保持 head_dim=128（head_dim == sum(axes_dims)）
    cfg = S3DiTConfig(hidden_size=hidden, depth=2, num_heads=heads, ffn_dim=ffn)
    assert cfg.head_dim == 128


def test_official_config():
    cfg = S3DiTConfig.official()
    assert cfg.head_dim == 128
    assert cfg.depth == 30
    assert cfg.num_heads == 30  # 论文 Table 2 的 32 已裁决为笔误
    assert cfg.ffn_dim == 10240
    assert cfg.hidden_size == 3840
