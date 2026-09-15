"""缩放配置测试：300M 配置与 100M 同架构，仅 scale 变化。"""

from zimage.models import S3DiTConfig


def test_300m_config_scale_only():
    c = S3DiTConfig(hidden_size=1024, depth=18, num_heads=8, ffn_dim=2730)
    assert c.head_dim == 128  # 与 100M 一致（head_dim 守恒）
    assert c.num_heads * c.head_dim == c.hidden_size
    # 只改 scale：RoPE / patch / VAE 常数不变
    assert c.rope_axes_dims == (32, 48, 48)
    assert c.patch_size == 2
    assert c.in_channels == 16


def test_300m_param_count_scale():
    from zimage.models import S3DiT

    small = S3DiT(S3DiTConfig(hidden_size=768, depth=8, num_heads=6, ffn_dim=2048)).num_parameters()
    large = S3DiT(S3DiTConfig(hidden_size=1024, depth=18, num_heads=8, ffn_dim=2730)).num_parameters()
    assert small < large
    assert 250_000_000 < large < 350_000_000  # ~300M
