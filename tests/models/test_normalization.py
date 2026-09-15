"""RMSNorm 数学性质与梯度测试。"""

import torch

from zimage.models.normalization import RMSNorm


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 3, 4, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_weight_gives_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64)
    rms = norm(x).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_scale_invariance():
    # RMSNorm(αx) == RMSNorm(x)：RMS 归一化对整体缩放不变
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64)
    assert torch.allclose(norm(x), norm(x * 3.7), atol=1e-4)


def test_rmsnorm_does_not_center():
    # 非零均值输入不改变符号结构（区别于 LayerNorm 的中心化）
    norm = RMSNorm(64)
    x = torch.ones(2, 64) * 5.0
    out = norm(x)
    # 权重为 1 时输出 rms=1，且全为正（未做减均值）
    assert torch.all(out > 0)


def test_rmsnorm_gradient():
    norm = RMSNorm(64)
    x = torch.randn(4, 64, requires_grad=True)
    out = norm(x)
    out.sum().backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
