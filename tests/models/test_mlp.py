"""SwiGLU 前馈网络测试。"""

import torch

from zimage.models.mlp import FeedForward


def test_feedforward_shape():
    ff = FeedForward(128, int(128 / 3 * 8))
    x = torch.randn(2, 7, 128)
    assert ff(x).shape == x.shape


def test_feedforward_gradient():
    ff = FeedForward(128, int(128 / 3 * 8))
    x = torch.randn(2, 7, 128, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
