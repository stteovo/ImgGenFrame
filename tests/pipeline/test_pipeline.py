"""最小 Z-Image pipeline 前向测试：forward / backward / BF16 / 变分辨率 / 变序列长度。"""

import pytest
import torch

from zimage.models import S3DiTConfig
from zimage.pipeline import ZImagePipeline


def _pipe():
    return ZImagePipeline(S3DiTConfig.tiny())


def test_forward_output_shape():
    pipe = _pipe()
    latent = torch.randn(2, 16, 16, 16)
    t = torch.rand(2)
    text = torch.randn(2, 8, 2560)
    out = pipe(latent, t, text)
    assert out.shape == latent.shape  # velocity 预测与输入 latent 同形


def test_backward_gradient_flows():
    pipe = _pipe()
    latent = torch.randn(2, 16, 16, 16, requires_grad=True)
    t = torch.rand(2)
    text = torch.randn(2, 8, 2560, requires_grad=True)
    out = pipe(latent, t, text)
    out.pow(2).mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert text.grad is not None and torch.isfinite(text.grad).all()


def test_bf16_forward():
    pipe = _pipe().to(torch.bfloat16)
    latent = torch.randn(2, 16, 16, 16, dtype=torch.bfloat16)
    t = torch.rand(2)
    text = torch.randn(2, 8, 2560, dtype=torch.bfloat16)
    out = pipe(latent, t, text)
    assert out.shape == latent.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("h,w", [(8, 8), (16, 16), (16, 32), (32, 16)])
def test_variable_resolution(h, w):
    pipe = _pipe()
    latent = torch.randn(1, 16, h, w)
    t = torch.rand(1)
    text = torch.randn(1, 8, 2560)
    out = pipe(latent, t, text)
    assert out.shape == (1, 16, h, w)


@pytest.mark.parametrize("seq_len", [1, 8, 32, 77])
def test_variable_sequence_length(seq_len):
    pipe = _pipe()
    latent = torch.randn(1, 16, 16, 16)
    t = torch.rand(1)
    text = torch.randn(1, seq_len, 2560)
    out = pipe(latent, t, text)
    assert out.shape == latent.shape


def test_freqs_are_position_dependent():
    # 同一 latent 在不同文本长度下，图像 token 的时间偏移不同（t = T+1）
    pipe = _pipe()
    latent = torch.randn(1, 16, 8, 8)
    t = torch.rand(1)
    out_a = pipe(latent, t, torch.randn(1, 4, 2560))
    out_b = pipe(latent, t, torch.randn(1, 9, 2560))
    assert not torch.allclose(out_a, out_b)  # 位置差异导致输出差异
