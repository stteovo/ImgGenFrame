"""多头注意力（QK-Norm / mask / RoPE 注入点）测试。"""

import torch

from zimage.models.attention import MultiHeadAttention


def test_attention_output_shape():
    attn = MultiHeadAttention(dim=128, num_heads=4, qk_norm=True)
    x = torch.randn(2, 10, 128)
    assert attn(x).shape == x.shape


def test_attention_gradient_flows():
    attn = MultiHeadAttention(dim=128, num_heads=4)
    x = torch.randn(2, 10, 128, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in attn.parameters():
        assert p.grad is not None


def test_attention_mask_applies():
    torch.manual_seed(0)
    attn = MultiHeadAttention(dim=64, num_heads=2)
    x = torch.randn(1, 4, 64)
    mask = torch.tensor([[True, True, False, False]])
    out = attn(x, attention_mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_attention_qk_norm_off():
    attn = MultiHeadAttention(dim=64, num_heads=2, qk_norm=False)
    x = torch.randn(1, 4, 64)
    assert attn(x).shape == x.shape


def test_attention_with_rotary_hook():
    attn = MultiHeadAttention(dim=64, num_heads=2)
    x = torch.randn(2, 6, 64, requires_grad=True)
    freqs = torch.ones(2, 6, 16, dtype=torch.complex64)  # head_dim//2 = 16
    out = attn(x, freqs_cis=freqs)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_head_dim_split():
    # dim/heads 必须整除
    attn = MultiHeadAttention(dim=256, num_heads=8)
    assert attn.head_dim == 32
