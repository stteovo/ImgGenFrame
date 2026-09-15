"""Flow Matching 训练原语测试：插值 / 目标 / MSE / 广播 / dtype / BF16。"""

import torch

from zimage.diffusion.flow_matching import (
    flow_matching_loss,
    linear_interpolate,
    sample_flow_pair,
    velocity_target,
)


def test_interpolate_t0_equals_x0():
    x0 = torch.randn(2, 3, 4)
    x1 = torch.randn(2, 3, 4)
    assert torch.allclose(linear_interpolate(x0, x1, 0.0), x0)


def test_interpolate_t1_equals_x1():
    x0 = torch.randn(2, 3, 4)
    x1 = torch.randn(2, 3, 4)
    assert torch.allclose(linear_interpolate(x0, x1, 1.0), x1)


def test_interpolate_t05_is_midpoint():
    x0 = torch.randn(2, 3, 4)
    x1 = torch.randn(2, 3, 4)
    assert torch.allclose(linear_interpolate(x0, x1, 0.5), (x0 + x1) / 2)


def test_velocity_target():
    x0 = torch.randn(5)
    x1 = torch.randn(5)
    assert torch.allclose(velocity_target(x0, x1), x1 - x0)


def test_zero_noise_flow_pair():
    # x0 = 0 时：x_t = t·x1，v = x1
    x1 = torch.randn(2, 4)
    x_t = linear_interpolate(torch.zeros_like(x1), x1, 0.25)
    assert torch.allclose(x_t, 0.25 * x1)


def test_flow_matching_loss_mean():
    v_pred = torch.randn(4, 5)
    v_target = torch.randn(4, 5)
    expected = (v_pred - v_target).pow(2).mean()
    assert torch.allclose(flow_matching_loss(v_pred, v_target), expected)


def test_flow_matching_loss_zero_when_equal():
    v = torch.randn(3, 3)
    assert flow_matching_loss(v, v) == 0.0


def test_flow_matching_loss_reductions():
    v_pred = torch.randn(3, 4)
    v_target = torch.randn(3, 4)
    diff = (v_pred - v_target).pow(2)
    assert torch.allclose(flow_matching_loss(v_pred, v_target, "none"), diff)
    assert torch.allclose(flow_matching_loss(v_pred, v_target, "sum"), diff.sum())


def test_deterministic_seed():
    x1 = torch.randn(3, 8, 8)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    _, x_t1, _ = sample_flow_pair(x1, 0.3, generator=gen1)
    _, x_t2, _ = sample_flow_pair(x1, 0.3, generator=gen2)
    assert torch.equal(x_t1, x_t2)


def test_batch_broadcasting():
    x0 = torch.randn(4, 3, 8, 8)
    x1 = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)  # [B]
    x_t = linear_interpolate(x0, x1, t)
    assert x_t.shape == x0.shape
    # 逐样本校验
    for b in range(4):
        expected = (1 - t[b]) * x0[b] + t[b] * x1[b]
        assert torch.allclose(x_t[b], expected)


def test_dtype_preserved():
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        x0 = torch.randn(2, 4, dtype=dtype)
        x1 = torch.randn(2, 4, dtype=dtype)
        x_t = linear_interpolate(x0, x1, 0.5)
        assert x_t.dtype == dtype


def test_bf16_loss_finite():
    x0 = torch.randn(2, 4, dtype=torch.bfloat16)
    x1 = torch.randn(2, 4, dtype=torch.bfloat16)
    _, x_t, v = sample_flow_pair(x1, 0.5)
    v_pred = v + 0.01 * torch.randn_like(v)
    loss = flow_matching_loss(v_pred, v)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.bfloat16
    assert x_t.dtype == torch.bfloat16
