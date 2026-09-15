"""Flow Matching 训练原语。

方向约定（与论文一致 [PAPER §4.3]，勿与 SD3/Flux 常见写法混用）：
  x_t = (1 - t)·x0 + t·x1       x0 = 高斯噪声，x1 = 原图（数据）
  v_target = x1 - x0            （velocity / 速度场）
  L = ||v_pred - v_target||²    （MSE，损失权重恒 1 [IMPLEMENTATION]）
"""

from __future__ import annotations

import torch


def _broadcast_t(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    """把 t（标量或 [B, ...]）广播到与 x 相同维度，便于逐元素插值。"""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    else:
        t = t.to(device=x.device, dtype=x.dtype)
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


def linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """线性插值路径 x_t = (1 - t)·x0 + t·x1。"""
    t = _broadcast_t(t, x0)
    return (1.0 - t) * x0 + t * x1


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """速度场目标 v = x1 - x0。"""
    return x1 - x0


def sample_flow_pair(
    x1: torch.Tensor,
    t: torch.Tensor | float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从数据 x1 与时间 t 采样一个 flow pair。

    返回 (x0, x_t, v_target)，其中 x0 ~ N(0, I) 与 x1 同形状/同 dtype。
    """
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    x_t = linear_interpolate(x0, x1, t)
    v = velocity_target(x0, x1)
    return x0, x_t, v


def flow_matching_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE 损失 L = ||v_pred - v_target||²。reduction ∈ {mean, sum, none}。"""
    loss = (v_pred - v_target).pow(2)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"未知 reduction: {reduction}")
