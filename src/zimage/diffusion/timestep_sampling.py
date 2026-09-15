"""时间采样与动态时间偏移（dynamic time shifting）。

[PAPER §4.3] logit-normal t 采样（following SD3）；Flux 式动态 shift 补偿
多分辨率 SNR 差异。常数 [OFFICIAL-CODE]：
  BASE_IMAGE_SEQ_LEN=256, MAX_IMAGE_SEQ_LEN=4096, BASE_SHIFT=0.5, MAX_SHIFT=1.15。
注：训练用动态 shift；发布权重推理用固定 shift（base=6.0、Turbo=3.0）
[OFFICIAL-CKPT]，两条路径必须显式区分。
"""

from __future__ import annotations

import torch


def sample_uniform_t(
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """均匀采样 t ~ U(0, 1)。"""
    return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logit-normal 采样：z ~ N(mean, std)，t = sigmoid(z)，集中在中间 t。

    [ASSUMPTION] σ=1.0：论文只写 "following SD3" 未给 σ（paper-facts NS-01 / open_questions A7），
    按 SD3 惯例取 σ=1.0，后续消融。
    """
    z = torch.randn(batch_size, device=device, dtype=dtype, generator=generator) * std + mean
    return torch.sigmoid(z)


def calculate_shift(
    image_seq_len: int | float,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Flux 式分辨率相关的 log-shift mu（与 diffusers `calculate_shift` 一致）。

    mu = m · seq_len + b，m = (max_shift-base_shift)/(max_seq_len-base_seq_len)，
    b = base_shift - m·base_seq_len。返回 log 空间偏移，作用于 exp(mu)。
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def apply_time_shift(t: torch.Tensor, mu: float) -> torch.Tensor:
    """指数型时间偏移（与 diffusers `_time_shift_exponential` sigma=1 一致）。

    t_shifted = exp(mu) / (exp(mu) + (1/t - 1)) = exp(mu)·t / (1 + (exp(mu)-1)·t)。
    保持端点 t=0→0、t=1→1，把中间时间向某一端偏移。
    """
    mu_t = torch.as_tensor(mu, device=t.device, dtype=t.dtype)
    e = torch.exp(mu_t)
    return e / (e + 1.0 / t - 1.0)
