"""3D Unified RoPE（temporal / height / width 三轴位置编码）。

[PAPER §4.1] 单流序列的位置编码：图像 token 沿空间维 (h, w) 展开，
文本 token 沿时间维 t 递增；编辑任务参考图/目标图共享空间坐标、时间维错位。
[OFFICIAL-CODE] 常数：theta=256.0、axes_dims=(32,48,48)（和=128=head_dim）、
axes_lens=(1536,512,512)；预计算复指数表 `freqs_cis`，每轴独立。

频率表构造（与 diffusers `RopeEmbedder.precompute_freqs_cis` 逐值一致）：
  对每个轴 d ∈ axes_dims、长度 e ∈ axes_lens：
    freqs_m = 1 / (theta ** (2m / d))，m = 0..d/2-1
    angle[i, m] = i * freqs_m（i=0..e-1）
    freqs_cis[i, m] = exp(j * angle[i, m]) = polar(1, angle)
调用 `rope(ids)`（ids: [N, 3] 的 (t, h, w) 坐标）按轴索引拼接，
得 [N, head_dim/2] 复指数，供注意力旋转 Q/K。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def create_coordinate_grid(
    size: Sequence[int],
    start: Sequence[int] = (0, 0, 0),
    device: torch.device | None = None,
) -> torch.Tensor:
    """3D 坐标网格 (t, h, w)，与官方 `create_coordinate_grid` 一致。

    size=(Ft, Ht, Wt)，start=(t0, h0, w0) → 返回 [Ft, Ht, Wt, 3]。
    """
    axes = [
        torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
        for x0, span in zip(start, size)
    ]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1)


def text_token_positions(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """文本 token 位置：沿时间轴 t=1..seq_len，h=w=0。

    [OFFICIAL-CODE] 基础模式 caption 位置 `_pad_with_ids(..., (cap_len,1,1), (1,0,0))`。
    """
    return create_coordinate_grid((seq_len, 1, 1), (1, 0, 0), device).flatten(0, 2)


def image_token_positions(
    h_tokens: int, w_tokens: int, t_offset: int = 0, device: torch.device | None = None
) -> torch.Tensor:
    """图像 token 位置：沿空间轴 (h, w) 展开，t 恒定于 t_offset。

    [OFFICIAL-CODE] 基础模式 image 位置 `_pad_with_ids(..., (1,Ht,Wt), (t_offset,0,0))`，
    单帧 Ft=1 时 t 为常数。展开顺序与 patchify 的 (Ht, Wt) 行主序一致。
    """
    return create_coordinate_grid((1, h_tokens, w_tokens), (t_offset, 0, 0), device).flatten(0, 2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对 Q/K 施加复指数旋转（与官方 `apply_rotary_emb` 逐值一致）。

    x: [B, T, H, D]（D 为 head_dim，偶数）；freqs_cis: [B, T, D//2] 复数。
    返回与 x 同 dtype。
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, T, 1, D//2]
    out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return out.type_as(x)


class RopeEmbedder(nn.Module):
    """3D 统一 RoPE 频率表（惰性预计算 + 缓存，避免 Python 逐 token 循环）。

    ids: [N, 3] 的 (t, h, w) 整数坐标 → [N, head_dim/2] 复指数。
    """

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: Sequence[int] = (32, 48, 48),
        axes_lens: Sequence[int] = (1536, 512, 512),
    ):
        super().__init__()
        self.theta = float(theta)
        self.axes_dims = tuple(int(d) for d in axes_dims)
        self.axes_lens = tuple(int(e) for e in axes_lens)
        if len(self.axes_dims) != 3 or len(self.axes_lens) != 3:
            raise ValueError("3D RoPE 必须恰好 3 条轴 (t, h, w)")
        if len(self.axes_dims) != len(self.axes_lens):
            raise ValueError("axes_dims 与 axes_lens 长度必须一致")
        if any(d % 2 != 0 for d in self.axes_dims):
            raise ValueError("每个轴维度必须是偶数（复指数成对旋转）")
        self._freqs_cis: list[torch.Tensor] | None = None

    @property
    def head_dim(self) -> int:
        return sum(self.axes_dims)

    @staticmethod
    def precompute_freqs_cis(
        axes_dims: Sequence[int],
        axes_lens: Sequence[int],
        theta: float = 256.0,
    ) -> list[torch.Tensor]:
        """逐轴预计算复指数表，返回 [len_i, d_i//2] complex64 列表。

        float64 计算角度 → float32 角度 → complex64，与官方一致。
        """
        freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            positions = torch.arange(e, dtype=torch.float64)
            angles = torch.outer(positions, freqs).float()  # [e, d//2]
            freqs_cis.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
        return freqs_cis

    def _get_freqs_cis(self, device: torch.device) -> list[torch.Tensor]:
        if self._freqs_cis is None:
            self._freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        if self._freqs_cis[0].device != device:
            self._freqs_cis = [f.to(device) for f in self._freqs_cis]
        return self._freqs_cis

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [N, 3] int（(t, h, w)）→ [N, head_dim/2] 复数。"""
        if ids.ndim != 2 or ids.shape[-1] != 3:
            raise ValueError(f"ids 必须为 [N, 3]，实际 {tuple(ids.shape)}")
        freqs_cis = self._get_freqs_cis(ids.device)
        parts = [freqs_cis[i][ids[:, i].long()] for i in range(3)]
        return torch.cat(parts, dim=-1)

    def embed_batched(self, ids: torch.Tensor) -> torch.Tensor:
        """批处理位置 → 频率：[B, N, 3] → [B, N, head_dim/2] 复数。"""
        if ids.ndim != 3 or ids.shape[-1] != 3:
            raise ValueError(f"ids 必须为 [B, N, 3]，实际 {tuple(ids.shape)}")
        B, N, _ = ids.shape
        return self.forward(ids.reshape(B * N, 3)).view(B, N, -1)
