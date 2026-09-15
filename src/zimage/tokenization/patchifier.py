"""patchify / unpatchify（图像 latent → token 的可逆变换）。

[OFFICIAL-CODE] patch=(2,2)、f_patch=1：latent [B, C, H, W] → tokens [B, H/2·W/2, C·4]。
分辨率无关、支持任意 H/W（只需被 patch 整除），保留空间元数据 (H_t, W_t)。
"""

from __future__ import annotations

from typing import Tuple

import torch


def patchify(latent: torch.Tensor, patch_size: int = 2) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """latent [B, C, H, W] → (tokens [B, N, C·p²], grid (H_t, W_t))。

    行主序展开（w 最快）；token 维度顺序为 [pH, pW, C]（与官方
    `_patchify_image` 的 `permute(1,3,5,2,4,6,0)` 一致，供权重对齐）。
    """
    B, C, H, W = latent.shape
    p = patch_size
    if H % p != 0 or W % p != 0:
        raise ValueError(f"latent 高宽必须被 patch_size={p} 整除，实际 {(H, W)}")
    H_t, W_t = H // p, W // p
    x = latent.view(B, C, H_t, p, W_t, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, H_t * W_t, C * p * p)
    return x, (H_t, W_t)


def unpatchify(
    tokens: torch.Tensor,
    grid: Tuple[int, int],
    patch_size: int = 2,
    out_channels: int | None = None,
) -> torch.Tensor:
    """tokens [B, N, C·p²]（token 维度 [pH, pW, C]）→ latent [B, C, H_t·p, W_t·p]，是 patchify 的逆变换。"""
    B, N, D = tokens.shape
    H_t, W_t = grid
    p = patch_size
    if N != H_t * W_t:
        raise ValueError(f"token 数 {N} 与 grid {grid} 不匹配（应 {H_t * W_t}）")
    C = out_channels if out_channels is not None else D // (p * p)
    if D != C * p * p:
        raise ValueError(f"token 维度 {D} 与 out_channels={C}、patch={p} 不匹配")
    x = tokens.view(B, H_t, W_t, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_t * p, W_t * p)
    return x
