"""归一化模块。

Z-Image 所有归一化统一使用 RMSNorm（eps=1e-5 [OFFICIAL-CODE]），
例外是 FinalLayer 使用 `nn.LayerNorm(elementwise_affine=False, eps=1e-6)`
[OFFICIAL-CODE] transformer_z_image.py，保持逐值一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    与 diffusers `RMSNorm` 等价：`x / rms(x) * weight`，不做中心化。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 计算保证数值稳定，输出 cast 回输入 dtype
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).add_(self.eps))
        return (x * rms * self.weight.float()).to(dtype)
