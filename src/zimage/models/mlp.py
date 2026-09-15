"""前馈网络（SwiGLU 门控 MLP）。

[OFFICIAL-CODE] diffusers FeedForward：`w1(dim→hidden)`, `w3(dim→hidden)`,
`silu(w1(x)) * w3(x)`, `w2(hidden→dim)`，hidden = int(dim/3*8)，全部 bias=False。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
