"""多头自注意力（含 QK-Norm 与 RoPE 注入点）。

[OFFICIAL-CODE] diffusers ZSingleStreamAttnProcessor 的行为：
  1. to_q / to_k / to_v（bias=False），unflatten 出 heads
  2. QK-Norm（RMSNorm，对 head 维）
  3. RoPE（复数旋转，`freqs_cis` 注入）
  4. SDPA（scaled dot-product attention）
RoPE 频率计算与 3D 轴分配属第 5 步（`rope-3d`），此处仅保留注入点：
当 `freqs_cis is None` 时不施加位置编码（第 4 步默认路径）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """复数旋转（与官方 `apply_rotary_emb` 一致）。

    x: [B, T, H, D]；freqs_cis: [B, T, D//2] 复数。返回与 x 同 dtype。
    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, T, 1, D//2]
    out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return out.type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # QK-Norm：对每个 head 的 head_dim 维做 RMSNorm [PAPER §4.1]
        self.norm_q = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x).unflatten(-1, (H, D))
        k = self.to_k(x).unflatten(-1, (H, D))
        v = self.to_v(x).unflatten(-1, (H, D))

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = D ** -0.5
        attn = q @ k.transpose(-2, -1) * scale

        if attention_mask is not None:
            # attention_mask: [B, T] bool，True=参与；False 位置置 -inf
            mask = attention_mask[:, None, None, :]  # [B, 1, 1, T]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(1, 2).flatten(2, 3)  # [B, T, D]
        return self.to_out(out)
