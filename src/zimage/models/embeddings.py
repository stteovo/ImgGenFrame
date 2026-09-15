"""嵌入模块：时间嵌入、图像 patchify 嵌入、文本 caption 嵌入。

全部逐值对齐 diffusers `transformer_z_image.py`：
  - `TimestepEmbedder`：正弦嵌入（freq=256, max_period=10000）+ MLP(mid=1024) → min(dim,256)
  - `x_embedder`：Linear(f_patch*patch*patch*in_channels → dim, bias=True)
  - `cap_embedder`：RMSNorm(cap_feat_dim) + Linear(cap_feat_dim → dim, bias=True)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .normalization import RMSNorm


class TimestepEmbedder(nn.Module):
    """正弦时间嵌入 + 两层 MLP（SiLU 激活）。

    [OFFICIAL-CODE] diffusers TimestepEmbedder：
      mlp = Linear(256→mid_size) → SiLU → Linear(mid_size→out_size)
    输入 t 由调用方先乘 `t_scale=1000`。
    """

    def __init__(self, out_size: int, mid_size: int = 1024, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class PatchEmbed(nn.Module):
    """图像 latent patchify 并线性嵌入。

    latent [B, C, H, W] → tokens [B, N, dim]，其中 N = (H/p)*(W/p)，patch 维度 = f*p*p*C。
    [OFFICIAL-CODE] 等价于 `all_x_embedder = Linear(f_patch*patch*patch*in_channels → dim)`。
    """

    def __init__(self, in_channels: int, patch_size: int, f_patch_size: int, dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.proj = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)

    def patchify(self, latent: torch.Tensor):
        """latent [B, C, H, W] → [B, N, patch_dim]，返回 grid (H_t, W_t)。"""
        B, C, H, W = latent.shape
        p = self.patch_size
        H_t, W_t = H // p, W // p
        x = latent.view(B, C, H_t, p, W_t, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, H_t * W_t, C * p * p)
        return x, (H_t, W_t)

    def forward(self, latent: torch.Tensor):
        tokens, grid = self.patchify(latent)
        return self.proj(tokens), grid


class CaptionEmbedder(nn.Module):
    """冻结 Qwen3-4B 文本特征 → 主干维度。

    [OFFICIAL-CODE] `cap_embedder = RMSNorm(cap_feat_dim) + Linear(cap_feat_dim → dim, bias=True)`。
    """

    def __init__(self, cap_feat_dim: int, dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(cap_feat_dim, eps=norm_eps)
        self.proj = nn.Linear(cap_feat_dim, dim, bias=True)

    def forward(self, cap_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(cap_feats))
