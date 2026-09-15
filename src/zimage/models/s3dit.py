"""S3-DiT：单流 diffusion transformer 主干。

结构逐值对齐 diffusers `ZImageTransformer2DModel`（[OFFICIAL-CODE]）：
  - 每模态 lightweight processor：noise_refiner（调制）/ context_refiner（无调制），各 n_refiner_layers=2 blocks
  - 单流拼接顺序（T2I 基础模式）：[x(noisy image), cap(text)]
  - 主块：Sandwich-Norm（每子层前后各一个 RMSNorm）+ QK-Norm + 低秩 adaLN（scale/gate，gate 经 tanh）
  - FinalLayer：LayerNorm(无仿射) × (1+modulation) → Linear → 输出 velocity latent
第 4 步不实现训练与 RoPE 频率计算（RoPE 注入点保留，第 5 步接线）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import S3DiTConfig
from .embeddings import CaptionEmbedder, PatchEmbed, TimestepEmbedder
from .mlp import FeedForward
from .normalization import RMSNorm


class S3DiTBlock(nn.Module):
    """单流 transformer 块（Sandwich-Norm + adaLN 调制）。

    [OFFICIAL-CODE] ZImageTransformerBlock：
      - 4 个 RMSNorm：attention_norm1/2、ffn_norm1/2（Sandwich-Norm）
      - 调制路径：adaLN_modulation = Linear(min(dim,256) → 4*dim)，
        输出 (scale_msa, gate_msa, scale_mlp, gate_mlp)，
        gate = tanh(gate)，scale = 1 + scale
      - attn:  x = x + gate_msa * norm2(attn(norm1(x) * scale_msa))
      - ffn:   x = x + gate_mlp * norm2(ffn(norm1(x) * scale_mlp))
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        sandwich_norm: bool = True,
        adaln_embed_dim: int = 256,
        modulation: bool = True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.modulation = modulation
        self.sandwich_norm = sandwich_norm

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        # Sandwich-Norm：子层输出端归一化；关闭时退化为普通 pre-norm（[EXPERIMENT] 消融 D5）
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()

        self.attention = MultiHeadAttention(dim, num_heads, qk_norm=qk_norm, norm_eps=norm_eps)
        self.feed_forward = FeedForward(dim, ffn_dim)

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True)
            )
        else:
            self.adaLN_modulation = None

    def forward(
        self,
        x: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)  # [B, 4*dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attention_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attention_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    """输出层：归一化 × (1+modulation) → 线性投影。

    [OFFICIAL-CODE] FinalLayer：norm_final = LayerNorm(无仿射, eps=1e-6)；
    scale = 1 + (SiLU → Linear(256→dim))(c)。
    """

    def __init__(self, dim: int, out_dim: int, adaln_embed_dim: int = 256):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, adaln_embed_dim), dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(c).unsqueeze(1)
        x = self.norm_final(x) * scale
        return self.linear(x)


class S3DiT(nn.Module):
    """单流 S3-DiT（T2I 基础模式）。

    前向：
      latent [B, C, H, W] + t [B] + cap_feats [B, T_cap, cap_feat_dim]
        → patchify+embed → noise_refiner（调制）
        → cap_embed + context_refiner（无调制）
        → 单流拼接 [x, cap] → 主块堆叠 → FinalLayer → unpatchify
        → velocity latent [B, C, H, W]
    """

    def __init__(self, config: S3DiTConfig):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.n_refiner_layers = config.n_refiner_layers
        self.patch_size = config.patch_size
        self.f_patch_size = config.f_patch_size
        self.out_channels = config.out_channels
        self.qk_norm = config.qk_norm
        self.sandwich_norm = config.sandwich_norm
        self.rope_axes_dims = config.rope_axes_dims
        self.rope_theta = config.rope_theta

        self.x_embedder = PatchEmbed(
            config.in_channels, config.patch_size, config.f_patch_size, config.hidden_size
        )
        self.t_embedder = TimestepEmbedder(
            config.adaln_in_dim,
            mid_size=config.timestep_embed_mid_size,
            frequency_embedding_size=config.frequency_embedding_size,
        )
        self.cap_embedder = CaptionEmbedder(config.cap_feat_dim, config.hidden_size, config.norm_eps)

        # 每模态 lightweight processor [PAPER §4.1]；noise_refiner 带调制、context_refiner 无调制 [OFFICIAL-CODE]
        self.noise_refiner = nn.ModuleList(
            [
                S3DiTBlock(
                    1000 + i,
                    config.hidden_size,
                    config.num_heads,
                    config.ffn_dim,
                    norm_eps=config.norm_eps,
                    qk_norm=config.qk_norm,
                    sandwich_norm=config.sandwich_norm,
                    adaln_embed_dim=config.adaln_embed_dim,
                    modulation=True,
                )
                for i in range(config.n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                S3DiTBlock(
                    i,
                    config.hidden_size,
                    config.num_heads,
                    config.ffn_dim,
                    norm_eps=config.norm_eps,
                    qk_norm=config.qk_norm,
                    sandwich_norm=config.sandwich_norm,
                    adaln_embed_dim=config.adaln_embed_dim,
                    modulation=False,
                )
                for i in range(config.n_refiner_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                S3DiTBlock(
                    i,
                    config.hidden_size,
                    config.num_heads,
                    config.ffn_dim,
                    norm_eps=config.norm_eps,
                    qk_norm=config.qk_norm,
                    sandwich_norm=config.sandwich_norm,
                    adaln_embed_dim=config.adaln_embed_dim,
                    modulation=True,
                )
                for i in range(config.depth)
            ]
        )

        # 输出维度 = f_patch * patch * patch * out_channels
        out_dim = config.f_patch_size * config.patch_size * config.patch_size * config.out_channels
        self.final_layer = FinalLayer(config.hidden_size, out_dim, config.adaln_embed_dim)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def unified_modality_ids(num_image_tokens: int, num_text_tokens: int) -> torch.Tensor:
        """单流序列的 token 边界信息：1=image，0=text。

        [IMPLEMENTATION] 用于 unpatchify 切片与第 5 步 RoPE 轴分配，
        是"保留 text/image token boundary"的显式载体。
        """
        return torch.cat(
            [
                torch.ones(num_image_tokens, dtype=torch.long),
                torch.zeros(num_text_tokens, dtype=torch.long),
            ]
        )

    def unpatchify(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """tokens [B, N, out_dim] → velocity latent [B, out_channels, H, W]。

        [OFFICIAL-CODE] unpatchify 的 "f h w pf ph pw c -> c (f pf) (h ph) (w pw)" 单帧特化。
        """
        assert self.f_patch_size == 1, "第 4 步仅支持单帧（T2I）"
        B = tokens.shape[0]
        H_t, W_t = grid
        p = self.patch_size
        C = self.out_channels
        x = tokens.view(B, H_t, W_t, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_t * p, W_t * p)
        return x

    def summarize(self) -> str:
        return "\n".join(
            [
                f"S3DiT(dim={self.dim}, depth={self.depth}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim}, ffn_dim={self.config.ffn_dim})",
                f"  noise_refiner: {self.n_refiner_layers} blocks (modulated)",
                f"  context_refiner: {self.n_refiner_layers} blocks (unmodulated)",
                f"  main layers: {self.depth} blocks (modulated)",
                f"  qk_norm={self.qk_norm}, sandwich_norm={self.sandwich_norm}",
                f"  rope_theta={self.rope_theta}, rope_axes_dims={self.rope_axes_dims}",
                f"  patch_size={self.patch_size}, in_channels={self.config.in_channels}, "
                f"out_channels={self.out_channels}",
                f"  total parameters: {self.num_parameters():,}",
            ]
        )

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        image_freqs_cis: torch.Tensor | None = None,
        caption_freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # t ∈ [0,1] → × t_scale 进正弦嵌入 [OFFICIAL-CODE]
        adaln_input = self.t_embedder(t * self.config.t_scale)

        # --- 图像分支：patchify + embed + noise_refiner（带调制） ---
        x_tokens, grid = self.x_embedder(latent)  # [B, N_img, dim]
        for layer in self.noise_refiner:
            x_tokens = layer(
                x_tokens,
                adaln_input=adaln_input,
                attention_mask=attention_mask,
                freqs_cis=image_freqs_cis,
            )

        # --- 文本分支：cap_embed + context_refiner（无调制） ---
        cap_tokens = self.cap_embedder(cap_feats)  # [B, T_cap, dim]
        for layer in self.context_refiner:
            cap_tokens = layer(cap_tokens, attention_mask=None, freqs_cis=caption_freqs_cis)

        # --- 单流拼接：[x, cap]（T2I 基础模式 [OFFICIAL-CODE]） ---
        unified = torch.cat([x_tokens, cap_tokens], dim=1)

        unified_freqs = None
        if image_freqs_cis is not None and caption_freqs_cis is not None:
            unified_freqs = torch.cat([image_freqs_cis, caption_freqs_cis], dim=1)
        unified_mask = None  # 主块默认全注意力；可变长 batch 的 padding mask 属训练阶段

        for layer in self.layers:
            unified = layer(
                unified,
                adaln_input=adaln_input,
                attention_mask=unified_mask,
                freqs_cis=unified_freqs,
            )

        # --- FinalLayer → velocity tokens → unpatchify（仅取图像 token 前缀） ---
        out = self.final_layer(unified, adaln_input)  # [B, T_total, out_dim]
        n_img = grid[0] * grid[1]
        velocity = self.unpatchify(out[:, :n_img], grid)
        return velocity
