"""S3-DiT 模型配置。

所有结构参数集中于此，实现与测试共用同一份配置来源。
官方 6B 常数与缩比推导见 `docs/reproduction/paper-facts.md` §2.2 与
`docs/reproduction/reproduction_matrix.md` 附录 B。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# [OFFICIAL-CODE] diffusers transformer_z_image.py 常量
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
# [OFFICIAL-CODE] 官方 6B 常数（HF Tongyi-MAI/Z-Image transformer/config.json）
OFFICIAL_DIM = 3840
OFFICIAL_LAYERS = 30
OFFICIAL_HEADS = 30  # 论文 Table 2 的 32 已裁决为笔误 [OFFICIAL-CKPT]
OFFICIAL_FFN_DIM = 10240
OFFICIAL_ROPE_AXES_DIMS = (32, 48, 48)
OFFICIAL_ROPE_AXES_LENS = (1536, 512, 512)
OFFICIAL_ROPE_THETA = 256.0
OFFICIAL_CAP_FEAT_DIM = 2560
OFFICIAL_IN_CHANNELS = 16
OFFICIAL_PATCH_SIZE = 2
OFFICIAL_F_PATCH_SIZE = 1


@dataclass(frozen=True)
class S3DiTConfig:
    """S3-DiT 结构配置。

    缩比约束（[IMPLEMENTATION]，见 paper-facts §2.2 / reproduction_matrix 附录 B）：
      - head_dim = hidden_size / num_heads == sum(rope_axes_dims)
      - ffn_dim ≈ hidden_size * 8/3（官方 FeedForward hidden = int(dim/3*8)）
    """

    # --- backbone ---
    hidden_size: int = 768
    depth: int = 8
    num_heads: int = 6
    ffn_dim: int = 2048  # = int(hidden_size / 3 * 8)

    # --- normalization ---
    qk_norm: bool = True
    sandwich_norm: bool = True
    norm_eps: float = 1e-5

    # --- 3D unified RoPE（第 5 步独立实现，此处仅参数化） ---
    rope_theta: float = OFFICIAL_ROPE_THETA
    rope_axes_dims: Tuple[int, int, int] = OFFICIAL_ROPE_AXES_DIMS
    rope_axes_lens: Tuple[int, int, int] = OFFICIAL_ROPE_AXES_LENS

    # --- timestep embedding（[OFFICIAL-CODE]） ---
    frequency_embedding_size: int = 256
    timestep_embed_mid_size: int = 1024
    t_scale: float = 1000.0

    # --- modality embedding（[OFFICIAL-CODE] / [OFFICIAL-CKPT]） ---
    in_channels: int = OFFICIAL_IN_CHANNELS  # Flux VAE latent 通道
    patch_size: int = OFFICIAL_PATCH_SIZE
    f_patch_size: int = OFFICIAL_F_PATCH_SIZE
    cap_feat_dim: int = OFFICIAL_CAP_FEAT_DIM  # Qwen3-4B hidden

    # --- adaLN（[OFFICIAL-CODE]） ---
    adaln_embed_dim: int = ADALN_EMBED_DIM  # min(dim, ADALN_EMBED_DIM)

    # --- modality processors（[PAPER §4.1] 每模态 2 blocks；[OFFICIAL-CODE] n_refiner_layers=2） ---
    n_refiner_layers: int = 2

    # --- 输出 ---
    out_channels: int = field(default=OFFICIAL_IN_CHANNELS)  # velocity 预测与输入 latent 同通道

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.f_patch_size * self.patch_size * self.patch_size * self.in_channels

    @property
    def adaln_in_dim(self) -> int:
        return min(self.hidden_size, self.adaln_embed_dim)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        if self.head_dim != sum(self.rope_axes_dims):
            raise ValueError(
                f"head_dim ({self.head_dim}) 必须等于 sum(rope_axes_dims) "
                f"({sum(self.rope_axes_dims)} = {'+'.join(map(str, self.rope_axes_dims))})，"
                "与官方 `assert head_dim == sum(axes_dims)` 一致 [OFFICIAL-CODE]"
            )
        if self.ffn_dim != int(self.hidden_size / 3 * 8):
            raise ValueError(
                f"ffn_dim ({self.ffn_dim}) 应等于 int(hidden_size/3*8) = "
                f"{int(self.hidden_size / 3 * 8)}（官方 FeedForward hidden_dim [OFFICIAL-CODE]）"
            )

    @classmethod
    def tiny(cls) -> "S3DiTConfig":
        """用于单元测试的最小配置（head_dim=128、axes_dims 守恒）。"""
        return cls(hidden_size=768, depth=8, num_heads=6, ffn_dim=2048)

    @classmethod
    def official(cls) -> "S3DiTConfig":
        """官方 6B 配置（仅实例化/权重对齐，不训练 [PAPER Table 2]）。"""
        return cls(
            hidden_size=OFFICIAL_DIM,
            depth=OFFICIAL_LAYERS,
            num_heads=OFFICIAL_HEADS,
            ffn_dim=OFFICIAL_FFN_DIM,
        )
