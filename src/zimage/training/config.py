"""训练配置（dataclass + YAML 加载）。

超参来源：论文未给出（open_questions A1–A8），取值标 [ASSUMPTION]（SD3/Flux 惯例），
后续以 [EXPERIMENT] 消融确定。配置一旦被实验引用即冻结（experiment-protocol 契约）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import yaml

from ..models.config import S3DiTConfig


@dataclass(frozen=True)
class TrainingSpec:
    # [ASSUMPTION] 论文未给优化器/lr/EMA 等超参（open_questions A1–A8）
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    batch_size: int = 4
    max_steps: int = 2000
    log_interval: int = 10
    save_interval: int = 500
    precision: str = "bf16"  # bf16 | fp32
    timestep_sampling: str = "logit_normal"  # logit_normal | uniform
    logit_normal_std: float = 1.0  # [ASSUMPTION] σ=1.0（open_questions A7）
    seed: int = 42
    dataset_version: str = "v001"  # 数据集版本（与 manifest 绑定，可追溯）


@dataclass(frozen=True)
class DataSpec:
    resolution: int = 256  # 图像分辨率（256 → latent 32×32 → 256 token）
    num_images: int = 16  # 第一实验：8~32 张过拟合
    text_len: int = 8
    cap_feat_dim: int = 2560
    latent_channels: int = 16
    latent_size: int = 32  # resolution // 8

    def __post_init__(self) -> None:
        if self.latent_size * 8 != self.resolution:
            raise ValueError(f"resolution={self.resolution} 需等于 latent_size*8={self.latent_size * 8}")


def _as_tuple(x: Any) -> Tuple:
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def load_model_config(path: str | Path) -> S3DiTConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return S3DiTConfig(
        hidden_size=raw["hidden_size"],
        depth=raw["depth"],
        num_heads=raw["num_heads"],
        ffn_dim=raw["ffn_dim"],
        qk_norm=raw.get("qk_norm", True),
        sandwich_norm=raw.get("sandwich_norm", True),
        norm_eps=raw.get("norm_eps", 1e-5),
        rope_theta=raw.get("rope_theta", 256.0),
        rope_axes_dims=_as_tuple(raw.get("rope_axes_dims", (32, 48, 48))),
        rope_axes_lens=_as_tuple(raw.get("rope_axes_lens", (1536, 512, 512))),
        in_channels=raw.get("in_channels", 16),
        patch_size=raw.get("patch_size", 2),
        cap_feat_dim=raw.get("cap_feat_dim", 2560),
        n_refiner_layers=raw.get("n_refiner_layers", 2),
    )


def load_training_config(path: str | Path) -> TrainingSpec:
    raw = yaml.safe_load(Path(path).read_text())
    return TrainingSpec(
        learning_rate=raw.get("learning_rate", 1.0e-4),
        weight_decay=raw.get("weight_decay", 0.01),
        betas=_as_tuple(raw.get("betas", (0.9, 0.999))),
        grad_clip=raw.get("grad_clip", 1.0),
        batch_size=raw.get("batch_size", 4),
        max_steps=raw.get("max_steps", 2000),
        log_interval=raw.get("log_interval", 10),
        save_interval=raw.get("save_interval", 500),
        precision=raw.get("precision", "bf16"),
        timestep_sampling=raw.get("timestep_sampling", "logit_normal"),
        logit_normal_std=raw.get("logit_normal_std", 1.0),
        seed=raw.get("seed", 42),
        dataset_version=raw.get("dataset_version", "v001"),
    )


def load_data_config(path: str | Path) -> DataSpec:
    raw = yaml.safe_load(Path(path).read_text())
    return DataSpec(
        resolution=raw.get("resolution", 256),
        num_images=raw.get("num_images", 16),
        text_len=raw.get("text_len", 8),
        cap_feat_dim=raw.get("cap_feat_dim", 2560),
        latent_channels=raw.get("latent_channels", 16),
        latent_size=raw.get("latent_size", raw.get("resolution", 256) // 8),
    )
