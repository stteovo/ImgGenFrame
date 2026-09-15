"""Data Profiling Engine mini（规则启发式）。

[PAPER §2.1] 画像项：分辨率/文件大小、pHash、压缩伪影、质量模型、信息熵、
美学、AIGC 检测、VLM tag、CN-CLIP 对齐。mini 版只实现**可解释的规则启发式**：
分辨率、文件大小、aHash（pHash 纯 torch 替代）、压缩比、信息熵、边界方差；
自训模型项（美学/AIGC/VLM/CN-CLIP）后续接公开模型并标 [IMPLEMENTATION]。
所有阈值标 [IMPLEMENTATION]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 数值特征（边界用例可测：纯色 → 熵 0，纯噪声 → 高熵）
# ---------------------------------------------------------------------------
def compute_entropy(x: torch.Tensor) -> float:
    """灰度归一化直方图的信息熵（香农熵，bits）。

    x: 任意 shape，值域 [0, 1]（或 [0, 255] 由调用方归一化）。
    """
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return 0.0
    hist = torch.histc(x, bins=256, min=0.0, max=1.0)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def compute_compression_ratio(raw_bytes: int, compressed_bytes: int) -> float:
    """压缩伪影启发式：原始大小 / 压缩后大小。

    [PAPER §2.1] "理想未压缩大小/实际大小比值"；比值越低说明已高度压缩（伪影风险）。
    """
    if compressed_bytes <= 0:
        return float("inf")
    return raw_bytes / compressed_bytes


def boundary_variance(x: torch.Tensor) -> float:
    """边界像素方差（[PAPER §2.1] 信息熵相关项之一，检测边缘伪影）。"""
    x = x.detach().float()
    h, w = x.shape[-2], x.shape[-1]
    if h < 2 or w < 2:
        return 0.0
    border = torch.cat([x[..., 0, :], x[..., -1, :], x[..., :, 0], x[..., :, -1]])
    return float(border.var())


def average_hash(x: torch.Tensor, hash_size: int = 8) -> int:
    """aHash（感知哈希的纯 torch 替代，pHash 占位 [IMPLEMENTATION]）。

    x: [C, H, W] 图像（任意 C），→ 8×8 灰度均值阈值 → 64-bit 整数。
    """
    x = x.detach().float()
    if x.ndim == 3:
        x = x.mean(dim=0, keepdim=True)  # 灰度
    x = x.view(1, 1, *x.shape[-2:])
    small = F.interpolate(x, size=(hash_size, hash_size), mode="area")
    small = small.flatten()
    mean = small.mean()
    bits = (small > mean).long()
    h = 0
    for i, b in enumerate(bits.tolist()):
        h |= b << i
    return h


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


# ---------------------------------------------------------------------------
# 画像结果 + 过滤规则
# ---------------------------------------------------------------------------
@dataclass
class ProfilingResult:
    height: int = 0
    width: int = 0
    entropy: float = 0.0
    boundary_var: float = 0.0
    compression_ratio: float = float("inf")
    ahash: int = 0
    raw: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "height": self.height,
            "width": self.width,
            "entropy": self.entropy,
            "boundary_var": self.boundary_var,
            "compression_ratio": self.compression_ratio,
            "ahash": self.ahash,
        }


def profile_image(
    image: torch.Tensor,
    raw_bytes: int | None = None,
    compressed_bytes: int | None = None,
) -> ProfilingResult:
    """image: [C, H, W] 或 [H, W]，值域 [0,1]。"""
    h, w = image.shape[-2], image.shape[-1]
    gray = image.mean(dim=0) if image.ndim == 3 else image
    ratio = float("inf")
    if raw_bytes is not None and compressed_bytes is not None:
        ratio = compute_compression_ratio(raw_bytes, compressed_bytes)
    return ProfilingResult(
        height=h,
        width=w,
        entropy=compute_entropy(gray),
        boundary_var=boundary_variance(gray),
        compression_ratio=ratio,
        ahash=average_hash(image),
    )


# 过滤规则（阈值 [IMPLEMENTATION]，动机见 paper-facts §2.1）
def passes_min_resolution(h: int, w: int, min_res: int = 256) -> bool:
    return h >= min_res and w >= min_res


def passes_entropy_threshold(entropy: float, min_entropy: float = 1.0) -> bool:
    """低熵（纯色/近纯色）样本信息量不足。"""
    return entropy >= min_entropy


def passes_compression_threshold(ratio: float, min_ratio: float = 8.0) -> bool:
    """压缩比过低 → 已重度压缩（伪影风险）[IMPLEMENTATION] 阈值。"""
    return ratio >= min_ratio
