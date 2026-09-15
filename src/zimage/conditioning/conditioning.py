"""Conditioning 抽象：Encoder / ConditioningOutput / ConditioningAdapter / UnifiedTokenSequence。

单流拼接顺序（T2I 基础模式）：[x(noisy image), cap(text)] [OFFICIAL-CODE]。
位置分配（3D RoPE，[OFFICIAL-CODE]）：
  - 文本 token：时间轴 t = 1..T_txt，h=w=0
  - 图像 token：空间轴 (h, w) 展开，t = T_txt + 1（时间维偏移到文本之后）
注意：序列顺序（image 在前）与位置时间顺序（text 在前）不一致，position_ids
按「位置语义」分配后，再按「拼接顺序 [image, text]」拼接。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..models.rope3d import image_token_positions, text_token_positions


@dataclass
class EncodedText:
    """文本编码器输出（[IMPLEMENTATION]，Qwen3 特征在第 9 步接入）。"""

    features: torch.Tensor  # [B, T_txt, cap_feat_dim]
    mask: torch.Tensor  # [B, T_txt] bool（True=有效）


@dataclass
class EncodedImage:
    """图像编码器输出（Flux VAE scaled latent）。"""

    latent: torch.Tensor  # [B, C, h, w]
    mask: Optional[torch.Tensor] = None  # 可选逐 token mask


@dataclass
class ConditioningOutput:
    """统一单流条件序列。

    - text_tokens / image_tokens：原始编码器特征（嵌入到模型维由 S3-DiT 完成）
    - modality_ids：1=image、0=text（按拼接顺序 [image, text]）
    - position_ids：[B, T_total, 3] 的 (t, h, w) 坐标
    """

    text_tokens: Optional[torch.Tensor] = None  # [B, T_txt, cap_feat_dim]
    text_mask: Optional[torch.Tensor] = None  # [B, T_txt] bool
    image_tokens: Optional[torch.Tensor] = None  # [B, C, h, w] scaled latent
    image_mask: Optional[torch.Tensor] = None  # [B, N_img] bool
    modality_ids: Optional[torch.Tensor] = None  # [B, T_total] long
    position_ids: Optional[torch.Tensor] = None  # [B, T_total, 3] long
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_text_tokens(self) -> int:
        return 0 if self.text_tokens is None else self.text_tokens.shape[1]

    @property
    def num_image_tokens(self) -> int:
        if self.image_tokens is None:
            return 0
        grid = self.metadata.get("image_grid")
        return grid[0] * grid[1] if grid is not None else 0

    @property
    def total_tokens(self) -> int:
        return self.num_image_tokens + self.num_text_tokens

    def has_text(self) -> bool:
        return self.text_tokens is not None

    def has_image(self) -> bool:
        return self.image_tokens is not None

    def is_empty(self) -> bool:
        return not self.has_text() and not self.has_image()


class TextEncoder(nn.Module):
    """文本编码器协议（Qwen3-4B 冻结，第 9 步接入；此处不写死实现）。"""

    def forward(self, *args, **kwargs) -> EncodedText:
        raise NotImplementedError


class ImageEncoder(nn.Module):
    """图像编码器协议（Flux VAE）。"""

    def forward(self, *args, **kwargs) -> EncodedImage:
        raise NotImplementedError


class ConditioningAdapter(nn.Module):
    """把编码器输出组织成统一单流条件（token / mask / modality / position）。"""

    def __init__(
        self,
        cap_feat_dim: int = 2560,
        latent_channels: int = 16,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        super().__init__()
        self.cap_feat_dim = cap_feat_dim
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size

    def forward(
        self,
        text: Optional[EncodedText] = None,
        image: Optional[EncodedImage] = None,
        semantic: Any = None,
    ) -> ConditioningOutput:
        if text is None and image is None:
            return ConditioningOutput(metadata={"empty": True})

        text_tokens = text.features if text is not None else None
        text_mask = text.mask if text is not None else None
        image_tokens = image.latent if image is not None else None

        t_txt = text_tokens.shape[1] if text_tokens is not None else 0

        # --- 位置 ID（按位置语义分配，再按拼接顺序 [image, text] 拼接） ---
        text_pos = text_token_positions(t_txt) if t_txt > 0 else None  # [T_txt, 3]
        image_pos = None
        image_grid = None
        image_mask = image.mask if image is not None else None
        if image_tokens is not None:
            _, _, h, w = image_tokens.shape
            p = self.patch_size
            H_t, W_t = h // p, w // p
            image_grid = (H_t, W_t)
            image_pos = image_token_positions(H_t, W_t, t_offset=t_txt + 1)  # [N_img, 3]
            if image_mask is None:
                image_mask = torch.ones(1, H_t * W_t, dtype=torch.bool, device=image_tokens.device)

        # --- modality_ids / position_ids（拼接顺序 [image, text]） ---
        modality_parts = []
        position_parts = []
        if image_pos is not None:
            n_img = image_pos.shape[0]
            modality_parts.append(torch.ones(n_img, dtype=torch.long, device=image_pos.device))
            position_parts.append(image_pos)
        if text_pos is not None:
            modality_parts.append(torch.zeros(t_txt, dtype=torch.long, device=text_pos.device))
            position_parts.append(text_pos)

        modality_ids = torch.cat(modality_parts, dim=0)[None]  # [1, T_total]
        position_ids = torch.cat(position_parts, dim=0)[None]  # [1, T_total, 3]

        metadata = {
            "image_grid": image_grid,
            "text_len": t_txt,
            "semantic": semantic is not None,
        }
        return ConditioningOutput(
            text_tokens=text_tokens,
            text_mask=text_mask,
            image_tokens=image_tokens,
            image_mask=image_mask,
            modality_ids=modality_ids,
            position_ids=position_ids,
            metadata=metadata,
        )
