"""Caption Pipeline：raw image → quality filtering → VLM caption → 结构化抽取 →
置信度评分 → dataset record。支持重新 caption（版本字段追溯）。

不写死 VLM（传入 `VLM` 协议对象）。质量过滤复用第 12 步 profiling。
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import torch

from ..data.profiling import passes_entropy_threshold, passes_min_resolution, profile_image
from .schema import CaptionRecord, QualityAssessment, SpanGroundedItem
from .vlm import VLM, VLMCaptionOutput


def _to_items(raw_list: List[dict], source: str) -> List[SpanGroundedItem]:
    items = []
    for d in raw_list:
        items.append(
            SpanGroundedItem(
                text=d.get("text", ""),
                span=d.get("span"),
                confidence=float(d.get("confidence", 1.0)),
                observed=bool(d.get("observed", True)),
                source=d.get("source", source),
            )
        )
    return items


def _score_confidence(out: VLMCaptionOutput, all_items: List[SpanGroundedItem]) -> float:
    # 综合置信度：所有抽取项置信度的均值；observed 项权重高于 inferred [IMPLEMENTATION]
    if not all_items:
        return 0.0
    weighted = sum(it.confidence * (1.0 if it.observed else 0.5) for it in all_items)
    norm = sum(1.0 if it.observed else 0.5 for it in all_items)
    return weighted / norm


class CaptionPipeline:
    def __init__(
        self,
        vlm: VLM,
        prompt_version: str = "v1",
        taxonomy_version: str = "v1",
        dataset_version: str = "v1",
    ):
        self.vlm = vlm
        self.prompt_version = prompt_version
        self.taxonomy_version = taxonomy_version
        self.dataset_version = dataset_version

    @staticmethod
    def image_hash(image: torch.Tensor) -> str:
        return hashlib.sha256(image.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]

    def caption_image(
        self,
        image_id: str,
        image: torch.Tensor,
        raw_bytes: Optional[int] = None,
        compressed_bytes: Optional[int] = None,
    ) -> CaptionRecord:
        # 1. 质量过滤（profiling 规则）
        prof = profile_image(image, raw_bytes=raw_bytes, compressed_bytes=compressed_bytes)
        reasons: List[str] = []
        if not passes_min_resolution(prof.height, prof.width):
            reasons.append("resolution_too_small")
        if not passes_entropy_threshold(prof.entropy):
            reasons.append("low_entropy")
        quality = QualityAssessment(
            passed=len(reasons) == 0,
            entropy=prof.entropy,
            resolution=(prof.height, prof.width),
            reasons=reasons,
        )

        # 2. VLM caption（OCR-first 由 VLM 侧保证）
        out = self.vlm.caption(image)

        # 3. 结构化抽取（observed vs inferred 由 VLM 输出标记）
        objects = _to_items(out.objects, "visual")
        attributes = _to_items(out.attributes, "visual")
        style = _to_items(out.style, "inference")
        lighting = _to_items(out.lighting, "inference")
        composition = _to_items(out.composition, "inference")
        ocr = [SpanGroundedItem(text=t, span=t, confidence=1.0, observed=True, source="ocr") for t in out.ocr]

        # 4. 置信度评分
        all_items = objects + attributes + style + lighting + composition + ocr
        confidence = _score_confidence(out, all_items)

        return CaptionRecord(
            image_id=image_id,
            caption_long=out.caption_long,
            caption_medium=out.caption_medium,
            caption_short=out.caption_short,
            tags=out.tags,
            objects=objects,
            attributes=attributes,
            style=style,
            lighting=lighting,
            composition=composition,
            ocr=ocr,
            watermark=SpanGroundedItem(text=out.watermark, observed=True, source="visual")
            if out.watermark
            else None,
            quality=quality,
            safety=out.safety,
            caption_confidence=confidence,
            vlm_model=self.vlm.name,
            prompt_version=self.prompt_version,
            taxonomy_version=self.taxonomy_version,
            image_hash=self.image_hash(image),
            dataset_version=self.dataset_version,
        )
