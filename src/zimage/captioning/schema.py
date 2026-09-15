"""Caption 数据 Schema：分层 + observed vs inferred + 置信度/span 保存。

分层（[PAPER §3.2]）：long/medium/short/tags 四类 caption + objects/attributes/
style/lighting/composition 结构化语义层；OCR 独立（原语言保留 [PAPER §3.1]）。
observed vs inferred（AGENTS.md §20）：OCR/可见对象=observed，风格/光照等主观判断=inferred。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SpanGroundedItem:
    """span-grounded 抽取项：文本 + 源 span + 置信度 + observed/inferred。"""

    text: str
    span: Optional[str] = None  # 在 VLM 原始输出中的 source span
    confidence: float = 1.0
    observed: bool = True  # True=observed（OCR/可见），False=inferred（推理）
    source: str = "visual"  # ocr | visual | inference | metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "span": self.span,
            "confidence": self.confidence,
            "observed": self.observed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpanGroundedItem":
        return cls(
            text=d["text"],
            span=d.get("span"),
            confidence=d.get("confidence", 1.0),
            observed=d.get("observed", True),
            source=d.get("source", "visual"),
        )


@dataclass
class QualityAssessment:
    """质量过滤结果（复用 profiling，第 12 步）。"""

    passed: bool
    entropy: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "entropy": self.entropy,
            "resolution": list(self.resolution),
            "reasons": self.reasons,
        }


@dataclass
class CaptionRecord:
    """单张图的完整 caption 记录（对齐 prompt schema）。"""

    image_id: str
    caption_long: str
    caption_medium: str
    caption_short: str
    tags: List[str] = field(default_factory=list)
    objects: List[SpanGroundedItem] = field(default_factory=list)
    attributes: List[SpanGroundedItem] = field(default_factory=list)
    style: List[SpanGroundedItem] = field(default_factory=list)
    lighting: List[SpanGroundedItem] = field(default_factory=list)
    composition: List[SpanGroundedItem] = field(default_factory=list)
    ocr: List[SpanGroundedItem] = field(default_factory=list)
    watermark: Optional[SpanGroundedItem] = None
    quality: Optional[QualityAssessment] = None
    safety: str = "unknown"
    caption_confidence: float = 0.0
    # 版本 / 溯源
    vlm_model: str = ""
    prompt_version: str = ""
    taxonomy_version: str = ""
    image_hash: str = ""
    dataset_version: str = ""

    def observed_items(self) -> List[SpanGroundedItem]:
        return [it for it in self._all_items() if it.observed]

    def inferred_items(self) -> List[SpanGroundedItem]:
        return [it for it in self._all_items() if not it.observed]

    def _all_items(self) -> List[SpanGroundedItem]:
        items = self.objects + self.attributes + self.style + self.lighting + self.composition + self.ocr
        if self.watermark is not None:
            items.append(self.watermark)
        return items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "caption_long": self.caption_long,
            "caption_medium": self.caption_medium,
            "caption_short": self.caption_short,
            "tags": self.tags,
            "objects": [it.to_dict() for it in self.objects],
            "attributes": [it.to_dict() for it in self.attributes],
            "style": [it.to_dict() for it in self.style],
            "lighting": [it.to_dict() for it in self.lighting],
            "composition": [it.to_dict() for it in self.composition],
            "ocr": [it.to_dict() for it in self.ocr],
            "watermark": self.watermark.to_dict() if self.watermark else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "safety": self.safety,
            "caption_confidence": self.caption_confidence,
            "vlm_model": self.vlm_model,
            "prompt_version": self.prompt_version,
            "taxonomy_version": self.taxonomy_version,
            "image_hash": self.image_hash,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptionRecord":
        def items(key: str) -> List[SpanGroundedItem]:
            return [SpanGroundedItem.from_dict(x) for x in d.get(key, [])]

        return cls(
            image_id=d["image_id"],
            caption_long=d["caption_long"],
            caption_medium=d["caption_medium"],
            caption_short=d["caption_short"],
            tags=list(d.get("tags", [])),
            objects=items("objects"),
            attributes=items("attributes"),
            style=items("style"),
            lighting=items("lighting"),
            composition=items("composition"),
            ocr=items("ocr"),
            watermark=SpanGroundedItem.from_dict(d["watermark"]) if d.get("watermark") else None,
            quality=QualityAssessment(**d["quality"]) if d.get("quality") else None,
            safety=d.get("safety", "unknown"),
            caption_confidence=d.get("caption_confidence", 0.0),
            vlm_model=d.get("vlm_model", ""),
            prompt_version=d.get("prompt_version", ""),
            taxonomy_version=d.get("taxonomy_version", ""),
            image_hash=d.get("image_hash", ""),
            dataset_version=d.get("dataset_version", ""),
        )
