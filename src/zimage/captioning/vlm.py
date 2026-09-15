"""VLM 抽象：不写死具体 VLM API。

- `VLMCaptionOutput`：VLM 结构化输出（OCR-first：先转录再描述 [PAPER §3.1]）。
- `VLM` 协议：任何实现只需返回结构化输出。
- `MockVLM`：确定性假 VLM，供测试与离线开发。
- 真实 Qwen-VL 类适配：包装 transformers 模型，prompt 模板版本化 [IMPLEMENTATION]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class VLMCaptionOutput:
    caption_long: str
    caption_medium: str
    caption_short: str
    tags: List[str] = field(default_factory=list)
    ocr: List[str] = field(default_factory=list)  # 原语言 OCR 文本（不翻译）
    objects: List[Dict] = field(default_factory=list)  # {text, span, confidence, observed}
    attributes: List[Dict] = field(default_factory=list)
    style: List[Dict] = field(default_factory=list)
    lighting: List[Dict] = field(default_factory=list)
    composition: List[Dict] = field(default_factory=list)
    watermark: Optional[str] = None
    quality: Optional[str] = None
    safety: str = "unknown"


class VLM(Protocol):
    """VLM 协议。"""

    name: str

    def caption(self, image) -> VLMCaptionOutput:
        ...


class MockVLM:
    """确定性假 VLM（离线测试用，不产生 hallucinated 事实）。"""

    name = "mock-vlm-0"

    def caption(self, image) -> VLMCaptionOutput:
        # 固定结构化输出；OCR 保留原语言（中文），不翻译 [PAPER §3.1]
        return VLMCaptionOutput(
            caption_long="一只棕色的小狗坐在草地上，背景是蓝天。",
            caption_medium="棕色小狗坐在草地上。",
            caption_short="一只狗。",
            tags=["dog", "grass", "outdoor"],
            ocr=["公园提示牌：请勿践踏草坪"],
            objects=[
                {"text": "dog", "span": "一只棕色的小狗", "confidence": 0.95, "observed": True},
                {"text": "grass", "span": "草地", "confidence": 0.9, "observed": True},
            ],
            attributes=[{"text": "brown", "span": "棕色", "confidence": 0.9, "observed": True}],
            style=[{"text": "photo", "span": None, "confidence": 0.6, "observed": False}],
            lighting=[{"text": "daylight", "span": None, "confidence": 0.5, "observed": False}],
            composition=[{"text": "center", "span": None, "confidence": 0.5, "observed": False}],
            watermark=None,
            quality="good",
            safety="safe",
        )


class TransformersVLM:
    """transformers 模型适配（真实 VLM 接入点，prompt 模板版本化 [IMPLEMENTATION]）。

    不写死 Qwen-VL：传入任意 `model`/`processor`；调用方负责生成 prompt 模板并版本化。
    """

    name: str

    def __init__(self, model, processor, model_name: str = "transformers-vlm", prompt_template: str = ""):
        self.model = model
        self.processor = processor
        self.name = model_name
        self.prompt_template = prompt_template

    def caption(self, image) -> VLMCaptionOutput:
        raise NotImplementedError(
            "真实 VLM 推理需 model.generate + 解析 JSON；接入时实现并冻结 prompt 版本"
        )
