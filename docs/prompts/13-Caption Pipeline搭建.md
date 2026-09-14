使用 caption-pipeline skill。

设计一个 production-oriented image caption pipeline。

目标：

raw image
→ quality filtering
→ VLM caption
→ structured semantic extraction
→ confidence scoring
→ dataset record

Schema：

{
    image_id,
    caption_long,
    caption_medium,
    caption_short,
    tags,
    objects,
    attributes,
    style,
    lighting,
    composition,
    OCR,
    watermark,
    quality,
    safety,
    caption_confidence
}

要求：

1. caption 不允许 hallucinated visual facts
2. OCR 单独处理
3. caption confidence 单独计算
4. 支持重新 caption
5. dataset versioning
6. parquet output

优先支持 Qwen-VL 类模型。

不要把具体 VLM API 写死。

创建：

docs/learning/04_captioning.md

解释：

为什么 image generation data 的 caption quality
可能比简单增加图片数量更加重要。