使用 conditioning skill。

实现 Z-Image conditioning abstraction。

需要支持：

Text
Image
Optional semantic conditions

要求设计统一接口：

ConditioningOutput

至少包含：

- text_tokens
- text_mask
- image_tokens
- image_mask
- modality_ids
- position_ids
- metadata

不要把 Qwen encoder 写死到 S3-DiT。

采用：

Encoder
↓
ConditioningAdapter
↓
UnifiedTokenSequence
↓
S3-DiT

要求测试：

- text only
- image only
- text + image
- empty condition
- variable text length
- variable image token count

解释：

为什么 Z-Image 可以 single-stream，
以及它与传统 MMDiT 的 conceptual difference。