"""conditioning 抽象：Encoder → ConditioningAdapter → UnifiedTokenSequence → S3-DiT。

设计目标（[IMPLEMENTATION]）：
  - 不把 Qwen3 / VAE / SigLIP 写死进 S3-DiT；编码器是协议，真实实现在第 9 步接入。
  - 统一 `ConditioningOutput`：单流序列的 token、mask、modality_ids、position_ids、metadata。
  - 各条件（text / image / semantic）独立可控开关（传 None 即关闭）。
"""
