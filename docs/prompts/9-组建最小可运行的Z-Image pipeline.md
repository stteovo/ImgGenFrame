现在使用：

- s3-dit
- rope-3d
- flow-matching
- vae-tokenization
- conditioning
- reproduction-audit

将前面所有模块组成最小可运行 Z-Image pipeline。

注意：

当前仍然禁止训练。

创建：

scripts/smoke_test.py

完成：

text
→ text encoder mock
→ image latent mock
→ unified tokens
→ S3-DiT
→ velocity prediction

必须能够：

- forward
- backward
- BF16
- variable resolution
- variable sequence length

运行：

pytest
ruff
mypy（如果项目启用）
smoke test

同时输出：

parameter count
activation shape
peak VRAM
forward time

最后创建：

docs/reproduction/milestone_01_s3dit.md

说明：

- 实现了什么
- 没实现什么
- 与论文对应关系
- 下一步