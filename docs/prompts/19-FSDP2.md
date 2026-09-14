使用 distributed-training skill。

现在将 training loop 从 single GPU architecture
扩展到 FSDP2。

要求：

1. 不改变模型数学定义
2. 支持 single GPU
3. 支持 multi-GPU
4. checkpoint 可恢复
5. rank-independent seed
6. distributed sampler
7. gradient accumulation
8. activation checkpointing
9. BF16
10. logging only on rank 0

首先实现：

2 GPU smoke test。

不要直接跑大训练。

验证：

single GPU loss
≈
multi GPU loss

允许 numerical tolerance。

记录：

docs/reproduction/distributed_training.md