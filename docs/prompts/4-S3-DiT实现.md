使用 s3-dit skill。

现在开始实现 Z-Image 的最小 faithful S3-DiT。

注意：

当前目标不是 6B。

先实现一个 tiny configurable architecture。

要求：

model configuration 必须参数化：

- hidden_size
- depth
- num_heads
- ffn_dim
- qk_norm
- sandwich_norm
- rope
- timestep embedding

首先实现：

src/zimage/models/

├── s3dit.py
├── attention.py
├── mlp.py
├── normalization.py
├── embeddings.py
└── config.py

不要实现训练。

要求：

1. PyTorch 原生实现
2. 不直接复制 FLUX/MMDiT architecture
3. 保持 single-stream design
4. 所有 modality 最终进入统一 token stream
5. 保留 text/image token boundary information
6. 支持 variable sequence length
7. 支持 arbitrary image token grid

首先写：

tests/models/

├── test_s3dit_shapes.py
├── test_attention.py
├── test_normalization.py
└── test_s3dit_forward.py

测试：

- batch size
- sequence length
- image resolution
- hidden dimension
- attention head
- gradient propagation

暂时不要追求性能。

完成后：

1. 运行全部测试
2. 检查参数量
3. 输出模型结构
4. 给我解释 S3-DiT 的 single-stream 为什么成立

如果遇到论文没有说明的地方，停止猜测并记录到 open_questions.md。