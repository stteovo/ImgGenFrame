# 数据集说明 — EXP-TRAIN-001

- **数据源**：合成 latent（非真实图像）。
- **构造**：`SyntheticLatentDataset`（`src/zimage/training/dataset.py`）。
  - latent：16 张 [16, 32, 32]，`torch.randn`，seed=0（N(0,1)）。
  - 文本：16 条 [8, 2560]，`torch.randn`，seed=0（mock 文本特征，维度对齐 Qwen3-4B hidden）。
- **样本数**：16（第一实验目标 8~32）。
- **用途**：验证 100M 模型过拟合能力，非训练数据（无 caption 语义）。
- **下一数据版本**：真实 T2I 数据（flickr30k / conceptual_captions 等）+ 真实文本编码器（第 16 步）。
