现在开始 Z-Image-Tiny。

目标：

在 RTX 5070 Ti 16GB 上从随机初始化训练一个约 100M 参数的 tiny S3-DiT。

不要尝试 6B。

创建：

configs/tiny/

├── model.yaml
├── training.yaml
└── data.yaml

模型目标：

~100M parameters

resolution:
256x256

task:
T2I

objective:
Flow Matching

precision:
BF16

optimizer:
AdamW

要求：

1. 先 overfit 一个极小数据集
2. 再训练 toy dataset
3. 再训练真实数据
4. 每一步都有明确验证

第一实验：

8~32 张图片。

目标：

模型能够过拟合。

必须生成：

training loss curve
sample images
checkpoint

如果不能 overfit：

不要继续扩大 dataset。

先 debug。