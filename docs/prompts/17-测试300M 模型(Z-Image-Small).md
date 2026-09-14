现在开始 Z-Image-Small。

规模：

~300M parameters

目标：

验证 scaling。

保持：

same architecture
same data pipeline
same objective

只改变：

model scale
compute
dataset size

建立：

experiments/scaling/

exp_100m
exp_300m

记录：

- params
- tokens
- image count
- training steps
- FLOPs estimate
- GPU hours
- loss
- validation metrics
- generation samples

不要修改多个变量。

目标：

建立最初的 scaling curve。