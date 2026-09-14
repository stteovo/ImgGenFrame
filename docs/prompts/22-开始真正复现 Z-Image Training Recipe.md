现在根据 reproduction matrix，
实现 Z-Image training curriculum。

严格按照论文中能够确认的内容：

Stage 1:
low-resolution pretraining

Stage 2:
omni-pretraining

Stage 3:
SFT

Stage 4:
D-DMD

Stage 5:
DMDR

Stage 6:
RL

但是：

当前只实现 Stage 1。

不要提前实现 Stage 2+。

要求：

每个 stage 都有：

- config
- dataset
- sampler
- checkpoint
- evaluation
- report

所有无法从论文确认的参数：

UNKNOWN

然后告诉我：

哪些参数需要我们通过 ablation 推断。