现在只做 Muon ablation。

Baseline：

Z-Image architecture
+
AdamW

Experiment：

same model
same dataset
same training steps
same seed policy

only change:

optimizer

AdamW
vs
Muon

比较：

training loss
gradient statistics
stability
throughput
final quality
GPU memory

输出：

docs/research/ablation_muon.md

结论必须基于实验。

如果没有显著收益，明确写没有显著收益。