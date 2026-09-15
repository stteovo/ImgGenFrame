# exp_300m 记录（M2 级，300M）

- **参数量**：301,474,112（301.47M）
- **配置**：dim=1024, depth=18, heads=8, ffn=2730（head_dim=128 守恒，同架构）
- **FLOPs 估算（前向）**：2.50e11（256²）
- **显存实测（bf16, RTX 5070 Ti 16GB）**：
  - 256²（264 token, batch4）：fwd 0.623GB / fwd+bwd 2.545GB，6.89 it/s
  - 512²（1032 token, batch4）：fwd+bwd 9.076GB，2.68 it/s
  - 1024²（4104 token, batch1）：fwd+bwd 13.233GB，0.98 it/s
- **训练步数 / GPU 小时 / loss / 验证指标 / 生成样本**：N/A（本步仅做显存/吞吐测试，训练属后续步骤）
- **结论**：300M 前向/反向在 16GB 内完成（至 1024²）；升级条件满足。

基准脚本：`scripts/scale_benchmark.py`；配置：`configs/300m/model.yaml`。
