# exp_100m 记录（M1 级，100M）

- **参数量**：95,661,120（95.66M）
- **数据**：16 张合成 latent（256² → 256 token/image），mock 文本（8 token）
- **训练步数**：2000；**处理 token** ≈ 2000 × batch4 × 264 token ≈ 2.1M
- **FLOPs 估算（前向）**：7.73e10（256²）
- **GPU 小时**：0.0265 h（95.4s @ 21 it/s）
- **loss**：2.338 → 0.245（≈9.5× 下降，收敛）
- **验证指标**：16 步 Euler 重建 MSE = 0.085
- **生成样本**：重建 latent（EXP-TRAIN-001，`experiments/20260915-tiny-overfit/`）

详见 `experiments/20260915-tiny-overfit/record.md`（EXP-TRAIN-001）。
