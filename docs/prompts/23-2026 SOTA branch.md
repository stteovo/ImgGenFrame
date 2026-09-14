现在创建 2026-SOTA experimental branch。

目标不是修改 faithful Z-Image implementation。

必须保持：

experiments/baseline/
    faithful Z-Image

experiments/sota/
    Z-Image + 2026 methods

第一批实验只研究：

1. Muon optimizer
2. parameter-free RMSNorm
3. image-only pretraining
4. improved real-data-dominant curriculum
5. progressive resolution training
6. improved distillation

主要参考：

- Z-Image
- LLaDA-Image
- Qwen-Image-2.x
- Qwen-Image-2.x-RL

要求：

每个改进单独做 ablation。

不要一次修改多个因素。

建立：

docs/research/2026_sota_matrix.md

格式：

Method
Source
Original purpose
Expected benefit
Compatibility with Z-Image
Risk
Experiment
Result

不要把“最新”当作“更好”。

必须用实验验证。