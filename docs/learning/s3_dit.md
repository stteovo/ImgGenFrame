# Learning Report — S3-DiT（单流 DiT backbone）

## 1. 做了什么

实现了最小可配置的 S3-DiT 单流主干（`src/zimage/models/`）：

- `config.py`：参数化结构配置（hidden_size / depth / num_heads / ffn_dim / qk_norm / sandwich_norm / rope / timestep embedding / patch / cap_feat_dim），含 `tiny()`（≈95.7M，M1 级）与 `official()`（6B，仅实例化/权重对齐）两类配置。
- `normalization.py`：RMSNorm（eps=1e-5，float32 计算）。
- `attention.py`：多头注意力 + QK-Norm + RoPE 注入点（`freqs_cis`，第 5 步接线）。
- `mlp.py`：SwiGLU 门控前馈（hidden = int(dim/3*8)）。
- `embeddings.py`：正弦时间嵌入 + patchify 嵌入 + caption 嵌入。
- `s3dit.py`：S3DiTBlock（Sandwich-Norm + 低秩 adaLN scale/gate）、FinalLayer、S3DiT 单流前向。

39 个 pytest 通过（形状 / batch / 序列长度 / 分辨率 / head 配置 / 梯度传播 / 参数量核对 / 结构输出）。未实现训练与 RoPE 频率计算（分别属第 6 / 5 步）。

## 2. 为什么需要

Z-Image 的核心架构创新（[PAPER §4.1, §1]）：文本、视觉语义、VAE 图像三类 token 在**序列维直接拼接**进入**同一个** transformer 主干，而不是用"文本编码器 → 交叉注意力 → 图像 DiT"的双流方案。

双流（如 SD3/FLUX 的 MMDiT 双分支）里文本与图像各走一条流、只在交叉注意力处交互；单流则让所有 token 从一开始就共享同一套自注意力，实现**逐层密集跨模态交互**，且**参数完全复用**（同一批注意力/FFN 权重服务所有模态），因此在同等参数量下表达力更强、参数效率更高。这就是"single-stream 为什么成立"的核心：早期融合 + 密集交互 + 参数复用。

## 3. 数学原理

单流本身是序列建模，无新算子；关键数学在块内调制与归一化：

- **RMSNorm**：`y = x / rms(x) · γ`，`rms(x)=√(mean(x²)+ε)`。不做中心化，计算更省（无均值），大模型训练中稳定性与 LayerNorm 相当。
- **QK-Norm**：对每个 head 的 Q、K 各自做 RMSNorm，约束注意力对数幅值，抑制长序列/大维度下的 logit 爆炸。
- **Sandwich-Norm**：每个 attention/FFN 子层的**输入端和输出端各一个** RMSNorm（块内共 4 个），把残差信号的幅度夹在两层归一化之间，稳定深层训练。
- **低秩 adaLN**：时间条件 `c` 经共享下投影到 256 维瓶颈，再每层独立上投影到 `4·d`，切出 `(scale_msa, gate_msa, scale_mlp, gate_mlp)`；应用为
  `gate=tanh(·)`、`scale=1+·`，
  `x = x + gate_msa ⊙ norm2(attn(norm1(x) ⊙ scale_msa))`（FFN 同理）。

## 4. Z-Image 中如何使用

- 输入侧每种模态各有一个 **lightweight modality processor**（各 2 个 transformer block）：`noise_refiner`（带调制，处理带噪图像 token）、`context_refiner`（无调制，处理文本 token）[PAPER §4.1 / OFFICIAL-CODE]。
- 单流拼接顺序（T2I 基础模式）：`[x(noisy image), cap(text)]` [OFFICIAL-CODE]。
- 位置编码用 3D Unified RoPE（第 5 步）；时间条件仅驱动 adaLN（文本走 token 路径，不经 adaLN）[OFFICIAL-CODE]。
- 6B 官方配置：dim=3840、30 层、30 heads（论文 Table 2 的 32 已裁决为笔误）、FFN=10240、head_dim=128。

## 5. 当前项目如何实现

- 结构逐值对齐 diffusers `ZImageTransformer2DModel`（见 `paper-facts.md §2.2` / 差异表 §6）。
- 缩比约束硬编码进 `S3DiTConfig.__post_init__`：`head_dim == sum(axes_dims)`、`ffn_dim == int(dim/3*8)`。
- `sandwich_norm` 保留开关：关闭时 `attention_norm2/ffn_norm2` 退化为 `Identity`（即普通 pre-norm），服务消融 D5。
- 单流边界信息由 `unified_modality_ids()` 显式承载（1=image、0=text），供 unpatchify 切片与第 5 步 RoPE 轴分配使用。

[IMPLEMENTATION] 的工程选择：
- RoPE 频率计算留到第 5 步，第 4 步 attention 仅保留 `freqs_cis` 注入点（`None` 时跳过）。
- patchify 用 `PatchEmbed.proj = Linear(patch_dim→dim)`，与官方 `all_x_embedder = Linear(f*p*p*C→dim)` 数学等价。
- 权重初始化用 PyTorch 默认（官方未指定自定义初始化，见 open_questions A15）。

## 6. 其他可能实现

- 双流 MMDiT（SD3/FLUX 式）：文本/图像各一支、交叉注意力交互——Z-Image 论文明确不采用（单流是其卖点）。
- patchify 用 `nn.Conv2d(kernel=patch, stride=patch)`：与 `Linear` 数学等价，本实现选 Linear 以贴合官方。
- 全秩 adaLN（每层独立 Linear(dim→4·dim)）：参数更多，Z-Image 用共享下投影省参数，且 256 维瓶颈是其显式设计。

## 7. Trade-offs

- 单流：跨模态交互强、参数省；代价是所有 token 长度叠加 → 自注意力序列变长（文本+图像共同计入 O(L²)），需序列长度感知组 batch 缓解（第 22 步）。
- Sandwich-Norm：稳定性↑，但每块 4 个 norm 增加少量参数与计算。
- 低秩 adaLN：参数量↓，但 256 维瓶颈限制了调制表达力（论文选择了这个平衡）。

## 8. 如何验证

- 形状：batch / 分辨率 / 序列长度 / head 配置参数化测试（`test_s3dit_shapes.py`）。
- 梯度：全模型前向后所有参数有有限梯度（`test_s3dit_forward.py`）。
- 数值：RMSNorm 单位 RMS / 缩放不变性；时间嵌入 t=0 闭式值（`test_normalization.py` / `test_embeddings.py`）。
- 参数量：`num_parameters()` 与逐参数求和一致，tiny ≈95.7M 落在 M1 区间。
- 后续（Stage 6）：加载官方 6B 权重与 diffusers `ZImagePipeline` 逐值对齐是最终裁决（金标准验证）。

## 9. Failure Modes

- heads 抄论文的 32（真实为 30，head_dim=128=32+48+48）。
- 用 SD3/Flux 的"常见"归一化顺序替代官方 Sandwich-Norm 顺序。
- 把 Qwen3-4B 的 2560 维特征直接进 3840 主干而不经 `cap_embedder` 对齐。
- 缩比时改 head_dim 或 FFN 比例，破坏与 6B 的可比性。
- FinalLayer 误用 RMSNorm（官方是 `LayerNorm(elementwise_affine=False, eps=1e-6)`）。

## 10. 当前仍然未知的问题

- 权重初始化方案（官方未指定）——见 `open_questions.md` A15。
- 训练期超参（优化器/lr/EMA/CFG dropout 等）——`open_questions.md` A 表，不属本步范围。
- RoPE 频率计算与 3D 轴分配的逐值对齐——第 5 步完成。
