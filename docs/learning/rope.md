# Learning Report — 3D Unified RoPE

## 1. 做了什么

实现 `src/zimage/models/rope3d.py`：3D Unified RoPE 频率表与旋转核心，含位置坐标生成（文本/图像 token），并附 `scripts/debug_rope3d.py`。16 个 pytest（`tests/models/test_rope3d.py`）通过，含与 diffusers `RopeEmbedder` 的逐值对齐测试。尚未接入 S3-DiT 前向（接线属第 9 步 smoke test 之前的集成）。

## 2. 为什么需要

单流 transformer 把文本、视觉语义、图像 token 拼进同一个序列，位置编码必须能区分"这是文本的第几个 token"与"这是图像的哪一行哪一列"。1D RoPE 只能编码一维序号，无法表达图像的空间结构；2D RoPE（h, w）能编码图像但无法区分文本与图像、也无法表达编辑任务中"参考图 vs 目标图"的时间先后。因此 Z-Image 用 **3D（temporal, height, width）** 统一编码三类 token。

## 3. 数学形式

对每个轴 d ∈ {32, 48, 48}（t/h/w，和=128=head_dim）：

```
freqs_m = theta^(-2m/d)，m = 0..d/2-1      （theta=256）
angle[i, m] = i · freqs_m                   （i 为该轴坐标）
freqs_cis[i, m] = exp(j · angle[i, m]) = cos + j·sin
```

三轴各自生成一张 `[轴长, d/2]` 复指数表；给定 token 的 (t,h,w) 坐标，按轴索引并拼接，得到 `[head_dim/2]` 复指数；对 Q/K 的 head 维做复数乘法完成旋转（等价于相邻维度两两旋转 angle）。三轴拼接后总维度 = (32+48+48)/2 = 64 个复数 = 128 实维 = head_dim。

## 4. Z-Image 中如何使用

- 图像 token：沿空间轴 (h, w) 展开（patchify 后按行主序），t 轴恒定于文本之后的偏移值。
- 文本 token：沿时间轴 t 递增（1..cap_len），h=w=0。
- 编辑任务（[PAPER §4.1]）：参考图与目标图共享空间 (h,w) 坐标，时间维错开 1 个单位，以区分两个上下文。
- 常数 [OFFICIAL-CKPT]：theta=256.0、axes_dims=[32,48,48]、axes_lens=[1536,512,512]；`head_dim == sum(axes_dims)` 为硬约束。

## 5. 当前项目如何实现

- `RopeEmbedder`（nn.Module）：惰性预计算 + 缓存三轴复指数表；`forward(ids)` 对 [N,3] 坐标按轴索引（gather，无逐 token Python 循环）。
- 位置生成：`text_token_positions` / `image_token_positions` / `create_coordinate_grid`，逐值对齐官方 `_pad_with_ids` 的坐标语义（文本 start=(1,0,0)；图像 start=(t_offset,0,0)，单帧 t 恒定）。
- `apply_rotary_emb`：复数旋转（float32 计算 → 回原 dtype），与官方一致；从 `attention.py` 迁移至本模块统一持有。

## 6. 其他可能实现

- 1D RoPE（如 GPT/LLaMA）：只能编码线性序号，无法表达空间结构。
- 2D RoPE（h, w）：能编码图像，但无法区分文本/图像与编辑时间轴。
- 可学习绝对位置嵌入：对变长/变分辨率不友好，RoPE 有相对位置性质与长度外推能力。
- 分头/分维度切分方式：也可把三轴交叠分配（非分段拼接），但官方明确按 (32,48,48) 分段拼接，我们保持一致。

## 7. Trade-offs

- 分段拼接（每轴独占维度）清晰、可解释、可对齐；代价是 head 内不同子空间只对单轴敏感（t 轴 32 维、h 轴 48 维、w 轴 48 维）。
- 复指数表预计算占用 `sum(axis_len · d/2)` complex64 ≈ (1536·16 + 512·24 + 512·24)·8 ≈ 393KB，可忽略。
- RoPE 相对位置性质利于长度外推；但 axes_lens 上限（t=1536）决定了最长可编码序列，超长需扩表（当前为 checkpoint 值）。

## 8. 如何验证

- 形状 / head_dim 兼容：输出 [N, head_dim/2]；`head_dim == sum(axes_dims)`。
- 确定性：同输入同输出（torch.equal）。
- 位置敏感性：不同坐标频率不同；旋转后向量不同。
- 轴分配：文本 t 递增 h=w=0；图像 t 恒定 h/w 展开；H/W 变化影响坐标表。
- 数值稳定性：|freqs_cis|≡1、全有限；旋转保持模长（正交）。
- 数值对齐：与 diffusers `RopeEmbedder` 同参数同 ids 下逐值一致（atol=1e-6）。

## 9. Failure Modes

- 轴分配颠倒（时间给图像、空间给文本）——由轴语义测试拦截。
- head_dim 与 axes_dims 和不一致（如 120≠128）——构造函数校验。
- 忘记编辑任务的时间偏移——第 9 步接线时需按官方 `cap_len+1` 偏移实现。
- axes_lens 用 diffusers 默认 (1024,512,512) 而非 checkpoint (1536,512,512)——paper-facts §6/C4 已裁决。

## 10. 当前仍然未知的问题

- 编辑任务的精确 temporal offset 语义（"错开 1 单位"）需读官方编辑推理代码确认精确值（rope-3d skill 已列，编辑阶段处理）。
- 超长序列（t>1536 或分辨率超 1k–1.5k）的外推行为未验证。
