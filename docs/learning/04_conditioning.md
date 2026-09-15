# Learning Report — Conditioning（条件抽象 + 低秩 adaLN + 单流 vs MMDiT）

## 1. 做了什么

实现 `src/zimage/conditioning/`：`ConditioningOutput` 统一条件序列结构、`TextEncoder`/`ImageEncoder` 协议、`ConditioningAdapter`（把编码器输出组织成统一单流条件，含 position_ids / modality_ids / mask / metadata）。并新增低秩 adaLN 结构测试与逐子层 scale/gate 数值对齐测试。128 项测试通过。

## 2. 为什么需要条件抽象

S3-DiT 是「单流」：文本、图像、（可选）语义 token 拼进同一个序列。如果不抽象，就必须在 S3-DiT 内部写死「Qwen3 怎么编码文本、VAE 怎么出 latent、SigLIP 怎么出语义」，导致：
- 换编码器/做 ablation 要改模型主体（违反单一职责）；
- 无法独立测试「只文本 / 只图像 / 空条件」这些组合。

抽象成 `Encoder → ConditioningAdapter → UnifiedTokenSequence → S3-DiT` 后，编码器是可替换协议，条件组合是可控开关，S3-DiT 只面对「统一 token 流」。

## 3. 数学原理（低秩 adaLN）

时间条件 c 经共享下投影到 256 维瓶颈，再每层独立上投影到 4·d：

```
c → t_embedder → [B, 256]（共享，跨层唯一）
           ↓ 每层独立 Linear(256 → 4·d)
   (scale_msa, gate_msa, scale_mlp, gate_mlp)
           ↓ gate = tanh(·)、scale = 1 + ·
   x = x + gate_msa ⊙ norm2(attn(norm1(x) ⊙ scale_msa))
   x = x + gate_mlp ⊙ norm2(ffn(norm1(x) ⊙ scale_mlp))
```

「低秩」：调制参数从「每层 d→4d 的全秩」降到「共享 256 瓶颈 + 每层 256→4d」，参数量与 6B 结构严格对应。注意 Z-Image 用的是 **scale + gate**（无 shift），与 diffusers 实现逐值一致。

## 4. Z-Image 中如何使用

- 文本（Qwen3-4B 冻结特征 2560 维）走 token 路径进流，**不经 adaLN**；adaLN 只由时间 t 驱动 [OFFICIAL-CODE]。
- 图像（Flux VAE scaled latent）patchify 后进流；语义（SigLIP）仅编辑任务加入。
- 单流拼接顺序（T2I）：`[x, cap]`；位置语义：文本沿时间轴 t=1..T，图像沿空间轴 (h,w)、t=T+1。
- 低秩 adaLN 是 [PAPER §4.1] 明确设计（共享 down-proj + 每层 up-proj）。

## 5. 当前项目如何实现

- `ConditioningAdapter` 不嵌入模型维（嵌入仍在 S3-DiT 的 cap_embedder/x_embedder），只负责 position_ids（复用 rope3d 的 `text_token_positions`/`image_token_positions`）、modality_ids、mask、metadata，避免与 S3-DiT 重复。
- 各条件独立开关：`forward(text=None, image=None, semantic=None)`，None 即关闭。
- 位置「拼接顺序 [image,text]」与「时间顺序 text 先」的差异已显式处理并测试。

[IMPLEMENTATION] 决定：batch 内变长（padding+mask）属训练阶段，本步 mask 恒 True；CFG 逻辑（训练 dropout / 推理双前向）留待第 9/10 步接入时实现。

## 6. 其他可能实现

- 双流 MMDiT（SD3/FLUX）：文本与图像各走一支 transformer，只在交叉注意力交互。
- 把 Qwen3 直接写进 S3-DiT：实现快，但换编码器/做 ablation 都要改主干。
- 全秩 adaLN（每层 Linear(d→4d)）：参数多、无共享瓶颈。

## 7. 单流 vs 传统 MMDiT 的概念差异

| | 单流 S3-DiT | 传统双流 MMDiT |
| --- | --- | --- |
| token 组织 | 文本+图像拼成**一条**序列，进同一自注意力 | 文本/图像各一条流，各自自注意力 |
| 跨模态交互 | **每层**自注意力都让任意 token 直接关注另一模态 token（早期+密集融合） | 只在**交叉注意力层**交互（稀疏、后期） |
| 参数 | 同一套 attention/FFN 权重服务所有模态，**参数复用** | 两条流各一套权重，参数接近翻倍 |
| 复杂度 | 序列长度=各模态之和，自注意力 O((T_txt+T_img)²) | 各流独立，交叉注意力 O(T_txt·T_img) |

Z-Image 之所以能单流：它把「模态差异」交给**轻量 modality-specific processor（每模态 2 blocks）做初始对齐**，再进统一主干；于是主干无需双流分支，密集交互 + 参数复用带来更高参数效率 [PAPER §1/§4.1]。

## 8. 如何验证

- 组合测试：text only / image only / text+image / empty / 变长文本 / 变图像 token 数。
- 位置语义：文本 t=1..T、图像 t=T+1、拼接顺序 [image,text] 的 modality_ids 断言。
- 低秩结构：共享 t_embedder 唯一、每层 up-proj 参数独立（data_ptr 断言）、Linear(256→4d)。
- scale/gate 数值：块前向与手动复现逐值一致（scale=1+scale、gate=tanh）。

## 9. Failure Modes

- 把 Qwen 写死进 S3-DiT（破坏抽象）。
- 位置顺序搞反：拼接顺序 [image,text] 与时间顺序 text 先混为一谈。
- 低秩 adaLN 做成每层独立 down-proj（参数翻倍、结构失真）。
- gate 忘 tanh、scale 忘 +1（与官方不一致）。

## 10. 当前仍然未知的问题

- CFG dropout 概率（训练期，[UNKNOWN]，open_questions A6）。
- 编辑任务的 semantic（SigLIP）接入与双时间条件（t vs t=1）——编辑阶段实现。
- 训练期 batch 内变长 padding 策略（open_questions B10）。
