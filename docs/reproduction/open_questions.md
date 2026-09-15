# Z-Image Open Questions（未公开参数 / 歧义 / 差异 / 待验证）

> 本文件是"复现中的不确定性"总账。任何实现不得绕过本文件：实现依赖 [UNKNOWN] 项时必须先在此登记假设并注明 [ASSUMPTION]。
> 更新规则：解决一项 → 划掉并注明裁决证据；发现新问题 → 追加并编号。

## A. 论文未公开的训练参数（[UNKNOWN]，禁止把惯例当论文值）

| # | 项目 | 论文说法 | 我们取值的依据 | 验证计划 | 状态 |
| --- | --- | --- | --- | --- | --- |
| A1 | 优化器（类型/β/eps） | 全文未提 | [ASSUMPTION] 参考 SD3/Flux 惯例（AdamW 类） | M1 阶段消融 | 待定 |
| A2 | 学习率与调度（峰值/warmup/decay 形式） | 全文未提 | [ASSUMPTION] 惯例 + 网格消融 | M1 阶段消融 | 待定 |
| A3 | weight decay / grad clip | 全文未提 | [ASSUMPTION] 惯例 | M1 消融 | 待定 |
| A4 | EMA | 全文未提 | [ASSUMPTION] 先不用，实测采样质量后再定 | 出像质量对照 | 待定 |
| A5 | 全局 batch size（及各阶段） | 全文未提 | [ASSUMPTION] 按显存最大化 | 吞吐/收敛对照 | 待定 |
| A6 | 训练期 CFG dropout 概率 | 全文未提 | [ASSUMPTION] 需消融确定 | M2 阶段消融 | 待定 |
| A7 | logit-normal t 采样参数 σ | 只写 "following SD3" | [ASSUMPTION] SD3 惯例 σ=1.0；文献核对后消融 | 分布直方图 + 收敛对照 | 待定 |
| A8 | 损失权重函数（是否随 t/SNR 加权） | 只写 MSE | [IMPLEMENTATION] 恒 1 | 消融（可选） | 待定 |
| A9 | 数据集规模与构成、各数据源配比 | "大规模内部数据"，无数值 | 不可复现；公开替代 + 版本记录 | — | 长期 |
| A10 | 文本截断长度 | 未说明 | [ASSUMPTION] 待读官方推理代码与 Qwen3 用法确认 | 与官方 pipeline 输出一致 | 待定 |
| A11 | resolution-mapping function | "映射到预设训练范围"，无数式 | [ASSUMPTION] 自设计（如面积分桶+就近 16 倍数） | 跨分辨率消融 | 待定 |
| A12 | 预训练期 T2I:I2I 混合比 | 无数值（编辑 SFT 期 4:1 [PAPER §4.7]） | [ASSUMPTION] 待定 | 消融 | 待定 |
| A13 | caption 各类目混合概率、原始 alt 混入概率 | "小概率"无数值 | [ASSUMPTION] 待定 | 消融 | 待定 |
| A14 | SFT 变体数量与合并权重 αi | 无数值 | [ASSUMPTION] 小规模实验确定 | merging 消融 | 待定 |
| A15 | 模型权重初始化方案 | 论文/官方推理代码均未指定自定义初始化 | [ASSUMPTION] PyTorch 默认初始化；官方 diffusers 亦无自定义 init | Stage 6 加载官方权重逐值对齐时反向校验 | 待定 |

## B. 歧义实现点（论文含糊，代码可查但需确认口径）

| # | 项目 | 歧义 | 当前证据 | 待确认 |
| --- | --- | --- | --- | --- |
| B1 | 单流拼接顺序 | 论文叙述 "text, semantic, image" | diffusers：T2I 为 [x, cap]，omni 为 [cap, x, siglip] [OFFICIAL-CODE] | 官方 repo 自身实现是否与 diffusers 一致 |
| B2 | adaLN "condition vectors" 指什么 | 论文未明说 | diffusers：adaLN 仅由 t 驱动；文本走 token 路径 [OFFICIAL-CODE] | 官方 repo 实现核对 |
| B3 | "9 步实际 8 次 forward"（README） | 步数语义含糊 | pipeline sigma 表 linspace(1, 1/N, N) 无 9→8 逻辑 [OFFICIAL-CODE] | 属文档表述，无需代码验证 |
| B4 | 每层 FFN/attn 的 scale/gate 应用点 | 论文只说"调制归一化输入输出" | 需逐行读块内 forward（norm1×scale → attn → norm2×gate 等） | Stage 1 精读 |
| B5 | t_embedder 的 mid_size 语义 | 论文未说明 | mid_size=1024 为 MLP 中间维 [OFFICIAL-CODE] | 复刻时核对 |
| B6 | "refiner" 的 layer-id 用途（0+/1000+/2000+） | 代码有 id 但作用不明 | 可能用于 PE 唯一性/adapter 挂载 | Stage 1 精读 |
| B7 | cap_pad_token / x_pad_token 初始化为零 | 论文未说明 | 代码为 nn.Parameter(zeros) [OFFICIAL-CODE] | 缩比复刻保持一致 |
| B8 | 第 4 步 RoPE 是否接线 | 位置编码属第 5 步独立模块 | [IMPLEMENTATION] 第 4 步 attention 仅保留 `freqs_cis` 注入点（None 时跳过） | 第 5 步实现并逐值对齐后接线 |
| B9 | 第 4 步 patchify 实现形式 | 官方为 `Linear(patch_dim→dim)` | [IMPLEMENTATION] `PatchEmbed.proj = Linear`，与官方数学等价 | 已测试（test_embeddings.py） |
| B10 | batch 内可变长序列（padding+mask） | 论文提序列长度感知组 batch | [IMPLEMENTATION] 第 4 步支持 batch 内等长；padding/组 batch 属训练阶段（第 22 步） | attention_mask 注入点已预留并测试 |

## C. 论文 vs 代码/权重 差异（已在 paper-facts §6 裁决）

| # | 差异 | 裁决 | 影响 |
| --- | --- | --- | --- |
| C1 | Table 2 heads=32 vs 代码/权重 30 | ✅ 论文笔误，用 30（head_dim=128） | 缩比模型 heads 推导以此为准 |
| C2 | 官方 repo 遗留 4ch/0.18215 VAE 常量 vs 实际 Flux VAE 16ch/0.3611 | ✅ Flux VAE 确认；遗留常量禁用 | 预处理必须用 0.3611/0.1159 |
| C3 | 论文训练动态 shift vs 权重固定 shift（base=6.0, Turbo=3.0） | ⚠️ 训练复现用动态；推理对齐用固定值 | 两个口径都要实现且显式区分 |
| C4 | diffusers 默认 axes_lens[0]=1024 vs checkpoint 1536 | ⚠️ 以 checkpoint 为准 | RoPE 预计算表长度 |

## D. 需要实验验证的问题（论文声称 → 我们的小模型验证）

| # | 待验证命题 | 论文依据 | 实验设计 | 状态 |
| --- | --- | --- | --- | --- |
| D1 | OCR 信息进 caption 与文字渲染能力强绑定 | §3.1 | caption 有/无 OCR 对照（100M/300M） | 计划 |
| D2 | 联合 T2I+I2I 训练不损害 T2I | §4.3 | 混合比消融 | 计划 |
| D3 | 序列长度感知组 batch 收益 | §4.2 | padding 率/吞吐对照 | 计划 |
| D4 | 动态 shift 对多分辨率训练的必要性 | §4.3 | shift 开/关对照 | 计划 |
| D5 | Sandwich-Norm vs 普通 pre-norm 稳定性 | §4.1 | 训练稳定性对照 | 计划 |
| D6 | 低秩 adaLN（共享 down-proj）vs 全秩 | §4.1 | 参数量/质量对照 | 计划 |
| D7 | tagged resampling 防长尾遗忘 | §4.4 | 有/无重采样 SFT 对照 | 计划 |
| D8 | model merging 优于单一 SFT 变体 | §4.4 | 合并前后基准对照 | 计划 |
| D9 | 模拟用户提示类 caption 提升真实 prompt 遵循 | §3.2 | caption 类目消融 | 计划 |
| D10 | Turbo 8 NFE 在 5070 Ti 的延迟/显存（官方声称 <16GB） | §1 | 本机基线实测 | Stage 1 |

## E. 资源与范围问题

| # | 问题 | 说明 |
| --- | --- | --- |
| E1 | 官方 repo 无训练代码 | 训练一切超参均无 [OFFICIAL-CODE] 可依 → 全部靠 A 表消融 |
| E2 | Z-Captioner / reward model / PE 未发布 | 全部用公开替代并标 [IMPLEMENTATION] |
| E3 | Z-Image-Omni-Base / Z-Image-Edit 未发布 | 编辑路线只能参考论文描述 + 官方推理代码中的 omni 模式接口 |
| E4 | 6B 全量预训练不可行 | 314K H800h；我们只做 6B 架构实例化 + 权重对齐 + LoRA |
| E5 | 官方 repo 的推理代码 vs diffusers 移植的一致性 | 两处实现需在 Stage 1 做输出对撞 |
