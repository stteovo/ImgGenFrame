# Z-Image 已验证事实库（Paper Facts）

> 本文件是全项目唯一的"事实来源"。任何技术讨论、实现决策、ablation 设计必须先对照本文件。
> 每条事实必须带标签与出处。标签定义见 `AGENTS.md`。
> **修改规则**：新增事实必须附来源；发现与官方来源冲突时，先更新本文件再讨论。

## 0. 主要来源

| 代号 | 来源 | 内容 |
| --- | --- | --- |
| [PAPER] | arXiv:2511.22699（Z-Image Team, Alibaba, 2025-12-01 提交） | 技术报告全文 |
| [OFFICIAL-CODE] | github.com/Tongyi-MAI/Z-Image（仅推理代码） | `src/config/model.py`、`inference.py`、README |
| [OFFICIAL-CODE] | HF PR #12703 / #12715（已并入 diffusers） | `ZImagePipeline` 实现 |
| [OFFICIAL-CKPT] | huggingface.co/Tongyi-MAI/Z-Image、Tongyi-MAI/Z-Image-Turbo | 权重与 config.json |

## 1. 模型家族 [PAPER]

- 4 个变体，DiT 均为同一 6B 架构：
  - **Z-Image-Omni-Base**：omni 预训练产物（生成+编辑），未发布（截至本文件撰写）。
  - **Z-Image**：omni → T2I SFT（50 步、CFG 开、建议 guidance 3.0–5.0、强烈建议负向提示词）。
  - **Z-Image-Turbo**：Z-Image 经 Decoupled-DMD + DMDR 蒸馏 + RL（8 NFE、CFG 关、<16GB VRAM 可推理）。
  - **Z-Image-Edit**：Z-Image 经编辑继续训练（未发布）。
- 全流程训练成本：Low-res 预训练 147.5K + Omni 预训练 142.5K + Post-training 24K = **314K H800 GPU 时 ≈ $628K** [PAPER Table 1]。
- SFT 基础模型带 CFG 推理约需 **100 NFE**（50 步 × 2）[PAPER §4.5]。

## 2. 架构（S3-DiT）

### 2.1 整体 [PAPER §4.1]

- **单流（single-stream）MM-DiT**：text tokens、visual semantic tokens（仅编辑任务）、image VAE tokens 在**序列维拼接**后统一进入 backbone；无双流分支。
- 输入侧：每种模态各有一个**轻量 modality-specific processor**（各由 **2 个 transformer block** 组成）做初始对齐，然后进入统一主干。
- 条件注入（adaLN 风格）：条件向量投影为 scale/gate 参数，调制 Attention 与 FFN 的归一化输入/输出；投影做**低秩分解**：一个**跨层共享的 down-projection** + **每层独立的 up-projection**。
- 归一化：**QK-Norm**（稳定 attention 激活）+ **Sandwich-Norm**（在每个 attention/FFN 块的输入与输出端都做归一化约束）；所有归一化统一使用 **RMSNorm**。
- 位置编码：**3D Unified RoPE**。image tokens 展开在空间维（h,w），text tokens 沿时间维递增。编辑任务：参考图与目标图共享空间 RoPE 坐标，时间维错开 1 个单位间隔；参考图（clean）与目标图（noisy）使用不同的 time-conditioning 值。

### 2.2 配置参数

| 项 | 值 | 标签 |
| --- | --- | --- |
| DiT 总参数 | 6.15B | [PAPER Table 2] |
| 层数 | 30 | [PAPER Table 2] = [OFFICIAL-CODE] |
| hidden dim | 3840 | [PAPER Table 2] = [OFFICIAL-CODE] |
| FFN 中间维 | 10240 | [PAPER Table 2] = [OFFICIAL-CODE] |
| attention heads | **30**（论文 Table 2 的 32 已裁决为笔误） | [OFFICIAL-CKPT] HF transformer/config.json n_heads=30；[OFFICIAL-CODE] `assert head_dim == sum(axes_dims)` → 3840/30=128=32+48+48 自洽 |
| RoPE 轴维 (dt,dh,dw) | (32,48,48)，和=128 | [PAPER Table 2] = [OFFICIAL-CKPT] axes_dims=[32,48,48] |
| RoPE theta | 256.0 | [OFFICIAL-CKPT] rope_theta=256.0 |
| RoPE 轴长度 (t,h,w) | (1536,512,512) | [OFFICIAL-CKPT] axes_lens=[1536,512,512]（diffusers 代码默认值曾为 (1024,512,512)，以 checkpoint 为准） |
| heads / kv-heads | 30 / 30（无 GQA，head_dim=128） | [OFFICIAL-CKPT] |
| RMSNorm eps | 1e-5 | [OFFICIAL-CKPT] norm_eps=1e-05 |
| QK-Norm | 开启（RMSNorm 实现，`qk_norm="rms_norm"`） | [PAPER §4.1] = [OFFICIAL-CODE] diffusers transformer_z_image.py |
| adaLN 共享嵌入维 | 256 = min(dim, ADALN_EMBED_DIM) | [OFFICIAL-CODE] diffusers：`t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)` |
| caption 特征维（Qwen3-4B hidden） | 2560 | [OFFICIAL-CKPT] transformer cap_feat_dim=2560 = text_encoder hidden_size |
| patch size（image latent） | (2,2)，f_patch=(1) | [OFFICIAL-CKPT] all_patch_size=[2], all_f_patch_size=[1]；x_embedder 输入 1×2×2×16=64 维 |
| 时间嵌入 | FREQUENCY_EMBEDDING_SIZE=256、MAX_PERIOD=10000、T_SCALE=1000（t∈[0,1]×1000 后进嵌入） | [OFFICIAL-CODE] diffusers TimestepEmbedder（正弦嵌入 + MLP, mid_size=1024） |
| 序列长度需为 32 的倍数 | SEQ_MULTI_OF=32（各模态独立填充，pad token 参与 attention mask） | [OFFICIAL-CODE] |
| modality processors | 3 组 refiner：noise_refiner（2 blocks，带调制，id 1000+）、context_refiner（2 blocks，无调制，id 0+）、siglip_refiner（可选，2 blocks，id 2000+） | [OFFICIAL-CODE] diffusers；[PAPER §4.1] 的"每模态 2 个 transformer block"即指此结构 |
| 序列拼接顺序 | 基础模式（T2I）：[x, cap]；omni 模式：[cap, x, siglip] | [OFFICIAL-CODE] `_build_unified_sequence` |
| 文本进入方式 | cap_embedder（RMSNorm+Linear 2560→3840）转 token 进流，**不经过 adaLN 路径**；adaLN 仅由 t 驱动 | [OFFICIAL-CODE]；[INFERENCE] 论文"input condition vectors"指 timestep 嵌入 |
| 编辑双条件 | clean/noisy token 分别用 t=1 / t 的时间嵌入，逐 token 选择（noise_mask + select_per_token）；FinalLayer scale=1.0+mod | [OFFICIAL-CODE] |

### 2.3 冻结组件 [PAPER §4.1, §4.2]

- **Text encoder：Qwen3-4B**（选其双语能力），**frozen**（论文 §4.2 明确 "they remain frozen"）。
- **VAE：Flux VAE**（选其重建质量），**frozen**。latent 16 通道（代码 transformer in_channels=16）、spatial 下采样 8×（代码 DEFAULT_VAE_SCALE_FACTOR=8）。
- **SigLIP 2**：**仅编辑任务**加入，用于从参考图提取 abstract visual semantics；T2I 时不使用。

## 3. 训练目标 [PAPER §4.3]

- **Flow Matching**：
  - 路径：`x_t = t·x1 + (1−t)·x0`，其中 x0 = 高斯噪声，x1 = 原图。
  - 目标速度场：`v_t = x1 − x0`。
  - 损失：`L = E[ ||u(x_t, y, t; θ) − (x1 − x0)||² ]`。
- **Timestep 采样**：logit-normal（沿 SD3），集中在中间 t。
- **动态时间偏移（dynamic shifting）**：多分辨率导致 SNR 不同，训练采用 Flux 式 dynamic time shifting [PAPER §4.3]。公式（diffusers 复刻 Flux `calculate_shift`）：`m=(MAX_SHIFT−BASE_SHIFT)/(MAX_IMAGE_SEQ_LEN−BASE_IMAGE_SEQ_LEN)`，`shift=m·seq_len+b`，常数 `BASE_IMAGE_SEQ_LEN=256、MAX_IMAGE_SEQ_LEN=4096、BASE_SHIFT=0.5、MAX_SHIFT=1.15` [OFFICIAL-CODE]。
- **推理 checkpoint 不使用动态 shift**：Turbo 固定 `shift=3.0`、base Z-Image 固定 `shift=6.0`、`use_dynamic_shifting=False` [OFFICIAL-CKPT]。即"论文训练用动态 shift / 发布权重用固定 shift"，且 base 与 Turbo 的 shift 不同——采样口径差异，详见 §6。
- **采样 sigma 表**（Z-Image 特有）：`linspace(1.0, 1/N, N)`，N=推理步数 [OFFICIAL-CODE] pipeline_z_image.py。
- **优化器 / lr / batch size / EMA / weight decay / CFG dropout：论文未给出 → [UNKNOWN]**（全文检索无 Adam/EMA/warmup/dropout 字样）。

## 4. 训练课程 [PAPER §4.2–4.4]

1. **Low-resolution pre-training**：仅 256²、仅 T2I、单阶段；占总预训练算力 **>50%**；主要习得基础视觉知识与跨模态对齐（含中文文字渲染基础）。
2. **Omni-pre-training**（多阶段）：
   - **任意分辨率**：原始分辨率经 "resolution-mapping function" 映射到预设训练范围（函数形式未给出 → [UNKNOWN]）。
   - **T2I + I2I 联合训练**：利用大规模弱对齐图像对；联合训练不损害 T2I 性能 [PAPER 观察]。
   - **多级双语 caption**：long/medium/short/tags/模拟用户提示 5 类；原始 alt text 以小概率混入；I2I 任务随机选目标图 caption 或差异 caption。
   - 结束时支持至 **1k–1.5k** 分辨率。
3. **PE-aware SFT**：
   - Distribution narrowing：高质量图 + 超详细 grounded captions 收窄分布。
   - Concept balancing：基于知识图谱 + BM25 rarity 的 tagged resampling，mini-batch 内动态加权。
   - Model merging：同初始化、不同能力偏好的多个 SFT 变体做**权重线性插值** `θ=Σαi·θi`。
   - 所有 prompt 先过 **Prompt Enhancer（固定 VLM + reasoning chain）** 再进 DiT。
4. **训练系统** [PAPER §4.2]：DiT 用 FSDP2；冻结的 VAE/TE 用 DP；所有 DiT 层 gradient checkpointing；torch.compile；**序列长度感知组 batch** + **动态 batch size**（长序列小 batch、短序列大 batch）。

## 5. 数据基础设施 [PAPER §2]

四个模块（本项目按 mini 版复现思路，见 reproduction_matrix）：

1. **Data Profiling Engine**：分辨率/文件大小元数据；pHash 去重；压缩伪影（理想未压缩大小/实际大小比值）；自训质量模型（偏色、模糊、水印、噪声）；信息熵（边界像素方差、JPEG 重编码 BPP）；美学打分；AIGC 检测分类器（引 Imagen 3 结论）；VLM 语义 tag + NSFW；CN-CLIP 图文对齐分；多级 caption（OCR 信息由同一 VLM 生成，不走独立 OCR 模块）。
2. **Cross-modal Vector Engine**：将 SD3 的 dedup 重构为图社区发现问题——k-NN 替代 range_search，k 近邻建图 + Leiden 社区检测（Traag 2019）；rapidsai 全 GPU，8×H800 上 1B 条目约 8 小时（含索引 + 100-NN）；兼做失败案例回溯检索与概念缺口采样。
3. **World Knowledge Topological Graph**：Wikipedia 实体 + 超链接建图；PageRank 剪枝 + VLM 可视化生成性过滤；caption tag embedding 层级化扩充（引 Vo et al. 2024）；节点按 BM25 分数 + 图层级关系计算采样权重，驱动语义级平衡采样。
4. **Active Curation Engine**：以 Z-Image 自身为诊断先验挖掘 hard case；human-in-the-loop 主动学习闭环（伪标签 → 人/机双验证 → 重训 captioner 与 reward model）。

**编辑对构造** [PAPER §2.5]：专家模型混合编辑；图表示组合——1 输入图 + N 个编辑版本 → C(N+1,2) 对；视频帧对（CN-CLIP 余弦相似度过滤）；可控文字渲染系统构造文字编辑对。

**Z-Captioner** [PAPER §3]：OCR 优先的 CoT captioning（先转录所有文字、保持原语言、再写 caption）；5 类 caption；条件化注入世界知识；编辑指令 = 3 步 CoT 差异 caption。

## 6. 论文 vs 官方代码 已发现的差异

| # | 论文 | 官方代码/权重 | 裁决（2026-09-14 更新） |
| --- | --- | --- | --- |
| 1 | 32 attention heads（Table 2） | `DEFAULT_TRANSFORMER_N_HEADS=30`；HF transformer/config.json `n_heads=30` | ✅ **已裁决：论文笔误，heads=30**。代码 `assert head_dim == sum(axes_dims)`（128=32+48+48）与 checkpoint 双重印证 [OFFICIAL-CKPT] |
| 2 | "Flux VAE"（16ch latent） | 官方 config 有 `DEFAULT_VAE_LATENT_CHANNELS=4、SCALING_FACTOR=0.18215` 等 SD 风格默认值；HF vae/config.json `_name_or_path=flux-dev`、`latent_channels=16`、`scaling_factor=0.3611` | ✅ **已裁决：Flux VAE 确认**（16ch、scaling 0.3611、shift_factor 0.1159）；repo 中 4ch/0.18215 为遗留死代码，**严禁使用** |
| 3 | 训练用 dynamic shifting | 发布 checkpoint 均 `use_dynamic_shifting=False`；base `shift=6.0`、Turbo `shift=3.0` | ⚠️ **训练用动态 shift [PAPER]、推理用固定 shift [OFFICIAL-CKPT]**，且 base≠Turbo。训练复现用动态 shift；对齐测试用各自固定值 |
| 4 | modality processor = 每模态 2 个 block | diffusers：noise_refiner / context_refiner / siglip_refiner 各 n_refiner_layers=2 | ✅ **已裁决**：对应关系成立；context_refiner 无调制、noise_refiner 带调制 |
| 5 | 单流序列"text + semantic + image 拼接" | diffusers：基础模式 `[x, cap]`，omni 模式 `[cap, x, siglip]` | ℹ️ 顺序与论文叙述不完全对应，以代码为准 [OFFICIAL-CODE] |
| 6 | README 示例 `num_inference_steps=9`（8 次 forward） | pipeline 源码无 9→8 特殊处理；sigma 表 `linspace(1.0, 1/N, N)` | ℹ️ 属文档表述（8 NFE 的 Euler 步数语义），非代码差异 |

## 7. UNKNOWN 清单（不得猜，列为 Stage 1/2 调查项）

- 优化器、学习率与调度、weight decay、warmup、梯度裁剪、EMA、全局 batch size（论文全文无 Adam/EMA/warmup 字样）。
- 数据集规模与构成、caption 长度分布、text encoder 输入截断长度。
- 训练期 CFG dropout 概率。
- resolution-mapping function 的具体形式。
- T2I/I2I 混合比（预训练期；编辑 SFT 期已知 4:1 [PAPER §4.7]）。
- logit-normal t 采样的参数（论文只写"following SD3"，未给 σ 值）。
- 蒸馏/RL 阶段的具体超参与数据量（细节在独立论文 arXiv:2511.22677、2511.13649）。
- 官方 repo（Tongyi-MAI/Z-Image）自身模型实现与 diffusers 实现的逐处一致性（diffusers 为 PR 移植，可能存在细微差异）。

## 8. 推理事实 [OFFICIAL-CODE/README]

- bf16 推荐；Turbo `num_inference_steps=9` 实际 8 次 DiT forward；Turbo `guidance_scale=0`。
- Z-Image 支持负向提示词与 `cfg_normalization` 开关（官方建议：通用风格化 False、写实 True）。
- 分辨率：512²–2048² 总像素面积、任意长宽比。
- Turbo 在 H800 上 sub-second 的前提：FlashAttention-3 + torch.compile [PAPER §1 脚注]。

---
*最后核对日期：2026-09-14。来源版本：arXiv v1 (2511.22699) PDF/HTML；官方 repo main@{2026-09 检索}；HF Tongyi-MAI/Z-Image(-Turbo) config.json；diffusers main transformer_z_image.py / pipeline_z_image.py。*
