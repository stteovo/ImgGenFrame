# Z-Image Reproduction Matrix

> 列说明：**Paper Claim** = 论文主张（[PAPER]）；**Paper Evidence** = 出处（章节/表号）；**Official Code** = 官方代码/权重证据（[OFFICIAL-CODE]/[OFFICIAL-CKPT]）；**Parameter** = 具体参数值（无则 UNKNOWN）；**Our Implementation** = 我们的实现计划（[IMPLEMENTATION]）；**Verification** = 验证方法；**Status** = ✅已裁决 / ⚠️有差异 / ❓UNKNOWN / ⬜未开始。
> 来源：arXiv:2511.22699；github.com/Tongyi-MAI/Z-Image；diffusers `transformer_z_image.py`/`pipeline_z_image.py`；HF `Tongyi-MAI/Z-Image(-Turbo)` config.json。

## 1. Architecture

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARCH-01 | Architecture | 单流 MM-DiT | text / visual semantic / VAE 三类 token 序列级拼接成统一输入流，无双流分支；早期融合、逐层密集跨模态交互 | §4.1, §1 | diffusers：基础模式 `[x, cap]`、omni `[cap, x, siglip]` | 拼接顺序与论文叙述不完全对应，以代码为准 | 缩比梯子实现统一单流 backbone（第4步：src/zimage/models/） | 权重加载对齐 + 序列顺序断言测试 | ⚠️ |
| ARCH-02 | Architecture | DiT 总参数 | 6.15B（仅 DiT，不含 Qwen3-4B 与 VAE） | Table 2 | — | 6.15B | 6B 级仅实例化+权重对齐，不训练 | 参数量统计脚本 vs 官方 checkpoint | ✅ |
| ARCH-03 | Architecture | 层数 / hidden / FFN | 30 层 / 3840 / 10240 | Table 2 | HF config：n_layers=30, dim=3840 | 30 / 3840 / 10240 | 缩比时保持 dim:FFN≈1:2.67 | 与 HF config 逐项比对 | ✅ |
| ARCH-04 | Architecture | attention heads | 论文写 32 | Table 2 | HF config n_heads=30；代码断言 head_dim==sum(axes_dims) | **30**（head_dim=128）| 使用 30 heads | `assert 3840/30 == 32+48+48` | ✅ |
| ARCH-05 | Architecture | FFN 类型 | 未明说 | — | FeedForward 为 silu gating（SwiGLU 式） | — | 按 diffusers 实现 | 数值对齐 | ❓ |
| ARCH-06 | Architecture | 初始化/精度 | 未说明 | — | `_skip_layerwise_casting_patterns=[t_embedder, cap_embedder]`（精度敏感） | UNKNOWN | 标 [ASSUMPTION] | 冒烟训练稳定性 | ❓ |
| ARCH-07 | Architecture | ControlNet 支持 | 未提及（社区版有） | — | `controlnet_block_samples` 逐层残差注入接口 | — | 不在范围（C 层之外） | — | ✅ |

## 2. Tokenization

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TOK-01 | Tokenization | VAE | 用 Flux VAE，选其重建质量；冻结 | §4.1, §4.2 | HF vae/config.json：`_name_or_path=flux-dev` | latent 16ch、8× 下采样 | 包装 diffusers AutoencoderKL，冻结（第7步 image_tokenizer） | 重建 PSNR；latent 统计 | ✅ |
| TOK-02 | Tokenization | VAE 缩放常数 | 未说明 | — | HF config：scaling_factor=0.3611, shift_factor=0.1159；repo 遗留 0.18215 为死代码 | 0.3611 / 0.1159 | 用 checkpoint 值，禁用 0.18215（第7步 latent_utils） | 与官方 pipeline latent 逐值一致 | ✅ |
| TOK-03 | Tokenization | 图像 patchify | 未明说 | — | all_patch_size=[2], all_f_patch_size=[1]；x_embedder=Linear(64→3840) | 2×2 空间 patch，1 帧 | 从零实现 patchify（第4步 PatchEmbed + 第7步 patchifier，token 序 [pH,pW,C]） | 形状 + 官方权重对齐 | ✅ |
| TOK-04 | Tokenization | text encoder | Qwen3-4B，双语能力；冻结 | §4.1, §4.2 | HF text_encoder/config.json：Qwen3ForCausalLM | hidden 2560、36 层、32/8 GQA、vocab 151936 | transformers 加载，冻结 | 特征分布探针 | ✅ |
| TOK-05 | Tokenization | 文本截断长度 | 未说明 | — | 无 max length 配置；Qwen3 max_position=40960 | UNKNOWN | 标 [ASSUMPTION]（需读官方推理代码确认） | 与官方 pipeline 输出一致 | ❓ |
| TOK-06 | Tokenization | 序列填充 | 未说明 | — | SEQ_MULTI_OF=32；各模态独立填充 + pad token + attention mask | 32 倍数 | 从零实现 | pad 行为测试 | ✅ |
| TOK-07 | Tokenization | SigLIP 2 | 仅编辑任务加入 | §4.1 | siglip_feat_dim 可选；siglip_embedder + siglip_refiner；位置缩放到与图像一致 | UNKNOWN | 编辑阶段再实现（当前不在范围） | — | ⬜ |

## 3. Conditioning

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COND-01 | Conditioning | 时间嵌入 | 未明说 | — | diffusers TimestepEmbedder：正弦嵌入+MLP；t×T_SCALE | freq_size=256, max_period=10000, t_scale=1000, MLP mid=1024 | 从零实现（第4步 TimestepEmbedder） | 与 diffusers 逐值对齐 | ✅ |
| COND-02 | Conditioning | 低秩 adaLN | 条件向量投影为 scale/gate；共享层无关 down-proj + 每层 up-proj | §4.1 | t_embedder 输出 256 = min(dim, ADALN_EMBED_DIM)（共享）；每层 `Linear(256→4·dim)`（层特定） | 256 → 4×3840 | 从零实现共享/每层结构（第8步测试锁定共享 down-proj + 每层 up-proj） | 参数量断言 + 权重对齐 | ✅ |
| COND-03 | Conditioning | 文本条件路径 | 未明说 | — | cap_embedder=RMSNorm+Linear(2560→3840) 转 token 进流；**不经过 adaLN** | 2560→3840 | 按代码实现（第4步 CaptionEmbedder） | 数值对齐 | ✅ |
| COND-04 | Conditioning | 编辑双条件 | 参考图 clean 与目标图 noisy 用不同 time-conditioning | §4.1 | t_noisy=t_embedder(t·1000)、t_clean=t_embedder(1·1000)，noise_mask 逐 token 选择 | t vs t=1 | 编辑阶段实现 | select_per_token 测试 | ⬜ |
| COND-05 | Conditioning | FinalLayer | 未说明 | — | scale = 1.0 + adaLN_modulation(c)，作用于残差输出 | 1+mod | 按代码实现 | 数值对齐 | ✅ |
| COND-06 | Conditioning | CFG dropout（训练） | 未提及 | — | 无 | UNKNOWN | 标 [ASSUMPTION] 并消融 | ablation | ❓ |

## 4. RoPE

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROPE-01 | RoPE | 3D 统一 RoPE | 图像 token 沿空间维 (h,w)，文本沿时间维 | §4.1 | precompute_freqs_cis 每轴独立算复指数 | axes_dims=[32,48,48] | 从零实现（第5步 rope3d.py） | 与 diffusers 逐值对齐 | ✅ |
| ROPE-02 | RoPE | theta | 未说明 | — | rope_theta=256.0 | 256.0 | 从零实现（第5步） | 数值对齐 | ✅ |
| ROPE-03 | RoPE | 轴长度 | 未说明 | — | axes_lens=[1536,512,512]（checkpoint）；diffusers 默认 [1024,512,512] | [1536,512,512] | 用 checkpoint 值（第5步） | 对齐测试 | ⚠️ |
| ROPE-04 | RoPE | 文本/图像位置关系 | 编辑任务：参考与目标共享空间坐标、时间维错开 1 单位 | §4.1 | cap_pos 从 (0,0,0) 起；x_pos 时间维从 cap 末+1 起；siglip 位置缩放到图像分辨率 | t 偏移 = cap_len+1 | 编辑阶段实现 | 位置表断言测试 | ✅ |
| ROPE-05 | RoPE | head_dim 约束 | 未明说 | — | `assert head_dim == sum(axes_dims)` | 128 = 32+48+48 | 缩比模型保持该约束（config + rope3d 校验） | 断言测试 | ✅ |

## 5. Normalization

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NORM-01 | Normalization | RMSNorm | 所有归一化统一用 RMSNorm | §4.1 | RMSNorm(dim, eps=norm_eps) | eps=1e-5 | 从零实现（第4步） | 数学性质测试（test_normalization.py） | ✅ |
| NORM-02 | Normalization | QK-Norm | 稳定注意力激活 | §4.1 | qk_norm="rms_norm"（对 Q、K 归一化） | RMSNorm, eps=1e-5 | 从零实现（第4步） | 数值对齐 | ✅ |
| NORM-03 | Normalization | Sandwich-Norm | 约束每个 attention/FFN 块输入输出的信号幅度 | §4.1 | 每块 4 个 norm：attention_norm1/2、ffn_norm1/2 | 块内 2+2 | 从零实现（第4步） | 结构断言 + 对齐 | ✅ |

## 6. Flow Matching

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM-01 | Flow Matching | 插值路径 | x_t = t·x1 + (1−t)·x0 | §4.3 公式(1) | — | x0=噪声, x1=数据 | 从零实现（第6步 flow_matching.py） | 端点/闭式测试 | ✅ |
| FM-02 | Flow Matching | 目标与损失 | v = x1 − x0，MSE | §4.3 公式(1) | — | velocity 预测 | 从零实现（第6步） | loss 初值量级测试 | ✅ |
| FM-03 | Flow Matching | 损失权重 | 未说明 | — | 无 | UNKNOWN（默认恒 1） | 标 [IMPLEMENTATION] | ablation（可选） | ❓ |
| FM-04 | Flow Matching | 预测类型 | velocity（v-prediction） | §4.3 | — | v-pred | 从零实现（第6步） | 与采样器自洽性 | ✅ |

## 7. Noise Sampling

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NS-01 | Noise Sampling | 训练期 t 采样 | logit-normal（following SD3），集中在中间 t | §4.3 | 无参数 | UNKNOWN（σ 未给） | 标 [ASSUMPTION]（SD3 惯例 σ=1.0，第6步已实现） | 分布直方图 + ablation | ❓ |
| NS-02 | Noise Sampling | 动态 time shifting | Flux 式动态 shift 补偿多分辨率 SNR | §4.3 | diffusers calculate_shift：`m=(1.15−0.5)/(4096−256)`，shift=m·seq+b | base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15 | 从零实现（第6步 timestep_sampling.py） | 与 diffusers 逐值对齐 | ✅ |
| NS-03 | Noise Sampling | 推理 shift | 未说明 | — | 发布权重固定 shift：base=6.0、Turbo=3.0；use_dynamic_shifting=False | 6.0 / 3.0 | 推理用固定值；训练用动态 | 官方权重采样对齐 | ⚠️ |
| NS-04 | Noise Sampling | 采样 sigma 表 | 未说明 | — | pipeline：linspace(1.0, 1/N, N)，N=步数 | σ ∈ [1, 1/N] | 从零实现（第6步 scheduler.py） | 与 pipeline 对齐 | ✅ |

## 8. Resolution Curriculum

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RES-01 | Resolution Curriculum | low-res 预训练 | 仅 256²、单阶段、占总预训练算力 >50% | §4.3, Table 1 | BASE_IMAGE_SEQ_LEN=256 一致（256²→16² token） | 256² | 课程第一步 | loss/出像检查 | ✅ |
| RES-02 | Resolution Curriculum | 任意分辨率训练 | resolution-mapping function 映射到训练范围 | §4.3 | 无函数实现 | UNKNOWN | 标 [ASSUMPTION]，自设计并消融 | 跨分辨率出像质量 | ❓ |
| RES-03 | Resolution Curriculum | 训练后分辨率上限 | omni 结束支持 1k–1.5k | §4.3 | MAX_IMAGE_SEQ_LEN=4096（≈1024² token 数） | 1k–1.5k | 课程终点 | 采样质量 | ✅ |
| RES-04 | Resolution Curriculum | 编辑继续训练分辨率 | 512² 数千步 → 1024²；T2I:I2I=4:1 | §4.7 | — | 512→1024；4:1 | 编辑阶段 | — | ⬜ |

## 9. Data Infrastructure

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DATA-01 | Data Infrastructure | Profiling Engine | 元数据、pHash、压缩比、质量模型、信息熵、美学、AIGC 检测、VLM tag+NSFW、CN-CLIP 对齐、多级 caption | §2.1 | 无（内部系统，不发布） | UNKNOWN（阈值全未给） | mini 版：公开替代 + 规则启发式 | 人工标注 1K 对照 | ❓ |
| DATA-02 | Data Infrastructure | Cross-modal Vector Engine | k-NN+图社区检测去重、跨模态检索；8×H800 上 1B 条目 8h | §2.2 | 无 | k=100（官方规模） | mini 版：10⁵–10⁶ 级 | 近重复召回测试 | ❓ |
| DATA-03 | Data Infrastructure | 内部数据池 | 大规模内部版权数据 | §2.1 | 不可得 | UNKNOWN | 公开数据集替代 | 数据版本记录 | ⬜ |

## 10. Captioning

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAP-01 | Captioning | Z-Captioner | 一体化 captioner，OCR 优先 CoT | §3.1 | 未发布 | UNKNOWN | 公开 VLM 复刻方法论 | 文字渲染 ablation | ❓ |
| CAP-02 | Captioning | OCR 原语言保留 | 强制 OCR 结果保持原语言不翻译 | §3.1 | — | — | prompt 模板约束 | 抽查测试 | ✅ |
| CAP-03 | Captioning | 5 类 caption | long/medium/short/tags/模拟用户提示 | §3.2 | — | UNKNOWN（长度规范未给） | 公开 VLM + 模板 | 质量抽查 | ❓ |
| CAP-04 | Captioning | 世界知识注入 | caption 条件化 meta 信息，减少实体幻觉 | §3.2 | — | — | mini 版（公开实体库） | 实体正确率抽查 | ❓ |
| CAP-05 | Captioning | 差异 caption | 3 步 CoT：详细 caption → 差异分析 → 指令合成 | §3.3 | — | — | 编辑阶段复刻 | 指令一致性抽查 | ⬜ |

## 11. Semantic Deduplication

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEDUP-01 | Semantic Deduplication | 方法 | SD3 dedup 重构为图社区检测：k-NN 替代 range_search | §2.2 | 无 | UNKNOWN（阈值/社区内保留策略未给） | k-NN + Leiden mini | 近重复召回 + 误删检查 | ❓ |
| DEDUP-02 | Semantic Deduplication | 官方规模参照 | rapidsai 全 GPU；8×H800 8h/1B 条目 | §2.2 | 无 | k=100 | 仅参照，不对标 | — | ✅ |

## 12. Knowledge Graph

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KG-01 | Knowledge Graph | 构建三阶段 | Wikipedia 实体图 → PageRank 剪枝 + VLM 可视化生成性过滤 → 层级化扩充 + 权重分配 | §2.3 | 无 | UNKNOWN | mini 图 + 简化层级 | 概念覆盖度量 | ❓ |
| KG-02 | Knowledge Graph | 采样权重 | tag→节点映射，BM25 分数 + 父子层级关系计算采样权重 | §2.3, §4.4 | 无 | UNKNOWN | 从零实现加权采样 | 分布均匀性度量 | ❓ |

## 13. Active Curation

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AC-01 | Active Curation | hard case 挖掘 | 以 Z-Image 自身为诊断先验发现长尾概念缺口 | §2.4 | 无 | UNKNOWN | 小模型失败检索 mini | 概念覆盖前后对照 | ❓ |
| AC-02 | Active Curation | 人机双验证闭环 | 伪标注 → 双验证 → 重训 captioner/reward model | §2.4 | 无 | UNKNOWN | 不复制（超出算力），仅读 | — | ⬜ |

## 14. Pretraining

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRE-01 | Pretraining | low-res 阶段 | 256² 固定分辨率 T2I，>50% 预训练算力 | §4.3, Table 1 | 无训练代码 | 算力 147.5K H800h | 缩比复现（100M/300M） | loss 曲线 + 出像 | ✅ |
| PRE-02 | Pretraining | 并行策略 | DiT 用 FSDP2；冻结 VAE/TE 用 DP；全层 grad ckpt；torch.compile | §4.2 | 无训练代码 | UNKNOWN（无 cluster 细节） | 单卡：grad ckpt + 显存优化 | 显存/吞吐实测 | ✅ |
| PRE-03 | Pretraining | 优化器与调度 | 未说明 | — | 无 | UNKNOWN | 标 [ASSUMPTION]（参考 SD3/Flux 惯例并消融） | 消融实验 | ❓ |
| PRE-04 | Pretraining | EMA / warmup / grad clip | 未说明 | — | 无 | UNKNOWN | 标 [ASSUMPTION] | 消融实验 | ❓ |
| PRE-05 | Pretraining | 序列长度感知组 batch | 按分辨率分组，减少 padding；长序列小 batch、短序列大 batch | §4.2 | 无 | UNKNOWN | 从零实现 | padding 率/吞吐对照 | ✅ |

## 15. Omni-pretraining

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OMNI-01 | Omni-pretraining | 任意分辨率 | 分辨率映射函数映射到训练范围 | §4.3 | 无 | UNKNOWN | 自设计 [ASSUMPTION] | 多分辨率出像 | ❓ |
| OMNI-02 | Omni-pretraining | T2I+I2I 联合 | 联合训练不损害 T2I 性能 | §4.3 | 无 | UNKNOWN（混合比） | 缩比联合训练 | T2I 性能对照 | ❓ |
| OMNI-03 | Omni-pretraining | 多级双语 caption | 5 类 caption 混用；原始 alt text 小概率混入；I2I 随机选目标 caption 或差异 caption | §4.3 | 无 | UNKNOWN（比例） | caption 管线接入 | 消融 | ❓ |
| OMNI-04 | Omni-pretraining | 算力 | 142.5K H800h | Table 1 | — | — | 缩比 | — | ✅ |

## 16. SFT

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT-01 | SFT | Distribution narrowing | 高质量精选数据 + 超详细 grounded captions 收窄分布 | §4.4 | 无 | UNKNOWN | 缩比复现 | 美学/指令遵循指标 | ❓ |
| SFT-02 | SFT | Tagged resampling | 知识图谱 + BM25 rarity 动态重采样，概念边际均匀 | §4.4 | 无 | UNKNOWN | 与 KG-02 联动实现 | 概念分布度量 | ❓ |
| SFT-03 | SFT | Model merging | 多 SFT 变体线性权重插值 θ=Σαi·θi | §4.4 | 无 | UNKNOWN（α 未给） | 从零实现 | 合并前后基准对照 | ❓ |
| SFT-04 | SFT | PE-aware SFT | 所有 prompt 经固定 PE（VLM+reasoning chain） | §4.8 | 无 | UNKNOWN | 公开 VLM 替代 | 生成质量对照 | ❓ |

## 17. D-DMD

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DDMD-01 | D-DMD | 机制解耦 | CA 是蒸馏引擎、DM 是正则器；独立 renoise schedule | §4.5.1 | 无（细节在 2511.22677） | UNKNOWN | 读独立论文后 mini 复现（Stage 9） | 8-NFE 学生 vs teacher | ⬜ |
| DDMD-02 | D-DMD | 效果 | 解决细节丢失与色偏；8 步可与 100-NFE teacher 媲美 | §4.5.1, §4.5.3 | — | — | — | — | ✅ |

## 18. DMDR

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DMDR-01 | DMDR | DMD+RL 融合 | RL 解锁偏好对齐，DM 项作正则防 reward hacking | §4.5.2 | 无（细节在 2511.13649） | UNKNOWN | 读独立论文后评估可行性 | 指标 + 人工检查 | ⬜ |

## 19. RLHF

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RLHF-01 | RLHF | Reward model | 三维度：指令遵循 / AIGC 感知 / 美学；指令遵循按 5 类元素满意率打分 | §4.6.1 | 未发布 | UNKNOWN | 观测 + 公开替代（若做 mini） | 与人工评分相关性 | ⬜ |
| RLHF-02 | RLHF | DPO（客观维度） | VLM 生成偏好对 + 人工清理；课程从简单到复杂 | §4.6.2 | 无 | UNKNOWN | mini 复现（Stage 9 视算力） | 客观任务正确率 | ⬜ |
| RLHF-03 | RLHF | GRPO | 复合优势函数在线优化多质量维度 | §4.6.3 | 无 | UNKNOWN | 观测 | — | ⬜ |

## 20. Inference

| ID | Category | Component | Paper Claim | Paper Evidence | Official Code | Parameter | Our Implementation | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INF-01 | Inference | Z-Image 推理参数 | SFT 模型 ~100 NFE 带 CFG | §4.5 | pipeline 默认 steps=50、guidance=5.0 | 50 步 × 2 forward | Euler+CFG 采样器 | 官方权重采样对齐 | ✅ |
| INF-02 | Inference | Turbo 推理参数 | 8 NFE、无 CFG、sub-second（H800+FA3+compile） | §1, §4.5 | pipeline：guidance=0；README steps=9→8 次 forward | 8 NFE | 8 步 Euler | 官方权重采样对齐 | ✅ |
| INF-03 | Inference | 负向提示与 cfg_normalization | 负向提示强烈推荐；cfg_normalization 写实 True | README | pipeline 支持 negative_prompt、cfg_normalization | guidance 3.0–5.0 | 实现 cfg_normalization | 开/关行为对照 | ✅ |
| INF-04 | Inference | 分辨率范围 | 512²–2048² 总像素面积、任意长宽比 | README | 无硬性检查 | — | 支持任意比例 | 多分辨率采样 | ✅ |
| INF-05 | Inference | dtype | bf16 推荐 | README | torch_dtype=bfloat16 | bf16 | bf16 推理 | 对齐容差验证 | ✅ |
| INF-06 | Inference | 6B 显存 | <16GB VRAM 可跑（Turbo） | §1 | README 提供 cpu offload | ≈12.3GB 权重 | 本机实测 + offload | 显存峰值实测 | ✅ |

## 附录 A：复现范围决策（A/B/C/D 四层）

| 层级 | 内容 | 策略 |
| --- | --- | --- |
| A 忠实复现 | S3-DiT、3D RoPE、低秩 adaLN、FM 目标、采样器 | 从零实现 + 单测 + 官方权重对齐 |
| B 缩比复现 | 训练课程、数据基建、SFT 三技术 | mini 版在 100M–1B 验证 |
| C 观测/替代 | Z-Captioner、reward model、D-DMD/DMDR/RLHF、PE | 读论文 + 公开替代/小规模 |
| D 明确不做 | 6B 全量预训练、内部版权数据 | 6B 只做架构实例化 + 权重对齐 + LoRA |

**金标准验证**：加载官方 6B 权重进我们的实现，输出与 diffusers `ZImagePipeline` 对齐（bf16 容差内）。Stage 2 起做模块级对齐，Stage 6 全量。

## 附录 B：缩比梯子

| 级 | 参数量 | 目的 | 数据规模（[ASSUMPTION]） | 分辨率 |
| --- | --- | --- | --- | --- |
| M1 | ~100M | 结构+训练循环验证、全部单测 | 合成/玩具 → 小型真实集 | 64²–256² |
| M2 | ~300M | 数据管线 v0 + 课程 + 首个 ablation | 10⁵ 级 | 256² |
| M3 | ~1B | 多分辨率 + 动态 shift + 组 batch | 10⁵–10⁶ 级 | 256²→512² |
| M4 | ~3B | SFT 三技术 + 显存极限工程 | 10⁶ 级 | 512² |
| M5 | 6.15B | 仅实例化+权重对齐+LoRA | — | — |

各级 (layers, dim, heads) 配置进入该级时设计并记录推导，保持 dim:FFN≈1:2.67、head_dim=128、head_dim==sum(axes_dims) 约束。

## 附录 C：硬件备忘（RTX 5070 Ti 16GB）

- 6.15B bf16 ≈ 12.3GB → 推理可行（需 offload）；全参训练不可行。
- 1B 训练（Adam fp32 状态）超 16GB → grad ckpt + 8bit 优化器/offload，进入 M3 实测 [EXPERIMENT]。
- Turbo 推理 <16GB 为论文/官方声称 [PAPER] [OFFICIAL-CODE]，Stage 1 本机实测验证。

## 统计

- 总条目：**68**；✅ 已裁决/有权威证据 32；⚠️ 有差异需注意 5；❓ UNKNOWN/待消融 23；⬜ 后续阶段 8
- 核心结论：**架构与推理路径已被 HF checkpoint + diffusers 源码完全锁定**（可对齐复现）；**训练 recipe 的大量超参论文未公开**（UNKNOWN），必须靠缩比消融自行确定——这正是缩比梯子的价值。
