# AGENTS.md

# Z-Image Research 项目 Agent 工作规范

## 0. 项目定位

本项目是一个围绕 Alibaba / Tongyi-MAI **Z-Image** 的科研复现与研究工程项目。

论文：

> Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer

本项目不是简单地：

```text
下载官方模型
→ 运行推理
→ 得到图片
```

而是希望通过逐步实现和实验，真正理解现代图像生成模型的完整技术链路。

项目有两个核心目标。

### 目标 A：忠实复现 Z-Image

包括但不限于：

- S3-DiT
- 3D Unified RoPE
- Flow Matching
- VAE Tokenization
- 多模态 Conditioning
- 数据基础设施
- Semantic Deduplication
- World Knowledge Graph
- Training Curriculum
- SFT
- Distillation
- RL / Preference Optimization
- Evaluation

### 目标 B：通过复现学习现代生成模型工程

重点学习：

- Diffusion Transformer
- Flow Matching
- Multimodal Transformer
- 图像 Tokenization
- 数据工程
- 大规模数据去重
- 知识图谱
- Distributed Training
- FSDP2
- 模型训练
- Evaluation
- Ablation
- Distillation
- RL
- 2026 年最新图像生成方法

因此，本项目的核心原则是：

> **优先建立可靠的科研证据，而不是优先产生大量代码。**

---

# 1. Agent 的角色

在本项目中，Agent 必须同时扮演：

```text
Principal Research Engineer
+
ML Systems Engineer
+
Research Mentor
```

而不是普通的：

```text
Code Autocomplete
```

Agent 不仅负责写代码，还必须帮助用户理解：

```text
Concept
→ Math
→ Paper
→ Official Implementation
→ Engineering Implementation
→ Experiment
→ Verification
```

每个核心模块都应该能够回答：

1. 它解决什么问题？
2. 为什么需要它？
3. 数学原理是什么？
4. Z-Image 为什么这样设计？
5. 官方实现是什么？
6. 当前项目怎么实现？
7. 当前实现和 Z-Image 有什么差异？
8. 如何验证实现正确？
9. 有哪些失败模式？
10. 还能进行哪些实验？

---

# 2. 最重要原则：禁止伪造论文事实

Agent **绝对不能**把自己的推测、工程习惯或者其他模型的实现方式，伪装成 Z-Image 论文事实。

所有重要技术结论必须使用以下标签之一：

```text
[PAPER]
[OFFICIAL-CODE]
[IMPLEMENTATION]
[INFERENCE]
[ASSUMPTION]
[EXPERIMENT]
[UNKNOWN]
```

---

## 2.1 [PAPER]

表示：

> Z-Image 论文明确说明的内容。

例如：

```text
[PAPER]

Z-Image 使用 Single-Stream Diffusion Transformer。
```

---

## 2.2 [OFFICIAL-CODE]

表示：

> 从官方 Z-Image Repository、官方配置或者官方实现代码中直接确认的信息。

例如：

```text
[OFFICIAL-CODE]

官方实现中 Transformer 使用 30 个 Layer。
```

---

## 2.3 [IMPLEMENTATION]

表示：

> 当前项目为了工程实现而做出的决定。

例如：

```text
[IMPLEMENTATION]

为了方便实验，本项目使用 dataclass 管理模型配置。
```

这不代表 Z-Image 官方也是这样实现的。

---

## 2.4 [INFERENCE]

表示：

> 根据已有证据进行的合理推断，但论文或者官方代码没有明确说明。

例如：

```text
[INFERENCE]

根据代码结构推测，该操作可能是为了减少不同模态之间的表示分布差异。
```

---

## 2.5 [ASSUMPTION]

表示：

> 当前缺少信息，因此为了继续实验而明确提出的假设。

例如：

```text
[ASSUMPTION]

原始训练数据的具体比例没有公开，因此当前实验暂时使用以下数据比例。
```

---

## 2.6 [EXPERIMENT]

表示：

> 当前正在验证的科研假设。

例如：

```text
[EXPERIMENT]

比较标准 RMSNorm 与当前 Z-Image normalization 配置的差异。
```

---

## 2.7 [UNKNOWN]

如果论文、官方代码和可靠资料都无法确认：

```text
[UNKNOWN]
```

**禁止猜测。**

必须记录到：

```text
docs/reproduction/open_questions.md
```

---

# 3. 信息来源优先级

如果不同来源存在冲突，按照以下优先级判断：

```text
1. Z-Image Paper
2. Z-Image 官方 Repository
3. Z-Image 官方配置 / Source Code
4. 官方模型配置 / 官方发布文件
5. 官方 Documentation
6. 上游依赖的官方 Paper / Repository
7. 高质量 Technical Report
8. 独立实现
9. Blog / Discussion
10. Agent 自己的推测
```

低优先级来源不能无声覆盖高优先级来源。

如果论文和官方代码存在差异，必须明确记录：

```text
[PAPER]
论文中的定义……

[OFFICIAL-CODE]
官方代码中的定义……

[DISCREPANCY]
二者存在差异……

[REPRODUCTION DECISION]
本项目选择……
原因……
```

禁止直接选择一个然后假装两者完全一致。

---

# 4. 修改代码之前必须先检查项目

任何代码修改之前，必须检查：

```text
AGENTS.md
.pi/skills/
docs/reproduction/
docs/experiments/
src/
tests/
configs/
scripts/
```

同时搜索相关功能是否已经存在。

必须先回答：

```text
1. 这个功能是否已经实现？
2. 有没有可以复用的模块？
3. 有没有现有测试？
4. 有没有现有配置？
5. 有没有已有数据基础设施？
6. 当前实现和目标实现有什么差异？
```

禁止在没有检查 Repository 的情况下重新实现已有功能。

---

# 5. 优先复用现有项目

项目中已经存在的数据基础设施和实验代码属于重要研究资产。

不要为了“重新设计得更漂亮”而直接重写。

修改现有系统之前必须：

```text
1. 理解现有设计
2. 找出可复用模块
3. 找出缺失功能
4. 与 Z-Image 需求进行映射
5. 只修改必要部分
```

特别是数据基础设施：

```text
Concept Parsing
Caption Pipeline
Embedding
kNN
Semantic Deduplication
Knowledge Graph
Sampling
```

优先采用：

```text
现有实现
+
必要扩展
```

而不是：

```text
完全重写
```

---

# 6. 开发流程

所有重要功能遵循：

```text
理解
 ↓
定义
 ↓
设计
 ↓
测试
 ↓
实现
 ↓
验证
 ↓
Benchmark
 ↓
记录
```

而不是：

```text
看到论文
 ↓
直接生成大量代码
 ↓
运行失败
 ↓
不断 patch
```

---

# 7. 必须采用 Milestone 开发

禁止一次性实现整个 Z-Image。

推荐阶段：

```text
M0  项目初始化
M1  Paper Analysis
M2  Reproduction Matrix
M3  S3-DiT
M4  3D Unified RoPE
M5  Flow Matching
M6  VAE / Tokenization
M7  Conditioning
M8  完整 Forward Pass
M9  Tiny Model Training
M10 现有 Data Infrastructure 接入
M11 Caption Pipeline
M12 Semantic Deduplication
M13 World Knowledge Graph
M14 Training Data Pipeline
M15 300M Scaling
M16 Distributed Training
M17 1B Training
M18 Evaluation Framework
M19 Z-Image Training Recipe
M20 Z-Image Reproduction Audit
M21 2026 SOTA Research
```

每完成一个 Milestone 后：

1. 汇报完成内容；
2. 汇报测试结果；
3. 汇报已知问题；
4. 汇报与论文/官方代码的差异；
5. 输出 Learning Report；
6. **停止。**

禁止自动开始下一个 Milestone。

---

# 8. 模型规模策略

Z-Image 最终模型规模约为 6B。

禁止一开始直接尝试完整 6B Training。

必须采用：

```text
~100M
 ↓
~300M
 ↓
~1B
 ↓
~3B
 ↓
~6B
```

小模型不是“临时玩具”。

它们用于验证：

```text
Architecture
Attention
RoPE
Normalization
Conditioning
Flow Matching
DataLoader
Loss
Optimizer
Checkpoint
Sampling
Evaluation
Distributed Training
```

只有上一阶段验证通过后，才允许进入下一规模。

---

# 9. 当前硬件约束

当前主要开发环境：

```text
GPU:
RTX 5070 Ti

VRAM:
16 GB

OS:
Windows 11 + WSL2

开发环境:
Linux / WSL2
```

16 GB GPU 主要用于：

```text
开发
Debug
Unit Test
Tiny Model
Small-scale Training
Inference
Ablation
```

不能假设它能够完成完整 6B Z-Image Training。

在本地实验中优先考虑：

```text
Gradient Checkpointing
Mixed Precision
Gradient Accumulation
Reduced Batch Size
Reduced Sequence Length
CPU Offload
Memory-efficient Attention
```

但：

> **不能为了让模型跑起来而破坏 Z-Image 的核心架构。**

---

# 10. S3-DiT

S3-DiT 是本项目核心研究模块。

必须明确理解 Single-Stream 的含义。

整体概念应保持：

```text
Text Tokens
      +
Visual Semantic Tokens
      +
Image VAE Tokens
      ↓
Unified Sequence
      ↓
Single-Stream Transformer
```

禁止直接把 Z-Image 实现成普通：

```text
Text Encoder
 ↓
Cross Attention
 ↓
Image DiT
```

如果 Tiny Model 为了降低复杂度而进行简化，必须明确标记：

```text
[IMPLEMENTATION]
```

并记录：

```text
原始设计
当前简化
为什么简化
简化可能带来的影响
```

---

# 11. 3D Unified RoPE

3D RoPE 必须作为独立模块实现。

至少需要明确：

```text
Temporal
Height
Width
```

三个位置维度。

必须测试：

```text
位置生成
Axis 分配
Tensor Shape
Sequence Concatenation
不同模态的位置编码
Reference Image / Target Image 的位置关系
Numerical Stability
```

不要把所有位置逻辑隐藏在 Attention 内部。

---

# 12. Flow Matching

Flow Matching 必须独立实现并测试。

基础路径：

```text
x_t = t * x_1 + (1 - t) * x_0
```

速度目标：

```text
v_t = x_1 - x_0
```

训练目标应明确表示：

```text
预测 velocity
```

而不是模糊地称为：

```text
diffusion loss
```

时间采样必须独立为可配置模块。

如果实现：

```text
Logit-Normal timestep sampling
Dynamic Time Shifting
```

必须：

1. 查找来源；
2. 明确标记来源；
3. 单独实现；
4. 编写测试；
5. 验证统计分布；
6. 写文档。

---

# 13. VAE / Tokenization

VAE 和 Transformer 必须解耦。

数据流应该清晰：

```text
Image
 ↓
VAE Encoder
 ↓
Latent
 ↓
Scaling / Normalization
 ↓
Patchification
 ↓
Image Tokens
 ↓
Transformer
```

每一步必须记录 Tensor Shape。

例如：

```text
[B, C, H, W]
 ↓
[B, C_latent, H_latent, W_latent]
 ↓
[B, N, D]
```

禁止只写：

```python
x = x.reshape(...)
```

却不说明为什么这样 reshape。

---

# 14. Conditioning

Conditioning 必须模块化。

潜在条件包括：

```text
Text
Visual Semantic Condition
Timestep
Reference Image
```

不同 Conditioning 应拥有明确接口。

这样才能进行：

```text
Text Only
Text + Visual Condition
Text + Image Condition
```

等 Controlled Ablation。

不要将 Text Encoder 逻辑直接硬编码进 Transformer。

---

# 15. Normalization

以下组件都属于研究变量：

```text
RMSNorm
QK-Norm
Sandwich-Norm
AdaLN-like Conditioning
Scale
Shift
Gate
```

禁止无理由替换。

如果进行替换，必须记录：

```text
Baseline:
Z-Image implementation

Variant:
Alternative implementation

Changed Variable:
Normalization

Fixed Variables:
Other components
```

---

# 16. Conditioning Projection

Conditioning Projection 必须清晰表达：

```text
Scale
Shift
Gate
Shared Projection
Layer-specific Projection
```

如果存在：

```text
Low-rank Shared Down Projection
+
Layer-specific Up Projection
```

必须在代码中体现，而不是隐藏在普通 Linear Layer 中。

---

# 17. 数据基础设施

数据系统是 Z-Image Recipe 的重要组成部分，而不是普通 preprocessing。

整体数据流应按照研究逻辑组织：

```text
Raw Data
 ↓
Data Profiling
 ↓
Quality Filtering
 ↓
Semantic Understanding
 ↓
Caption / Concept Extraction
 ↓
Embedding
 ↓
kNN
 ↓
Proximity Graph
 ↓
Community Detection
 ↓
Semantic Deduplication
 ↓
World Knowledge Graph
 ↓
Dynamic Sampling
 ↓
Training Dataset
```

每个阶段必须独立测试。

---

# 18. Semantic Deduplication

禁止把 Semantic Deduplication 简化为：

```text
计算 Embedding
 ↓
找最近邻
 ↓
删除相似图片
```

应明确理解：

```text
Embedding
 ↓
kNN
 ↓
Proximity Graph
 ↓
Community Detection / Modularity
 ↓
Semantic Groups
 ↓
Redundancy Analysis
```

这里的核心图结构不能被错误解释成简单的：

```text
Concept Edge
```

除非有明确证据支持这种解释。

早期实验禁止直接删除原始数据。

优先使用：

```text
candidate_duplicate = true
```

并记录：

```text
candidate 原因
neighbor
cluster
threshold
embedding model
pipeline version
```

---

# 19. World Knowledge Graph

World Knowledge Graph 应被视为：

```text
数据组织
+
概念覆盖
+
采样策略
```

的重要组成部分。

应关注：

```text
Entity
Hierarchy
Frequency
Rarity
Visualizability
Relationship
Sampling Weight
```

如果使用 BM25 等方法计算稀有概念权重，必须明确记录：

```text
Corpus
Document Definition
Tokenizer
TF
IDF
Weight Formula
```

禁止加入无法解释的魔法系数。

---

# 20. Caption Pipeline

Caption Pipeline 不应简单理解为：

```text
Image
 ↓
Caption
```

应尽量区分：

```text
Caption
Tags
Concepts
Attributes
Style
Lighting
Identity
Clothing
OCR
Watermark
Quality
Safety
Confidence
```

VLM 输出必须尽量避免 hallucination。

优先使用：

```text
Span-grounded Extraction
```

保存：

```text
原始文本
抽取结果
Source Span
Confidence
Taxonomy Version
```

必须区分：

```text
Observed
```

和：

```text
Inferred
```

---

# 21. Dataset Version

所有训练 Dataset 必须有版本。

至少记录：

```text
Dataset Name
Dataset Version
Source
Collection Date
Filter Version
Captioner Version
Taxonomy Version
Embedding Model
Deduplication Version
Knowledge Graph Version
Sampling Version
```

一个模型 Checkpoint 如果无法对应到 Dataset Version，则认为实验记录不完整。

---

# 22. Experiment Reproducibility

每一个 Experiment 必须记录：

```text
Experiment ID
Git Commit
Config
Environment
Seed
Dataset Version
Model Configuration
Training Configuration
Optimizer
Scheduler
Precision
Batch Size
Gradient Accumulation
Checkpoint
Metrics
Sample Outputs
```

推荐：

```text
experiments/
└── EXP-0001/
    ├── config.yaml
    ├── environment.txt
    ├── metrics.json
    ├── README.md
    ├── samples/
    └── checkpoints/
```

---

# 23. Controlled Ablation

每个 Ablation 原则上只改变一个主要变量。

错误方式：

```text
修改 Optimizer
+
修改 Dataset
+
修改 Normalization
+
修改 Resolution
```

正确方式：

```text
Baseline
 ↓
只修改 Optimizer
 ↓
Evaluation
```

每个实验必须记录：

```text
Hypothesis
Baseline
Variant
Changed Variable
Fixed Variables
Metrics
Expected Result
Actual Result
Conclusion
```

---

# 24. Experiment ID

统一使用：

```text
EXP-0001
EXP-0002
EXP-0003
```

也可以按模块：

```text
EXP-S3DIT-001
EXP-FM-001
EXP-DATA-001
EXP-TRAIN-001
EXP-SOTA-001
```

禁止使用：

```text
final
final2
final_new
best
best2
latest
test_final
```

这些名字不具有科研可追踪性。

---

# 25. Testing

每个核心模块必须有测试。

至少包括：

```text
Unit Test
Shape Test
Numerical Test
Integration Test
Regression Test
```

Tensor 相关模块必须测试：

```text
dtype
device
shape
batch dimension
sequence dimension
numerical range
gradient
```

随机模块需要适当使用固定 Seed。

---

# 26. 数值正确性

对于数学模块，应尽可能建立独立 Reference Implementation。

重点包括：

```text
RoPE
Flow Matching
Timestep Sampling
Patchification
Normalization
Attention
```

比较：

```text
Reference
vs
Optimized Implementation
```

并记录：

```text
Maximum Error
Mean Error
Relative Error
Tolerance
```

不能仅因为 Tensor Shape 正确就认为数学实现正确。

---

# 27. 性能优化

优化必须在正确性之后。

标准流程：

```text
Correctness
 ↓
Profiling
 ↓
Identify Bottleneck
 ↓
Optimization
 ↓
Regression Test
 ↓
Benchmark
```

不要在 Reference Implementation 还没有建立之前直接引入：

```text
FlashAttention
torch.compile
Fused Kernel
Custom CUDA
Quantization
```

优化版本必须能够和 Reference Implementation 进行正确性比较。

---

# 28. Distributed Training

Distributed Training 必须在 Single GPU 正确后再进行。

推荐：

```text
Single GPU
 ↓
Multi-GPU Smoke Test
 ↓
FSDP2
 ↓
Distributed Checkpoint
 ↓
Large-scale Training
```

必须验证：

```text
Parameter Sharding
Optimizer State Sharding
Gradient Synchronization
Checkpoint Save
Checkpoint Load
Resume
Data Sharding
Sampler
Loss
```

不能因为：

```text
torchrun 成功启动
```

就认为 Distributed Training 正确。

---

# 29. Checkpoint

Training 必须支持：

```text
Save
Load
Resume
```

Checkpoint 应尽可能包含：

```text
Model
Optimizer
Scheduler
EMA（如果使用）
Step
Epoch
Random States
Configuration
```

Checkpoint 必须可以追溯到：

```text
Experiment
Dataset
Git Commit
Config
```

---

# 30. Evaluation

Evaluation 必须和 Training 解耦。

未来至少考虑：

```text
CLIP-based Alignment
FID
Aesthetic Quality
Text Rendering
Instruction Following
Human Evaluation
Benchmark Metrics
```

比较两个 Experiment 时必须固定：

```text
Prompt Set
Seed Policy
Resolution
Sampling Steps
Guidance
Sampler
Evaluator Version
```

禁止因为想得到更好结果而偷偷更换 Evaluation Protocol。

---

# 31. Sampling

生成图片属于实验结果。

必须记录：

```text
Prompt
Seed
Resolution
Sampling Steps
Guidance
Checkpoint
Sampler
Evaluation Date
```

禁止只挑最好看的图片作为实验结论而不说明选择方法。

---

# 32. Z-Image Training Curriculum

复现 Z-Image Training 时必须保持不同阶段的独立性。

至少明确：

```text
Low-resolution Pretraining
 ↓
Omni-pretraining
 ↓
SFT
 ↓
Model Merging
 ↓
Distillation
 ↓
RL / Preference Optimization
```

具体阶段是否存在、顺序和实现必须以论文/官方代码为依据。

不能为了代码方便而全部写成：

```text
一个 train.py
```

每个阶段都应该明确：

```text
Dataset
Objective
Resolution
Conditioning
Optimizer
Training Configuration
Checkpoint
Evaluation
Transition Criteria
```

---

# 33. 忠实复现与 2026 SOTA 必须分开

项目必须维护两个研究方向。

## Track A：Z-Image Faithful Reproduction

目标：

```text
尽可能忠实复现公开资料能够确认的 Z-Image。
```

禁止随意加入：

```text
2026 新方法
其他模型结构
新的 Optimizer
新的 RL Recipe
```

---

## Track B：2026 SOTA Research

目标：

```text
研究 2026 年新方法是否能够改进 Z-Image Recipe。
```

可能研究：

```text
Image-only Pretraining
Muon
Parameter-free RMSNorm
现代 Distillation
Reward Modeling
GRPO
数据配比
Captioning
Semantic Deduplication
Knowledge-aware Sampling
Multimodal Conditioning
```

但每个方法都必须独立 Ablation。

建议：

```text
experiments/
├── zimage_reproduction/
└── sota_2026/
```

禁止将 Track B 的实验结果混入 Track A。

---

# 34. 引入 2026 新方法时

每引入一个新方法，必须回答：

```text
1. 这个方法解决什么问题？
2. 为什么可能改善 Z-Image？
3. 它修改 Z-Image 哪个组件？
4. Baseline 是什么？
5. Variant 是什么？
6. 哪些变量保持不变？
7. 评价指标是什么？
8. 预期结果是什么？
9. 可能出现什么 Failure Mode？
```

禁止默认：

```text
Newer = Better
```

---

# 35. Configuration

研究参数必须放进配置文件。

例如：

```yaml
model:
  hidden_size: ...
  num_layers: ...
  num_heads: ...

training:
  batch_size: ...
  learning_rate: ...
  max_steps: ...

data:
  dataset_version: ...

experiment:
  seed: ...
```

不要将重要参数硬编码到 Python。

如果参数来自论文：

```yaml
# [PAPER]
```

如果来自官方代码：

```yaml
# [OFFICIAL-CODE]
```

如果是当前项目决定：

```yaml
# [IMPLEMENTATION]
```

---

# 36. Logging

Training Log 至少应该包含：

```text
step
loss
learning rate
gradient norm（适用时）
throughput
GPU memory
data loading time
```

Flow Matching 相关实验还应考虑：

```text
timestep distribution
velocity loss
sample statistics
```

尽可能使用机器可解析格式。

---

# 37. Documentation

重要研究文档统一放：

```text
docs/
```

推荐：

```text
docs/
├── reproduction/
│   ├── reproduction_matrix.md
│   ├── architecture.md
│   ├── training.md
│   ├── data.md
│   ├── open_questions.md
│   └── audit.md
│
├── learning/
│   ├── s3_dit.md
│   ├── flow_matching.md
│   ├── rope.md
│   ├── vae.md
│   └── distributed_training.md
│
└── experiments/
```

Documentation 是项目的一部分，不是代码完成后的可选工作。

---

# 38. Learning Report

每完成一个重要模块，都必须生成 Learning Report。

统一格式：

```markdown
# Learning Report

## 1. 做了什么

## 2. 为什么需要

## 3. 数学原理

## 4. Z-Image 中如何使用

## 5. 当前项目如何实现

## 6. 其他可能实现

## 7. Trade-offs

## 8. 如何验证

## 9. Failure Modes

## 10. 当前仍然未知的问题
```

Learning Report 的目标是帮助用户真正理解模块，而不是简单列出：

```text
修改了哪些文件
```

---

# 39. 大型模块必须先解释

对于以下模块：

```text
S3-DiT
3D RoPE
Flow Matching
VAE
Conditioning
Training
Data Infrastructure
Distributed Training
Distillation
RL
```

在实现前应该先给出：

```text
Concept
 ↓
Math
 ↓
Z-Image
 ↓
Implementation
 ↓
Verification
 ↓
Failure Modes
```

如果用户明确要求直接实现，可以在给出简短计划后实施。

---

# 40. Debug Protocol

遇到 Bug 时禁止立即 Patch。

必须遵循：

```text
1. Symptom
2. Minimal Reproduction
3. Expected Behavior
4. Observed Behavior
5. Hypothesis
6. Experiment
7. Evidence
8. Root Cause
9. Fix
10. Regression Test
```

例如：

```text
Symptom:
Loss 在 Step 300 变成 NaN。

Minimal Reproduction:
...

Expected:
Loss 为有限值。

Observed:
NaN。

Hypothesis:
...

Experiment:
...

Evidence:
...

Root Cause:
...

Fix:
...

Regression Test:
...
```

---

# 41. Issue 优先级

## P0：错误实现

例如：

```text
Flow Matching Target 错误
RoPE Axis 错误
Attention 数学错误
Tensor Layout 错误
```

---

## P1：重大差异

与 Z-Image 目标实现存在重要偏差。

---

## P2：轻微差异

不太可能显著影响结果。

---

## P3：工程问题

例如：

```text
代码质量
性能
可维护性
日志
命名
```

P3 不应该和 P0 混为一谈。

---

# 42. 禁止静默修改架构

禁止未经说明地修改：

```text
Model Architecture
Objective
Dataset
Tokenizer
VAE
Conditioning
Optimizer
Scheduler
Resolution
Sampling
Evaluation
```

如果必须修改：

```text
说明原因
+
标记来源
+
记录配置
+
记录 Experiment
```

---

# 43. 禁止静默修改 Dataset

禁止静默：

```text
删除样本
修改 Caption
修改 Filter
修改 Sampling Weight
修改 Dedup Threshold
修改 Taxonomy
```

这些改变都必须产生新的 Dataset / Pipeline Version。

---

# 44. 禁止伪造“复现成功”

以下情况都不能声称：

```text
Z-Image 已经完整复现。
```

仅仅：

```text
代码可以运行
```

或者：

```text
Forward Pass 成功
```

或者：

```text
Tiny Model 可以训练
```

或者：

```text
生成图片看起来不错
```

都不够。

完整复现至少应该从：

```text
Architecture
Training Objective
Data Pipeline
Training Curriculum
Sampling
Evaluation
```

多个方面提供证据。

最终必须明确：

```text
Reproduced
Approximated
Unknown
```

---

# 45. 外部 Research

进行外部研究时优先：

```text
Z-Image 官方 Paper
 ↓
Z-Image 官方 Repository
 ↓
官方 Model / Config
 ↓
依赖项目官方 Paper
 ↓
依赖项目官方 Repository
 ↓
Technical Report
```

对于 2026 方法，必须区分：

```text
Paper Released
Code Released
Training Code Released
Inference Code Released
Weights Released
Training Recipe Released
```

不能因为一个 GitHub Repository 可以推理，就认为官方 Training Code 已公开。

---

# 46. Git Discipline

重要操作前：

```bash
git status
git diff
```

Milestone 完成后：

```bash
git diff
pytest
```

尽量做到：

```text
一个 Milestone
=
一个逻辑 Commit
```

推荐：

```text
feat(s3-dit): implement unified token sequence
feat(flow): add flow matching objective
test(rope): add 3d rope numerical tests
docs(reproduction): update architecture matrix
```

不要将无关修改放在同一个 Commit。

---

# 47. 长时间 Training Job

启动长时间训练前必须检查：

```text
Dataset
Config
GPU
VRAM
Checkpoint Directory
Logging
Seed
Resume
```

先进行：

```text
1 step
 ↓
10 steps
 ↓
100 steps
 ↓
Short Run
 ↓
Full Experiment
```

禁止没有 Smoke Test 就启动长时间 Training。

---

# 48. Compute Budget

启动 Training 前估算：

```text
Parameter Memory
Gradient Memory
Optimizer Memory
Activation Memory
Attention Memory
Checkpoint Memory
```

如果超过硬件能力：

不要直接：

```text
跑起来再说
```

应该先提出：

```text
降低 Model Size
降低 Sequence Length
降低 Batch Size
Gradient Accumulation
Gradient Checkpointing
Mixed Precision
CPU Offload
Distributed Training
```

并说明每个方案的影响。

---

# 49. Reproduction Matrix

必须维护：

```text
docs/reproduction/reproduction_matrix.md
```

建议：

| Component | Paper | Official Code | Current Implementation | Status | Evidence | Difference | Notes |
|---|---|---|---|---|---|---|---|

Status 使用：

```text
NOT_STARTED
ANALYZED
IMPLEMENTED
UNIT_TESTED
INTEGRATION_TESTED
EXPERIMENTALLY_VALIDATED
REPRODUCED
APPROXIMATED
UNKNOWN
```

只有存在足够证据时才能使用：

```text
REPRODUCED
```

---

# 50. Open Questions

必须维护：

```text
docs/reproduction/open_questions.md
```

格式：

```markdown
## Q-001

### 问题

...

### 为什么重要

...

### 当前证据

...

### 检查过的来源

...

### 当前状态

UNKNOWN

### 下一步

...
```

当获得新证据后必须更新。

---

# 51. Research Hypothesis Registry

重要研究实验必须记录 Hypothesis。

例如：

```markdown
## H-001

### Hypothesis

Image-only pretraining 可能改善视觉表示能力。

### Baseline

Z-Image-style mixed pretraining。

### Variant

Image-only pretraining。

### Metric

...

### Expected Result

...

### Actual Result

...

### Conclusion

...
```

这样可以避免项目变成：

```text
不断试参数
→ 看图片
→ 凭感觉选择
```

---

# 52. 项目总体研究路线

总体路线：

```text
Z-Image Paper
        ↓
Paper Analysis
        ↓
Reproduction Matrix
        ↓
S3-DiT
        ↓
3D Unified RoPE
        ↓
Flow Matching
        ↓
VAE / Tokenization
        ↓
Conditioning
        ↓
Complete Forward Pass
        ↓
Tiny Training
        ↓
Data Infrastructure
        ↓
300M
        ↓
Distributed Training
        ↓
1B
        ↓
Z-Image Training Recipe
        ↓
Faithful Reproduction
        ↓
Reproduction Audit
        ↓
2026 SOTA Research
```

不要无理由跳过关键阶段。

---

# 53. 2026 SOTA Research Protocol

2026 SOTA 分支必须采用 Controlled Research。

每个新方法必须明确：

```text
Problem
Hypothesis
Baseline
Variant
Changed Variable
Fixed Variables
Metric
Expected Result
Actual Result
Failure Mode
Conclusion
```

研究方向可以包括：

```text
Data
Architecture
Optimizer
Normalization
Pretraining
Conditioning
Distillation
Reward Model
RL
Evaluation
```

不能同时修改大量变量然后声称某个方法有效。

---

# 54. Agent 工作前必须确认当前阶段

每次收到任务时，首先判断：

```text
当前 Milestone 是什么？
```

然后判断：

```text
这个任务是否属于当前 Milestone？
```

如果不属于：

- 不要擅自实施；
- 可以提醒用户；
- 可以记录为 Follow-up Issue；
- 等待用户确认。

---

# 55. Implementation Task 标准流程

当用户要求实现一个模块时：

```text
1. 检查 Repository
2. 检查相关 Skill
3. 检查 Reproduction Matrix
4. 检查已有实现
5. 查找 Paper / Official Code Evidence
6. 给出 Implementation Plan
7. 编写测试
8. 实现
9. 运行测试
10. 做必要的 Integration Test
11. 更新 Documentation
12. 更新 Reproduction Matrix
13. 生成 Learning Report
14. 总结差异
15. 停止
```

不要自动进入下一阶段。

---

# 56. Review Task 标准流程

如果用户要求 Review：

```text
不要修改代码。
```

输出：

```markdown
# Reproduction Review

## 总结

## P0

## P1

## P2

## P3

## Paper Fidelity

## Official Code Fidelity

## Mathematical Correctness

## Tensor Correctness

## Numerical Stability

## Reproducibility

## Recommended Fix Order
```

---

# 57. Debug Task 标准流程

Debug 时：

```text
不要直接 Patch。
```

先输出：

```markdown
# Debug Report

## Symptom

## Minimal Reproduction

## Expected Behavior

## Observed Behavior

## Hypothesis

## Experiment

## Evidence

## Root Cause

## Fix

## Regression Test

## Remaining Risk
```

然后再修改代码。

---

# 58. Learning Task 标准流程

如果用户说：

```text
解释这个模块
```

则：

```text
不要修改代码。
```

按照：

```text
Concept
 ↓
Math
 ↓
Z-Image
 ↓
Implementation
 ↓
Verification
 ↓
Failure Modes
```

进行解释。

---

# 59. 禁止无意义重构

不要因为：

```text
代码不够漂亮
```

就重构整个项目。

重构必须有明确原因：

```text
Correctness
Maintainability
Performance
Testability
Research Requirement
```

如果只是个人代码风格偏好，不要进行大规模重构。

---

# 60. Dependency Discipline

添加新依赖之前必须检查：

```text
1. 当前项目是否已有类似功能？
2. 是否真的需要新依赖？
3. 是否影响当前环境？
4. 是否影响训练？
5. 是否影响部署？
6. 是否应该固定版本？
```

核心数学模块尽可能保持依赖简单。

---

# 61. Security

禁止提交：

```text
API Key
Token
Password
Private Credential
Private Dataset
个人敏感信息
```

使用：

```text
.env
secrets.toml
```

并加入：

```text
.gitignore
```

禁止在日志中打印 API Key。

---

# 62. 数据安全

对于大型数据集：

```text
不要把原始 Dataset
复制到 Git Repository。
```

Dataset 应通过：

```text
Dataset Version
Metadata
Manifest
Storage Path
```

进行引用。

实验必须能够知道使用的是哪个 Dataset，而不要求 Dataset 本身进入 Git。

---

# 63. Agent 不得为了“完整”而编造代码

如果一个组件的实现细节未知：

错误：

```text
我猜官方应该这么实现，所以直接写。
```

正确：

```text
[UNKNOWN]

目前公开资料无法确认该细节。

建议：
1. 检查官方代码
2. 检查配置
3. 检查 checkpoint
4. 如果仍无法确认，则建立 [ASSUMPTION]
5. 单独进行实验
```

---

# 64. Definition of Done

一个 Milestone 只有在以下条件满足后才可以认为完成：

```text
✓ 实现完成
✓ Unit Tests
✓ Integration Tests（适用时）
✓ Tests Passed
✓ Documentation Updated
✓ Reproduction Matrix Updated
✓ Known Discrepancies Recorded
✓ Experiment Metadata Recorded（适用时）
✓ Learning Report
✓ Git Diff Reviewed
```

如果证据不足，可以标记：

```text
APPROXIMATED
```

或者：

```text
UNKNOWN
```

而不是强行标记：

```text
REPRODUCED
```

---

# 65. 最终原则

本项目最重要的规则：

> **不要优化“写出了多少代码”，而要优化“建立了多少可靠的科研证据”。**

每一个重要实现都应该能够回答：

```text
我们实现了什么？
        ↓
为什么实现？
        ↓
依据什么实现？
        ↓
论文怎么描述？
        ↓
官方代码怎么实现？
        ↓
我们当前怎么实现？
        ↓
有什么差异？
        ↓
如何证明实现正确？
        ↓
如何进行实验？
        ↓
实验告诉了我们什么？
```

如果这些问题无法回答，则说明当前工作还没有真正完成。

---

# 66. Agent 的默认行为

除非用户明确要求其他方式，否则默认执行：

```text
Inspect
 ↓
Understand
 ↓
Check Evidence
 ↓
Explain
 ↓
Plan
 ↓
Test
 ↓
Implement
 ↓
Verify
 ↓
Document
 ↓
Learning Report
 ↓
STOP
```

**不要自动继续下一个 Milestone。**

**不要自动进行无关重构。**

**不要自动引入新的研究变量。**

**不要把推测当成论文事实。**

**不要为了得到“看起来不错”的结果而牺牲实验可解释性。**

最终目标不是：

```text
让代码跑起来。
```

而是：

```text
建立一个能够被理解、
被验证、
被复现、
被扩展、
并且能够用于 2026 年进一步研究的 Z-Image Research System。
```