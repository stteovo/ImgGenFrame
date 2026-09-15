# Learning Report — Caption Pipeline（多级 caption 与结构化抽取）

## 1. 做了什么

实现 `src/zimage/captioning/`：`schema.py`（分层 schema + observed/inferred + span/置信度）、`vlm.py`（VLM 协议 + MockVLM + transformers 适配点，不写死 Qwen-VL）、`pipeline.py`（质量过滤 → VLM → 结构化抽取 → 置信度 → record）、`io.py`（parquet 读写 + 版本化）。11 项测试通过。

## 2. 为什么需要

T2I 模型不是"看图生成图"，而是"从文字到图"。文字（caption）是模型学习图像分布的**条件入口**：caption 若不准确，模型学到的是错误的图文对应；caption 若太笼统（"一张照片"），模型没有可学习的信息增益。Z-Captioner 的核心判断是：**信息增益率决定训练效率上限**——高质量 caption 让同样数量的图片训练出更强的模型。

## 3. 为什么 caption quality > 图片数量

1. **样本复杂度 = 条件信息量**：一张图配一条精确 caption，模型学的是"文字→像素"的高价值映射；配一条垃圾 caption，等于白学。
2. **瓶颈在语义不在数量**：现代大模型的容量已能吸收海量数据，但若文字条件是噪声，再多图也只能学到模糊分布。
3. **长尾与文字渲染**：OCR 信息进 caption 直接绑定文字渲染能力 [PAPER §3.1 观察]——没有 OCR 的 caption，模型永远画不好图里的字。
4. **模拟用户提示**：真实用户 prompt 是"不完整、只关注局部"的，只用完整长 caption 训练会导致分布失配；多类 caption 收窄这一 gap。

## 4. 核心设计（对齐 Z-Captioner 方法论）

- **OCR-first CoT**：先转录图中全部文字（**保持原语言不翻译** [PAPER §3.1]），再写 caption。
- **5 类 caption**：long / medium / short / tags / 模拟用户提示 [PAPER §3.2]。
- **observed vs inferred**（AGENTS.md §20）：可见对象/OCR = observed（有 span、高置信）；风格/光照/构图 = inferred（主观，低置信）。禁止把推断当事实写进 caption。
- **span-grounded 抽取 + 置信度**：每个抽取项保存 source span、confidence、observed 标记、taxonomy 版本。
- **dataset versioning**：image_hash + prompt_version + taxonomy_version + dataset_version 绑定，支持重新 caption 且可追溯。

## 5. 当前实现要点

- VLM 是协议：`MockVLM`（确定性，离线测试）、`TransformersVLM`（真实接入点，prompt 模板版本化）——不写死 Qwen-VL [IMPLEMENTATION]。
- 质量过滤复用第 12 步 profiling（低熵/低分辨率 → `quality.passed=False`）。
- 置信度综合评分：observed 项权重高于 inferred [IMPLEMENTATION]。
- parquet 输出（pandas/pyarrow），嵌套字段 JSON 序列化。

## 6. 与官方 Z-Captioner 的差距

官方 Z-Captioner 权重未发布，我们是"方法论复刻 + 公开 VLM 替代" [IMPLEMENTATION]，**不等价**。已知差距：OCR 精度、CoT 质量、世界知识注入能力依赖所选 VLM。

## 7. 验证

- 分层：long>medium>short；tags/objects/attributes 存在。
- observed/inferred 分离：objects/OCR observed、style/lighting/composition inferred。
- OCR 原语言保留（中文不翻译）。
- span/置信度保存、schema dict 往返、parquet 往返、重 caption 版本可追溯、低熵质量标记。

## 8. Failure Modes

- 跳过 OCR-first，直接让 VLM 看图说话 → 文字渲染数据退化。
- 翻译 OCR 文本（论文明确禁止）。
- 把 inferred（风格/光照）当 observed 写进 caption。
- caption 未与图像 hash/版本绑定，更新后无法对应。
- 宣称"实现了 Z-Captioner"（官方未发布）。

## 9. 下一步

- 接入真实 VLM（Qwen-VL 类），冻结 prompt 模板版本，做质量抽查。
- 5 类 caption 的混合比例与训练消融（caption 粒度对文字渲染/指令遵循的影响）。
- 差异 caption（编辑指令 3 步 CoT，编辑阶段）。
