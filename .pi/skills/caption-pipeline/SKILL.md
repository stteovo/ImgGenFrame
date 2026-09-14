---
name: caption-pipeline
description: "Reproduction of Z-Image's multi-level captioning approach (paper §3 Z-Captioner: OCR-first CoT captioning, five caption types, world-knowledge conditioning, three-step difference captions) using public VLM substitutes. Use when building caption generation, caption mixing strategies, or editing-instruction synthesis."
---

# Caption Pipeline（多级 caption 管线）

## Purpose
复现 Z-Captioner 的**方法论**（OCR 优先、多级 caption、世界知识注入、差异 caption）以支持我们的多级/双语 caption 训练数据 [PAPER §3、§4.3 omni 阶段]；官方 Z-Captioner 权重未发布，只能用公开 VLM 替代 [IMPLEMENTATION]。

## Scope
- OCR-first CoT：先转录图中全部文字（保持原语言）再写 caption [PAPER §3.1]。
- 5 类 caption：long/medium/short/tags/模拟用户提示 [PAPER §3.2]；长短互补、模拟用户提示"不完整、只关注局部" [PAPER §3.2]。
- 世界知识注入：条件化 meta 信息减少幻觉 [PAPER §3.2]（公开实体识别可做 mini 版）。
- 差异 caption（编辑指令）：3 步 CoT = 详细 caption → 差异分析 → 指令合成 [PAPER §3.3]。
- 不包含：训练时 caption 的采样/混合比例（归 `training`）、图结构（`knowledge-graph`）。

## Inputs
- paper-facts.md §5 末段（Z-Captioner 事实）
- 选定 VLM 的能力清单与 prompt 模板（模板必须版本化）
- 现有 `src/zimage/data/` caption 相关代码（先检查再动手）

## Outputs
- caption 生成脚本（OCR-first 模板、5 类型、差异 caption）+ 测试
- 模板版本文件（与数据版本绑定）
- caption 质量抽查报告

## Rules
1. 改代码前先检查现有实现。
2. 明确宣称边界：我们是"复刻方法论 + 公开 VLM 替代"，**不得声称与 Z-Captioner 等价**。
3. OCR 优先顺序不可省（先转录→再描述 [PAPER §3.1]）；原文语言保留，不翻译 [PAPER §3.1]。
4. 5 类 caption 的生成 prompt 模板冻结版本，改动即新版本。
5. 每张图 captions 与图片 hash 绑定，保证可追溯。

## Verification
- 模板测试：OCR 输出不含翻译、出现在 caption 中；模拟用户提示确实"不完整"。
- 质量抽查：每类 caption 抽样人工检查（信息覆盖、幻觉、语言正确性），记录评分。
- 效果验证（训练后）：caption 粒度对文字渲染/指令遵循影响的 ablation（在 100M/300M 级）。

## Learning
- 练习：解释"为什么 OCR 信息进 caption 与文字渲染能力绑定" [PAPER §3.1 的实验观察]。
- 练习：长/短 caption 分别服务什么目标（长=精确映射，短/模拟提示=适配真实用户 [PAPER §3.2]）。
- 记录：我们的替代 VLM 与官方 Z-Captioner 的已知能力差距。

## Common failure modes
- 直接让 VLM "看图说话"，跳过 OCR-first 步骤 → 文字渲染数据退化。
- 翻译 OCR 文本（论文明确禁止 [PAPER §3.1]）。
- caption 未与图像版本绑定，数据更新后无法对应。
- 宣称"实现了 Z-Captioner"（官方未发布，只能标 [IMPLEMENTATION]）。
