使用 evaluation skill。

建立统一 image generation evaluation framework。

至少支持：

- CLIPScore
- FID
- GenEval
- text rendering evaluation
- aesthetic score

建立固定 benchmark：

evaluation/prompts/

├── basic.jsonl
├── composition.jsonl
├── counting.jsonl
├── spatial.jsonl
├── Chinese_text.jsonl
├── English_text.jsonl
├── long_prompt.jsonl
└── style.jsonl

要求：

每个模型使用完全相同 prompts。

禁止：

为了某个模型表现好而修改 prompt。

输出：

evaluation/<model>/<experiment>/

metrics.json
samples/
report.md