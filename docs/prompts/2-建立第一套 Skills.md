请现在创建下面这些 OpenCode Skills。

不要实现业务代码。

只创建 Skill 文档，并确保每个 Skill 都有明确的：

- Purpose
- Scope
- Inputs
- Outputs
- Rules
- Verification
- Learning section
- Common failure modes

目录：

.opencode/skills/

├── paper-analysis/
│   └── SKILL.md
│
├── reproduction-audit/
│   └── SKILL.md
│
├── s3-dit/
│   └── SKILL.md
│
├── rope-3d/
│   └── SKILL.md
│
├── flow-matching/
│   └── SKILL.md
│
├── vae-tokenization/
│   └── SKILL.md
│
├── conditioning/
│   └── SKILL.md
│
├── data-infrastructure/
│   └── SKILL.md
│
├── caption-pipeline/
│   └── SKILL.md
│
├── semantic-dedup/
│   └── SKILL.md
│
├── knowledge-graph/
│   └── SKILL.md
│
├── training/
│   └── SKILL.md
│
├── distributed-training/
│   └── SKILL.md
│
├── experiment/
│   └── SKILL.md
│
├── evaluation/
│   └── SKILL.md
│
├── distillation/
│   └── SKILL.md
│
└── learning/
    └── SKILL.md

要求：

- Skill 文档必须围绕 Z-Image reproduction。
- 不要泛泛描述 AI。
- 不要把未知信息编造成事实。
- 优先引用论文和官方代码。
- 每个 Skill 都要求 Agent 在修改代码前先检查现有实现。
- 每个 Skill 都必须包含 verification protocol。
- learning Skill 要求 Agent 在完成任务后给我解释关键知识点。

完成后输出：
1. 创建了哪些 Skill
2. 每个 Skill 的职责
3. Skill 之间的依赖关系

不要写模型代码。