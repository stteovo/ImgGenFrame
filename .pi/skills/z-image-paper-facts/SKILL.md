---
name: z-image-paper-facts
description: Z-Image reproduction fact lookup. Use when any claim about Z-Image architecture, training, data pipeline, or paper/code discrepancies is needed — reads docs/reproduction/paper-facts.md and enforces evidence tags [PAPER]/[OFFICIAL-CODE]/[UNKNOWN]. Trigger on questions like "论文怎么说 RoPE/head 数/VAE/text encoder/flow matching/SFT/蒸馏".
---

# Z-Image Paper Facts

Use when reasoning about Z-Image (arXiv:2511.22699) facts, before writing or reviewing any implementation.

## Workflow

1. **Read first**: `docs/reproduction/paper-facts.md` is the single source of truth. Check it before answering any technical question about Z-Image.
2. **Tag discipline**: every stated fact must carry a tag — `[PAPER]` (cite section/table), `[OFFICIAL-CODE]` (cite file), `[OFFICIAL-CKPT]`, `[INFERENCE]` (show the reasoning chain), `[ASSUMPTION]`, `[EXPERIMENT]`. Never present guesses as facts.
3. **Unknown means blocked**: if a needed fact is in the `UNKNOWN` list (paper-facts §7), do NOT implement against it. Add an investigation task instead (Stage 1 reading or paper check).
4. **Discrepancies**: if the paper, official code (github.com/Tongyi-MAI/Z-Image), and HF checkpoint config disagree, the resolution goes into paper-facts §6 before any implementation decision. Known example: paper Table 2 says 32 attention heads, official code says 30 (head_dim=128) — currently unresolved, check HF `config.json` in Stage 1.
5. **Update, don't drift**: new verified facts must be added to paper-facts.md with source; if you find a conflict, update the fact base first, then discuss the decision.

## Primary sources

- Paper: arXiv:2511.22699 (HTML: https://arxiv.org/html/2511.22699v1)
- Official inference repo: https://github.com/Tongyi-MAI/Z-Image (notably `src/config/model.py`)
- Diffusers support: `ZImagePipeline` (merged PRs #12703, #12715)
- Checkpoints: HF `Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo`
- Distillation papers: arXiv:2511.22677 (Decoupled-DMD), arXiv:2511.13649 (DMDR)
