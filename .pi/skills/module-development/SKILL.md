---
name: module-development
description: "Test-first module development workflow for the Z-Image reproduction codebase. Use when implementing or reviewing any core module (RoPE, norms, adaLN, S3-DiT blocks, flow matching, samplers, data pipeline) — enforces pytest-first, numerical parity with official implementations, scale-ladder gating, and the mentor seven-question review. Trigger words: implement module, write tests, RoPE, adaLN, DiT block, trainer, sampler."
---

# Module Development Workflow

Use when creating or reviewing any core module in `src/zimage/`.

## Order of operations (strict)

1. **Fact check**: load the `z-image-paper-facts` skill; confirm the design against `docs/reproduction/paper-facts.md`. Block on `[UNKNOWN]` facts.
2. **Tests first**: write pytest tests in `tests/` covering shapes, mathematical properties, edge cases, and — for math modules — numerical parity against a reference (diffusers implementation or known-correct library).
3. **Implement minimally** in the agreed repo layout (`src/zimage/models|encoders|vae|scheduling|data|training`). No speculative code for future stages.
4. **Parity harness**: math modules (RoPE, norms, adaLN, FM objective, schedulers/samplers) must match the official implementation numerically within tolerance; record max-error stats in the experiment record.
5. **Mentor review**: answer the seven questions below; the answers go into the stage's experiment record or docs notes.

## Scale-ladder gate

Before a module is used in training, confirm the current ladder step (100M → 300M → 1B → 3B → 6B) and that the previous step's exit criteria (see `docs/reproduction/roadmap.md`) are met. Never skip ahead to look complete.

## Mentor seven questions (mandatory per module)

1. What problem does this module solve in Z-Image?
2. Why this design (vs simpler alternatives)?
3. What is the math?
4. Why can't a plain/simpler implementation replace it?
5. Where does our implementation agree with the paper?
6. What is our own engineering choice (`[IMPLEMENTATION]`)?
7. How is it verified (done so far, still to do)?

## Code conventions

- No comments unless requested; keep modules small and composable.
- Config values referenced by experiments live in frozen files under `configs/`.
- Mark every engineering choice not dictated by the paper with `[IMPLEMENTATION]` in the stage notes.
