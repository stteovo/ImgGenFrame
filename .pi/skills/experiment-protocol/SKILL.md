---
name: experiment-protocol
description: "Reproducible experiment record protocol for Z-Image reproduction. Use when running any training run, benchmark, ablation, probing script, or inference baseline — covers experiments/&lt;date&gt;-&lt;slug&gt;/ layout, seeds, environment capture, metrics, samples. Trigger words: experiment, training run, record, reproducibility, ablation, baseline."
---

# Experiment Protocol

Use before launching ANY run that produces numbers or images (training, probing, benchmarking, sampling baselines).

## Required record layout

Every run gets `experiments/<YYYYMMDD>-<slug>/` containing:

| File | Content |
| --- | --- |
| `record.md` | git commit hash, motivation and hypotheses (with evidence tags), config copy or frozen path under `configs/` |
| `environment.txt` | `uv pip freeze`, torch/CUDA/driver versions, GPU name, generated at launch |
| `dataset.md` | data source, version/hash, filter rules, sample count (when data is involved) |
| `metrics/` | loss curves and eval metrics as jsonl/csv |
| `samples/` | fixed prompts + fixed seeds outputs (images or arrays) |
| `checkpoints/` | resumable checkpoints + saving policy note |

## Rules

1. **Seeds**: fix and record all random sources — torch, numpy, python random, dataloader generator/worker seeds. A run without recorded seeds is invalid.
2. **Git commit**: record the exact commit; uncommitted changes must be committed or listed before launching. (Stage 0 pending decision: dedicated git repo in this directory.)
3. **Configs freeze**: once a config file is referenced by an experiment, it is immutable — new runs create new configs.
4. **Baseline before change**: any ablation must reference the baseline experiment directory and change exactly one variable.
5. **Hardware context**: note VRAM peak and throughput (it/s or tokens/s) for every training run — needed to plan the next ladder step.
