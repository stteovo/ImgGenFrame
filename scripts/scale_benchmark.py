"""缩放基准：params / FLOPs 估算 / forward·backward 峰值 VRAM / 吞吐。

用法：
    .venv/bin/python scripts/scale_benchmark.py --config configs/tiny/model.yaml
    .venv/bin/python scripts/scale_benchmark.py --config configs/300m/model.yaml --h 32 --w 32
单卡 16GB 环境，显存工程组合实测（bf16 / 分辨率 / batch）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage.models import S3DiTConfig  # noqa: E402
from zimage.pipeline import ZImagePipeline  # noqa: E402
from zimage.training.config import load_model_config  # noqa: E402


def flops_estimate(cfg: S3DiTConfig, seq_total: int) -> float:
    """前向 FLOPs 解析估算 [IMPLEMENTATION]（MAC×2=FLOPs，忽略 refiner/embed 小项）。"""
    dim, ffn = cfg.hidden_size, cfg.ffn_dim
    n_blocks = cfg.depth + 2 * cfg.n_refiner_layers  # 主块 + refiner 块近似
    attn_per_token = 4 * dim * dim + 2 * seq_total * dim
    ffn_per_token = 6 * dim * ffn
    per_block = attn_per_token + ffn_per_token
    return 2.0 * n_blocks * seq_total * per_block  # 前向（MAC→FLOPs）


def measure(cfg: S3DiTConfig, batch: int, h: int, w: int, text_len: int, device: str) -> dict:
    pipe = ZImagePipeline(cfg).to(device, dtype=torch.bfloat16)
    n_params = pipe.model.num_parameters()
    latent = torch.randn(batch, 16, h, w, device=device, dtype=torch.bfloat16)
    t = torch.rand(batch, device=device)
    text = torch.randn(batch, text_len, 2560, device=device, dtype=torch.bfloat16)
    n_img = (h // 2) * (w // 2)
    seq_total = n_img + text_len

    # forward
    pipe.eval()
    with torch.no_grad():
        _ = pipe(latent, t, text)
        if device == "cuda":
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(5):
            _ = pipe(latent, t, text)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) / 5 * 1000
        fwd_vram = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    # forward + backward
    pipe.train()
    latent.requires_grad_(True)
    text.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.perf_counter()
    out = pipe(latent, t, text)
    out.pow(2).mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) * 1000
    bwd_vram = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    return {
        "params": n_params,
        "flops_fwd": flops_estimate(cfg, seq_total),
        "fwd_ms": round(fwd_ms, 2),
        "fwd_bwd_ms": round(fwd_bwd_ms, 2),
        "it_per_s": round(1000 / fwd_bwd_ms, 2),
        "fwd_vram_gb": round(fwd_vram, 3),
        "bwd_vram_gb": round(bwd_vram, 3),
        "seq_total": seq_total,
        "n_img_tokens": n_img,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--h", type=int, default=32)  # latent 高（256² 图 → 32）
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--text-len", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    r = measure(cfg, args.batch, args.h, args.w, args.text_len, args.device)
    print(f"=== {Path(args.config)} ===")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
