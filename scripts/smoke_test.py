"""最小可运行 Z-Image pipeline 冒烟测试。

验证：forward / backward / BF16 / 变分辨率 / 变序列长度，
并输出参数量、activation shape、峰值 VRAM、forward time。

用法：
    .venv/bin/python scripts/smoke_test.py                # 默认 tiny 配置，auto 设备
    .venv/bin/python scripts/smoke_test.py --device cuda   # 强制 GPU（测峰值 VRAM）
    .venv/bin/python scripts/smoke_test.py --hidden 3840 --depth 30 --heads 30 --ffn 10240  # 官方 6B 尺寸
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--text-len", type=int, default=8)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = S3DiTConfig(
        hidden_size=args.hidden,
        depth=args.depth,
        num_heads=args.heads,
        ffn_dim=args.ffn,
    )
    pipe = ZImagePipeline(config)
    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    pipe = pipe.to(device=device, dtype=dtype)

    n_params = pipe.model.num_parameters()
    print("=" * 60)
    print(pipe.model.summarize())

    latent = torch.randn(args.batch, 16, args.h, args.w, device=device, dtype=dtype)
    t = torch.rand(args.batch, device=device)
    text = torch.randn(args.batch, args.text_len, 2560, device=device, dtype=dtype)

    # --- forward ---
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    pipe.eval()
    with torch.no_grad():
        # warmup
        _ = pipe(latent, t, text)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(latent, t, text)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_time = time.perf_counter() - t0

    print(f"  forward output (velocity) shape: {tuple(out.shape)}")
    print(f"  dtype: {out.dtype}, device: {out.device}")
    print(f"  forward time: {fwd_time * 1000:.2f} ms")

    # --- backward ---
    pipe.train()
    latent.requires_grad_(True)
    text.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    out = pipe(latent, t, text)
    out.pow(2).mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  backward OK, grad finite: "
          f"latent={torch.isfinite(latent.grad).all().item()}, "
          f"text={torch.isfinite(text.grad).all().item()}")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  peak VRAM (backward): {peak:.3f} GB")
    else:
        print("  peak VRAM: N/A (CPU)")

    # --- 变分辨率 / 变序列长度冒烟 ---
    for h, w in [(8, 8), (16, 32), (32, 16)]:
        l = torch.randn(1, 16, h, w, device=device, dtype=dtype)
        o = pipe(l, t[:1], torch.randn(1, args.text_len, 2560, device=device, dtype=dtype))
        assert o.shape == (1, 16, h, w)
    for n in [1, 32, 77]:
        o = pipe(latent[:1], t[:1], torch.randn(1, n, 2560, device=device, dtype=dtype))
        assert o.shape == (1, 16, args.h, args.w)
    print(f"  variable resolution / sequence length: OK")
    print("=" * 60)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
