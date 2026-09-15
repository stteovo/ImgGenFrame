"""Z-Image-Tiny 训练入口（100M，Flow Matching，256²）。

用法：
    .venv/bin/python scripts/train_tiny.py --smoke          # 1/10/100 step 冒烟
    .venv/bin/python scripts/train_tiny.py --overfit        # 过拟合 16 张 latent，产出 loss 曲线+checkpoint
    .venv/bin/python scripts/train_tiny.py --steps 500      # 自定义步数
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage.diffusion.scheduler import FlowEulerScheduler  # noqa: E402
from zimage.models import S3DiTConfig  # noqa: E402
from zimage.pipeline import ZImagePipeline  # noqa: E402
from zimage.training.config import (  # noqa: E402
    load_data_config,
    load_model_config,
    load_training_config,
)
from zimage.training.dataset import SyntheticLatentDataset  # noqa: E402
from zimage.training.trainer import FlowMatchingTrainer, save_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build(config_dir: str = "tiny", device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg: S3DiTConfig = load_model_config(ROOT / "configs" / config_dir / "model.yaml")
    train_spec = load_training_config(ROOT / "configs" / config_dir / "training.yaml")
    data_spec = load_data_config(ROOT / "configs" / config_dir / "data.yaml")

    torch.manual_seed(train_spec.seed)
    model = ZImagePipeline(model_cfg).to(device)
    if train_spec.precision == "bf16" and device == "cuda":
        model = model.to(torch.bfloat16)
    dataset = SyntheticLatentDataset(
        num_images=data_spec.num_images,
        latent_size=data_spec.latent_size,
        latent_channels=data_spec.latent_channels,
        text_len=data_spec.text_len,
        cap_feat_dim=data_spec.cap_feat_dim,
        seed=0,
        device=device,
    )
    return model, train_spec, data_spec, dataset, device


def smoke(device: str) -> None:
    print("=== 1/10/100 step 冒烟测试 ===")
    for steps in (1, 10, 100):
        model, train_spec, data_spec, dataset, dev = build(device=device)
        trainer = FlowMatchingTrainer(model, train_spec, device=dev)
        logs = trainer.train(dataset, max_steps=steps, log_interval=steps)
        last = logs[-1]["loss"]
        assert torch.isfinite(torch.tensor(last)), f"{steps} step loss 非有限"
        print(f"  {steps:>3} steps → loss = {last:.4f}  OK")
    print("SMOKE PASSED")


def reconstruct(model, x1, text, num_steps: int = 16) -> torch.Tensor:
    """用训练好的模型从噪声 Euler 采样，重建干净 latent。"""
    model.eval()
    dtype = next(model.parameters()).dtype
    x1 = x1.to(dtype)
    text = text.to(dtype)
    B = x1.shape[0]
    sched = FlowEulerScheduler(num_steps)
    x = torch.randn_like(x1)
    with torch.no_grad():
        for i in range(num_steps):
            t_i = float(sched.timesteps[i])
            t = torch.full((B,), t_i, device=x1.device, dtype=dtype)
            v = model(x, t, text)
            x = sched.step(v, x, step_index=i)
    return x


def overfit(device: str, out_dir: str | None = None, steps: int | None = None) -> None:
    model, train_spec, data_spec, dataset, dev = build(device=device)
    trainer = FlowMatchingTrainer(model, train_spec, device=dev)
    max_steps = steps or train_spec.max_steps

    if out_dir is None:
        out_dir = f"experiments/{time.strftime('%Y%m%d')}-tiny-overfit"
    out = Path(out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"=== overfit {data_spec.num_images} 张 latent，{max_steps} steps ===")
    print(model.model.summarize())
    n_params = model.model.num_parameters()

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    logs = trainer.train(
        dataset,
        max_steps=max_steps,
        log_interval=train_spec.log_interval,
        save_path=out / "checkpoints" / "checkpoint.pt",
        save_interval=train_spec.save_interval,
    )
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    trainer.save_checkpoint(out / "checkpoints" / "checkpoint.pt")
    save_metrics(out / "metrics" / "training_metrics.json", logs)

    print(f"  first loss = {logs[0]['loss']:.4f}, last loss = {logs[-1]['loss']:.4f}")
    print(f"  it/s = {max_steps / elapsed:.2f}, elapsed = {elapsed:.1f}s")
    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  peak VRAM = {peak:.3f} GB")

    # --- 过拟合重建验证：从噪声恢复训练 latent ---
    model.eval()
    x1 = dataset.latents[:2]
    text = dataset.texts[:2]
    recon = reconstruct(model, x1, text, num_steps=16)
    mse = (recon - x1).pow(2).mean().item()
    print(f"  reconstruction MSE (16-step Euler, 2 样本) = {mse:.6f}")

    # --- 生成 loss 曲线 PNG（本地产物，不入库） ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [l["step"] for l in logs]
        losses = [l["loss"] for l in logs]
        plt.figure()
        plt.plot(steps, losses)
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"Z-Image-Tiny overfit (100M, {n_params:,} params)")
        plt.savefig(out / "metrics" / "loss_curve.png", dpi=120)
        plt.close()
        print(f"  loss curve saved: {out}/metrics/loss_curve.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (loss curve PNG skipped: {e})")

    print(f"=== 产出：{out} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.smoke:
        smoke(args.device)
    else:
        overfit(args.device, out_dir=args.out, steps=args.steps)
