"""Flow Matching 训练器：train_step / 训练循环 / checkpoint save·load·resume。

目标 [PAPER §4.3]：x_t=(1−t)x0+t·x1，v=x1−x0，MSE；t ~ logit-normal（σ=1.0 [ASSUMPTION]）。
checkpoint 含 model / optimizer / step / 各随机状态 / 配置，支持续训。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..diffusion.flow_matching import flow_matching_loss, linear_interpolate
from ..diffusion.timestep_sampling import sample_logit_normal_t, sample_uniform_t
from .config import TrainingSpec


class FlowMatchingTrainer:
    def __init__(
        self,
        model: nn.Module,
        spec: TrainingSpec,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.spec = spec
        self.device = torch.device(device)
        self.step = 0
        self.dataset_version = spec.dataset_version
        self.manifest_hash = ""
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            betas=spec.betas,
            weight_decay=spec.weight_decay,
        )

    # ------------------------------------------------------------------
    # 训练步
    # ------------------------------------------------------------------
    def _sample_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.spec.timestep_sampling == "uniform":
            return sample_uniform_t(batch_size, device=device, dtype=dtype)
        return sample_logit_normal_t(
            batch_size, device=device, dtype=dtype, std=self.spec.logit_normal_std
        )

    def _autocast_ctx(self):
        if self.spec.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def train_step(self, x1: torch.Tensor, text_feats: torch.Tensor) -> float:
        """x1 [B,C,H,W] 干净 latent，text_feats [B,T,D] → FM loss。"""
        self.model.train()
        x1 = x1.to(self.device)
        text_feats = text_feats.to(self.device)
        B = x1.shape[0]

        t = self._sample_t(B, self.device, x1.dtype)
        x0 = torch.randn_like(x1)
        x_t = linear_interpolate(x0, x1, t)
        v_target = x1 - x0

        with self._autocast_ctx():
            v_pred = self.model(x_t, t, text_feats)
            loss = flow_matching_loss(v_pred, v_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.spec.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.spec.grad_clip)
        self.optimizer.step()
        self.step += 1
        return loss.item()

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def set_dataset_info(self, dataset_version: str, manifest_hash: str) -> None:
        """把数据集版本写入训练器，随 checkpoint 保存（可追溯）。"""
        self.dataset_version = dataset_version
        self.manifest_hash = manifest_hash

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "dataset_version": self.dataset_version,
            "manifest_hash": self.manifest_hash,
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "training_spec": self.spec.__dict__,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.dataset_version = ckpt.get("dataset_version", "unknown")
        self.manifest_hash = ckpt.get("manifest_hash", "")
        torch.random.set_rng_state(ckpt["rng_cpu"])
        if ckpt.get("rng_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    def train(
        self,
        dataset,
        max_steps: int,
        log_interval: int = 10,
        save_path: str | Path | None = None,
        save_interval: int = 500,
    ) -> list[dict]:
        """在 dataset 上训练 max_steps 步，返回 [{step, loss}] 日志。"""
        logs: list[dict] = []
        for _ in range(max_steps):
            x1, text = dataset.sample_batch(self.spec.batch_size)
            loss = self.train_step(x1, text)
            if self.step % log_interval == 0 or self.step == 1:
                logs.append({"step": self.step, "loss": loss})
            if save_path is not None and self.step % save_interval == 0:
                self.save_checkpoint(save_path)
        return logs


def save_metrics(path: str | Path, metrics: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
