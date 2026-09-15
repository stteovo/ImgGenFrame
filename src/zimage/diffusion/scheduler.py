"""Euler ODE 采样器（velocity 预测）。

Z-Image sigma 表：sigma = linspace(1.0, 1/N, N)（末端补 0）[OFFICIAL-CODE]
`pipeline_z_image.get_default_z_image_sigmas`。
本项目沿用论文时间约定：t = 1 - sigma，采样时 t 从 0→1，v = x1 - x0。
Euler 步进：x_{t+dt} = x_t + dt · v（对速度场做一阶 ODE 积分）。
"""

from __future__ import annotations

import torch


class FlowEulerScheduler:
    """Flow Matching 的 Euler ODE 采样器。

    例：
        sched = FlowEulerScheduler(num_inference_steps=8)
        x = noise  # t=0
        for i in range(8):
            v = model(x, sched.timesteps[i])   # velocity
            x = sched.step(v, x, step_index=i)
        # x == 预测的数据 x1
    """

    def __init__(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigmas = torch.cat([sigmas, torch.zeros(1)])  # 末端 sigma=0（t=1）
        self.sigmas = sigmas  # [N+1] 递减 1.0 → 0
        self.timesteps = 1.0 - sigmas  # [N+1] 递增 0 → 1

    def step(self, velocity: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        """单步 Euler：x_{t+dt} = x_t + dt · v。velocity 为模型预测的速度场 v=x1-x0。"""
        t = self.timesteps[step_index]
        t_next = self.timesteps[step_index + 1]
        dt = t_next - t  # > 0
        return sample + dt * velocity
