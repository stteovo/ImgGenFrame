"""Euler ODE 采样器测试。"""

import torch

from zimage.diffusion.scheduler import FlowEulerScheduler


def test_timesteps_and_sigmas():
    sched = FlowEulerScheduler(num_inference_steps=8)
    assert sched.sigmas.shape == (9,)
    assert sched.timesteps.shape == (9,)
    assert sched.sigmas[0] == 1.0
    assert sched.sigmas[-1] == 0.0
    assert sched.timesteps[0] == 0.0
    assert sched.timesteps[-1] == 1.0
    # sigma 递减、t 递增，且 t = 1 - sigma
    assert torch.allclose(sched.timesteps, 1.0 - sched.sigmas)


def test_euler_step_exact_value():
    sched = FlowEulerScheduler(num_inference_steps=4)
    sample = torch.randn(2, 3)
    velocity = torch.randn(2, 3)
    out = sched.step(velocity, sample, step_index=0)
    dt = sched.timesteps[1] - sched.timesteps[0]
    assert torch.allclose(out, sample + dt * velocity)


def test_full_denoise_recovers_data():
    # 恒定速度 v = x1 - x0 时 Euler 对线性 ODE 精确，N 步后应回到 x1
    torch.manual_seed(0)
    x0 = torch.randn(2, 4)  # 初始噪声
    x1 = torch.randn(2, 4)  # 目标数据
    v = x1 - x0  # 真实速度场（恒定）

    sched = FlowEulerScheduler(num_inference_steps=16)
    x = x0.clone()
    for i in range(sched.num_inference_steps):
        x = sched.step(v, x, step_index=i)
    assert torch.allclose(x, x1, atol=1e-5)


def test_step_dtype_preserved():
    sched = FlowEulerScheduler(num_inference_steps=8)
    sample = torch.randn(2, 3, dtype=torch.bfloat16)
    velocity = torch.randn(2, 3, dtype=torch.bfloat16)
    out = sched.step(velocity, sample, step_index=2)
    assert out.dtype == torch.bfloat16
