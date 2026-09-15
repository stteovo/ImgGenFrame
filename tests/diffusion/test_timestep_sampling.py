"""时间采样与动态 shift 测试：logit-normal 统计分布 / shift 数值对齐。"""

import math

import pytest
import torch

from zimage.diffusion.timestep_sampling import (
    apply_time_shift,
    calculate_shift,
    sample_logit_normal_t,
    sample_uniform_t,
)


def _normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def test_uniform_t_in_unit_interval():
    torch.manual_seed(0)
    t = sample_uniform_t(10000)
    assert t.shape == (10000,)
    assert (t >= 0).all() and (t < 1).all()


def test_logit_normal_shape_and_range():
    t = sample_logit_normal_t(1000)
    assert t.shape == (1000,)
    assert (t > 0).all() and (t < 1).all()


def test_logit_normal_deterministic_seed():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(sample_logit_normal_t(100, generator=g1), sample_logit_normal_t(100, generator=g2))


def test_logit_normal_concentrates_in_middle():
    # logit-normal（σ=1）把概率集中在中间 t，两端稀疏
    torch.manual_seed(0)
    t = sample_logit_normal_t(200_000)
    p_mid = ((t > 0.4) & (t < 0.6)).float().mean()
    p_edges = ((t < 0.1) | (t > 0.9)).float().mean()
    assert p_mid > p_edges


def test_logit_normal_matches_theoretical_cdf():
    # 经验 CDF ≈ Φ(logit(t))（mean=0, std=1 的 logit-normal）
    torch.manual_seed(0)
    t = sample_logit_normal_t(300_000)
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        empirical = (t < q).float().mean()
        theoretical = _normal_cdf(torch.logit(torch.tensor(q)))
        assert abs(empirical - theoretical) < 0.02, f"q={q}: {empirical} vs {theoretical}"


def test_logit_normal_mean_near_half():
    torch.manual_seed(0)
    t = sample_logit_normal_t(200_000)
    assert abs(t.mean().item() - 0.5) < 0.01


def test_calculate_shift_endpoints():
    # base_seq_len → base_shift；max_seq_len → max_shift
    assert abs(calculate_shift(256) - 0.5) < 1e-6
    assert abs(calculate_shift(4096) - 1.15) < 1e-6


def test_calculate_shift_is_linear():
    mu_small = calculate_shift(256)
    mu_mid = calculate_shift((256 + 4096) // 2)
    mu_big = calculate_shift(4096)
    assert abs(mu_mid - (mu_small + mu_big) / 2) < 1e-6


def test_apply_time_shift_endpoints():
    t = torch.tensor([0.0, 1.0])
    out = apply_time_shift(t, mu=1.0)
    assert torch.allclose(out, t, atol=1e-6)  # 端点不变


def test_apply_time_shift_monotonic_in_mu():
    t = torch.linspace(0.01, 0.99, 50)
    out_small = apply_time_shift(t, mu=0.5)
    out_big = apply_time_shift(t, mu=1.15)
    assert (out_small > 0).all() and (out_small < 1).all()
    # 更大 mu → 更多时间被推向 1（大分辨率需要更大 shift）
    assert (out_big >= out_small).all()


def test_shift_alignment_with_diffusers():
    """与 diffusers calculate_shift + _time_shift_exponential 逐值对齐。"""
    mod = pytest.importorskip("diffusers.pipelines.z_image.pipeline_z_image")
    sched_mod = pytest.importorskip("diffusers.schedulers.scheduling_flow_match_euler_discrete")

    for seq_len in (256, 1024, 2048, 4096):
        assert abs(calculate_shift(seq_len) - mod.calculate_shift(seq_len)) < 1e-6

    t = torch.linspace(0.05, 0.95, 11)
    sched = sched_mod.FlowMatchEulerDiscreteScheduler()
    for mu in (0.5, 1.15):
        ref = sched._time_shift_exponential(mu, 1.0, t)
        ours = apply_time_shift(t, mu)
        assert torch.allclose(ours, ref, atol=1e-6)


def test_apply_time_shift_matches_flux_formula():
    # t_shifted = e^mu · t / (1 + (e^mu - 1) · t)
    t = torch.linspace(0.0, 1.0, 101)
    mu = 0.8
    e = math.exp(mu)
    expected = e * t / (1 + (e - 1) * t)
    assert torch.allclose(apply_time_shift(t, mu), expected, atol=1e-6)
