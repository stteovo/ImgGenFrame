"""S3-DiT 完整前向：梯度传播、参数量核对、结构输出。"""

import torch

from zimage.models import S3DiT, S3DiTConfig


def test_full_forward_gradient_flows():
    cfg = S3DiTConfig.tiny()
    model = S3DiT(cfg)
    latent = torch.randn(2, cfg.in_channels, 16, 16, requires_grad=True)
    t = torch.rand(2)
    cap = torch.randn(2, 8, cfg.cap_feat_dim, requires_grad=True)

    out = model(latent, t, cap)
    out.pow(2).mean().backward()

    assert latent.grad is not None
    assert cap.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert torch.isfinite(cap.grad).all()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"


def test_param_count_consistency_and_scale():
    model = S3DiT(S3DiTConfig.tiny())
    total = model.num_parameters()
    assert total == sum(p.numel() for p in model.parameters())
    # tiny 配置 ≈ 95.7M，落在 100M 级缩比梯子（M1）区间
    assert 50_000_000 < total < 150_000_000


def test_param_count_by_component():
    model = S3DiT(S3DiTConfig.tiny())
    by_name = {n: p.numel() for n, p in model.named_parameters()}
    # adaLN 调制投影参数存在于主块与 noise_refiner，context_refiner 无调制
    assert any("adaLN_modulation" in n for n in by_name)
    assert not any("context_refiner" in n and "adaLN_modulation" in n for n in by_name)


def test_summarize_outputs_structure():
    model = S3DiT(S3DiTConfig.tiny())
    s = model.summarize()
    assert "S3DiT(" in s
    assert "noise_refiner" in s
    assert "context_refiner" in s
    assert "main layers" in s
    assert "total parameters" in s
    print("\n=== S3-DiT 模型结构 ===\n" + s)
