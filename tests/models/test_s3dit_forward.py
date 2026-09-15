"""S3-DiT 完整前向：梯度传播、参数量核对、结构输出。"""

import torch

from zimage.models import S3DiT, S3DiTConfig
from zimage.models.s3dit import S3DiTBlock


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


def test_patchify_unpatchify_roundtrip():
    # 模型级 patchify（PatchEmbed）与 unpatchify（S3DiT）必须互为逆变换，
    # token 维度顺序 [pH, pW, C] 与官方一致（回归防护：早前此处理不一致）
    cfg = S3DiTConfig.tiny()
    model = S3DiT(cfg)
    latent = torch.randn(2, cfg.in_channels, 16, 16)
    tokens, grid = model.x_embedder.patchify(latent)
    assert tokens.shape == (2, 64, 64)
    restored = model.unpatchify(tokens, grid)
    assert torch.equal(restored, latent)


def test_low_rank_adaln_shared_down_proj():
    # [PAPER §4.1] 低秩 adaLN：共享 down-proj（t_embedder，跨层唯一）+ 每层 up-proj
    cfg = S3DiTConfig.tiny()
    model = S3DiT(cfg)
    dim = cfg.hidden_size

    # 共享 down-proj：t_embedder 唯一实例，输出 min(dim,256)=256
    assert model.t_embedder is not None
    assert model.t_embedder.mlp[-1].out_features == 256

    # 每层 up-proj：Linear(256 → 4·dim)，层间参数独立
    for layer in model.layers:
        lin = layer.adaLN_modulation[0]
        assert lin.in_features == 256
        assert lin.out_features == 4 * dim
    ptrs = [layer.adaLN_modulation[0].weight.data_ptr() for layer in model.layers]
    assert len(set(ptrs)) == len(ptrs)  # 无共享（每层独立）

    # noise_refiner 同样带独立 up-proj；context_refiner 无调制
    assert all(layer.adaLN_modulation is not None for layer in model.noise_refiner)
    assert all(layer.adaLN_modulation is None for layer in model.context_refiner)


def test_block_matches_manual_adaln_application():
    # scale = 1 + scale，gate = tanh(gate)，逐子层应用点与官方一致
    torch.manual_seed(0)
    dim, heads, ffn = 64, 2, int(64 / 3 * 8)
    block = S3DiTBlock(layer_id=0, dim=dim, num_heads=heads, ffn_dim=ffn, modulation=True)
    x = torch.randn(2, 5, dim)
    c = torch.randn(2, min(dim, 256))  # adaLN 输入维 = min(dim, 256)
    out = block(x, adaln_input=c)

    mod = block.adaLN_modulation(c).unsqueeze(1)  # [2,1,4·dim]
    s_msa, g_msa, s_mlp, g_mlp = mod.chunk(4, dim=2)
    g_msa, g_mlp = g_msa.tanh(), g_mlp.tanh()
    s_msa, s_mlp = 1.0 + s_msa, 1.0 + s_mlp

    attn_out = block.attention(block.attention_norm1(x) * s_msa)
    x2 = x + g_msa * block.attention_norm2(attn_out)
    x3 = x2 + g_mlp * block.ffn_norm2(block.feed_forward(block.ffn_norm1(x2) * s_mlp))
    assert torch.allclose(out, x3, atol=1e-5)


def test_context_refiner_has_no_modulation():
    cfg = S3DiTConfig.tiny()
    model = S3DiT(cfg)
    dim = cfg.hidden_size
    x = torch.randn(1, 4, dim)
    # context_refiner 无调制：不传 adaln_input 也能前向
    for layer in model.context_refiner:
        out = layer(x)
        assert out.shape == x.shape
