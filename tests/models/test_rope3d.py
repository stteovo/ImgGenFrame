"""3D Unified RoPE 测试：形状 / 确定性 / 位置敏感性 / 轴分配 / 数值对齐。

参考实现：diffusers `transformer_z_image.RopeEmbedder`（[OFFICIAL-CODE]）。
"""

import pytest
import torch

from zimage.models.rope3d import (
    RopeEmbedder,
    apply_rotary_emb,
    create_coordinate_grid,
    image_token_positions,
    text_token_positions,
)

AXES_DIMS = (32, 48, 48)
AXES_LENS = (1536, 512, 512)
THETA = 256.0
HEAD_DIM = 128  # = sum(axes_dims)


def _rope():
    return RopeEmbedder(theta=THETA, axes_dims=AXES_DIMS, axes_lens=AXES_LENS)


def test_shape():
    rope = _rope()
    ids = torch.randint(0, 64, (17, 3))
    out = rope(ids)
    assert out.shape == (17, HEAD_DIM // 2)
    assert out.dtype == torch.complex64


def test_head_dim_compatibility():
    rope = _rope()
    assert rope.head_dim == HEAD_DIM
    assert rope.head_dim == sum(AXES_DIMS)


def test_deterministic():
    rope = _rope()
    ids = torch.randint(0, 64, (8, 3))
    assert torch.equal(rope(ids), rope(ids))


def test_position_sensitivity():
    rope = _rope()
    ids_a = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ids_b = torch.tensor([[5, 0, 0], [0, 5, 0], [0, 0, 5], [2, 3, 4]])
    for a, b in zip(ids_a, ids_b):
        fa = rope(a[None])
        fb = rope(b[None])
        assert not torch.allclose(fa, fb), f"不同位置 {a} vs {b} 应产生不同频率"


def test_rotation_is_position_sensitive():
    # 同一向量在不同位置旋转后结果不同
    rope = _rope()
    x = torch.randn(1, 1, 1, HEAD_DIM)  # [B, T, H, D]
    ids0 = torch.tensor([[0, 0, 0]])
    ids1 = torch.tensor([[3, 5, 7]])
    r0 = apply_rotary_emb(x, rope(ids0)[None])  # [1,1,D//2] -> unsqueeze
    r1 = apply_rotary_emb(x, rope(ids1)[None])
    assert not torch.allclose(r0, r1)


def test_rotation_preserves_norm_and_shape():
    rope = _rope()
    x = torch.randn(2, 4, 3, HEAD_DIM)
    ids = torch.randint(0, 64, (2, 4, 3))
    freqs = rope(ids.flatten(0, 1)).unflatten(0, (2, 4))  # [2, 4, D//2]
    out = apply_rotary_emb(x, freqs)
    assert out.shape == x.shape
    # 旋转是正交变换，保持模长
    assert torch.allclose(out.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-3)


def test_text_token_positions_on_temporal_axis():
    pos = text_token_positions(5)
    assert pos.shape == (5, 3)
    assert pos[:, 0].tolist() == [1, 2, 3, 4, 5]  # t 递增
    assert (pos[:, 1] == 0).all()  # h=0
    assert (pos[:, 2] == 0).all()  # w=0


def test_image_token_positions_on_spatial_axes():
    pos = image_token_positions(h_tokens=2, w_tokens=3, t_offset=9)
    assert pos.shape == (6, 3)
    assert (pos[:, 0] == 9).all()  # t 恒定
    # 行主序展开：(h=0,w=0..2) 后 (h=1,w=0..2)
    expected_hw = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert pos[:, 1:].tolist() == expected_hw


def test_hw_change_updates_positions():
    pos_small = image_token_positions(2, 2, t_offset=0)
    pos_big = image_token_positions(4, 4, t_offset=0)
    assert pos_small.shape == (4, 3)
    assert pos_big.shape == (16, 3)
    # 大网格覆盖更多空间坐标
    assert pos_big[:, 1].max() == 3
    assert pos_big[:, 2].max() == 3


def test_text_vs_image_axis_assignment():
    # 文本沿时间轴、图像沿空间轴（不同 axis 编码）
    text = text_token_positions(4)
    image = image_token_positions(2, 2, t_offset=5)
    assert (text[:, 1:] == 0).all()  # 文本 h,w 恒 0
    assert (image[:, 0] == 5).all()  # 图像 t 恒 5
    assert (image[:, 1:] >= 0).all()


def test_numerical_stability_unit_magnitude():
    rope = _rope()
    ids = torch.randint(0, 512, (100, 3))
    freqs = rope(ids)
    assert torch.isfinite(freqs.real).all() and torch.isfinite(freqs.imag).all()
    assert torch.allclose(freqs.abs(), torch.ones_like(freqs.real), atol=1e-6)


def test_invalid_ids_shape_raises():
    rope = _rope()
    with pytest.raises(ValueError):
        rope(torch.zeros(4))  # 不是 [N,3]


def test_invalid_axis_count_raises():
    with pytest.raises(ValueError):
        RopeEmbedder(theta=THETA, axes_dims=(32, 48), axes_lens=(512, 512))


def test_odd_axis_dim_raises():
    with pytest.raises(ValueError):
        RopeEmbedder(theta=THETA, axes_dims=(31, 48, 48), axes_lens=(512, 512, 512))


def test_numerical_alignment_with_diffusers():
    """与 diffusers RopeEmbedder 逐值对齐（[OFFICIAL-CODE] 参考实现）。"""
    diffusers_rope = pytest.importorskip(
        "diffusers.models.transformers.transformer_z_image"
    ).RopeEmbedder(theta=THETA, axes_dims=list(AXES_DIMS), axes_lens=list(AXES_LENS))
    ours = _rope()

    ids = torch.randint(0, 128, (37, 3))
    ref = diffusers_rope(ids)
    out = ours(ids)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6)


def test_coordinate_grid_matches_expected():
    grid = create_coordinate_grid((2, 2, 2), (0, 0, 0))
    assert grid.shape == (2, 2, 2, 3)
    # 原点与角点坐标
    assert grid[0, 0, 0].tolist() == [0, 0, 0]
    assert grid[1, 1, 1].tolist() == [1, 1, 1]
