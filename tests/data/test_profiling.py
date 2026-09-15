"""Profiling 规则独立测试：熵/压缩比/边界方差/aHash/过滤规则（边界用例）。"""

import torch

from zimage.data.profiling import (
    average_hash,
    boundary_variance,
    compute_compression_ratio,
    compute_entropy,
    hamming_distance,
    passes_compression_threshold,
    passes_entropy_threshold,
    passes_min_resolution,
    profile_image,
)


def test_entropy_pure_color_is_zero():
    x = torch.ones(1, 32, 32) * 0.5  # 纯色
    assert compute_entropy(x) == 0.0


def test_entropy_noise_is_high():
    torch.manual_seed(0)
    x = torch.rand(1, 32, 32)  # 均匀噪声 → 高熵（接近 8 bit 上限）
    assert compute_entropy(x) > 6.0


def test_entropy_bimodal_matches_theory():
    # 一半 0 一半 1 → 熵 = 1 bit
    x = torch.cat([torch.zeros(512), torch.ones(512)])
    assert abs(compute_entropy(x) - 1.0) < 0.01


def test_compression_ratio_formula():
    assert compute_compression_ratio(1000, 100) == 10.0
    assert compute_compression_ratio(1000, 0) == float("inf")


def test_boundary_variance_flat_vs_edges():
    flat = torch.zeros(1, 16, 16)
    assert boundary_variance(flat) == 0.0
    torch.manual_seed(0)
    noisy = torch.rand(1, 16, 16)
    assert boundary_variance(noisy) > 0.0


def test_average_hash_deterministic():
    torch.manual_seed(0)
    img = torch.rand(3, 64, 64)
    assert average_hash(img) == average_hash(img)


def test_average_hash_sensitive_to_content():
    torch.manual_seed(0)
    a = torch.rand(3, 64, 64)  # 噪声
    y = torch.linspace(0, 1, 64).view(1, 64, 1).expand(3, 64, 64)  # 渐变
    assert average_hash(a) != average_hash(y)


def test_hamming_distance():
    assert hamming_distance(0b0000, 0b1111) == 4
    assert hamming_distance(0b1010, 0b1010) == 0


def test_profile_image_shape_and_fields():
    img = torch.rand(3, 256, 256)
    r = profile_image(img, raw_bytes=300_000, compressed_bytes=30_000)
    assert r.height == 256 and r.width == 256
    assert r.compression_ratio == 10.0
    assert "entropy" in r.to_dict()


def test_filter_rules():
    assert passes_min_resolution(256, 256)
    assert not passes_min_resolution(128, 256)
    assert passes_entropy_threshold(3.0, min_entropy=1.0)
    assert not passes_entropy_threshold(0.2, min_entropy=1.0)
    assert passes_compression_threshold(10.0)
    assert not passes_compression_threshold(2.0)
