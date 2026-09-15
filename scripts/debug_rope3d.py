"""3D Unified RoPE 调试/可视化脚本。

打印不同位置 token 的频率表 / 嵌入差异，验证三轴语义：
  - 文本 token 沿时间轴 t 递增（h=w=0）
  - 图像 token 沿空间轴 (h, w) 展开（t 恒定）
运行：`.venv/bin/python scripts/debug_rope3d.py`（PYTHONPATH 需含 src）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage.models.rope3d import (  # noqa: E402
    RopeEmbedder,
    apply_rotary_emb,
    image_token_positions,
    text_token_positions,
)


def show_positions(title: str, pos: torch.Tensor, rope: RopeEmbedder) -> None:
    print(f"\n=== {title} ===")
    print(f"  坐标表 shape={tuple(pos.shape)}（(t, h, w)）")
    for p in pos[:6]:
        print(f"  {p.tolist()}")


def show_freq_difference(rope: RopeEmbedder, pos_a: torch.Tensor, pos_b: torch.Tensor) -> None:
    fa = rope(pos_a[None])
    fb = rope(pos_b[None])
    angle_a = torch.angle(fa)
    angle_b = torch.angle(fb)
    # 各轴贡献：t 轴占前 16 维（32/2），h 轴占 24 维（48/2），w 轴占 24 维
    t_a, t_b = angle_a[:, :16], angle_b[:, :16]
    h_a, h_b = angle_a[:, 16:40], angle_b[:, 16:40]
    w_a, w_b = angle_a[:, 40:], angle_b[:, 40:]
    print(f"  位置 {pos_a.tolist()} vs {pos_b.tolist()}")
    print(f"    Δangle(t)  max={ (t_a - t_b).abs().max().item():.4f}")
    print(f"    Δangle(h)  max={ (h_a - h_b).abs().max().item():.4f}")
    print(f"    Δangle(w)  max={ (w_a - w_b).abs().max().item():.4f}")


def main() -> None:
    torch.manual_seed(0)
    rope = RopeEmbedder(theta=256.0, axes_dims=(32, 48, 48), axes_lens=(1536, 512, 512))
    print(f"head_dim={rope.head_dim}（= sum(axes_dims)=32+48+48）")
    print(f"freqs_cis 输出最后一维 = head_dim/2 = {rope.head_dim // 2}（复数）")

    # 文本 token：时间轴
    text = text_token_positions(6)
    show_positions("文本 token 位置（沿时间轴）", text, rope)

    # 图像 token：空间轴（2x2 网格，t 偏移 7）
    image = image_token_positions(2, 2, t_offset=7)
    show_positions("图像 token 位置（沿空间轴，t 偏移=7）", image, rope)

    # 轴语义差异演示
    print("\n=== 轴语义差异（同一 token 沿不同轴移动） ===")
    origin = torch.tensor([0, 0, 0])
    show_freq_difference(rope, origin, torch.tensor([1, 0, 0]))  # 仅 t 变
    show_freq_difference(rope, origin, torch.tensor([0, 1, 0]))  # 仅 h 变
    show_freq_difference(rope, origin, torch.tensor([0, 0, 1]))  # 仅 w 变

    # 旋转演示
    print("\n=== 旋转演示（同一向量在不同位置） ===")
    x = torch.randn(1, 1, 1, rope.head_dim)
    r0 = apply_rotary_emb(x, rope(origin[None])[None])
    r1 = apply_rotary_emb(x, rope(torch.tensor([[3, 5, 7]]))[None])
    print(f"  |Δ|(rot(x, 000) - rot(x, 3,5,7)) = {(r0 - r1).abs().max().item():.6f}")
    print(f"  |rot(x)| == |x| : {torch.allclose(r1.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-3)}")


if __name__ == "__main__":
    main()
