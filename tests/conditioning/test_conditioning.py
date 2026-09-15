"""Conditioning 抽象测试：text only / image only / text+image / empty / 变长。"""

import torch

from zimage.conditioning.conditioning import (
    ConditioningAdapter,
    ConditioningOutput,
    EncodedImage,
    EncodedText,
)


def _text(n, b=2, feat_dim=2560):
    return EncodedText(
        features=torch.randn(b, n, feat_dim),
        mask=torch.ones(b, n, dtype=torch.bool),
    )


def _image(h, w, b=2, c=16):
    return EncodedImage(latent=torch.randn(b, c, h, w))


def test_text_only():
    adapter = ConditioningAdapter()
    out = adapter(text=_text(8))
    assert out.has_text() and not out.has_image()
    assert out.text_tokens.shape == (2, 8, 2560)
    assert out.position_ids.shape == (1, 8, 3)
    assert (out.modality_ids == 0).all()
    # 文本沿时间轴 t=1..8，h=w=0
    assert out.position_ids[0, :, 0].tolist() == list(range(1, 9))
    assert (out.position_ids[0, :, 1:] == 0).all()


def test_image_only():
    adapter = ConditioningAdapter()
    out = adapter(image=_image(16, 32))  # latent 16×32 → grid (8,16) → 128 tokens
    assert out.has_image() and not out.has_text()
    assert out.image_tokens.shape == (2, 16, 16, 32)
    assert out.position_ids.shape == (1, 128, 3)
    assert (out.modality_ids == 1).all()
    # 图像沿空间轴，t 恒定 = 0+1=1
    assert (out.position_ids[0, :, 0] == 1).all()


def test_text_plus_image():
    adapter = ConditioningAdapter()
    out = adapter(text=_text(5), image=_image(8, 8))  # latent 8×8 → grid(4,4)=16 tokens
    assert out.has_text() and out.has_image()
    # 拼接顺序 [image, text]：modality 前 16 为 1，后 5 为 0
    assert out.modality_ids[0].tolist() == [1] * 16 + [0] * 5
    assert out.position_ids.shape == (1, 21, 3)
    # 图像位置：t = text_len+1 = 6
    assert (out.position_ids[0, :16, 0] == 6).all()
    # 文本位置：t = 1..5
    assert out.position_ids[0, 16:, 0].tolist() == [1, 2, 3, 4, 5]


def test_empty_condition():
    adapter = ConditioningAdapter()
    out = adapter(text=None, image=None)
    assert out.is_empty()
    assert out.total_tokens == 0
    assert out.metadata["empty"] is True


def test_variable_text_length():
    adapter = ConditioningAdapter()
    for n in (1, 3, 17, 33):
        out = adapter(text=_text(n, b=1))
        assert out.position_ids.shape == (1, n, 3)
        assert out.num_text_tokens == n


def test_variable_image_token_count():
    adapter = ConditioningAdapter()
    for h, w in ((8, 8), (16, 16), (16, 32), (32, 16), (32, 48)):
        out = adapter(image=_image(h, w, b=1))
        n_img = (h // 2) * (w // 2)
        assert out.num_image_tokens == n_img
        assert out.position_ids.shape == (1, n_img, 3)


def test_position_ids_text_before_image_in_time():
    # 位置时间顺序：文本 t=1..T，图像 t=T+1（即使拼接时图像在前）
    adapter = ConditioningAdapter()
    out = adapter(text=_text(4, b=1), image=_image(8, 8, b=1))
    # 图像 token 时间轴 = 5（4+1）
    assert (out.position_ids[0, :16, 0] == 5).all()
    # 文本 token 时间轴 = 1..4
    assert out.position_ids[0, 16:, 0].tolist() == [1, 2, 3, 4]


def test_metadata_records_grid_and_semantic():
    adapter = ConditioningAdapter()
    out = adapter(text=_text(3, b=1), image=_image(16, 16, b=1), semantic="siglip")
    assert out.metadata["image_grid"] == (8, 8)
    assert out.metadata["text_len"] == 3
    assert out.metadata["semantic"] is True


def test_semantic_optional_flag_off_by_default():
    adapter = ConditioningAdapter()
    out = adapter(text=_text(3, b=1))
    assert out.metadata["semantic"] is False
