"""Caption Pipeline 测试：分层 / observed vs inferred / 置信度 / span / parquet 往返。"""

import torch

from zimage.captioning.io import read_parquet, write_parquet
from zimage.captioning.pipeline import CaptionPipeline
from zimage.captioning.schema import CaptionRecord, SpanGroundedItem
from zimage.captioning.vlm import MockVLM


def _image(h=256, w=256):
    torch.manual_seed(0)
    return torch.rand(3, h, w)


def test_pipeline_end_to_end_with_mock_vlm():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("img_001", _image())
    assert rec.image_id == "img_001"
    assert rec.caption_long and rec.caption_medium and rec.caption_short
    assert rec.tags
    assert 0.0 <= rec.caption_confidence <= 1.0
    assert rec.image_hash
    assert rec.prompt_version == "v1"


def test_layered_captions_present():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("x", _image())
    # 分层：long/medium/short/tags + 结构化层
    assert len(rec.caption_long) > len(rec.caption_medium) > len(rec.caption_short)
    assert rec.tags and rec.objects and rec.attributes


def test_observed_vs_inferred_separation():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("x", _image())
    obs = rec.observed_items()
    inf = rec.inferred_items()
    # objects/attributes/OCR 是 observed；style/lighting/composition 是 inferred
    assert all(it.observed for it in obs)
    assert all(not it.observed for it in inf)
    assert any(it.source == "ocr" for it in obs)
    assert any(it.source == "inference" for it in inf)


def test_ocr_separate_and_preserves_original_language():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("x", _image())
    ocr_texts = [it.text for it in rec.ocr]
    assert any("公园" in t for t in ocr_texts)  # 中文原样保留，不翻译
    assert all(it.observed for it in rec.ocr)


def test_span_and_confidence_saved():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("x", _image())
    obj = rec.objects[0]
    assert obj.span is not None  # source span 保存
    assert 0.0 <= obj.confidence <= 1.0


def test_quality_filter_marks_low_entropy():
    pipe = CaptionPipeline(MockVLM())
    img = torch.ones(3, 256, 256)  # 纯色 → 低熵
    rec = pipe.caption_image("solid", img)
    assert rec.quality is not None
    assert rec.quality.passed is False
    assert "low_entropy" in rec.quality.reasons


def test_quality_filter_passes_normal_image():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("normal", _image())
    assert rec.quality.passed is True


def test_recaption_tracks_version():
    pipe_v1 = CaptionPipeline(MockVLM(), prompt_version="v1")
    pipe_v2 = CaptionPipeline(MockVLM(), prompt_version="v2")
    img = _image()
    r1 = pipe_v1.caption_image("img", img)
    r2 = pipe_v2.caption_image("img", img)
    assert r1.image_hash == r2.image_hash  # 同图同 hash
    assert r1.prompt_version != r2.prompt_version  # 重 caption 版本可追溯


def test_schema_dict_roundtrip():
    pipe = CaptionPipeline(MockVLM())
    rec = pipe.caption_image("x", _image())
    d = rec.to_dict()
    rec2 = CaptionRecord.from_dict(d)
    assert rec2.image_id == rec.image_id
    assert rec2.objects[0].text == rec.objects[0].text
    assert rec2.caption_confidence == rec.caption_confidence


def test_parquet_roundtrip(tmp_path):
    pipe = CaptionPipeline(MockVLM())
    recs = [pipe.caption_image(f"img_{i}", _image()) for i in range(3)]
    p = tmp_path / "captions.parquet"
    write_parquet(recs, p)
    loaded = read_parquet(p)
    assert len(loaded) == 3
    assert loaded[0].image_id == recs[0].image_id
    assert loaded[0].objects[0].text == recs[0].objects[0].text
    assert loaded[0].caption_confidence == recs[0].caption_confidence


def test_span_grounded_item_roundtrip():
    it = SpanGroundedItem(text="dog", span="一只小狗", confidence=0.9, observed=True, source="visual")
    it2 = SpanGroundedItem.from_dict(it.to_dict())
    assert it2 == it
