"""Data → Training 闭环测试：版本化 manifest + 采样权重 + checkpoint 可追溯。"""

import torch

from zimage.data.dataset_version import (
    DatasetManifest,
    WeightedTrainingDataset,
    build_dataset,
)
from zimage.models import S3DiTConfig
from zimage.pipeline import ZImagePipeline
from zimage.training.config import TrainingSpec
from zimage.training.trainer import FlowMatchingTrainer


def test_build_dataset_manifest_fields():
    m = build_dataset(version="v001", num_unique=12, num_dup_groups=2, dup_size=3, seed=0)
    assert m.version == "v001"
    assert m.manifest_hash
    assert m.provenance["dedup"] == "knn-louvain"
    assert len(m.items) == 12 + 6
    for it in m.items:
        for k in ("image_id", "image_hash", "quality_passed", "concepts", "community_id", "candidate_duplicate", "sampling_weight"):
            assert k in it


def test_manifest_save_load_and_hash_stable(tmp_path):
    m = build_dataset(version="v001", num_unique=8, seed=0)
    p = tmp_path / "manifest.json"
    m.save(p)
    m2 = DatasetManifest.load(p)
    assert m2.version == m.version
    assert m2.manifest_hash == m.manifest_hash
    assert len(m2.items) == len(m.items)


def test_near_duplicates_marked_and_zero_weight():
    m = build_dataset(version="v001", num_unique=10, num_dup_groups=2, dup_size=3, seed=1)
    # 近重复组内：1 个代表 + 2 个 candidate_duplicate（权重 0）
    dup_items = [it for it in m.items if it["candidate_duplicate"]]
    assert len(dup_items) == 2 * 2
    assert all(it["sampling_weight"] == 0.0 for it in dup_items)


def test_weighted_dataset_sampling():
    m = build_dataset(version="v001", num_unique=12, seed=0)
    latents = torch.randn(len(m.items), 16, 32, 32)
    texts = torch.randn(len(m.items), 8, 2560)
    ds = WeightedTrainingDataset(m, latents, texts)
    x, t = ds.sample_batch(4)
    assert x.shape == (4, 16, 32, 32)
    assert t.shape == (4, 8, 2560)


def test_end_to_end_checkpoint_traceable(tmp_path):
    # 闭环：build v001 → train → checkpoint 记录 dataset_version + manifest_hash
    m = build_dataset(version="v001", num_unique=8, num_dup_groups=1, dup_size=3, seed=0)
    latents = torch.randn(len(m.items), 16, 32, 32)
    texts = torch.randn(len(m.items), 8, 2560)
    ds = WeightedTrainingDataset(m, latents, texts)

    cfg = S3DiTConfig(hidden_size=128, depth=2, num_heads=1, ffn_dim=341)
    torch.manual_seed(0)
    model = ZImagePipeline(cfg)
    trainer = FlowMatchingTrainer(model, TrainingSpec(dataset_version="v001"), device="cpu")
    trainer.set_dataset_info("v001", m.manifest_hash)

    for _ in range(3):
        x, t = ds.sample_batch(2)
        trainer.train_step(x, t)

    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path)

    # 新训练器加载，验证 dataset 信息可追溯
    model2 = ZImagePipeline(cfg)
    trainer2 = FlowMatchingTrainer(model2, TrainingSpec(dataset_version="v001"), device="cpu")
    trainer2.load_checkpoint(ckpt_path)
    assert trainer2.dataset_version == "v001"
    assert trainer2.manifest_hash == m.manifest_hash
