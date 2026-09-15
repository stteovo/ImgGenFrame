"""Dataset 版本化 + Data→Training 闭环编排。

把数据管线（profile → quality → caption → concept → embedding → kNN → Louvain →
KG mapping → sampling weight）与训练侧（WeightedTrainingDataset）接通。
dataset 必须版本化：训练 config 写 `dataset_version`，checkpoint 记录
`dataset_version` + `manifest_hash`，不依赖"当前目录里的图片"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

from ..captioning.pipeline import CaptionPipeline
from ..captioning.vlm import MockVLM, VLM
from .knowledge_graph import Concept, ConceptGraph, compute_sampling_weights
from .profiling import passes_entropy_threshold, passes_min_resolution, profile_image
from .semantic_dedup import run_dedup


@dataclass
class DatasetManifest:
    """数据集清单（可序列化，版本可追溯）。"""

    version: str
    provenance: Dict[str, str] = field(default_factory=dict)
    items: List[Dict] = field(default_factory=list)  # 每张图一条
    manifest_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps(
            {"version": self.version, "provenance": self.provenance, "items": self.items},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def finalize(self) -> "DatasetManifest":
        self.manifest_hash = self.compute_hash()
        return self

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetManifest":
        return cls(
            version=d["version"],
            provenance=d.get("provenance", {}),
            items=d.get("items", []),
            manifest_hash=d.get("manifest_hash", ""),
        )

    def save(self, path) -> None:
        import pathlib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "DatasetManifest":
        import pathlib

        d = json.loads(pathlib.Path(path).read_text())
        return cls.from_dict(d)


def _default_kg() -> ConceptGraph:
    g = ConceptGraph()
    g.add_concept(Concept("animal"))
    g.add_concept(Concept("dog", parent="animal"))
    g.add_concept(Concept("cat", parent="animal"))
    g.add_concept(Concept("golden_retriever", parent="dog"))
    g.add_concept(Concept("tibetan_mastiff", parent="dog"))
    return g


def build_dataset(
    version: str = "v001",
    num_unique: int = 24,
    num_dup_groups: int = 3,
    dup_size: int = 3,
    latent_size: int = 32,
    seed: int = 0,
    vlm: Optional[VLM] = None,
) -> DatasetManifest:
    """端到端构建版本化数据集（合成原始图 + latent + 文本；真实数据后续替换）。"""
    vlm = vlm or MockVLM()
    rng = torch.Generator().manual_seed(seed)

    # 合成：唯一图（8×8 低频网格最近邻上采样，池化后仍可区分）+ 近重复组
    import torch.nn.functional as F

    n = num_unique + num_dup_groups * dup_size
    grid = torch.rand(n, 3, 8, 8, generator=rng)  # 低频结构
    raw_images = F.interpolate(grid, size=(256, 256), mode="nearest")
    latents = torch.randn(n, 16, latent_size, latent_size, generator=rng)
    texts = torch.randn(n, 8, 2560, generator=rng)

    # 近重复组：每组占 dup_size 个连续槽位（首槽为 base，其余为复制+微噪声）
    for g in range(num_dup_groups):
        base_idx = num_unique + g * dup_size
        base_img = raw_images[base_idx].clone()
        base_lat = latents[base_idx].clone()
        for k in range(1, dup_size):
            idx = base_idx + k
            raw_images[idx] = base_img + 0.001 * torch.randn(3, 256, 256, generator=rng)
            latents[idx] = base_lat + 0.001 * torch.randn(16, latent_size, latent_size, generator=rng)

    # 1. profile + quality filter；2. caption；3. concept（tags）
    pipeline = CaptionPipeline(vlm)
    items: List[Dict] = []
    concepts_per_image: List[List[str]] = []
    valid_indices: List[int] = []
    for i in range(n):
        prof = profile_image(raw_images[i])
        passed = passes_min_resolution(prof.height, prof.width) and passes_entropy_threshold(prof.entropy)
        rec = pipeline.caption_image(f"img_{i:04d}", raw_images[i])
        items.append(
            {
                "index": i,
                "image_id": f"img_{i:04d}",
                "image_hash": rec.image_hash,
                "quality_passed": passed,
                "caption_short": rec.caption_short,
                "concepts": rec.tags,
                "caption_confidence": rec.caption_confidence,
            }
        )
        concepts_per_image.append(rec.tags)
        if passed:
            valid_indices.append(i)

    # 4. embedding（确定性连续：area-pool 到 8×8 再展平 → 近重复图得近相同向量）
    # 5. kNN/Louvain 去重
    import torch.nn.functional as F

    pooled = F.avg_pool2d(raw_images, kernel_size=32, stride=32)  # [n, 3, 8, 8]
    embeddings = pooled.flatten(1) - 0.5  # 中心化（均匀 [0,1) 均值 0.5），零均值 → 余弦可区分
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    dedup = run_dedup(embeddings.numpy(), k=5, min_similarity=0.7, seed=seed)
    dedup_by_index = {c.index: c for c in dedup}

    # 6. KG mapping + sampling weight
    kg = _default_kg()
    corpus = [[c for c in tags if c in kg.concepts] for tags in concepts_per_image]
    concept_weights = compute_sampling_weights(kg, corpus)

    for i, it in enumerate(items):
        c = dedup_by_index[i]
        it["community_id"] = c.community_id
        it["candidate_duplicate"] = c.candidate_duplicate
        it["duplicate_score"] = c.duplicate_score
        it["representative_score"] = c.representative_score
        # 采样权重 = 该图概念的 KG 权重最大值；重复候选权重减半（不直接删除）
        cw = max((concept_weights.get(t, 0.0) for t in it["concepts"]), default=1.0)
        it["sampling_weight"] = round(cw * (0.0 if c.candidate_duplicate else 1.0), 4)

    manifest = DatasetManifest(
        version=version,
        provenance={
            "vlm_model": vlm.name,
            "embedding_model": "synthetic-deterministic",
            "dedup": "knn-louvain",
            "kg": "mini-concept-graph",
            "prompt_version": "v1",
            "taxonomy_version": "v1",
        },
        items=items,
    ).finalize()
    return manifest


class WeightedTrainingDataset:
    """按 manifest 采样权重采样的训练集（数据与版本绑定，不依赖目录）。"""

    def __init__(self, manifest: DatasetManifest, latents: torch.Tensor, texts: torch.Tensor):
        self.manifest = manifest
        self.latents = latents
        self.texts = texts
        weights = torch.tensor([it["sampling_weight"] for it in manifest.items], dtype=torch.float32)
        if weights.sum() <= 0:
            weights = torch.ones_like(weights)
        self.probs = weights / weights.sum()

    def __len__(self) -> int:
        return len(self.manifest.items)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.multinomial(self.probs, batch_size, replacement=True)
        return self.latents[idx], self.texts[idx]
