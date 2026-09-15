"""Semantic Dedup（kNN proximity graph + 社区检测）。

[PAPER §2.2] Cross-modal Vector Engine 的 mini 版：k-NN 替代 range_search 建图，
社区检测（Leiden；本实现用 python-louvain 的 Louvain [IMPLEMENTATION]，Leiden 的
近亲，leidenalg 未安装）→ 语义簇 → 冗余分析。
**不删除数据**：仅标记 candidate_duplicate 并选 representative。
官方规模（k=100、8×H800、1B 条目 8h）仅作参照，不做性能对标。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import community as community_louvain
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class DedupCandidate:
    index: int
    image_id: str
    community_id: int
    similarity: float  # 到 k 近邻的平均余弦相似度
    duplicate_score: float  # 高=更冗余
    representative_score: float  # 高=更适合作簇代表（到质心相似度）
    candidate_duplicate: bool  # 非代表 → True（标记而非删除）

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "image_id": self.image_id,
            "community_id": self.community_id,
            "similarity": self.similarity,
            "duplicate_score": self.duplicate_score,
            "representative_score": self.representative_score,
            "candidate_duplicate": self.candidate_duplicate,
        }


def build_knn_graph(embeddings: np.ndarray, k: int, min_similarity: float = 0.0) -> nx.Graph:
    """embeddings [N, D] → kNN proximity graph（节点=索引，边权=余弦相似度）。

    min_similarity：弱边剪枝阈值 [IMPLEMENTATION]，避免低相似度的随机向量被
    Louvain 误聚类；宁可漏删不可错删长尾。
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    dists, idxs = nn.kneighbors(embeddings)  # [N, k+1]（含自身）

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j, d in zip(idxs[i], dists[i]):
            if j == i:
                continue
            sim = float(1.0 - d)  # cosine 距离 → 相似度
            if sim > min_similarity:
                G.add_edge(int(i), int(j), weight=sim)
    return G


def detect_communities(graph: nx.Graph, seed: int = 0) -> Dict[int, int]:
    """Louvain 社区检测（[IMPLEMENTATION] Louvain 替代 Leiden）。"""
    return community_louvain.best_partition(graph, weight="weight", random_state=seed)


def _neighbor_similarity(embeddings: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    dists, idxs = nn.kneighbors(embeddings)
    sims = []
    for i in range(len(embeddings)):
        s = [1.0 - d for j, d in zip(idxs[i], dists[i]) if j != i]
        sims.append(float(np.mean(s)) if s else 0.0)
    return np.array(sims)


def run_dedup(
    embeddings: np.ndarray,
    k: int = 10,
    min_similarity: float = 0.0,
    image_ids: Sequence[str] | None = None,
    seed: int = 0,
) -> List[DedupCandidate]:
    """完整去重分析 → candidate 列表（不删除数据）。"""
    n = embeddings.shape[0]
    if image_ids is None:
        image_ids = [str(i) for i in range(n)]

    graph = build_knn_graph(embeddings, k, min_similarity=min_similarity)
    partition = detect_communities(graph, seed=seed)
    neigh_sim = _neighbor_similarity(embeddings, k)

    # 每个社区的质心
    communities: Dict[int, List[int]] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(int(node))

    # 代表 = 到社区质心余弦相似度最高者
    representative_of: Dict[int, int] = {}
    centroid_sim: Dict[int, float] = {}
    for cid, members in communities.items():
        centroid = embeddings[members].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims = {}
        for m in members:
            v = embeddings[m]
            v = v / (np.linalg.norm(v) + 1e-8)
            sims[m] = float(np.dot(v, centroid))
        rep = max(sims, key=sims.get)
        representative_of[cid] = rep
        centroid_sim.update(sims)

    candidates = []
    for i in range(n):
        cid = partition[i]
        is_rep = representative_of[cid] == i
        candidates.append(
            DedupCandidate(
                index=i,
                image_id=image_ids[i],
                community_id=cid,
                similarity=float(neigh_sim[i]),
                duplicate_score=float(neigh_sim[i]),  # 冗余度 = 近邻相似度 [IMPLEMENTATION]
                representative_score=centroid_sim[i],
                candidate_duplicate=(not is_rep),
            )
        )
    return candidates


def cluster_statistics(candidates: List[DedupCandidate]) -> Dict:
    """簇统计：数量、规模分布、疑似重复数。"""
    sizes: Dict[int, int] = {}
    for c in candidates:
        sizes[c.community_id] = sizes.get(c.community_id, 0) + 1
    size_vals = list(sizes.values())
    n_dup = sum(1 for c in candidates if c.candidate_duplicate)
    return {
        "num_images": len(candidates),
        "num_clusters": len(sizes),
        "cluster_sizes": sorted(size_vals, reverse=True),
        "max_cluster_size": max(size_vals) if size_vals else 0,
        "singleton_clusters": sum(1 for s in size_vals if s == 1),
        "suspected_duplicates": n_dup,
        "duplicate_ratio": n_dup / len(candidates) if candidates else 0.0,
    }


def save_candidates(candidates: List[DedupCandidate], path) -> None:
    """输出 dedup_candidates.parquet（不删除原始数据）。"""
    import pandas as pd

    df = pd.DataFrame([c.to_dict() for c in candidates])
    df.to_parquet(path, index=False)
