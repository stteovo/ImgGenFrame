"""Semantic Dedup 测试：proximity graph + 社区检测 + candidate 标记（不删除）。"""

import numpy as np

from zimage.data.semantic_dedup import (
    build_knn_graph,
    cluster_statistics,
    detect_communities,
    run_dedup,
)


def _synthetic(n_unique=20, n_groups=4, group_size=3, dim=32, seed=0):
    """构造已知近重复对：n_unique 个随机向量 + n_groups 组（组内近重复）。"""
    rng = np.random.default_rng(seed)
    embs = [rng.normal(size=(n_unique, dim)).astype(np.float32)]
    for _ in range(n_groups):
        base = rng.normal(size=(dim,)).astype(np.float32)
        group = np.stack([base + 0.001 * rng.normal(size=dim) for _ in range(group_size)])
        embs.append(group.astype(np.float32))
    return np.concatenate(embs, axis=0)


def test_build_knn_graph_nodes_and_edges():
    embs = _synthetic()
    G = build_knn_graph(embs, k=5)
    assert G.number_of_nodes() == len(embs)
    assert G.number_of_edges() > 0
    # 边权为相似度（含负值可能，cosine）
    assert all("weight" in d for _, _, d in G.edges(data=True))


def test_community_detection_partitions_all_nodes():
    embs = _synthetic()
    G = build_knn_graph(embs, k=5)
    part = detect_communities(G, seed=0)
    assert set(part.keys()) == set(range(len(embs)))


def test_near_duplicates_same_community():
    # 组内近重复应聚到同一社区（召回测试）
    embs = _synthetic(n_unique=20, n_groups=3, group_size=3, dim=64, seed=1)
    cands = run_dedup(embs, k=5, seed=0)
    # 每个 group 的 3 个成员（索引 20..28）应两两同社区
    for g in range(3):
        cids = {cands[20 + g * 3 + i].community_id for i in range(3)}
        assert len(cids) == 1, f"group {g} 未聚到同一社区：{cids}"


def test_candidate_duplicate_marked_not_deleted():
    embs = _synthetic(n_unique=20, n_groups=3, group_size=3, dim=64, seed=1)
    cands = run_dedup(embs, k=5, seed=0)
    # 每社区恰一个代表，其余标记 candidate_duplicate
    for g in range(3):
        group = [cands[20 + g * 3 + i] for i in range(3)]
        n_rep = sum(1 for c in group if not c.candidate_duplicate)
        n_dup = sum(1 for c in group if c.candidate_duplicate)
        assert n_rep == 1 and n_dup == 2
    # 总数不变（未删除）
    assert len(cands) == len(embs)


def test_cluster_statistics():
    embs = _synthetic(n_unique=20, n_groups=4, group_size=3, dim=32, seed=0)
    cands = run_dedup(embs, k=5, seed=0)
    stats = cluster_statistics(cands)
    assert stats["num_images"] == 32
    assert stats["num_clusters"] > 0
    assert stats["suspected_duplicates"] > 0
    assert 0.0 <= stats["duplicate_ratio"] <= 1.0
    # 组内成员应为疑似重复（每个 group 2 个，共 8 个）
    assert stats["suspected_duplicates"] >= 8


def test_deterministic_given_seed():
    embs = _synthetic(seed=0)
    c1 = run_dedup(embs, k=5, seed=42)
    c2 = run_dedup(embs, k=5, seed=42)
    assert [x.community_id for x in c1] == [x.community_id for x in c2]
