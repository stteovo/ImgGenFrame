"""Cross-modal Vector Engine mini（实现见 `semantic_dedup.py`，第 14 步）。

[PAPER §2.2] kNN 替代 range_search 建图 + Leiden 社区检测 → 语义簇去重。
"""

from .semantic_dedup import (  # noqa: F401
    DedupCandidate,
    build_knn_graph,
    cluster_statistics,
    detect_communities,
    run_dedup,
    save_candidates,
)
