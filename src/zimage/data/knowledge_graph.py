"""World Knowledge Topological Graph mini（视觉概念图 + 采样权重）。

[PAPER §2.3] 概念组织与语义级均衡采样：BM25 稀有度 + 父子层级 + 可视化生成性过滤
→ sampling_weight，服务 long-tail concept balancing（衔接 SFT tagged resampling §4.4）。

BM25 口径（可解释，[IMPLEMENTATION] 简化）：
  - Corpus：全部 caption（每条 = 一个 document）
  - Document：单条 caption 的 concept/tag token 集合
  - Tokenizer：空白/逗号分词（调用方传入已 tokenize 的 doc）
  - IDF(c) = log((N − df + 0.5) / (df + 0.5) + 1)（BM25 IDF 变体）
  - rarity(c) = IDF(c)（稀有概念 IDF 高）
  - sampling_weight(c) = visualizability(c) · rarity(c)^α · hierarchy(c)
    hierarchy(c) = 1 + ln(1 + n_children(c))（子概念越多越"中心"，父概念受控）
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Concept:
    name: str
    parent: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    visualizability: float = 1.0  # [0,1]：能否被可视化生成（VLM 过滤 [IMPLEMENTATION]）
    rarity: float = 1.0  # 手动稀有度先验（可选；默认由 BM25 IDF 计算）


class ConceptGraph:
    """概念层级图：parent/child + synonym + related。"""

    def __init__(self) -> None:
        self.concepts: Dict[str, Concept] = {}

    def add_concept(self, c: Concept) -> None:
        self.concepts[c.name] = c

    def add_parent(self, child: str, parent: str) -> None:
        if child in self.concepts and parent in self.concepts:
            self.concepts[child].parent = parent

    def children(self, name: str) -> List[str]:
        return [c.name for c in self.concepts.values() if c.parent == name]

    def depth(self, name: str) -> int:
        d = 0
        cur = name
        seen = set()
        while cur in self.concepts and self.concepts[cur].parent and cur not in seen:
            seen.add(cur)
            cur = self.concepts[cur].parent
            d += 1
        return d

    def roots(self) -> List[str]:
        return [c.name for c in self.concepts.values() if c.parent is None]

    def __len__(self) -> int:
        return len(self.concepts)


def compute_bm25_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """BM25 IDF（文档频率的倒数，稀有概念得分高）。"""
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        df.update(set(doc))
    idf = {}
    for term, d in df.items():
        idf[term] = math.log((N - d + 0.5) / (d + 0.5) + 1.0)
    return idf


def dataset_frequency(corpus: List[List[str]]) -> Dict[str, int]:
    """每个概念在语料中的文档频率 df。"""
    df: Counter = Counter()
    for doc in corpus:
        df.update(set(doc))
    return dict(df)


def compute_sampling_weights(
    graph: ConceptGraph,
    corpus: List[List[str]],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """sampling_weight(c) = visualizability · rarity^α · (1 + ln(1 + n_children))。

    返回未归一化的权重（调用方按需归一化）；weight=0 表示不应采样（不可见/缺失）。
    """
    idf = compute_bm25_idf(corpus)
    freq = dataset_frequency(corpus)
    weights: Dict[str, float] = {}
    for name, c in graph.concepts.items():
        rarity = idf.get(name, max(idf.values()) if idf else 1.0)
        hierarchy = 1.0 + math.log(1.0 + len(graph.children(name)))
        w = c.visualizability * (rarity ** alpha) * hierarchy * c.rarity
        weights[name] = w
    return weights


def concept_report(graph: ConceptGraph, corpus: List[List[str]]) -> List[dict]:
    """逐概念记录：entity/hierarchy/frequency/rarity/visualizability/sampling_weight。"""
    idf = compute_bm25_idf(corpus)
    freq = dataset_frequency(corpus)
    weights = compute_sampling_weights(graph, corpus)
    rows = []
    for name, c in graph.concepts.items():
        rows.append(
            {
                "entity": name,
                "parent": c.parent,
                "depth": graph.depth(name),
                "n_children": len(graph.children(name)),
                "n_synonyms": len(c.synonyms),
                "n_related": len(c.related),
                "visualizability": c.visualizability,
                "frequency": freq.get(name, 0),
                "rarity": round(idf.get(name, 0.0), 4),
                "sampling_weight": round(weights[name], 4),
            }
        )
    return rows
