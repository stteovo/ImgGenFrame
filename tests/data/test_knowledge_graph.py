"""Knowledge Graph 测试：BM25 稀有度 / 层级 / 可视化过滤 / 采样权重 / 报告。"""

import math

from zimage.data.knowledge_graph import (
    Concept,
    ConceptGraph,
    compute_bm25_idf,
    compute_sampling_weights,
    concept_report,
    dataset_frequency,
)


def _graph():
    g = ConceptGraph()
    g.add_concept(Concept("animal"))
    g.add_concept(Concept("dog", parent="animal"))
    g.add_concept(Concept("cat", parent="animal"))
    g.add_concept(Concept("golden_retriever", parent="dog", visualizability=0.9))
    g.add_concept(Concept("abstract_emotion", visualizability=0.0))  # 不可视觉化
    g.add_concept(Concept("tibetan_mastiff", parent="dog"))  # 稀有
    return g


def _corpus():
    # dog 高频、cat 中频、golden_retriever 低频、tibetan_mastiff 极稀
    return [
        ["dog"] * 1,
        ["dog"] * 1,
        ["dog", "cat"] * 1,
        ["cat"] * 1,
        ["golden_retriever"] * 1,
    ]


def test_bm25_idf_rare_terms_higher():
    corpus = _corpus()
    idf = compute_bm25_idf(corpus)
    assert idf["dog"] < idf["cat"] < idf["golden_retriever"]


def test_dataset_frequency_counts_docs():
    freq = dataset_frequency(_corpus())
    assert freq["dog"] == 3
    assert freq["cat"] == 2
    assert freq["golden_retriever"] == 1


def test_sampling_weight_rare_concept_oversampled():
    g = _graph()
    corpus = _corpus()
    w = compute_sampling_weights(g, corpus)
    # 稀有的 tibetan_mastiff 权重大于高频 dog（rarity 主导）
    assert w["tibetan_mastiff"] > w["dog"]


def test_sampling_weight_non_visualizable_zero():
    g = _graph()
    w = compute_sampling_weights(g, _corpus())
    assert w["abstract_emotion"] == 0.0


def test_hierarchy_parent_gets_children_weight():
    g = _graph()
    w = compute_sampling_weights(g, _corpus())
    # 有子概念的父节点（dog 有 2 子）层级因子 > 叶子（cat 无子）
    assert g.children("dog") and not g.children("cat")
    # dog 层级因子高于 cat
    assert (1 + math.log(1 + len(g.children("dog")))) > (1 + math.log(1))


def test_concept_report_fields():
    g = _graph()
    rows = concept_report(g, _corpus())
    by_name = {r["entity"]: r for r in rows}
    r = by_name["tibetan_mastiff"]
    for k in ("entity", "parent", "depth", "n_children", "visualizability", "frequency", "rarity", "sampling_weight"):
        assert k in r
    assert r["frequency"] == 0  # 未出现在语料中（仍按最大 IDF 给稀有度）
    assert r["parent"] == "dog"


def test_hierarchy_depth():
    g = _graph()
    assert g.depth("animal") == 0
    assert g.depth("dog") == 1
    assert g.depth("golden_retriever") == 2
    assert g.roots() == ["animal", "abstract_emotion"]


def test_synonyms_and_related_recorded():
    g = ConceptGraph()
    g.add_concept(Concept("dog", synonyms=["canine", "puppy"], related=["pet"]))
    rows = concept_report(g, [["dog"]])
    r = rows[0]
    assert r["n_synonyms"] == 2
    assert r["n_related"] == 1
