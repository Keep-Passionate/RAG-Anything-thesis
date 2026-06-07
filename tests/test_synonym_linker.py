"""L2 同义边纯计算层单测。运行：pytest tests/test_synonym_linker.py -v

只测 find_synonym_pairs（不碰文件/图），验证 cos 与 Jaccard 双门控逻辑。
"""
import numpy as np

from raganything.graph_fusion.synonym_linker import find_synonym_pairs, _jaccard


def test_jaccard_basic():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a", "b", "c"}, {"a"}) == 1 / 3


def test_finds_true_synonym_rejects_cos_decoy():
    # Apple 与 Apple Inc：向量近 + 邻居高度重叠 -> 应判同义
    # Banana：向量也接近 Apple（cos 干扰项），但邻居完全不同 -> Jaccard 应拒掉
    names = ["Apple", "Apple Inc", "Banana", "Orange"]
    matrix = np.array([
        [1.00, 0.00, 0.0],
        [0.99, 0.01, 0.0],   # ~ Apple
        [0.98, 0.02, 0.0],   # cos 干扰项，与 Apple 很像
        [0.00, 0.00, 1.0],   # 无关
    ], dtype=np.float32)
    neighbor_of = {
        "Apple":     {"iPhone", "TimCook", "Mac"},
        "Apple Inc": {"iPhone", "TimCook", "iPad"},   # 与 Apple 交2并4 -> J=0.5
        "Banana":    {"Fruit", "Yellow", "Smoothie"}, # 与 Apple 交0 -> J=0
        "Orange":    {"Fruit"},
    }
    pairs = find_synonym_pairs(names, matrix, neighbor_of, tau=0.90, theta=0.15)
    found = {frozenset((a, b)) for a, b, *_ in pairs}

    assert frozenset(("Apple", "Apple Inc")) in found          # 真同义被找到
    assert frozenset(("Apple", "Banana")) not in found         # cos 干扰项被 Jaccard 拒掉
    assert frozenset(("Apple Inc", "Banana")) not in found


def test_theta_gate_rejects_low_jaccard():
    # cos 极高但邻居无交集 -> 被 theta 拒
    names = ["A", "B"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    neighbor_of = {"A": {"x", "y"}, "B": {"p", "q"}}
    assert find_synonym_pairs(names, matrix, neighbor_of, tau=0.9, theta=0.1) == []


def test_tau_gate_rejects_low_cosine():
    # 邻居完全相同但 cos=0 -> 被 tau 拒
    names = ["A", "B"]
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    neighbor_of = {"A": {"x", "y"}, "B": {"x", "y"}}
    assert find_synonym_pairs(names, matrix, neighbor_of, tau=0.5, theta=0.1) == []


def test_skip_already_connected():
    # 已经互为邻居（已有边）的对不再加同义边
    names = ["A", "B"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    neighbor_of = {"A": {"B", "x"}, "B": {"A", "x"}}
    assert find_synonym_pairs(names, matrix, neighbor_of, tau=0.9, theta=0.01) == []


def test_deterministic_order_by_cosine():
    # 返回按 cos 降序，结果与输入顺序无关
    names = ["A", "B", "C"]
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],   # A-B cos 高
        [0.91, 0.0, 0.0],   # A-C / B-C cos 较低
    ], dtype=np.float32)
    nb = {
        "A": {"x", "y", "z"},
        "B": {"x", "y", "w"},
        "C": {"x", "y", "v"},
    }
    pairs = find_synonym_pairs(names, matrix, nb, tau=0.85, theta=0.1)
    coss = [c for *_, c, _ in [(p[0], p[1], p[2], p[3]) for p in pairs]]
    assert coss == sorted(coss, reverse=True)
