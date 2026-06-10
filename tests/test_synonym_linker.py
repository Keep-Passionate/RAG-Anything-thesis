"""L2 同义边纯计算层单测。运行：pytest tests/test_synonym_linker.py -v

只测 find_synonym_pairs（不碰文件/图），验证 cos 与 Jaccard 双门控逻辑。
"""
import numpy as np

from raganything.graph_fusion.synonym_linker import (
    find_synonym_pairs,
    _jaccard,
    _is_enumeration_variant,
    _names_name_match,
    _acronym_of,
)


# ---- 枚举判别守卫：用 doc63(Comcast 10-K)真实例子验证 ----

def test_enum_guard_rejects_real_false_positives():
    """这些都是 doc63 里观察到的"仅差编号/序号"的不同实体，必须判为枚举变体。"""
    bad = [
        ("0.250% Notes Due 2027", "0.250% Notes Due 2029"),
        ("Exhibit 10.22", "Exhibit 10.23"),
        ("Page 102", "Page 103"),
        ("Section 4.02(a)", "Section 4.02(b)"),
        ("Comcast of Sacramento I, LLC", "Comcast of Sacramento II, LLC"),
        ("Fiscal Year 2020", "Fiscal Year 2019"),
        ("¥29.7 Billion RMB", "¥26.6 Billion RMB"),
        ("October 12 2021", "January 1, 2021"),
        ("CIK 0001166559", "CIK 0000733125"),
        ("Treasury Regulation §1.409A-1(c)(2)(i)(A)",
         "Treasury Regulation §1.409A-1(c)(2)(i)(B)"),
    ]
    for a, b in bad:
        assert _is_enumeration_variant(a, b), f"应判枚举变体: {a!r} vs {b!r}"


def test_enum_guard_keeps_real_true_synonyms():
    """这些是 doc63 里真正的同义（缩写/大小写/单复数/后缀），不能被误伤。"""
    good = [
        ("SEC", "Securities and Exchange Commission"),
        ("RSUs", "Restricted Stock Units"),
        ("MD&A", "Managements Discussion And Analysis"),
        ("Opinion of Counsel", "Opinion Of Counsel"),
        ("Table Of Contents", "Table of Contents"),
        ("Debt Rating", "Debt Ratings"),
        ("Atairos Group", "Atairos Group, Inc."),
        ("S&P 500 Stock Index", "Standard & Poors 500 Stock Index"),
    ]
    for a, b in good:
        assert not _is_enumeration_variant(a, b), f"误伤真同义: {a!r} vs {b!r}"


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


# ---- Step3 规则1：文档作用域守卫 ----

def test_same_doc_guard_rejects_cross_doc_pair():
    # Alpha、Alphabet 向量近 + 邻居重叠（本应判同义），但来自不同文档 -> 文档作用域应拒绝
    names = ["Alpha", "Alphabet"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Alpha": {"x", "y"}, "Alphabet": {"x", "y"}}
    doc_of = {"Alpha": {"doc1.pdf"}, "Alphabet": {"doc2.pdf"}}
    # 不开守卫：能找到（旧版行为）
    assert find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1) != []
    # 开守卫：跨文档对被拒
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        doc_of=doc_of, require_same_doc=True,
    ) == []


def test_same_doc_guard_keeps_shared_doc_pair():
    # 两实体有共同来源文档（doc2）-> 守卫放行
    names = ["Alpha", "Alphabet"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Alpha": {"x", "y"}, "Alphabet": {"x", "y"}}
    doc_of = {"Alpha": {"doc1.pdf", "doc2.pdf"}, "Alphabet": {"doc2.pdf"}}
    pairs = find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        doc_of=doc_of, require_same_doc=True,
    )
    assert {frozenset((a, b)) for a, b, *_ in pairs} == {frozenset(("Alpha", "Alphabet"))}


def test_same_doc_guard_allows_when_doc_unknown():
    # 文档元数据缺失（空集）时不应误杀 -> 放行
    names = ["Alpha", "Alphabet"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Alpha": {"x", "y"}, "Alphabet": {"x", "y"}}
    doc_of = {"Alpha": set(), "Alphabet": set()}
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        doc_of=doc_of, require_same_doc=True,
    ) != []


# ---- Step3 规则4：每节点同义边预算 ----

def test_max_per_node_caps_degree():
    # Alpha 同时与 Beta、Gamma、Delta 同义（cos 递减）；后三者邻居互不重叠故彼此不成对。
    # max_per_node=1 时 Alpha 只保留 cos 最高的一条 Alpha-Beta。
    names = ["Alpha", "Beta", "Gamma", "Delta"]
    matrix = np.array([
        [1.00, 0.00, 0.0],
        [0.98, 0.20, 0.0],   # Alpha-Beta cos 最高
        [0.95, 0.31, 0.0],   # Alpha-Gamma 次之
        [0.92, 0.39, 0.0],   # Alpha-Delta 再次
    ], dtype=np.float32)
    nb = {
        "Alpha": {"x", "y", "z", "w"},
        "Beta": {"x", "p"},      # 与 Alpha 交 {x}，与 Gamma/Delta 不交
        "Gamma": {"y", "q"},     # 与 Alpha 交 {y}
        "Delta": {"z", "r"},     # 与 Alpha 交 {z}
    }
    # 不限预算：Alpha 与 Beta、Gamma、Delta 三条边
    full = find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1)
    a_full = [p for p in full if "Alpha" in (p[0], p[1])]
    assert len(a_full) == 3
    # 预算=1：Alpha 只剩 cos 最高的 Alpha-Beta
    capped = find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1, max_per_node=1)
    a_cap = [p for p in capped if "Alpha" in (p[0], p[1])]
    assert len(a_cap) == 1
    assert frozenset((a_cap[0][0], a_cap[0][1])) == frozenset(("Alpha", "Beta"))


# ---- Step3 类型过滤：跳过 person 治人名假阳性 ----

def test_skip_types_rejects_person_pairs():
    # 同姓不同人：cos+Jaccard 双高（共同作者邻居重合）本会误连；类型过滤(person)应拒
    names = ["Weizhi Zhang", "Weizhi Chen"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Weizhi Zhang": {"Coauthor", "Paper"}, "Weizhi Chen": {"Coauthor", "Paper"}}
    type_of = {"Weizhi Zhang": "person", "Weizhi Chen": "person"}
    # 不过滤：会误连（旧版行为）
    assert find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1) != []
    # 过滤 person：被拒
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        type_of=type_of, skip_types={"person"},
    ) == []


def test_skip_types_keeps_nonperson_pairs():
    # 非 person 类型(method)不受 person 过滤影响
    names = ["LightRAG", "Light RAG"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"LightRAG": {"x", "y"}, "Light RAG": {"x", "y"}}
    type_of = {"LightRAG": "method", "Light RAG": "method"}
    pairs = find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        type_of=type_of, skip_types={"person"},
    )
    assert {frozenset((a, b)) for a, b, *_ in pairs} == {frozenset(("LightRAG", "Light RAG"))}


# ---- 精度守卫 A：同类型才连 ----

def test_require_same_type_rejects_cross_type():
    names = ["Alpha", "Beta"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Alpha": {"x", "y"}, "Beta": {"x", "y"}}
    type_of = {"Alpha": "method", "Beta": "concept"}
    # 不开守卫：会连
    assert find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1) != []
    # 开 same_type：跨类型被拒
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        type_of=type_of, require_same_type=True,
    ) == []


def test_require_same_type_keeps_same_type():
    names = ["Alpha", "Beta"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Alpha": {"x", "y"}, "Beta": {"x", "y"}}
    type_of = {"Alpha": "method", "Beta": "method"}
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1,
        type_of=type_of, require_same_type=True,
    ) != []


# ---- 精度守卫 B：名字必须沾边 ----

def test_acronym_of():
    assert _acronym_of("SEC", "Securities and Exchange Commission")
    assert _acronym_of("RAG", "Retrieval Augmented Generation")
    assert not _acronym_of("CAT", "Securities and Exchange Commission")
    assert not _acronym_of("S", "Securities")  # 太短(<2 字母)


def test_name_match_accepts_true_synonym_forms():
    assert _names_name_match("Hybrid Retrieval", "Hybrid Retrieval Mechanism")  # 包含
    assert _names_name_match("SEC", "Securities and Exchange Commission")        # 缩写
    assert _names_name_match("Rating", "Ratings")                                # 单复数
    assert _names_name_match("color", "colour")                                  # 拼写变体
    assert _names_name_match("LightRAG", "Light RAG")                            # 字符高相似


def test_name_match_rejects_different_qualifier():
    # 换词的不同概念：共享一个词但既不包含也不缩写也不高相似 -> 拒
    assert not _names_name_match("Net income", "Operating income")
    assert not _names_name_match("training loss", "validation loss")


def test_require_name_match_rejects_desc_similar_unrelated():
    # 名字不沾边但 cos+jaccard 高（描述样板相似）-> name_match 应拒
    names = ["Net income", "Operating income"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Net income": {"x", "y"}, "Operating income": {"x", "y"}}
    # 不开守卫：会连（共享 income，默认不查名字）
    assert find_synonym_pairs(names, matrix, nb, tau=0.9, theta=0.1) != []
    # 开 name_match：被拒
    assert find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1, require_name_match=True,
    ) == []


def test_require_name_match_keeps_containment():
    names = ["Hybrid Retrieval", "Hybrid Retrieval Mechanism"]
    matrix = np.array([[1, 0, 0], [0.99, 0, 0]], dtype=np.float32)
    nb = {"Hybrid Retrieval": {"x", "y"}, "Hybrid Retrieval Mechanism": {"x", "y"}}
    pairs = find_synonym_pairs(
        names, matrix, nb, tau=0.9, theta=0.1, require_name_match=True,
    )
    assert len(pairs) == 1
