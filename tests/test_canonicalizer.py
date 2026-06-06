"""L1 名称规范化单元测试。运行：pytest tests/test_canonicalizer.py -v"""
import unicodedata

from raganything.graph_fusion import normalize_entity_name, normalize_chunk_results


def test_strip_and_collapse_whitespace():
    assert normalize_entity_name("  Figure   2 ") == "Figure 2"


def test_strip_trailing_punctuation():
    assert normalize_entity_name("Figure 2.") == "Figure 2"
    assert normalize_entity_name("Section 3:") == "Section 3"
    assert normalize_entity_name("model,") == "model"


def test_protect_symbol_entities():
    # 万无一失：绝不能把 C++ / C# 砍坏
    assert normalize_entity_name("C++") == "C++"
    assert normalize_entity_name("C#") == "C#"
    assert normalize_entity_name("F#") == "F#"


def test_does_not_lowercase():
    # 万无一失版不小写（小写有 WHO/who 风险，留给 L2）
    assert normalize_entity_name("DAE") == "DAE"
    assert normalize_entity_name("COVID-19") == "COVID-19"


def test_unicode_nfc_equivalence():
    # é 的组合式(NFD)与预组合式应归一为同一字符串
    a = normalize_entity_name(unicodedata.normalize("NFD", "café"))
    b = normalize_entity_name("café")
    assert a == b


def test_variants_collapse_to_same_key():
    assert normalize_entity_name("Figure 2 ") == normalize_entity_name("Figure  2")
    assert normalize_entity_name("Figure 2.") == normalize_entity_name("Figure 2")


def test_none_and_empty():
    assert normalize_entity_name(None) is None
    assert normalize_entity_name("   ") == ""


def test_normalize_chunk_results():
    nodes = {"Figure 2. ": [{"entity_name": "Figure 2. ", "entity_type": "image"}]}
    edges = {("Figure 2. ", "DAE"): [{"src_id": "Figure 2. ", "tgt_id": "DAE"}]}
    (n, e), = normalize_chunk_results([(nodes, edges)])
    assert "Figure 2" in n
    assert n["Figure 2"][0]["entity_name"] == "Figure 2"
    assert ("Figure 2", "DAE") in e
    assert e[("Figure 2", "DAE")][0]["src_id"] == "Figure 2"
