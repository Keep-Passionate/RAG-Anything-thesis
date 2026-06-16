"""doc_anchor（被点名元素接地）纯函数单测：引用检测 / 按标号定位 / 注入格式化。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_anchor import (  # noqa: E402
    detect_element_reference,
    find_referenced_element,
    format_anchor_note,
    load_content_items,
)


def test_detect_reference_positive():
    cases = {
        "What does Table 8 show for the BERT model?": ("table", "8"),
        "According to Table 10, what is the highest F1 score?": ("table", "10"),
        "In Figure 4, what is the function of Word LSTM-B?": ("figure", "4"),
        "According to Figure 3, what is the sequence?": ("figure", "3"),
        "See Fig. 2 for the architecture.": ("figure", "2"),
    }
    for q, want in cases.items():
        assert detect_element_reference(q) == want, q


def test_detect_reference_negative():
    """数量题（归 DSG）、无标号、非表图引用都不应命中。"""
    neg = [
        "How many tables are in the document?",   # 数量题：table 后无标号
        "How many figures excluding appendix?",    # 同上
        "What is the table of contents about?",     # table of：无标号
        "What is the main content of Section III?", # section：刻意不处理
        "What is the BLEU score of the model?",     # 普通内容题
    ]
    for q in neg:
        assert detect_element_reference(q) is None, q


def test_label_boundary_no_prefix_match():
    """Table 8 不应误命中 Table 80（词边界）。"""
    items = [
        {"type": "table", "table_caption": ["Table 80: unrelated"], "table_body": "X"},
        {"type": "table", "table_caption": ["Table 8: target"], "table_body": "Y"},
    ]
    out = find_referenced_element(items, "table", "8")
    assert "target" in out and "Y" in out
    assert "unrelated" not in out


def test_find_table_returns_caption_and_body():
    items = [
        {"type": "text", "text": "intro"},
        {"type": "table", "table_caption": ["Table 1: Performance"],
         "table_body": "<table><tr><td>93.14</td></tr></table>"},
    ]
    out = find_referenced_element(items, "table", "1")
    assert "Performance" in out and "93.14" in out


def test_find_figure_returns_caption_only():
    """figure 只返回 caption(+footnote) 文本，不含图像路径。"""
    items = [
        {"type": "image", "img_path": "/x/fig3.jpg",
         "image_caption": ["Figure 3: Label generation sequence"],
         "image_footnote": ["best viewed in color"]},
    ]
    out = find_referenced_element(items, "figure", "3")
    assert "Label generation sequence" in out
    assert "best viewed in color" in out
    assert "fig3.jpg" not in out


def test_find_missing_returns_empty():
    items = [{"type": "table", "table_caption": ["Table 2: x"], "table_body": "b"}]
    assert find_referenced_element(items, "table", "9") == ""   # 标号不存在
    assert find_referenced_element([], "figure", "1") == ""      # 空列表


def test_format_anchor_note():
    note = format_anchor_note("table", "8", "Table 8: ...\n<rows>")
    assert note.startswith("[") and note.endswith("]")
    assert "Table 8" in note and "<rows>" in note
    # 空内容安全返回 ''
    assert format_anchor_note("table", "8", "") == ""


def test_load_content_items_safe_on_missing():
    """定位不到 content_list 时安全返回 []，不抛异常。"""
    assert load_content_items("/nonexistent/nope.pdf") == []
    assert load_content_items("") == []
