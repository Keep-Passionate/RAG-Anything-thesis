"""doc_locate（页码定位接地）纯函数单测：定位意图检测 / 索引构造安全退化 / 注入格式化。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_locate import (  # noqa: E402
    build_heading_page_index,
    detect_location_intent,
    format_locate_note,
)


def test_location_intent_positive():
    pos = [
        "On which page is the table of contents of the report?",
        "On which page do the researchers discuss future work?",
        "From which page does the document start to state the situation?",
        "On what page is the signature located?",
        "Which page does the paper introduce the corpus statistics on?",
    ]
    for q in pos:
        assert detect_location_intent(q), q


def test_location_intent_negative():
    neg = [
        "How many pages does the document have?",   # 数页数，不是定位
        "What is the summary of page 20?",          # 问某页内容，不是"在第几页"
        "Who is the author of the paper?",
    ]
    for q in neg:
        assert not detect_location_intent(q), q


def test_build_safe_on_missing():
    """定位不到 content_list 时安全返回 ''，不抛异常。"""
    assert build_heading_page_index("/nonexistent/nope.pdf") == ""
    assert build_heading_page_index("") == ""


def test_format_locate_note():
    note = format_locate_note('"1 Introduction" -> page 1\n"2 Methods" -> page 3')
    assert "Introduction" in note and "page 3" in note
    assert note.startswith("[") and note.endswith("]")
    assert format_locate_note("") == ""
