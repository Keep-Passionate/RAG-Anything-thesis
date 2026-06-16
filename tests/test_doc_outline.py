"""doc_outline（文档前页接地）纯函数单测：前页意图检测 / 前页抽取 / 注入格式化。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_outline import (  # noqa: E402
    detect_structure_intent,
    format_outline_note,
    load_outline,
)


def test_frontmatter_intent_positive():
    pos = [
        "Where does the paper publish at?",
        "Where was this document published?",
        "When was the document approved?",
        "What is the publication date of the report?",
        "Which conference was this paper presented at?",
        "Are all authors from the same affiliation?",
        "What is the affiliation of the authors?",
    ]
    for q in pos:
        assert detect_structure_intent(q) == "frontmatter", q


def test_frontmatter_intent_negative():
    """普通内容题、以及已移除的章节题，都不应触发（章节分支实测帮倒忙已删）。"""
    neg = [
        "What is the BLEU score of the model?",
        "Who is the CEO of the company?",
        "How many pages does the document have?",   # DSG 的统计题，不是前页题
        "What method is published in this paper?",  # published 但非"在哪发表"
        "How many parts does the report consist of?",  # 章节题：已不再处理（曾帮倒忙）
        "What is the major topic of section 2.5?",     # 章节题：已不再处理
    ]
    for q in neg:
        assert detect_structure_intent(q) is None, q


def test_load_outline_safe_on_missing():
    """定位不到 content_list（如随便给个 pdf 路径）时安全返回 ''，不抛异常。"""
    assert load_outline("/nonexistent/nope.pdf") == ""
    assert load_outline("") == ""


def test_format_outline_note():
    note = format_outline_note("Proceedings of EMNLP 2019, Hong Kong")
    assert "EMNLP 2019" in note and note.startswith("[") and note.endswith("]")
    # 空文本安全返回 ''
    assert format_outline_note("") == ""
    assert format_outline_note(None) == ""
