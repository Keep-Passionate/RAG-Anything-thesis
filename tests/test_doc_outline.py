"""doc_outline（文档结构接地）纯函数单测：意图检测 / 结构抽取 / 注入格式化。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_outline import (  # noqa: E402
    detect_structure_intent,
    format_outline_note,
    load_outline,
)


def test_structure_intent_frontmatter():
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


def test_structure_intent_sections():
    pos = [
        "How many sections does the document have?",
        "How many parts are in the report?",
        "Which section discusses the methodology?",
        "What is the topic of section 2.5?",
    ]
    for q in pos:
        assert detect_structure_intent(q) == "sections", q


def test_structure_intent_negative():
    """普通内容题不应误触发（误触发只是多一段无害说明，但仍尽量精准）。"""
    neg = [
        "What is the BLEU score of the model?",
        "Who is the CEO of the company?",
        "How many pages does the document have?",   # 这是 DSG 的统计题，不是结构题
        "What method is published in this paper?",  # published 但非"在哪发表"
    ]
    for q in neg:
        assert detect_structure_intent(q) is None, q


def test_load_outline(tmp_path):
    p = tmp_path / "x_content_list.json"
    p.write_text(json.dumps([
        {"type": "text", "text": "Proceedings of EMNLP 2019, Hong Kong", "page_idx": 0},
        {"type": "text", "text_level": 1, "text": "1 Introduction", "page_idx": 1},
        {"type": "text", "text": "body text on page 2", "page_idx": 2},
        {"type": "text", "text_level": 1, "text": "2 Methods", "page_idx": 3},
        {"type": "text", "text_level": 2, "text": "2.1 Setup", "page_idx": 3},
    ]), encoding="utf-8")
    o = load_outline(str(p))   # locate 失败时退 None；这里直接喂路径需 locate 命中
    # load_outline 内部用 locate_content_list(pdf_path)，单测里 pdf_path 无法定位，
    # 故这里只验证：传一个非 content_list 路径时安全返回 None（不抛异常）。
    assert load_outline(str(tmp_path / "nope.pdf")) is None


def test_format_outline_note():
    o = {
        "frontmatter": "Proceedings of EMNLP 2019, Hong Kong",
        "sections": [(1, "1 Introduction"), (2, "2.1 Setup"), (1, "2 Methods")],
        "n_sections": 2,
    }
    fm = format_outline_note(o, "frontmatter")
    assert "EMNLP 2019" in fm and fm.startswith("[") and fm.endswith("]")
    sec = format_outline_note(o, "sections")
    assert "Introduction" in sec and "Total level-1 sections: 2" in sec
    # 空/不匹配安全返回 ''
    assert format_outline_note(None, "sections") == ""
    assert format_outline_note({"frontmatter": ""}, "frontmatter") == ""
