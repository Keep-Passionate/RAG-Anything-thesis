"""doc_meta（文档统计量注入）纯函数单测：意图检测 / 文本统计 / 说明格式化。"""

import json
import sys
from pathlib import Path

# doc_meta 在 reproduce/（脚本目录，非包），手动加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_meta import (  # noqa: E402
    count_elements,
    detect_count_intent,
    detect_meta_intent,
    find_page_reference,
    format_stats_note,
    text_stats,
)


# ---------------------------------------------------------------------------
# detect_meta_intent
# ---------------------------------------------------------------------------

def test_meta_intent_positive():
    """DocBench 真实 meta-data 题应被识别。"""
    positives = [
        "How many pages does the document have in total?",
        "What is the most common abbreviation in the report?",
        "What is the top 3 most frequent words mentioned in the report?",
        "How many words are there in the document?",
        "On which page does the document present the signature of US District Judge?",
        "What is the brief summary of page 20?",
        "这篇文档总共有多少页？",
    ]
    for q in positives:
        assert detect_meta_intent(q), q


def test_meta_intent_negative():
    """内容题不应误触发（误触发只是多一段无害说明，但仍应尽量精准）。"""
    negatives = [
        "What is the BLEU score for CodeBERT when pre-trained with MLM+RTD objectives?",
        "Which model achieved the highest F1 score in the English WSJ dataset?",
        "Was the document approved before it was revised?",
        "What types of payments will the employee receive upon separation?",
    ]
    for q in negatives:
        assert not detect_meta_intent(q), q


# ---------------------------------------------------------------------------
# text_stats
# ---------------------------------------------------------------------------

def test_text_stats_counts():
    text = "the the the and and report to CDP CDP HIV"
    s = text_stats(text)
    assert s["words"] == 10                      # 空白切分
    assert s["top_words"] == ["the", "and", "cdp"]
    assert s["top_abbrevs"] == [("CDP", 2), ("HIV", 1)]


def test_text_stats_empty():
    s = text_stats("")
    assert s["words"] == 0
    assert s["top_words"] == []
    assert s["top_abbrevs"] == []


# ---------------------------------------------------------------------------
# format_stats_note
# ---------------------------------------------------------------------------

def test_format_stats_note():
    note = format_stats_note(
        {"pages": 9, "words": 8849, "top_words": ["and", "the", "to"],
         "top_abbrevs": [("CDP", 31), ("HIV", 7)]}
    )
    assert "total pages = 9" in note
    assert "8849" in note
    assert "and, the, to" in note
    assert "CDP (31x)" in note
    assert note.startswith("[") and note.endswith("]")
    assert "figures" not in note  # 无元素计数时不出现 v3 字段


def test_format_stats_note_with_elements():
    note = format_stats_note(
        {"pages": 9, "words": 100, "top_words": ["a"], "top_abbrevs": [],
         "figures": 5, "tables": 3, "equations": 2}
    )
    assert "figures = 5" in note and "tables = 3" in note and "equations = 2" in note


# ---------------------------------------------------------------------------
# v3：计数题 / 页码引用 / 元素计数
# ---------------------------------------------------------------------------

def test_count_intent():
    assert detect_count_intent("How many figures are there in the paper (excluding Appendix)?")
    assert detect_count_intent("How many tables does the document contain?")
    assert detect_count_intent("文档里有几张图？")
    assert not detect_count_intent("How many pages does the document have in total?")
    assert not detect_count_intent("What is shown in Figure 4?")


def test_find_page_reference():
    assert find_page_reference("What is the brief summary of page 20?") == 20
    assert find_page_reference("第 7 页的主要内容是什么？") == 7
    assert find_page_reference("On which page does the signature appear?") is None
    assert find_page_reference("") is None


def test_count_elements(tmp_path):
    p = tmp_path / "x_content_list.json"
    p.write_text(json.dumps(
        [{"type": "image"}, {"type": "image"}, {"type": "table"},
         {"type": "text"}, {"type": "equation"}]
    ), encoding="utf-8")
    assert count_elements(p) == {"figures": 2, "tables": 1, "equations": 1}
    assert count_elements(tmp_path / "missing.json") is None
