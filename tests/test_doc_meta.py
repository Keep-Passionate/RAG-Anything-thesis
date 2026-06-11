"""doc_meta（文档统计量注入）纯函数单测：意图检测 / 文本统计 / 说明格式化。"""

import sys
from pathlib import Path

# doc_meta 在 reproduce/（脚本目录，非包），手动加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_meta import detect_meta_intent, format_stats_note, text_stats  # noqa: E402


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
