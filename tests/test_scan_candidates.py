"""scan_candidates 纯函数单测：页码定位检测 / 多项清单答案检测（零数据、零 API）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from scan_candidates import (  # noqa: E402
    _answer_is_multi,
    is_list_answer,
    is_page_location,
)


def test_page_location_positive():
    for q in [
        "On which page does the signature appear?",
        "What page is the executive summary on?",
        "On what page can the balance sheet be found?",
        "What is the page number of the conclusion?",
    ]:
        assert is_page_location(q), q


def test_page_location_negative():
    """泛指 page 名词、数量题、概述题都不该命中。"""
    for q in [
        "What does the home page of the website show?",
        "How many pages does the report have?",
        "Summarize the first page.",
    ]:
        assert not is_page_location(q), q


def test_list_answer_by_question_cue():
    assert is_list_answer("List all the subsidiaries mentioned.", "")
    assert is_list_answer("What are the three pillars of the strategy?", "")
    assert is_list_answer("Name the board members.", "")


def test_list_answer_by_answer_structure():
    assert _answer_is_multi("Alice\nBob\nCarol\nDave")          # 换行多项
    assert _answer_is_multi("1. Risk A 2. Risk B 3. Risk C")    # 编号
    assert _answer_is_multi("apples, oranges, bananas, pears")  # 逗号 >=4 段
    assert _answer_is_multi("foo; bar; baz")                    # 分号 >=3 段


def test_list_answer_negatives():
    assert not _answer_is_multi("$1,234,567.89")   # 纯货币逗号
    assert not _answer_is_multi("42")              # 单值
    assert not _answer_is_multi("New York, USA")   # 两段，不够阈值
    assert not is_list_answer("When was it founded?", "1998")
