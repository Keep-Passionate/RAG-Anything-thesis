"""modality（模态路由/重排视觉保位）纯函数单测。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from modality import (  # noqa: E402
    count_image_evidence,
    detect_visual_intent,
    guard_rerank_results,
    is_visual_chunk,
)


# ---------------------------------------------------------------------------
# detect_visual_intent（从 query.py 迁来，行为保持）
# ---------------------------------------------------------------------------

def test_visual_intent():
    v, t = detect_visual_intent("In Figure 4, what is the function of Word LSTM-B?")
    assert v and not t
    v, t = detect_visual_intent("What is the highest F1 according to Table 10?")
    assert t
    v, t = detect_visual_intent("How many authors are listed in the paper?")
    assert not v and not t
    v, t = detect_visual_intent("图2显示了什么趋势？")
    assert v


# ---------------------------------------------------------------------------
# count_image_evidence（与 raganything 的 Image Path 标记格式一致）
# ---------------------------------------------------------------------------

def test_count_image_evidence():
    ctx = (
        "Entity: Figure 2\nImage Path: /data/output/images/fig2.jpg\n"
        "some text...\nImage Path: ./imgs/chart_01.PNG\n"
        "Image Path: not_an_image.txt\n"  # 非图片扩展名不算
    )
    assert count_image_evidence(ctx) == 2
    assert count_image_evidence("") == 0
    assert count_image_evidence("plain text only") == 0


# ---------------------------------------------------------------------------
# is_visual_chunk
# ---------------------------------------------------------------------------

def test_is_visual_chunk():
    assert is_visual_chunk("caption...\nImage Path: a/b.png")
    assert is_visual_chunk("<table><td>1</td></table>")
    assert is_visual_chunk("| col1 | col2 | col3 |\n| 1 | 2 | 3 |")  # markdown 表
    assert not is_visual_chunk("ordinary prose paragraph about revenue")
    assert not is_visual_chunk("a | b")  # 偶发管道符不算表


# ---------------------------------------------------------------------------
# guard_rerank_results
# ---------------------------------------------------------------------------

def _mk(scores):
    """按给定顺序构造重排结果（已按分降序）。"""
    return [{"index": i, "relevance_score": s} for i, s in scores]


def test_guard_promotes_visual_for_table_question():
    docs = [
        "prose A", "prose B", "prose C",
        "| h1 | h2 | h3 |\n| 1 | 2 | 3 |",   # idx3 表块（被排到截断线下）
        "caption\nImage Path: x/y.jpg",        # idx4 图块（截断线下）
    ]
    results = _mk([(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.4), (4, 0.3)])
    out = guard_rerank_results(results, docs, "What does Table 3 show?", top_n=3, min_visual=2)
    kept = [r["index"] for r in out]
    assert len(kept) == 3
    assert 3 in kept and 4 in kept          # 两个视觉块都被提进来
    assert kept[0] == 0                      # 最高分文本块保住
    assert all(r["index"] != 2 for r in out) and all(r["index"] != 1 for r in out)


def test_guard_noop_for_text_question():
    docs = ["prose A", "prose B", "Image Path: x.png"]
    results = _mk([(0, 0.9), (1, 0.8), (2, 0.1)])
    out = guard_rerank_results(results, docs, "Who is the CEO?", top_n=2, min_visual=2)
    assert [r["index"] for r in out] == [0, 1]  # 纯文本题：原样截断，零影响


def test_guard_noop_when_already_enough_visual():
    docs = ["| a | b | c |\n| 1 | 2 | 3 |", "Image Path: x.jpg", "prose"]
    results = _mk([(0, 0.9), (1, 0.8), (2, 0.7)])
    out = guard_rerank_results(results, docs, "According to the table...", top_n=2, min_visual=2)
    assert [r["index"] for r in out] == [0, 1]  # 截断内已有 2 个视觉块，不动


def test_guard_handles_no_truncation():
    docs = ["prose"]
    results = _mk([(0, 0.9)])
    assert guard_rerank_results(results, docs, "Table?", top_n=None) == results
    assert guard_rerank_results(results, docs, "Table?", top_n=5) == results
