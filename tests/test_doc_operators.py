"""doc_operators（确定性算子工具库）路由单测：验证 dispatch 的优先级/命中与 query.py 等价。"""
import sys
from pathlib import Path

# doc_operators 在 reproduce/（脚本目录，非包），手动加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from doc_operators import (  # noqa: E402
    Ctx,
    build_route_prompt,
    dispatch,
    neural_dispatch,
    parse_route_response,
)


def _ctx():
    return Ctx(
        pdf_path="x.pdf",
        doc_stats={"_text": "Revenue grew. revenue again. CDP CDP.",
                   "pages": 9, "words": 100, "top_words": ["the"],
                   "top_abbrevs": [("CDP", 2)], "figures": 3, "tables": 1,
                   "equations": 0, "figures_body": 3, "tables_body": 1},
        locate_index='(section) "Related Work" -> page 5',
        kw_visual=False, kw_table=False,
    )


def test_meta_pages():
    note, fired = dispatch("How many pages does the document have in total?", _ctx())
    assert "meta_stats" in fired and "total pages = 9" in note


def test_mention_has_top_priority():
    note, fired = dispatch('How many times does the document mention "revenue"?', _ctx())
    assert fired[0] == "mention_count" and "appears 2 times" in note


def test_locate_fires():
    _, fired = dispatch("On which page is the related work discussed?", _ctx())
    assert "locate" in fired


def test_meta_and_locate_stack():
    # "on which page" 含 page → meta 命中；locate 叠加（统计类与定位类可共存）
    _, fired = dispatch("On which page does the paper discuss future work?", _ctx())
    assert "locate" in fired and "meta_stats" in fired


def test_visual_suppresses_meta():
    c = _ctx()
    c.kw_table = True
    _, fired = dispatch("What is the value on page 3 of the table?", c)
    assert "meta_stats" not in fired  # 图/表意图题避让统计注入


def test_content_question_no_fire():
    note, fired = dispatch("What is the BLEU score of the model?", _ctx())
    assert not fired and note == ""


# ---------------------------------------------------------------------------
# 神经路由（function-calling）：route_fn 注入 mock LLM，验证选算子→确定性执行
# ---------------------------------------------------------------------------

def test_route_prompt_lists_tools():
    p = build_route_prompt("How many pages?")
    assert "mention_count" in p and "locate" in p and "Tools to call:" in p


def test_parse_route_response():
    assert parse_route_response("meta_stats") == ["meta_stats"]
    assert parse_route_response("none") == []
    # 多个 + 保持注册表优先级顺序
    assert parse_route_response("locate, meta_stats") == ["meta_stats", "locate"]


def test_neural_dispatch_meta():
    note, fired = neural_dispatch("How many pages?", _ctx(), lambda p: "meta_stats")
    assert "meta_stats" in fired and "total pages = 9" in note


def test_neural_dispatch_none():
    note, fired = neural_dispatch("What is the BLEU score?", _ctx(), lambda p: "none")
    assert not fired and note == ""


def test_neural_dispatch_locate_stacks():
    note, fired = neural_dispatch("which page is future work", _ctx(),
                                  lambda p: "meta_stats, locate")
    assert "locate" in fired and "meta_stats" in fired
