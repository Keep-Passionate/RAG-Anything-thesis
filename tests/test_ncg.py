"""NCG（表格数值计算接地）纯函数单测：意图检测 / JSON解析 / 安全求值。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce"))

from ncg import (  # noqa: E402
    detect_calc_intent,
    parse_ncg_json,
    safe_eval,
)


def test_calc_intent():
    pos = [
        "How much did revenue change from 2019 to 2020?",
        "Calculate the decline in gross profit.",
        "What is the percentage increase in stores?",
        "What is the difference between owned and franchise stores?",
        "How many more shares does B Blundy hold?",
    ]
    for q in pos:
        assert detect_calc_intent(q), q
    neg = [
        "Who is the CEO of the company?",
        "What is the main content of Section III?",
        "How many pages does the document have?",
        # 收紧后这些不再误触发（去掉了 exceed/average/total number of 等宽词）
        "Does the company's revenue exceed 1 billion?",
        "What is the average tenure of board members?",
        "What is the total number of authors?",
    ]
    for q in neg:
        assert not detect_calc_intent(q), q


def test_parse_ncg_json():
    # 容忍 JSON 前后有解释文字
    p = parse_ncg_json('here: {"numbers": {"a": 12.3, "b": 15.1}, "formula": "b - a"} done')
    assert p == {"numbers": {"a": 12.3, "b": 15.1}, "formula": "b - a"}
    # 空/无效
    assert parse_ncg_json("") is None
    assert parse_ncg_json("no json here") is None
    # 缺字段
    assert parse_ncg_json('{"formula": "a+b"}') is None


def test_safe_eval_correct():
    assert safe_eval("b - a", {"a": 12.3, "b": 15.1}) == 2.8
    assert safe_eval("(b - a) / a * 100", {"a": 100, "b": 130}) == 30.0
    assert safe_eval("a + b - c", {"a": 1, "b": 2, "c": 3}) == 0
    assert safe_eval("-a", {"a": 5}) == -5


def test_safe_eval_rejects_unsafe():
    # 恶意代码 / 未声明变量 / 空式 一律返回 None，绝不执行
    assert safe_eval('__import__("os").system("ls")', {}) is None
    assert safe_eval("open('x')", {}) is None
    assert safe_eval("a ** b", {"a": 2, "b": 3}) is None  # 幂运算不在白名单
    assert safe_eval("unknown_var + 1", {}) is None
    assert safe_eval("", {"a": 1}) is None
