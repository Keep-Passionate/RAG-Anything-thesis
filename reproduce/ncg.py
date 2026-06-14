"""NCG：表格数值计算接地（ENABLE_NCG）——治 multimodal-t 财报"计算类"题。

为什么 DSG 解决不了这类题：DSG 算的是文档【全局标量】（页数/词频/缩写），完全不碰
"表格内的数值计算"。财报计算题（算差值/变化/百分比，如"2019→2020 毛利润下降多少"）
需要：① 找对表 ② 抽对数字 ③ 多步计算——DSG 三步都不做；而 paperbase 全开 VLM 让
模型看图【心算】，LLM 多步心算大数极易出错。

NCG 用 Program-of-Thoughts（PoT）思路（与 DSG 同哲学：把计算外包给确定性程序）：
让模型先从检索到的表格里【列出数字 + 写算式】（不让它直接算），再用【受限求值器】
程序执行，得到精确结果，作为"计算辅助"注入，帮正常生成答对。

设计（纯函数可单测 + 一处 LLM 调用在 query.py）：
  detect_calc_intent(q)          : 关键词识别计算类题（零成本）
  build_extraction_prompt(q,ctx) : 构造让模型输出 JSON{numbers,formula} 的提示（含 few-shot）
  parse_ncg_json(text)           : 解析模型输出的 JSON
  safe_eval(formula, numbers)    : 受限算式求值（ast 安全解析，只允许 + - * / () 和数字/已声明变量）

边界（论文要写清）：只治表格数值计算，不治图片题（multimodal-f 仍靠 VLM）；
受检索召回 + MinerU 表格解析质量制约（对的表没检索到、或解析乱→抽数错）。
文献：Program-of-Thoughts(arXiv:2211.12588)、TableCall(2026)、PAL(arXiv:2211.10435)。
"""

import ast
import json
import operator
import re

# 命中任一关键词即认为是"需要数值计算"的题。
_CALC_KWS = (
    "calculate", "how much did", "how many more", "how many fewer",
    "difference between", "increase", "decrease", "decline", "growth",
    "change from", "change in", "percentage", "percent", " ratio",
    "average", "sum of", "total number of", "exceed", "more than",
)


def detect_calc_intent(question: str) -> bool:
    """问题是否需要数值计算。零 LLM 成本。"""
    q = (question or "").lower()
    return any(k in q for k in _CALC_KWS)


_FEWSHOT = (
    "Example 1:\n"
    "Question: By how much did revenue change from 2019 (12.3M) to 2020 (15.1M)?\n"
    'Output: {"numbers": {"rev_2019": 12.3, "rev_2020": 15.1}, "formula": "rev_2020 - rev_2019"}\n'
    "Example 2:\n"
    "Question: What percentage increase in stores from 100 to 130?\n"
    'Output: {"numbers": {"a": 100, "b": 130}, "formula": "(b - a) / a * 100"}'
)


def build_extraction_prompt(question: str, context: str, max_ctx: int = 6000) -> str:
    """构造让模型从表格抽数字 + 列算式的提示（PoT；不让它直接算）。"""
    ctx = (context or "")[:max_ctx]
    return (
        "You are given a question that requires NUMERICAL CALCULATION over a document's "
        "tables, plus the retrieved context. Do NOT compute the final answer yourself. "
        "Instead, extract the exact numbers needed from the context and express the "
        "calculation as a formula.\n\n"
        f"{_FEWSHOT}\n\n"
        "Rules: each value in \"numbers\" must be a plain number found in the context; "
        "\"formula\" may ONLY use the declared number names and the operators + - * / and "
        "parentheses. Output ONLY a JSON object with keys \"numbers\" (name->number) and "
        '"formula" (string). If the needed numbers are NOT clearly in the context, output '
        '{"numbers": {}, "formula": ""}.\n\n'
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nOutput:"
    )


def parse_ncg_json(text: str):
    """从模型输出里解析 {numbers, formula}。失败返回 None。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)  # 容忍模型在 JSON 前后加解释
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(d, dict) and isinstance(d.get("numbers"), dict) and "formula" in d:
        return d
    return None


# ---------------------------------------------------------------------------
# 受限求值：只允许数字、已声明变量、+ - * / 和括号。绝不 exec 任意代码。
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node, names):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand, names))
    raise ValueError("disallowed expression")


def safe_eval(formula: str, numbers: dict):
    """安全求值 formula（只允许 + - * / ()、数字、numbers 里声明的变量）。失败返回 None。"""
    if not formula:
        return None
    try:
        names = {k: float(v) for k, v in (numbers or {}).items()}
        tree = ast.parse(formula, mode="eval")
        result = _eval_node(tree, names)
        return round(result, 4) if isinstance(result, float) else result
    except Exception:
        return None


def format_calc_note(numbers: dict, formula: str, value) -> str:
    """把程序算出的结果拼成"计算辅助"注入文本（中性，模型可参考）。"""
    return (
        f"[Programmatic calculation aid (verify against the question): using values "
        f"{numbers} and formula '{formula}', the computed result is {value}. "
        "Use it only if it directly answers the question.]"
    )
