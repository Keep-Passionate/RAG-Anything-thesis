"""doc_outline：文档前页接地（ENABLE_DOC_OUTLINE）——治"问发表/批准/作者单位"等前页元信息题。

与 DSG（doc_meta）平行、解耦：DSG 算全局标量（页数/词频/缩写），doc_outline 专补一类
DSG 够不到的题——答案印在文档"门面"（标题页 / 首页 footer）上的元信息：
  发表在哪 / 出版日期 / 文档何时批准 / 作者是否同一单位。
这类答案既不在检索块里、也不是统计量，而在 MinerU content_list 的前页（page_idx 0~1）正文里。
做法与 DSG 同哲学（确定性接地）：把"门面信息"确定性地抄给模型，让它照着读、不再瞎猜。

设计取舍（重要，论文要写）：早期版本还有"章节树"分支治"有几节 / 某节讲啥"，但 54/74 篇
实测它对 10-K 财报的 "how many parts" 帮倒忙——正式的 Part I~IV 与 MinerU 的 text_level
标题层级对不上、赢0砸1，是纯负担，已**彻底移除**；只保留实测有效、零副作用的前页注入。
（复盘见 commit 46de7aa。这是"宁可不碰、不可帮倒忙"原则的一次落地。）

纯函数可单测 + query.py 一处注入：
  detect_structure_intent(q) -> "frontmatter" | None   （关键词，零 LLM 成本）
  load_outline(pdf_path)     -> str                    （前 1~2 页正文，找不到则 ""）
  format_outline_note(text)  -> str                    （中性注入文本）
"""

import json
import re

# 前页元信息意图：发表 / 出版 / 批准 / 出处 / 作者单位。
# 稳妥优先：只收高确定性短语，要求 author 上下文等约束，避免误命中正文里普通的
# "published / affiliation / section"。
_FRONT_RES = (
    re.compile(r"\bwhere\b.{0,40}\bpublish", re.IGNORECASE),
    re.compile(r"\bwhen\b.{0,40}\b(approved|published|revised|dated|issued)\b", re.IGNORECASE),
    re.compile(r"\b(publication|published)\s+(venue|date|year)\b", re.IGNORECASE),
    re.compile(r"\b(which|what)\s+(conference|journal|venue|proceedings)\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+(publication|approval|issue)\b", re.IGNORECASE),
    re.compile(r"\b(authors?|institution|organi[sz]ation)\b.{0,40}\baffiliat", re.IGNORECASE),
    re.compile(r"\baffiliat\w*\b.{0,40}\bauthors?\b", re.IGNORECASE),
)


def detect_structure_intent(question: str):
    """前页元信息题返回 'frontmatter'，否则 None。零 LLM 成本。

    （保留返回字符串而非 bool，兼容既有调用与诊断脚本 `== 'frontmatter'` 的写法。）
    """
    q = question or ""
    return "frontmatter" if any(r.search(q) for r in _FRONT_RES) else None


def load_outline(pdf_path: str, max_chars: int = 1500) -> str:
    """从 MinerU content_list 取前 1~2 页正文（标题页 / 首页 footer）。找不到则返回 ''（无害退回）。"""
    try:
        from doc_meta import locate_content_list  # 复用 DSG 的解析定位工具

        cl = locate_content_list(pdf_path)
        if not cl:
            return ""
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return ""

    parts = []
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = (it.get("text") or "").strip()
        pg = it.get("page_idx")
        if txt and isinstance(pg, int) and pg <= 1:  # 前两页（0、1）的正文
            parts.append(txt)
    return "\n".join(parts)[:max_chars]


def format_outline_note(front_text: str) -> str:
    """把前页正文拼成中性的"提取的前页信息"注入文本。空则返回 ''。"""
    if not front_text:
        return ""
    return (
        "[Document front matter (first pages, extracted programmatically — publication "
        "venue / date / approval / author affiliation is usually printed here):\n"
        f"{front_text}]"
    )
