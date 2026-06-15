"""doc_outline：文档结构接地（ENABLE_DOC_OUTLINE）——治"找结构/前页元信息"题。

与 DSG（doc_meta）平行、解耦：DSG 算全局标量（页数/词频/缩写），doc_outline 治
"在文档结构里定位"的题——发表在哪/出版日期/文档何时批准/有几节几个 part/某节讲什么。
这类答案既不在检索块里、也不是统计量，而在 MinerU content_list 的结构信息里：
  - 前页元信息（venue/日期/批准/出处）→ page_idx 0~1 的正文文本（标题页/footer）
  - 章节大纲 / 节数 → text_level 标题层级

为什么 DSG 不能覆盖：DSG 注入的是"全局标量"，不含"第一页 footer 原文"或"章节标题树"；
本模块补的正是这两块确定性结构信息。设计哲学与 DSG 一致（确定性接地），但**独立文件、
独立开关、独立结果名**，与 doc_meta 平行，不改 DSG 一行，便于单独消融。

设计（纯函数可单测 + query.py 一处注入）：
  detect_structure_intent(q) -> "frontmatter" | "sections" | None  （关键词，零成本）
  load_outline(pdf_path)     -> {"frontmatter": str, "sections": [(level,text)], "n_sections": int}
  format_outline_note(o, intent) -> str

边界：text_level 抽章节有噪声（MinerU 偶尔漏标/错标），故注入措辞中性、标注"提取的"、
仅供参考；默认关、找不到 content_list 静默退回原行为（无害）。
"""

import json
import re

# 前页元信息意图：发表/出版/批准/出处——答案多在标题页或首页 footer。
# 稳妥优先：只收高确定性短语，避免误命中正文里的普通"published/section"。
_FRONT_RES = (
    re.compile(r"\bwhere\b.{0,40}\bpublish", re.IGNORECASE),
    re.compile(r"\bwhen\b.{0,40}\b(approved|published|revised|dated|issued)\b", re.IGNORECASE),
    re.compile(r"\b(publication|published)\s+(venue|date|year)\b", re.IGNORECASE),
    re.compile(r"\b(which|what)\s+(conference|journal|venue|proceedings)\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+(publication|approval|issue)\b", re.IGNORECASE),
    # 作者单位/隶属——印在标题页(前页正文),实测题例 "are all authors from the
    # same affiliation"。要求 author 上下文，避免误命中正文里普通的 affiliation。
    re.compile(r"\b(authors?|institution|organi[sz]ation)\b.{0,40}\baffiliat", re.IGNORECASE),
    re.compile(r"\baffiliat\w*\b.{0,40}\bauthors?\b", re.IGNORECASE),
)

# 章节结构意图：数节/数 part，或"哪节讲 X / 某节主题"。
_SECTION_RES = (
    re.compile(r"\bhow many\s+(sections?|parts?|chapters?|subsections?)\b", re.IGNORECASE),
    re.compile(r"\b(which|what)\s+section\b.{0,40}\b(discuss|describe|cover|about|topic|present)", re.IGNORECASE),
    re.compile(r"\b(topic|subject|content)\s+of\s+section\b", re.IGNORECASE),
)


def detect_structure_intent(question: str):
    """返回 'frontmatter' / 'sections' / None。零 LLM 成本。"""
    q = question or ""
    if any(r.search(q) for r in _FRONT_RES):
        return "frontmatter"
    if any(r.search(q) for r in _SECTION_RES):
        return "sections"
    return None


def load_outline(pdf_path: str, max_front: int = 1500):
    """从 MinerU content_list 取前页正文 + 章节标题树。找不到则返回 None（无害退回）。"""
    try:
        from doc_meta import locate_content_list  # 复用 DSG 的解析定位工具

        cl = locate_content_list(pdf_path)
        if not cl:
            return None
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return None

    front_parts, sections = [], []
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = (it.get("text") or "").strip()
        pg = it.get("page_idx")
        lvl = it.get("text_level")
        # 前页（page_idx 0~1）正文：标题页 / 首页 footer 常含 venue/日期/出处
        if txt and isinstance(pg, int) and pg <= 1:
            front_parts.append(txt)
        # 章节标题：text_level 有值即为标题层级
        if lvl and txt:
            sections.append((lvl, txt))

    if not front_parts and not sections:
        return None
    front = "\n".join(front_parts)[:max_front]
    n_sections = sum(1 for lvl, _ in sections if lvl == 1)
    return {"frontmatter": front, "sections": sections, "n_sections": n_sections}


def format_outline_note(outline: dict, intent: str) -> str:
    """按意图把结构信息拼成中性的"提取的结构信息"注入文本。无可用信息则返回 ''。"""
    if not outline:
        return ""
    if intent == "frontmatter" and outline.get("frontmatter"):
        return (
            "[Document front matter (first pages, extracted programmatically — "
            "publication venue / date / approval info is usually here):\n"
            f"{outline['frontmatter']}]"
        )
    if intent == "sections" and outline.get("sections"):
        titles = "; ".join(f"L{lvl}:{t}" for lvl, t in outline["sections"][:40])
        return (
            "[Document section outline (level:title, extracted programmatically): "
            f"{titles}. Total level-1 sections: {outline['n_sections']}.]"
        )
    return ""
