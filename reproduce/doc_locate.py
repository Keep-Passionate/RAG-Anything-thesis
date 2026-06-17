"""doc_locate：页码定位接地（ENABLE_DOC_LOCATE）——治"X 在第几页"的题。

确定性接地框架的第三类「Localization 定位」（与 Counting / Statistics 并列）：
  问"目录 / 相关工作 / 未来工作 在第几页"时，答案是文档结构里某个元素的页码。
  检索把文档切块后丢失了"页码归属"，LLM 只能猜；但 MinerU 的 content_list 里每个
  元素都带 page_idx——我们把【章节标题 → 页码】索引直接交给模型，把"在第几页"
  从"猜"变成"查表"。这是 DSG"算 LLM 算不准的量"思想在"定位"维度上的延伸：
  LLM 不擅长在长文里报出准确页码，那就把页码索引算好喂给它。

设计（与 doc_meta / doc_anchor 平行，独立文件 + 独立开关，可单独消融）：
  detect_location_intent(q)      -> bool   （问"在第几页"类，纯关键词，零 LLM 成本）
  build_heading_page_index(pdf)  -> str    （章节标题→物理页码 的索引文本）
  format_locate_note(index)      -> str    （中性注入文本）

边界（承袭"宁可不碰、不可帮倒忙"）：只注入【标题级元素（text_level 有值）的页码】，
不去猜正文内容的位置（正文定位有噪、易错）；content_list 找不到则返回 ''（不注入、
零风险退化）。页码统一用 page_idx + 1（解析的物理页，1 起算）。
"""

import json
import re

# "X 在第几页" 的两种问法：① on/at/from which|what page；② which page ... is/are/located/discuss
_LOC_RE = re.compile(
    r"\b(on|at|from)\s+(which|what)\s+page\b"
    r"|\bwhich\s+page\b[^?]*\b(is|are|does|do|located|show|present|discuss|list|introduce)\b",
    re.IGNORECASE,
)


def detect_location_intent(question: str) -> bool:
    """问题是否在问"某内容/章节在第几页"。零 LLM 成本。"""
    return bool(_LOC_RE.search(question or ""))


def build_heading_page_index(pdf_path: str, max_items: int = 40, max_chars: int = 1500) -> str:
    """读 content_list，构造"章节标题 → 物理页码"索引文本。找不到/无标题则返回 ''。"""
    try:
        from doc_meta import locate_content_list  # 复用 DSG 的解析定位工具

        cl = locate_content_list(pdf_path)
        if not cl:
            return ""
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return ""

    lines = []
    for it in items:
        if not isinstance(it, dict) or not it.get("text_level"):
            continue
        txt = " ".join(str(it.get("text", "")).split()).strip()
        pg = it.get("page_idx")
        if txt and isinstance(pg, int):
            lines.append(f'"{txt[:80]}" -> page {pg + 1}')
        if len(lines) >= max_items:
            break
    if not lines:
        return ""
    return "\n".join(lines)[:max_chars]


def format_locate_note(index_text: str) -> str:
    """把"章节标题→页码"索引拼成中性的注入文本。空则返回 ''。"""
    if not index_text:
        return ""
    return (
        "[Section-heading→page-number index, extracted programmatically from the parsed "
        "document. Use it to answer on-which-page questions (pages are physical, 1-based):\n"
        f"{index_text}]"
    )
