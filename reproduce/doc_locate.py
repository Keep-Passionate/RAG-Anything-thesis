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
import os
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


def _locate_elements_on() -> bool:
    """ENABLE_LOCATE_ELEMENTS：保守增强开关。开了才在标题外【追加】表/图/公式的页码行。"""
    return os.getenv("ENABLE_LOCATE_ELEMENTS", "").strip().lower() in ("1", "true", "yes", "on")


def _caption_of(it: dict) -> str:
    """尽量取元素的标题/说明文字（表/图 caption），取不到返回 ''。"""
    for k in ("text", "img_caption", "table_caption", "caption"):
        v = it.get(k)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if v:
            return " ".join(str(v).split()).strip()
    return ""


def build_heading_page_index(pdf_path: str, max_items: int = 40, max_chars: int = 1500) -> str:
    """读 content_list，构造"章节标题 → 物理页码"索引文本。找不到/无标题则返回 ''。

    保守增强（仅当 ENABLE_LOCATE_ELEMENTS=true）：在标题之后【追加】表/图/公式的
    "(类型 #序号) 标题 -> 页码"行，使"第一张表 / 图 2 / 某张表 在第几页"也能由模型查表作答。
    —— 不放宽触发条件（detect_location_intent 不变）、不改默认输出（开关关时与原版逐字一致），
       只是给模型【更全的参照表】、由模型自己对照，故不因放宽而引入新的误判（保守）。
    """
    try:
        from doc_meta import locate_content_list  # 复用 DSG 的解析定位工具

        cl = locate_content_list(pdf_path)
        if not cl:
            return ""
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return ""

    elements_on = _locate_elements_on()
    if elements_on:
        max_chars = 3000  # 放宽容量以容纳表/图行
    head_prefix = "(section) " if elements_on else ""  # 关时与原版逐字一致（零回归）

    lines = []
    for it in items:  # 1) 章节标题（原有逻辑不变）
        if not isinstance(it, dict) or not it.get("text_level"):
            continue
        txt = " ".join(str(it.get("text", "")).split()).strip()
        pg = it.get("page_idx")
        if txt and isinstance(pg, int):
            lines.append(f'{head_prefix}"{txt[:80]}" -> page {pg + 1}')
        if len(lines) >= max_items:
            break

    if elements_on:  # 2) 表/图/公式（保守增强：只列确定的元素+序号，不做模糊猜测）
        seq = {"table": 0, "image": 0, "equation": 0}
        label = {"table": "table", "image": "figure", "equation": "equation"}
        added = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            t, pg = it.get("type"), it.get("page_idx")
            if t not in seq or not isinstance(pg, int):
                continue
            seq[t] += 1
            cap = _caption_of(it)
            cap = f' "{cap[:60]}"' if cap else ""
            lines.append(f'({label[t]} #{seq[t]}){cap} -> page {pg + 1}')
            added += 1
            if added >= max_items:
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


# ===========================================================================
# 内容定位器（ENABLE_LOCATE_CONTENT）：补"非标题"定位——搜元素正文/图表标题里的目标关键词→返回页
# 治"France/Asia/签名/执行官/独立董事 在第几页"这类目标不是章节标题、Locate 标题表够不着的题。
# 保守：抽不到清晰目标就放弃、只给候选页让模型自选、独立开关可回退。
# ===========================================================================
_LOC_TARGET_RES = [
    re.compile(r"\binformation\s+(?:about|of|on)\s+(.+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    re.compile(r"\bsituations?\s+of\s+(.+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    re.compile(r"\babout\s+(.+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    re.compile(r"\b(?:presents?|details?|lists?|discuss(?:es)?|introduces?|outlines?|"
               r"reports?)\s+(?:the\s+)?(.+?)\s*(?:section|part)?\s*[\?\.!]*\s*$", re.IGNORECASE),
    re.compile(r"\bpage\s+(?:of|for)\s+(?:the\s+)?['\"]?(.+?)['\"]?\s*(?:section)?\s*[\?\.!]*\s*$",
               re.IGNORECASE),
]
_LOC_REJECT = re.compile(r"^(it|them|this|that|these|those|the\s+(document|report|paper))\b",
                         re.IGNORECASE)


def _content_locate_on() -> bool:
    return os.getenv("ENABLE_LOCATE_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")


def extract_location_target(question: str):
    """从'X 在第几页'类问题抽出要定位的内容关键词；抽不到/太模糊则 None（保守）。"""
    q = question or ""
    for rx in _LOC_TARGET_RES:
        m = rx.search(q)
        if m:
            t = " ".join(m.group(1).split()).strip(" '\"‘’“”")
            if 2 <= len(t) <= 50 and len(t.split()) <= 6 and not _LOC_REJECT.match(t):
                return t
    return None


def content_locate_note(pdf_path: str, question: str, max_pages: int = 6) -> str:
    """ENABLE_LOCATE_CONTENT：在 content_list 所有元素正文 + 图表标题里搜目标关键词，
    返回'目标 出现在第 N、M 页'的候选注入。关 / 无 content_list / 抽不到目标 / 没搜到 → ''。"""
    if not _content_locate_on():
        return ""
    target = extract_location_target(question)
    if not target:
        return ""
    try:
        from doc_meta import locate_content_list  # 复用解析定位
        cl = locate_content_list(pdf_path)
        if not cl:
            return ""
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return ""
    tl = target.lower()
    pages = []
    for it in items:
        if not isinstance(it, dict):
            continue
        blob = (str(it.get("text", "")) + " " + _caption_of(it)).lower()
        pg = it.get("page_idx")
        if isinstance(pg, int) and tl in blob and (pg + 1) not in pages:
            pages.append(pg + 1)
        if len(pages) >= max_pages:
            break
    if not pages:
        return ""
    plist = ", ".join(str(p) for p in sorted(pages))
    return (f'[Content locator (programmatic): the text "{target}" appears on '
            f"page(s) {plist} (physical, 1-based); use it to answer on-which-page questions.]")
