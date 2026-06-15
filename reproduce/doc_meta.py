"""文档统计量注入（ENABLE_DOC_META）——专打 DocBench 的 meta-data 题。

为什么（20 篇/127 题实测）：meta-data 题（总页数 / 词数 / 最常见缩写 / 高频词 /
"第 X 页讲了什么"）准确率只有 42–54%，是全部题型里最大的可攻失分块。这些题的
答案不在"检索到的内容片段"里，而在文档的【全局统计量】里——LLM 靠检索几乎必错；
但我们手里就有 PDF 原件，这些量可以程序化【精确计算】（training-free、零 LLM 成本）。

做法（三个纯函数 + 一个 I/O 函数，query.py 接线）：
  1. detect_meta_intent(q)   : 关键词判断"这道题问的是不是文档统计量"（零成本）；
  2. compute_doc_stats(pdf)  : 读 PDF 一次（PyMuPDF 优先——MinerU 自带依赖；pypdf 兜底），
                               返回 总页数 / 近似词数 / 高频词 top3 / 高频缩写 top3；
  3. format_stats_note(s)    : 把统计量拼成附加在【送给模型的问题】后的英文说明。
                               结果 JSON 里的 question 字段必须保持原文——评测器
                               （eval_by_type / llm_answer_evaluator）按问题原文匹配题型。

掉分风险控制（首轮 30 篇实测后的两条修正）：
  - 注入避让图/表题：题干带 "on page N" 的表格题也含 "page" 关键词，首轮被注入后
    mm-t 85%→81% 轻微回退。统计量对这类题本无帮助，query.py 里对命中图/表意图
    （detect_visual_intent）的题跳过注入——meta 收益保留、表题不再被扰。
  - 措辞中性化：不再写"trust over retrieved text"（见 format_stats_note 内注释）。
词数是从提取文本算的近似值，说明里如实标注 approximate。
首轮战绩（30篇/177题）：meta 题 37%→53%（+5题），机制完全对应（详见 memory）。
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

# 命中任一关键词即认为是"文档统计量"题。"page" 故意放宽（带页码的题总页数都有用，
# 比如 "summary of page 20" 在 12 页文档上应答"不存在第 20 页"）。
_META_KWS = (
    "page",
    "how many words", "number of words", "word count", "words are there",
    "abbreviation", "most frequent", "most common word",
    "页", "多少字", "词数", "缩写", "高频词",
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_ABBREV_RE = re.compile(r"\b[A-Z]{2,8}\b")

# 页眉/标题常被排版成全大写的普通英语词（'APPROVED FOR PUBLIC RELEASE' 等），
# 不是缩写，从缩写榜单里剔除。
_COMMON_CAPS = frozenset(
    """THE AND OF TO IN ON AT BY FOR WITH FROM NOT ALL ANY NEW OR AS IS ARE BE
    PUBLIC RELEASE APPROVED DRAFT FINAL CONFIDENTIAL UNCLASSIFIED CLASSIFIED
    REPORT ANNUAL TOTAL PAGE SECTION TABLE FIGURE APPENDIX CONTENTS NOTES NOTE
    INTRODUCTION SUMMARY OVERVIEW ITEM PART CHAPTER INDEX ABSTRACT REFERENCES
    UNITED STATES DEPARTMENT BUREAU OFFICE COURT CASE NO VS VERSUS
    JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER
    NOVEMBER DECEMBER""".split()
)


def detect_meta_intent(question: str) -> bool:
    """纯关键词判断问题是否在问文档统计量（页数/词数/缩写/高频词）。零 LLM 成本。"""
    q = (question or "").lower()
    return any(k in q for k in _META_KWS)


# v3：计数题（"how many figures/tables..."）。这类题含 figure/table 关键词，会被
# query.py 的"避让图/表题"门拦下，但它们恰恰是统计量能精确回答的——所以单独识别、
# 在 query.py 里覆盖避让门（归因实锤：'how many figures excluding appendix' 曾因此丢分）。
_COUNT_RE = re.compile(
    r"how\s+many\s+(?:figures?|images?|tables?|equations?|charts?|pictures?|illustrations?)",
    re.IGNORECASE,
)
_COUNT_KWS_ZH = ("几张图", "多少张图", "几幅图", "几个表", "多少个表", "几个公式", "多少个公式")


def detect_count_intent(question: str) -> bool:
    """问题是否在数"文档里有几张图/几个表/几个公式"。"""
    q = question or ""
    return bool(_COUNT_RE.search(q)) or any(k in q for k in _COUNT_KWS_ZH)


# v3：页码引用（"summary of page 20" / "第 7 页"）。识别出具体页码后，
# query.py 会把该页开头文本一并注入——把"第 N 页讲什么"从猜变成读。
_PAGE_NUM_RE = re.compile(r"page\s+(\d{1,4})", re.IGNORECASE)
_PAGE_NUM_ZH_RE = re.compile(r"第\s*(\d{1,4})\s*页")


def find_page_reference(question: str):
    """提取问题里引用的页码（1 起算）。没有具体页码返回 None。"""
    m = _PAGE_NUM_RE.search(question or "") or _PAGE_NUM_ZH_RE.search(question or "")
    return int(m.group(1)) if m else None


# v4：关键词计数题（"how many times does the document mention 'X'"）。DSG 现在只数
# 高频词 top3，这里加"数指定词/短语"——程序精确数，是 DSG 从"数高频"到"数指定"的
# 自然延伸（错题例：ASX_LOV "how many times mention 'net working capital'"）。
_MENTION_TARGET_RE = re.compile(
    r'["“‘\']([^"”’\']{1,60})["”’\']'  # 引号内（优先）
)


def detect_mention_count_intent(question: str) -> bool:
    """问题是否在问"某词/短语在文档里出现多少次"。零 LLM 成本。"""
    q = (question or "").lower()
    return "how many times" in q and (
        "mention" in q or "appear" in q or "occur" in q or "use the word" in q
    )


def extract_mention_target(question: str):
    """从问题抽出要数的目标词/短语（取引号内内容）。抽不到返回 None。

    只接受引号内的明确目标——无引号的"how many times X"边界模糊、易抽错，宁可不处理。
    """
    m = _MENTION_TARGET_RE.search(question or "")
    return m.group(1).strip() if m else None


def count_mentions(text: str, target: str) -> int:
    """数 target（不区分大小写）在 text 中出现的次数。纯函数。"""
    if not text or not target:
        return 0
    return len(re.findall(re.escape(target), text, flags=re.IGNORECASE))


def text_stats(text: str) -> dict:
    """从提取文本计算 词数 / 高频词 top3 / 高频缩写 top3（纯函数，可单测）。

    - words      : 近似词数（空白切分，与 DocBench 标注的统计口径未必一致，标注 approximate）
    - top_words  : 出现最多的 3 个词（小写后计数，停用词不剔除——DocBench 标准答案
                   本身就是 "and; the; to" 这种）
    - top_abbrevs: 出现最多的 3 个全大写缩写（2–8 字母），带出现次数。
                   排除"被排版成全大写的普通词"（页眉 'APPROVED FOR PUBLIC RELEASE'
                   会让 FOR/PUBLIC 混进榜单）：真缩写（CDP/HIV）几乎只以全大写出现，
                   普通词在正文里必有大量小写形式——小写出现明显多于全大写就剔除。
    """
    words = _WORD_RE.findall(text)
    freq = Counter(w.lower() for w in words)
    abbrevs = Counter(_ABBREV_RE.findall(text))
    real_abbrevs = [
        (a, c) for a, c in abbrevs.most_common()
        if a not in _COMMON_CAPS          # 页眉常见普通词，排除
        and freq[a.lower()] <= c * 1.5    # 正文里大量以小写出现的也是普通词，排除
    ]
    return {
        "words": len(text.split()),
        "top_words": [w for w, _ in freq.most_common(3)],
        "top_abbrevs": real_abbrevs[:3],
    }


def compute_doc_stats(pdf_path: str):
    """读 PDF 计算统计量。PyMuPDF 优先（MinerU 自带），pypdf 兜底；都失败返回 None。

    v3：若能找到 MinerU 的解析产物 content_list（建索引时已生成），一并精确计数
    图/表/公式元素（找不到则静默跳过，统计量退回 v2 行为）。
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            pages = doc.page_count
            text = "\n".join(p.get_text() for p in doc)
    except Exception:
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages = len(reader.pages)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return None
    stats = text_stats(text)
    stats["pages"] = pages
    stats["_text"] = text  # 供 count_mentions 复用（带下划线=内部用，不进 note）
    cl = locate_content_list(pdf_path)
    if cl:
        elements = count_elements(cl)
        if elements:
            stats.update(elements)
    return stats


def locate_content_list(pdf_path: str):
    """定位 MinerU 解析产物 <stem>_content_list.json（索引时生成在解析输出目录）。

    依次尝试：PARSE_OUTPUT_DIR 环境变量（默认 ./output，与 index.py 的 --output
    默认一致）下的标准路径，再做一次有界 glob。找不到返回 None（v3 计数静默关闭）。
    """
    stem = Path(pdf_path).stem
    root = Path(os.getenv("PARSE_OUTPUT_DIR", "./output"))
    for cand in (
        root / stem / "auto" / f"{stem}_content_list.json",
        root / stem / f"{stem}_content_list.json",
    ):
        if cand.exists():
            return cand
    if root.exists():
        hits = list(root.glob(f"*/*/{stem}_content_list.json"))
        if hits:
            return hits[0]
    return None


# 附录/参考文献起始标题——其后的图表算"附录",治 "how many figures excluding appendix"
_APPENDIX_RE = re.compile(r"\b(references?|appendix|appendices|bibliography)\b", re.IGNORECASE)


def count_elements(content_list_path):
    """从 MinerU content_list 精确计数图/表/公式元素。解析失败返回 None。

    附加"不含附录"计数(figures_body/tables_body):找到第一个"标题级且文本含
    References/Appendix"的元素的页码,只数它之前的图表——治 "excluding appendix" 题。
    """
    try:
        with open(content_list_path, encoding="utf-8") as f:
            items = [it for it in json.load(f) if isinstance(it, dict)]
    except Exception:
        return None
    # 找附录起始页
    appendix_page = None
    for it in items:
        if it.get("text_level") and _APPENDIX_RE.search(str(it.get("text", ""))):
            appendix_page = it.get("page_idx")
            break

    def cnt(t, body_only=False):
        els = [it for it in items if it.get("type") == t]
        if body_only and appendix_page is not None:
            return sum(1 for e in els if (e.get("page_idx") or 0) < appendix_page)
        return len(els)

    return {
        "figures": cnt("image"),
        "tables": cnt("table"),
        "equations": cnt("equation"),
        "figures_body": cnt("image", body_only=True),
        "tables_body": cnt("table", body_only=True),
    }


def extract_page_text(pdf_path: str, page_no: int, max_chars: int = 1200) -> str:
    """抽取第 page_no 页（1 起算）开头文本。页码越界/读取失败返回空串。

    用途：问题点名"page N"时把该页内容直接给模型——"第 N 页讲什么"从猜变成读；
    页码超出总页数时返回空串，模型只看到统计量里的 total pages，自然答"不存在"。
    """
    if not page_no or page_no < 1:
        return ""
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            if page_no > doc.page_count:
                return ""
            text = doc[page_no - 1].get_text()
        return " ".join(text.split())[:max_chars]
    except Exception:
        return ""


def format_stats_note(stats: dict) -> str:
    """生成附加在问题后的统计量说明（英文——DocBench 是英文语料）。"""
    tw = ", ".join(stats["top_words"]) or "n/a"
    aw = ", ".join(f"{a} ({c}x)" for a, c in stats["top_abbrevs"]) or "n/a"
    # 措辞刻意中性：首轮实测用了 "trust these numbers over any retrieved text"，
    # 导致部分非统计题（题干带 page 的表格题）被带偏、轻微掉分。统计量只做补充信息，
    # 不得贬低检索内容的可信度。
    parts = [
        f"total pages = {stats['pages']}",
        f"approximate word count (from extracted text) = {stats['words']}",
        f"most frequent words = {tw}",
        f"most frequent abbreviations = {aw}",
    ]
    if "figures" in stats:  # v3：content_list 可用时的精确元素计数
        fb = stats.get("figures_body", stats["figures"])
        tb = stats.get("tables_body", stats["tables"])
        parts.append(
            f"parsed element counts: figures = {stats['figures']} "
            f"(excluding appendix/references: {fb}); tables = {stats['tables']} "
            f"(excluding appendix: {tb}); equations = {stats['equations']}"
        )
    return (
        "[Supplementary document statistics, computed programmatically from the PDF: "
        + "; ".join(parts)
        + ". Use them only when the question asks about such document-level statistics.]"
    )
