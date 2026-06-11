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

import re
from collections import Counter

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
    """读 PDF 计算统计量。PyMuPDF 优先（MinerU 自带），pypdf 兜底；都失败返回 None。"""
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
    return stats


def format_stats_note(stats: dict) -> str:
    """生成附加在问题后的统计量说明（英文——DocBench 是英文语料）。"""
    tw = ", ".join(stats["top_words"]) or "n/a"
    aw = ", ".join(f"{a} ({c}x)" for a, c in stats["top_abbrevs"]) or "n/a"
    # 措辞刻意中性：首轮实测用了 "trust these numbers over any retrieved text"，
    # 导致部分非统计题（题干带 page 的表格题）被带偏、轻微掉分。统计量只做补充信息，
    # 不得贬低检索内容的可信度。
    return (
        "[Supplementary document statistics, computed programmatically from the PDF: "
        f"total pages = {stats['pages']}; "
        f"approximate word count (from extracted text) = {stats['words']}; "
        f"most frequent words = {tw}; "
        f"most frequent abbreviations = {aw}. "
        "Use them only when the question asks about such document-level statistics.]"
    )
