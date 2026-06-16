"""doc_anchor：被点名元素接地（ENABLE_DOC_ANCHOR）——治"问某一张具体表/图"的题。

与 DSG（doc_meta）、doc_outline 平行、解耦，同属"确定性接地"这条主线：
  DSG          算检索拿不到的【全局标量】（页数 / 词频 / 缩写）；
  doc_outline  抄印在门面上的【前页元信息】（发表 / 批准 / 作者单位）；
  doc_anchor   取问题【点名的那一个元素】（"according to Table 8" / "in Figure 3"）。

为什么这是检索的结构性盲区（与 DSG 同源）：问题指名 "Table 8" 时，语义检索常把那张表
排不进 top-k——表体几乎全是数字、caption 又短，和问句的语义相似度低，于是"对的表没被
检索到"（呼应原论文 A.5 的 text-centric bias）。但 caption 里白纸黑字写着 "Table 8"，
我们可以【按标号确定性定位】：扫 MinerU content_list 里 caption 含该标号的浮动元素，把它的
解析内容原样抄给模型。这正是 NCG 复盘留下的未竟一招——"瓶颈是找对表、不是算"：题目点名
时标号即坐标，无需语义猜测，也无需任何计算。

设计取舍（论文要写，承袭项目"宁可不碰、不可帮倒忙"的原则）：
  - 只处理【带 caption 的浮动元素】table / figure——二者字段统一（*_caption），按标号匹配
    干净确定。section / heading 引用（"Section III 讲什么"）【刻意不做】：靠 MinerU 的
    text_level 切章节实测有噪（doc_outline 的章节树分支正因对 10-K "how many parts" 帮倒忙
    被移除，commit 46de7aa）。
  - figure 只注入 caption（+ footnote）文本；图像内容仍走 paperbase 既有的 VLM 通道，本模块
    不碰 VLM、不重蹈"选择性看图"的 EMR 覆辙。table 注入完整表体（caption + body）。
  - 纯加法、可退化：定位不到该元素时返回空、不注入，对 baseline 零风险（与 DSG/outline 一致）。

纯函数可单测 + query.py 一处注入：
  detect_element_reference(q)            -> ("table"|"figure", label) | None  （正则，零 LLM 成本）
  load_content_items(pdf)               -> list                              （读一次 content_list）
  find_referenced_element(items, k, l)  -> str                               （按标号取该元素内容）
  format_anchor_note(k, l, content)     -> str                               （中性注入文本）
"""

import json
import re

# 点名引用：<table|figure 及缩写> + 阿拉伯标号。只认数字标号——table/figure 几乎都用
# 阿拉伯数字（Table 8 / Figure 3）；罗马数字主要出现在 section（本模块刻意不处理），
# 放进来反而会把 "civil"/"mix" 这类全字母词误当罗马数字。\b 词边界避免 8 命中 80。
_REF_RE = re.compile(r"\b(tables?|figures?|fig)\s*\.?\s*(\d{1,3})\b", re.IGNORECASE)
_KIND = {"table": "table", "figure": "figure", "fig": "figure"}


def detect_element_reference(question: str):
    """问题是否点名了某张具体表/图。命中返回 ("table"|"figure", 标号字符串)，否则 None。

    "how many tables"（数量题，归 DSG）不会命中——它的 table 后面没有紧跟标号。
    """
    m = _REF_RE.search(question or "")
    if not m:
        return None
    head = m.group(1).lower().rstrip("s")  # tables->table, figures->figure
    kind = _KIND.get(head)
    return (kind, m.group(2)) if kind else None


def load_content_items(pdf_path: str):
    """读 MinerU content_list 的元素列表（解析失败 / 找不到则返回 []，无害退化）。"""
    try:
        from doc_meta import locate_content_list  # 复用 DSG 的解析定位逻辑

        cl = locate_content_list(pdf_path)
        if not cl:
            return []
        with open(cl, encoding="utf-8") as f:
            items = json.load(f)
        return [it for it in items if isinstance(it, dict)]
    except Exception:
        return []


def _caption_text(item: dict) -> str:
    """取元素 caption 文本（table_caption / image_caption / img_caption，list 或 str）。"""
    cap = (
        item.get("table_caption")
        or item.get("image_caption")
        or item.get("img_caption")
        or []
    )
    if isinstance(cap, list):
        cap = " ".join(str(c) for c in cap)
    return str(cap).strip()


def find_referenced_element(items, kind: str, label: str, max_chars: int = 3000) -> str:
    """在 content_list 里按标号定位 table/figure，返回其文本内容（caption [+body/footnote]）。

    匹配口径：该类型元素的 caption 里出现 "<table|tab|figure|fig> <label>"（词边界，
    标号精确，避免 Table 8 命中 Table 80）。命中第一处即返回；找不到返回 ''（不注入）。
    """
    typ = "image" if kind == "figure" else "table"
    head = "fig(?:ure)?" if kind == "figure" else "tab(?:le)?"
    label_re = re.compile(rf"\b{head}\.?\s*{re.escape(label)}\b", re.IGNORECASE)
    for it in items:
        if it.get("type") != typ:
            continue
        cap = _caption_text(it)
        if not (cap and label_re.search(cap)):
            continue
        if kind == "table":
            body = it.get("table_body") or it.get("text") or ""
            out = (cap + "\n" + str(body)).strip()
        else:  # figure：只给 caption（+ footnote）文本，图像本体仍由 VLM 通道处理
            foot = it.get("image_footnote") or it.get("img_footnote") or []
            if isinstance(foot, list):
                foot = " ".join(str(f) for f in foot)
            out = (cap + ("\n" + str(foot) if foot else "")).strip()
        return out[:max_chars]
    return ""


def format_anchor_note(kind: str, label: str, content: str) -> str:
    """把定位到的元素拼成中性的注入文本。content 为空则返回 ''。"""
    if not content:
        return ""
    name = f"{kind.capitalize()} {label}"
    return (
        f"[{name}, located by its label in the parsed document — this is the exact "
        f"element the question refers to:\n{content}]"
    )
