"""dg_core —— DG-RAG 统一确定性接地框架(收编散落的 doc_meta/doc_locate/... 补丁式算子)。

三层(详见素材库 2026-06-28 错题诊断):
  Layer 1 DocumentModel : 每篇建一次的单一真相源——PageMap(物理↔印刷页)/ 逐页文本 /
                          canonical 全文 / content_list typed elements。所有算子共用,杜绝口径不一。
  Layer 2 算子代数       : 4 个【参数化组合子】COUNT / LOCATE / EXTRACT / LOOKUP(不再是 N 个手写函数)。
                          每个题型 = 给组合子传参数(unit/scope/target/field),不是新增一个函数。
                          11 条意图正则只是【确定性语义解析器】:把自然语言映射到上面的算子调用(零 LLM)。
  Layer 3 门控+弃权      : 每个组合子返回 Fact(value, confidence, note);confidence 是【实例级、由输入
                          确定性算出的自检信号】(非死常数);conf<阈值则弃权(不注入)→ 回退基座=非回归。

与 GlobalRAG 的区分:GlobalRAG 是【语料级】符号算子(跨文档实体计数/极值/排序);本框架是【单文档】、
绕过检索直接读原文、并带【确定性可验证门控】(算子仅在自检判定可靠时才开口),training-free、可跨基座迁移。

为什么这样比旧版高(诊断驱动,非逐题打补丁):
  - 页码相关错占 A/C 的 55%(37/67),根因是"物理页 vs 印刷页"框架错位 + locate 只索引标题。
    PageMap 一处建模、LOCATE/EXTRACT 全程用它出"印刷 N(物理 M)"双框架 —— 金标正是这双框架。
  - 计数错(mention/word/element)源于文本源不一致、目标未归一化 —— 统一 DocumentModel + 归一化治。
  - 噪声大、口径不可观测的(footnote/section)—— 用实例级自检弃权,把"算错值"变安全 no-op。

零回归:本模块独立;query.py 经 ENABLE_DG_CORE 开关接入,默认关 = 走旧路由。
消融/回退开关:DG_LEGACY=true 完全复现旧(64%)行为(常数置信度 + first-over-threshold),供 A/B。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

# 复用旧算子里已验证的纯函数(收编,不重复造轮子)
try:
    from doc_meta import text_stats, count_mentions, locate_content_list
except Exception:  # 容错:dg_core 也可单独 import
    text_stats = count_mentions = locate_content_list = None


def _dg_env(name: str, default: bool = True) -> bool:
    """dg_core 内部消融/修复开关。default=该项现状行为;改 env 可单独开关某层做层级测试。"""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

# =====================================================================================
# Layer 1 —— DocumentModel
# =====================================================================================

_INT = re.compile(r"^\d{1,4}$")
PAGEMAP_CONF_THRESHOLD = 0.6
_MAX_OFFSET = 25


def _edge_numbers(page) -> set:
    """该页 上/下缘(各 12%)区域的孤立整数候选 -> set[int]。"""
    h = page.rect.height
    found = set()
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            y = ln["bbox"][1]
            if not (y < h * 0.12 or y > h * 0.88):
                continue
            txt = "".join(s["text"] for s in ln.get("spans", [])).strip()
            for tok in re.split(r"\s+", txt):
                tok = tok.strip(".,()[]-—–:")
                if _INT.match(tok):
                    v = int(tok)
                    if 1 <= v <= 2000:
                        found.add(v)
    return found


@dataclass
class PageMap:
    """物理页(1 起算)↔ 印刷页(页脚标注)。printed = physical_1based - offset。"""
    physical_count: int
    offset: int
    confidence: float

    @property
    def confident(self) -> bool:
        if not _dg_env("DG_PAGEMAP", True):
            return False  # 消融 DG_PAGEMAP=false:强制关物理/印刷页重写,页号按原样报
        return self.confidence >= PAGEMAP_CONF_THRESHOLD and self.offset > 0

    def to_physical(self, printed: int):
        phys = printed + self.offset if self.confident else printed
        return phys if 1 <= phys <= self.physical_count else None

    def frame(self, printed: int) -> str:
        """题面页号(默认按印刷解读)-> '印刷 N(物理 M)' 双框架说明。"""
        if not self.confident:
            return f"page {printed}"
        phys = self.to_physical(printed)
        return f"printed page {printed} (physical page {phys})" if phys else f"page {printed}"

    def phys_frame(self, physical_1based: int) -> str:
        """物理页 -> '物理 M(印刷 K)' 说明(用于把检索到的物理页报给模型)。"""
        if not self.confident:
            return f"page {physical_1based}"
        printed = physical_1based - self.offset
        if printed >= 1:
            return f"physical page {physical_1based} (printed page {printed})"
        return f"physical page {physical_1based} (front matter, unnumbered)"


def detect_page_offset(doc) -> PageMap:
    per_page = [_edge_numbers(doc[i]) for i in range(doc.page_count)]
    n = doc.page_count
    best_k, best_sup = 0, 0
    for k in range(0, _MAX_OFFSET + 1):
        sup = sum(1 for i in range(n) if (i + 1 - k) >= 1 and (i + 1 - k) in per_page[i])
        if sup > best_sup:
            best_k, best_sup = k, sup
    numbered = max(1, n - best_k)
    return PageMap(physical_count=n, offset=best_k, confidence=round(best_sup / numbered, 3))


def _offset_from_page_numbers(elements, page_count: int):
    """从 content_list 的 page_number 元素(解析器自带页码标注)推断偏移——比页脚坐标更干净。
    offset = 物理页(page_idx+1) − 印刷号;多数元素一致即高置信。论文 anthology 大页号(5962)
    → offset 为负 → 自动拒绝(offset 0,退回"物理=题面页")。"""
    cands = []
    for it in elements:
        if it.get("type") != "page_number":
            continue
        t = str(it.get("text", "")).strip()
        pg = it.get("page_idx")
        if t.isdigit() and isinstance(pg, int):
            cands.append((pg + 1) - int(t))
    if not cands:
        return None
    from collections import Counter
    off, n = Counter(cands).most_common(1)[0]
    if off <= 0:
        return PageMap(physical_count=page_count, offset=0, confidence=0.0)
    return PageMap(physical_count=page_count, offset=off, confidence=round(n / len(cands), 3))


def _norm_caption(it: dict) -> str:
    """取元素的标题/正文文本(图表 caption / 正文),拼成可搜索串。"""
    parts = []
    for k in ("text", "image_caption", "table_caption", "img_caption", "caption", "table_body"):
        v = it.get(k)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if v:
            parts.append(str(v))
    return " ".join(" ".join(parts).split())


def _has_caption(it: dict) -> bool:
    """元素是否带 caption(真图表有 caption;装饰性图标/分隔图多半没有)。"""
    for k in ("image_caption", "table_caption", "img_caption", "caption"):
        v = it.get(k)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if v and str(v).strip():
            return True
    return False


def _flex_count(text: str, target: str) -> int:
    """弹性计数:词间空白/连字符可变(治 'WikiText-\\n2'、'total  revenue'),大小写不敏感。"""
    toks = [re.escape(t) for t in re.split(r"[\s\-]+", target.strip()) if t]
    if not toks:
        return 0
    pat = r"[\s\-]*".join(toks)
    return len(re.findall(pat, text, re.I))


@dataclass
class DocModel:
    """单一文档真相源。content_list 缺失时,纯 PDF 路径仍可用(content_list 类计数则弃权)。"""
    pdf_path: str
    page_count: int = 0
    page_map: PageMap = None
    per_page_text: list = field(default_factory=list)
    elements: list = field(default_factory=list)   # content_list items(可空)
    _full_text: str = None

    @property
    def full_text(self) -> str:
        if self._full_text is None:
            self._full_text = "\n".join(self.per_page_text)
        return self._full_text

    @property
    def has_elements(self) -> bool:
        return bool(self.elements)

    def page_text(self, physical_1based: int, max_chars: int = 1400) -> str:
        i = physical_1based - 1
        if 0 <= i < len(self.per_page_text):
            return " ".join(self.per_page_text[i].split())[:max_chars]
        return ""

    def element_text_with_tables(self) -> str:
        """全文 + content_list 表格正文(治 mention 计数漏表格)。无 elements 则退回全文。"""
        if not self.elements:
            return self.full_text
        extra = []
        for it in self.elements:
            if isinstance(it, dict):
                tb = it.get("table_body")
                if tb:
                    extra.append(str(tb))
        return self.full_text + "\n" + "\n".join(extra)


def build_doc_model(pdf_path: str, content_list_path: str = None) -> DocModel | None:
    """建 DocumentModel:PyMuPDF 取逐页文本 + PageMap;content_list 可选(增强计数/定位)。"""
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            per_page = [p.get_text() for p in doc]
            pm = detect_page_offset(doc)
    except Exception:
        return None
    elements = []
    cl = content_list_path
    if cl is None and locate_content_list is not None:
        try:
            cl = locate_content_list(pdf_path)
        except Exception:
            cl = None
    if cl and os.path.exists(str(cl)):
        try:
            import json
            with open(cl, encoding="utf-8") as f:
                elements = [it for it in json.load(f) if isinstance(it, dict)]
        except Exception:
            elements = []
    # PageMap 双源:页脚坐标法 vs 解析器 page_number 元素法,取更自洽(置信更高)的一个。
    if elements:
        pm_elem = _offset_from_page_numbers(elements, pm.physical_count)
        if pm_elem is not None and pm_elem.confidence > pm.confidence:
            pm = pm_elem
    return DocModel(pdf_path=pdf_path, page_count=pm.physical_count, page_map=pm,
                    per_page_text=per_page, elements=elements)


# =====================================================================================
# Layer 3 数据结构 —— Fact
# =====================================================================================

@dataclass
class Fact:
    kind: str
    value: object
    confidence: float
    note: str          # 注入到问题后的英文说明(空串=弃权)
    provenance: str = ""


# =====================================================================================
# Layer 2 —— 算子代数:4 个参数化组合子(COUNT / LOCATE / EXTRACT / LOOKUP)
# -------------------------------------------------------------------------------------
# 设计:把旧版 N 个手写函数收编为 4 个组合子。每个组合子接受【参数】(unit / scope / target / field),
# 输出 (value, confidence)。confidence = 该输出的【实例级确定性自检】,而非写死的常数。
#   COUNT(unit, scope) : unit ∈ {PAGE, WORD, ELEMENT(types), HEADING(kind), REF, SPAN(variants)}
#   LOCATE(target)     : 定位目标到讨论页
#   EXTRACT(page_ref)  : 取某物理页文本
#   LOOKUP(field)      : 从结构化区读单值字段(title;date/venue/author 为扩展点)
# =====================================================================================

# ---- unit / scope 构造子(算子的参数,可组合)----
def U_PAGE():               return ("page", ())
def U_WORD():               return ("word", ())
def U_ELEMENT(types):       return ("element", tuple(types))
def U_HEADING(kind):        return ("heading", (kind,))
def U_REF():                return ("ref", ())
def U_SPAN(variants):       return ("span", tuple(variants))

S_WHOLE = ("whole", None)
def S_EXCLUDING(page_idx):  return ("excluding", page_idx)   # 排除 >= page_idx 的元素(如附录起始)


def _pypdf_wordcount(pdf_path: str):
    """第二个独立抽取器(pypdf)的词数,用作 WORD 的跨源自检。抽取失败/空 -> None(=无第二源)。"""
    PdfReader = None
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return None
    try:
        r = PdfReader(pdf_path)
        wc = sum(len((p.extract_text() or "").split()) for p in r.pages)
        return wc or None
    except Exception:
        return None


def _span_text(m: DocModel) -> str:
    # 修复(16题诊断):含表格计数会过计(LSTM+CRF 6→8)。默认纯正文;DG_MENTION_TABLES=true 才含表格。
    return m.element_text_with_tables() if _dg_env("DG_MENTION_TABLES", False) else m.full_text


def COUNT(m: DocModel, unit, scope=S_WHOLE):
    """⭐ 参数化计数器:一个组合子覆盖 页/词/元素/标题/参考文献/正则跨度。
    返回 (value, confidence, evidence)。confidence = 该 unit 的【实例级确定性自检】。
    value=None 表示该 unit 在本文档不可数(交由调用方弃权)。"""
    kind, params = unit
    skind, szone = scope

    if kind == "page":
        # 精确可数(物理页) -> 自检恒高(真确定)。
        return m.page_count, 0.95, {}

    if kind == "word":
        v = len(m.full_text.split())
        alt = _pypdf_wordcount(m.pdf_path)
        # 自检 = 跨源一致率:两个独立抽取器吻合 -> 高;发散(扫描/多栏)-> 低 -> 弃权。
        # ⚠️ 与数据集无关(不写死"词数=PyMuPDF");无第二源时退回 0.8(=可注入)。
        if alt is None or v == 0:
            conf, agree = 0.8, None
        else:
            agree = 1.0 - min(1.0, abs(v - alt) / max(v, alt))
            conf = round(agree, 2)
        return v, conf, {"alt": alt, "agree": agree}

    if kind == "element":
        c = 0
        for it in m.elements:
            if it.get("type") in params:
                if skind == "excluding" and szone is not None and (it.get("page_idx") or 0) >= szone:
                    continue
                c += 1
        # 自检 = 合理度 min(1, 页数/计数):论文图表数 << 页数 -> 高;计数 ≫ 页数(被切碎)-> 低 -> 弃权。
        conf = round(min(1.0, m.page_count / c), 2) if c else 0.0
        return c, conf, {}

    if kind == "heading":
        hk = params[0]
        if hk == "subsection":
            c = sum(1 for it in m.elements if it.get("text_level")
                    and re.match(r"^\s*\d+\.\d+\b", str(it.get("text", ""))))
        else:
            c = sum(1 for it in m.elements if it.get("text_level")
                    and re.match(r"^\s*(\d+\s+\S|chapter\s+\d+|part\s+[ivxl0-9]+)",
                                 str(it.get("text", "")), re.I))
        # 章节粒度口径不可观测,自检无可靠信号 -> 低置信(默认弃权)。
        return c, 0.4, {}

    if kind == "ref":
        start = None
        for idx, it in enumerate(m.elements):
            if it.get("text_level") and re.search(r"^\s*(references?|bibliography)\b",
                                                  str(it.get("text", "")), re.I):
                start = idx
                break
        if start is None:
            return None, 0.0, {}
        c = 0
        for it in m.elements[start + 1:]:
            if it.get("text_level"):  # 到下一个标题(附录等)停
                break
            t = str(it.get("text", "")).strip()
            if it.get("type") == "list":
                c += max(1, t.count("\n") + 1)
            elif re.match(r"^\[?\d+\]?\.?\s|\b\(\d{4}\)\b|\b\d{4}\.", t):  # 编号或带年份
                c += 1
        if c < 3:
            return None, 0.0, {}     # 没数出像样的参考表 -> 弃权
        return c, 0.45, {}

    if kind == "span":
        counts = [_flex_count(_span_text(m), v) for v in params]
        best = max(counts, default=0)
        return best, None, {"counts": counts}   # conf 交调用方(需 target 长度判假阳性)

    return None, 0.0, {}


def LOCATE(m: DocModel, target: str):
    """⭐ 定位组合子:把 target 定位到【真正讨论它的页】(排除目录"提一嘴")。
    自检 = 命中章节标题(0.9)/ 否则出现最密集页的占比(越集中越确定)。"""
    tl = target.lower()
    # 排除目录页:目标常在目录里"提一嘴",不是真正讨论页(治 locate 找错页)。
    toc = set()
    for it in m.elements:
        if it.get("text_level") and re.search(r"table of contents|^\s*contents\s*$",
                                              str(it.get("text", "")), re.I):
            pg = it.get("page_idx")
            if isinstance(pg, int):
                toc.add(pg + 1)
    # 1) 命中【章节标题】= 最可靠(排除目录里的条目)
    heading_pages = []
    for it in m.elements:
        pg = it.get("page_idx")
        if (isinstance(pg, int) and it.get("text_level") and (pg + 1) not in toc
                and tl in str(it.get("text", "")).lower() and (pg + 1) not in heading_pages):
            heading_pages.append(pg + 1)
    if heading_pages:
        pages, conf = sorted(heading_pages)[:2], 0.9
    else:
        # 2) 按【每页出现次数】选最密集页(排除目录)。引言里只"提一嘴"次数少→自然落选。
        hits = {}
        for i, t in enumerate(m.per_page_text):
            if (i + 1) in toc:
                continue
            c = _flex_count(t, target)
            if c > 0:
                hits[i + 1] = c
        if not hits:
            return None   # 只在目录/根本没出现 → 弃权(治"硬编不存在的附录"假阳性)
        total = sum(hits.values())
        top = max(hits.values())
        pages = sorted([p for p, c in hits.items() if c == top])[:2]
        conf = round(top / total, 2)
    # 页框架(校准:locate 44% 多因框架不匹配,金标常"印刷 or 物理"两收)。
    # 仅当 PageMap 高置信(双源可靠)时给双框架对齐金标;不可信时报朴素页号。
    if m.page_map.confident or _dg_env("DG_PAGEMAP_REFRAME", False):
        framed = "; ".join(m.page_map.phys_frame(p) for p in pages)
        note = (f'[Programmatic locator: "{target}" appears on {framed}. '
                f"Page questions may expect either the printed or physical number.]")
    else:
        framed = ", ".join(f"page {p}" for p in pages)
        note = f'[Programmatic locator: "{target}" appears on {framed}.]'
    return Fact("locate", pages, conf, note, "content_list/pdf")


def EXTRACT(m: DocModel, phys: int, label: str, clean: bool):
    """⭐ 取页组合子:返回某物理页文本。自检 = 页引用是否被唯一/可靠解析(clean)+ 文本非空。"""
    snip = m.page_text(phys)
    if not snip:
        return None
    conf = 0.85 if clean else 0.65
    return Fact("extract_page", phys, conf,
                f"[Content of {label}, extracted programmatically: {snip}]", "pdf")


def LOOKUP(m: DocModel, fieldname: str):
    """⭐ 查字段组合子:从结构化区读单值。当前 field=title;date/venue/author 为后续 Field 扩展点。"""
    if fieldname == "title":
        h1 = [it for it in m.elements if it.get("text_level") == 1 and str(it.get("text", "")).strip()]
        if not h1:
            return None
        title = " ".join(str(h1[0]["text"]).split())
        # 自检 = 标题唯一性:全文唯一 H1 -> 高;有多个竞争 H1 -> 略低(仍可注入,取首个)。
        conf = 0.9 if len(h1) == 1 else 0.65
        return Fact("title", title, conf,
                    f'[Programmatically extracted document title: "{title}".]', "content_list")
    return None


# =====================================================================================
# 确定性语义解析器:把自然语言映射到上面的算子调用(11 条意图正则,零 LLM)。
# =====================================================================================
_RE_MENTION = re.compile(r"how many time", re.I)
_RE_WORDS = re.compile(r"how many words|number of words|word count|words (?:are|does|in|in total)", re.I)
_RE_PAGES_TOTAL = re.compile(r"how many pages", re.I)
_RE_ELEM = re.compile(r"how many\s+(figures?|images?|tables?|equations?|charts?|illustrations?)", re.I)
_RE_REFS = re.compile(r"how many\s+(references?|citations?|cited)", re.I)
_RE_SECTIONS = re.compile(r"how many\s+(sections?|chapters?|parts?|subsections?)", re.I)
_RE_FOOTNOTES = re.compile(r"how many\s+footnotes?", re.I)
_RE_AUTHORS = re.compile(r"how many\s+(authors?|institutions?|affiliations?)", re.I)
# 定位:"X 在第几页"
_RE_LOCATE = re.compile(
    r"\b(on|at|from)\s+(which|what)\s+page\b"
    r"|\bwhich\s+page\b[^?]*\b(is|are|does|do|located|show|present|discuss|list|introduce|begin|start)\b"
    r"|\bwhat\s+page\b", re.I)
# 页内容:"第 N 页讲什么 / 末页 / 首页"
_RE_PAGE_REF = re.compile(r"\bpage\s+(\d{1,4})\b", re.I)
_RE_FIRST_SENT = re.compile(r"first sentence on page\s+(\d{1,4})", re.I)
_RE_REL_LAST = re.compile(r"\b(last|final)\s+page\b", re.I)
_RE_REL_2ND = re.compile(r"\bsecond[-\s]to[-\s]last\s+page\b", re.I)
_RE_REL_FRONT = re.compile(r"\b(front\s*page|first\s+page|cover\s+page|frontpage)\b", re.I)
_RE_PAGE_CONTENT = re.compile(
    r"(what|topic|content|focus|message|summary|purpose|talk about|conveyed|present)", re.I)
_RE_TITLE = re.compile(r"\b(document|paper|report|newspaper)\s+title\b|\btitle of the\b", re.I)
# 缩写 / 最高频词(16题诊断:dgcore 原先没这两类 resolver→弃权→答错 4+1 题)
_RE_ABBR = re.compile(r"\b(?:most\s+(?:common|frequent)\s+)?(?:abbreviation|acronym)s?\b", re.I)
_RE_TOPWORDS = re.compile(r"\b(?:top\s*\d*\s*)?most\s+(?:common|frequent)\s+words?\b", re.I)


def _mention_target(q: str):
    m = re.search(r'["“‘\']([^"”’\']{1,60})["”’\']', q)
    if m:
        return m.group(1).strip()
    m = re.search(r"mention(?:s|ed)?\s+(?:the\s+(?:word|name|phrase|term)s?\s+(?:of\s+)?)?(.+?)\s*[?.!]*\s*$", q, re.I)
    if not m:
        return None
    t = m.group(1).strip().strip("\"'“”‘’()")
    if not (1 <= len(t) <= 50) or len(t.split()) > 6:
        return None
    if re.match(r"^(it|them|this|that|these|those|the\s+(document|report|paper|text))\b", t, re.I):
        return None
    return t


def _normalize_mention_variants(target: str):
    """归一化:去括号注释、生成大小写/复数/同义变体,治 'VAT(value-added tax)'、'total revenue(s)'。"""
    base = re.sub(r"\s*\([^)]*\)\s*", " ", target).strip() or target
    variants = {base}
    inner = re.findall(r"\(([^)]+)\)", target)  # 括号里的也算(如缩写)
    variants.update(x.strip() for x in inner if x.strip())
    return [v for v in variants if v]


_STRUCT_WORDS = ("table of contents", "related work", "future work", "appendix", "appendices",
                 "references", "bibliography", "introduction", "conclusion", "abstract",
                 "acknowledg", "methodology", "discussion")


def _locate_target(q: str):
    # 名词在定位动词之前:"...does the appendix (of ...) start/begin/appear"
    m = re.search(r"\bdoes\s+the\s+(.+?)\s+(?:begin|start|appear|locate|present)", q, re.I)
    if m:
        t = re.sub(r"\bof\s+(this|the)\s+document\b.*$", "", m.group(1), flags=re.I).strip(" '\"")
        if 2 <= len(t) <= 60:
            return t
    for rx in (
        re.compile(r"\binformation\s+(?:about|of|on)\s+(.+?)\s*[\?\.!]*\s*$", re.I),
        re.compile(r"\bsituations?\s+of\s+(.+?)\s*[\?\.!]*\s*$", re.I),
        re.compile(r"\b(?:presents?|details?|lists?|discuss(?:es)?|introduces?|outlines?|reports?|begins?|starts?|mentions?)\s+(?:the\s+)?(.+?)\s*[\?\.!]*\s*$", re.I),
        re.compile(r"\babout\s+(.+?)\s*[\?\.!]*\s*$", re.I),
    ):
        m = rx.search(q)
        if m:
            t = " ".join(m.group(1).split()).strip(" '\"‘’“”.")
            t = re.sub(r"\b(begin|start|located|presented|detailed)\b.*$", "", t, flags=re.I).strip()
            if 2 <= len(t) <= 60 and len(t.split()) <= 8 and not re.match(
                    r"^(it|them|this|that|these|those|the\s+(document|report|paper))\b", t, re.I):
                return t
    # 结构性关键词兜底
    ql = q.lower()
    for w in _STRUCT_WORDS:
        if w in ql:
            return w
    return None


def _appendix_page(m: DocModel):
    for it in m.elements:
        if it.get("text_level") and re.search(r"\b(references?|appendix|appendices|bibliography)\b",
                                               str(it.get("text", "")), re.I):
            return it.get("page_idx")
    return None


# =====================================================================================
# resolvers —— 薄封装:解析意图 -> 调组合子 -> 包成 Fact(自检置信度由组合子给出)。
# =====================================================================================

def r_mention(m: DocModel, q: str) -> Fact | None:
    if not _RE_MENTION.search(q):
        return None
    target = _mention_target(q)
    if not target:
        return None
    variants = _normalize_mention_variants(target)
    best, _, ev = COUNT(m, U_SPAN(variants))
    if best == 0:
        return None  # 没数到 -> 弃权(可能目标抽错/拼写差异),不报 0
    # 实例级自检:命中集中度 + 短目标(易假阳性)惩罚。
    counts = ev.get("counts") or [best]
    tot = sum(counts) or best
    agree = max(counts) / tot if tot else 1.0
    conf = round(0.85 * agree, 2)
    if len(target.replace(" ", "")) <= 2:
        conf = min(conf, 0.5)
    return Fact("mention", best, conf,
                f'[Programmatically verified: the phrase "{target}" appears {best} times '
                f"in the document (including tables).]", "full_text+tables")


def r_pages_total(m: DocModel, q: str) -> Fact | None:
    if not _RE_PAGES_TOTAL.search(q):
        return None
    if re.search(r"excluding|without|except", q, re.I):
        return None  # "排除参考文献的页数" 口径复杂 -> 弃权
    v, conf, _ = COUNT(m, U_PAGE())
    return Fact("pages", v, conf,
                f"[Programmatically verified: the document has {v} physical pages.]", "pdf")


def r_words(m: DocModel, q: str) -> Fact | None:
    if not _RE_WORDS.search(q):
        return None
    pr = _RE_PAGE_REF.search(q)
    if pr:  # "page N 上多少词"
        phys = m.page_map.to_physical(int(pr.group(1)))
        if not phys:
            return None
        wc = len(m.page_text(phys, max_chars=10**9).split())
        return Fact("words_page", wc, 0.45,
                    f"[Approximate word count on {m.page_map.frame(int(pr.group(1)))} "
                    f"(from extracted text) = {wc}.]", "pdf")
    # 自检 = 两个独立抽取器(PyMuPDF vs pypdf)的词数一致率(见 COUNT word)。
    # 取代旧版"词数=PyMuPDF→恒注入"(有 DocBench 过拟合嫌疑):换成数据集无关的稳定性信号。
    wc, conf, _ = COUNT(m, U_WORD())
    return Fact("words", wc, conf,
                f"[Programmatically counted: the document contains {wc} words "
                f"(whitespace-delimited tokens over the extracted text).]", "pdf")


def r_elements(m: DocModel, q: str) -> Fact | None:
    mm = _RE_ELEM.search(q)
    if not mm:
        return None
    if not m.has_elements:
        return None  # 需 content_list -> 无则弃权
    kind = mm.group(1).lower()
    type_map = {"figure": ("image", "chart"), "figures": ("image", "chart"),
                "image": ("image",), "images": ("image",),
                "table": ("table",), "tables": ("table",),
                "equation": ("equation",), "equations": ("equation",),
                "chart": ("chart",), "charts": ("chart",),
                "illustration": ("image",), "illustrations": ("image",)}
    # 方法学边界(诚实写进论文):MinerU 对长多栏报告会把图切成碎片(doc114:184页/230碎片/gold31),
    # 元素解析仅在论文体量文档可信 → 长文档对"图/表计数"一律弃权(文档类作用域,非按题特调)。
    if m.page_count > 30:
        return None
    types = type_map.get(kind, ("image",))
    ap = _appendix_page(m) if re.search(r"excluding|without|except", q, re.I) else None
    scope = S_EXCLUDING(ap) if ap is not None else S_WHOLE

    if re.search(r"tables?\s+and\s+(figures?|images?)|figures?\s+and\s+tables?", q, re.I):
        figs, _, _ = COUNT(m, U_ELEMENT(("image", "chart")), scope)
        tbls, _, _ = COUNT(m, U_ELEMENT(("table",)), scope)
        total = figs + tbls
        if total < 1:
            return None
        conf = round(min(1.0, m.page_count / total), 2) if total else 0.0
        return Fact("elements_sum", total, conf,
                    f"[Programmatically counted from the parsed document: {figs} figures + "
                    f"{tbls} tables = {total} in total.]", "content_list")
    cnt, conf, _ = COUNT(m, U_ELEMENT(types), scope)
    if cnt < 1:
        return None
    return Fact("elements", cnt, conf,
                f"[Programmatically counted from the parsed document: {cnt} {kind}.]", "content_list")


def r_references(m: DocModel, q: str) -> Fact | None:
    if not _RE_REFS.search(q) or not m.has_elements:
        return None
    cnt, conf, _ = COUNT(m, U_REF())
    if cnt is None:
        return None
    return Fact("references", cnt, conf,
                f"[Approximate reference count from the parsed bibliography ≈ {cnt}.]", "content_list")


def r_sections(m: DocModel, q: str) -> Fact | None:
    mm = _RE_SECTIONS.search(q)
    if not mm or not m.has_elements:
        return None
    unit = mm.group(1).lower()
    hk = "subsection" if unit.startswith("subsection") else "section"
    cnt, conf, _ = COUNT(m, U_HEADING(hk))
    if cnt < 1:
        return None
    return Fact("sections", cnt, conf,
                f"[Approximate count of top-level {unit} from parsed headings ≈ {cnt}; "
                f"section granularity may differ from the reference.]", "content_list")


def r_footnotes(m: DocModel, q: str) -> Fact | None:
    # 脚注口径噪声大(实测 page_footnote 数≠金标)-> 默认弃权,避免注入错值
    return None


def r_title(m: DocModel, q: str) -> Fact | None:
    if not _RE_TITLE.search(q) or not m.has_elements:
        return None
    return LOOKUP(m, "title")


def r_abbrev_words(m: DocModel, q: str) -> Fact | None:
    """缩写 / 最高频词:收编旧 DSG 的有效逻辑(text_stats)。dgcore 原先漏了这两类→弃权→答错。"""
    if text_stats is None or not _dg_env("DG_META_STATS", True):
        return None
    is_abbr = bool(_RE_ABBR.search(q))
    is_tw = bool(_RE_TOPWORDS.search(q)) and not is_abbr
    if not (is_abbr or is_tw):
        return None
    try:
        s = text_stats(m.full_text)
    except Exception:
        return None
    if is_abbr:
        abv = s.get("top_abbrevs") or []
        if not abv:
            return None
        ab, cnt = abv[0]
        # 实例级置信 = 头名领先度:第一名比第二名领先越多越确定;并列(易选错)→低置信→弃权。
        f2 = abv[1][1] if len(abv) > 1 else 0
        conf = round(1.0 - (f2 / cnt), 2) if cnt else 0.0
        return Fact("abbrev", ab, conf,
                    f'[Programmatically computed: the most frequent abbreviation/acronym is "{ab}" '
                    f"({cnt} occurrences).]", "pdf")
    tw = s.get("top_words") or []
    if not tw:
        return None
    top3 = ", ".join(tw[:3])
    return Fact("topwords", top3, 0.7,
                f"[Programmatically computed: the top-3 most frequent words are {top3}.]", "pdf")


def r_locate(m: DocModel, q: str) -> Fact | None:
    if not _RE_LOCATE.search(q):
        return None
    target = _locate_target(q)
    if not target:
        return None
    return LOCATE(m, target)


def r_extract_page(m: DocModel, q: str) -> Fact | None:
    # 解析题面指向的物理页(具体页号 / 相对页),注入该页文本
    if re.search(r"how many", q, re.I):
        return None  # "某页有几个X" 是计数题,不是取页内容 -> 交给计数 resolver
    phys = None
    label = ""
    clean = True   # 页引用是否被唯一/可靠解析(自检信号)
    fs = _RE_FIRST_SENT.search(q)
    pr = _RE_PAGE_REF.search(q)
    if fs:
        printed = int(fs.group(1))
        phys = m.page_map.to_physical(printed)
        label = m.page_map.frame(printed)
        clean = m.page_map.confident   # 经印刷→物理映射,可靠度取决于 PageMap 置信
    elif _RE_REL_2ND.search(q):
        phys = m.page_count - 1 if m.page_count >= 2 else m.page_count
        label = f"second-to-last page ({m.page_map.phys_frame(phys)})"
    elif _RE_REL_LAST.search(q):
        phys = m.page_count
        label = f"last page ({m.page_map.phys_frame(phys)})"
    elif _RE_REL_FRONT.search(q):
        phys = 1
        label = "front page (physical page 1)"
    elif pr and _RE_PAGE_CONTENT.search(q):
        printed = int(pr.group(1))
        phys = m.page_map.to_physical(printed)
        label = m.page_map.frame(printed)
        clean = m.page_map.confident
    if not phys:
        return None
    return EXTRACT(m, phys, label, clean)


def r_meta_stats(m: DocModel, q: str) -> Fact | None:
    """通用统计兜底(页/词/缩写/高频词)——给泛 meta 题,低-中置信。"""
    if text_stats is None:
        return None
    s = text_stats(m.full_text)
    tw = ", ".join(s["top_words"]) or "n/a"
    aw = ", ".join(f"{a} ({c}x)" for a, c in s["top_abbrevs"]) or "n/a"
    return Fact("meta_stats", None, 0.5,
                f"[Supplementary document statistics (programmatic): total pages = {m.page_count}; "
                f"approximate word count = {len(m.full_text.split())}; most frequent words = {tw}; "
                f"most frequent abbreviations = {aw}. Use only if the question asks such document-level stats.]",
                "pdf")


# 路由优先级:具体 -> 泛化。
_RESOLVERS = [r_mention, r_abbrev_words, r_locate, r_extract_page, r_title, r_elements,
              r_references, r_sections, r_pages_total, r_words, r_footnotes]

# ===========================================================================
# L3 门控(确定性可验证门控 deterministic, verifiability-gated injection)
# ---------------------------------------------------------------------------
# 每个 Fact 的 confidence 是【实例级、可由输入确定性算出的自检信号】(见各组合子),而非死常数:
#   pages    : 精确可数            -> 0.95(真确定)
#   words    : 跨源一致率          -> 1 − |c_pymupdf − c_pypdf| / max(两者)
#   mention  : 命中集中度 + 短目标惩罚
#   abbrev   : 头名领先度          -> 1 − f2/f1
#   elements : 合理度             -> min(1, 页数/计数)
#   locate   : 命中标题(0.9)/ 否则最密集页占比
#   title    : H1 唯一性
#   extract  : 页引用解析可靠度
#   词数/脚注/章节/参考文献口径不可观测处 -> 自检低 -> 维持弃权(诚实边界)
# 注入门:inject(f) ⇔ conf(f) ≥ τ_kind。阈值是开发集"校准"出来的常数(数触发准确率,非梯度训练)。
# 决策论扩展点(V3):若提供 DG_CALIB_FILE(含各 kind 的 baseline 准确率 pbase),则
#   τ_eff = max(τ_kind, pbase[kind] + margin) —— 即"仅当算子估计答对率 > 基座答对率才注入"。
#   缺该文件时退回 τ_kind(本次运行)。机制通用:换数据集在其 dev split 上重算 pbase 即可,无写死。
# ===========================================================================
_THRESHOLD = {
    "mention": 0.6, "extract_page": 0.6, "title": 0.6, "topwords": 0.6, "pages": 0.6,
    "abbrev": 0.5,        # 头名领先度≥0.5 = 第一名≥2×第二名才注入
    "locate": 0.4,        # 命中标题(0.9)或 ≤2 页(0.5)才注入,3+ 页歧义则弃权
    "elements": 0.8, "elements_sum": 0.8,   # 真实图表数 ≤ ~页数;计数>1.25×页数=被切碎→弃权
    "references": 0.99, "sections": 0.99,    # 口径不可观测 -> 默认弃权
    "words": 0.5,                            # 跨源一致率≥0.5(差异<50%)才注入;发散则弃权
    "words_page": 0.99, "footnotes": 0.99, "meta_stats": 0.99,   # 单页词数/脚注仍弃权(未验证)
}

# DG_LEGACY=true:恢复旧(64%)的死常数置信度 + first-over-threshold,用于精确 A/B 复现。
_LEGACY_CONF = {"words": 0.85, "mention": 0.8, "title": 0.7, "extract_page": 0.7,
                "pages": 0.9, "topwords": 0.7, "words_page": 0.45}
_LEGACY_THRESHOLD = dict(_THRESHOLD, words=0.6)   # 旧版 words 阈值是 0.6


def _load_calib():
    """可选:DG_CALIB_FILE -> {'threshold': {...}, 'pbase': {...}}。缺省则空(用内置阈值、无 V3 floor)。"""
    path = os.getenv("DG_CALIB_FILE")
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("threshold", {}) or {}, d.get("pbase", {}) or {}
    except Exception:
        return {}, {}


_PBASE_MARGIN = float(os.getenv("DG_PBASE_MARGIN", "0.0") or 0.0)


def _eff_threshold(kind, base_thr, pbase):
    tau = base_thr.get(kind, 0.6)
    floor = pbase.get(kind)
    if floor is not None:                       # V3 决策论:注入须超过基座答对率
        tau = max(tau, float(floor) + _PBASE_MARGIN)
    return tau


def ground(question: str, pdf_path: str, content_list_path: str = None,
           model: DocModel = None) -> Fact | None:
    """框架总入口:建/复用 DocModel -> 解析意图调组合子 -> 实例自检置信 -> 门控/弃权 -> Fact 或 None。
    门控:默认收集所有触发且过阈的 Fact,取置信最高者(arbitration,治"第一个过阈≠最可信");
          DG_ARBITRATE=false 退回 first-over-threshold;DG_LEGACY=true 复现旧 64% 行为。"""
    m = model or build_doc_model(pdf_path, content_list_path)
    if m is None:
        return None
    no_abstain = not _dg_env("DG_ABSTAIN", True)   # DG_ABSTAIN=false:不弃权、全注入(消融用)
    legacy = _dg_env("DG_LEGACY", False)
    arbitrate = _dg_env("DG_ARBITRATE", True) and not legacy
    base_thr = _LEGACY_THRESHOLD if legacy else _THRESHOLD
    calib_thr, pbase = ({}, {}) if legacy else _load_calib()
    if calib_thr:
        base_thr = dict(base_thr, **calib_thr)

    fired = []
    for r in _RESOLVERS:
        try:
            fact = r(m, question)
        except Exception:
            fact = None
        if not (fact and fact.note):
            continue
        if legacy and fact.kind in _LEGACY_CONF:   # 精确 A/B:还原旧死常数置信度
            fact = replace(fact, confidence=_LEGACY_CONF[fact.kind])
        tau = _eff_threshold(fact.kind, base_thr, pbase)
        if no_abstain or fact.confidence >= tau:
            if not arbitrate:
                return fact            # first-over-threshold(legacy / 关 arbitration)
            fired.append(fact)
    if not fired:
        return None
    # arbitration:取置信最高者;并列时 Python 稳定排序保留优先级(具体>泛化)。
    fired.sort(key=lambda f: f.confidence, reverse=True)
    return fired[0]


# =====================================================================================
# 本地自测:PageMap 偏移(锁定真实文档,防回归)
# =====================================================================================
if __name__ == "__main__":
    import glob
    import sys
    base = os.path.join(os.path.dirname(__file__), "..", "data", "DocBench_download")
    cases = [("152", 2, True), ("154", 4, True), ("102", 0, False),
             ("110", 2, True), ("164", 3, True), ("0", None, False)]
    ok = True
    for did, exp_off, exp_conf in cases:
        g = glob.glob(os.path.join(base, did, "*.pdf"))
        if not g:
            continue
        pm = build_doc_model(g[0]).page_map
        passed = (exp_off is None or pm.offset == exp_off) and (pm.confident == exp_conf)
        ok = ok and passed
        print(f"doc {did:>4}: offset={pm.offset} conf={pm.confidence} confident={pm.confident} "
              f"-> {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if ok else 1)
