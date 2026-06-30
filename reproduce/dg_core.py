"""dg_core —— DG-RAG:免训练、零额外 LLM 调用的确定性接地层(单文档、绕过检索、读原文)。

形式化(论文口径):对一道问题,我们做三步,
    parse  : 问题 q ──▶ 类型化查询 Q       (确定性语义解析器;零 LLM)
    eval   : ⟦Q⟧(D) ──▶ Fact(value, s)     (在文档模型 D 上解释执行;s = 可验证自检分)
    gate   : 当 s ≥ τ 才注入,否则弃权回退基座(确定性可验证门控;期望非回归)

查询语言只有 4 个组合子(算子代数 O),与具体题型无关——题型只是给组合子传不同参数:
    Q ::= Count(unit, scope)        计数:unit ∈ {Page, Word, Element(t), Heading(k), Ref, Span(x)}
        | Locate(target)            定位:把 target 定位到真正讨论它的页
        | Extract(ref)              取页:把页引用(字面/相对/印刷)投到物理页取文本
        | Lookup(field)             查域:从结构化区读单值(title/authors/date/abbrev/...)

三层落地:
  L1 DocumentModel(build_doc_model):每篇建一次的单一真相源——PageMap(物理↔印刷)/逐页文本/
     canonical 全文 / content_list typed elements。所有算子共用,杜绝口径不一。
  L2 算子代数(COUNT/LOCATE/EXTRACT/LOOKUP)+ 解析文法(_GRAMMAR)+ 解释器(evaluate)。
  L3 门控(ground 内 gate):单一决策论门控,无手设阈值、无魔法数字——仅当该算子 kind 在留出 dev split 上
     答对率 p_op > 基座 p_base(+margin)才采纳其注入,否则弃权回退基座 → 期望非回归。p_op/p_base 由
     calibrate_v3 在 dev 上数频率(Laplace 平滑,小样本不被一条幸运样本采纳)、存 DG_CALIB_FILE;非梯度训练
     → training-free,可在任意数据集 dev split 重跑。无 calib=观测/全开(校准跑本身);有 calib 但某 kind 未被
     dev 观测到=保守弃权(不裸奔注入)。实例级自检 confidence 仅用于多命中时仲裁取最自洽者(非门控阈值)。

与 GlobalRAG 区分:GlobalRAG 是【语料级】符号算子(跨文档实体计数/极值/排序);本框架是【单文档】、
绕检索读原文、且带【确定性可验证门控】(算子仅在自检判定可靠时开口),training-free、跨基座可迁移。

零回归:query.py 经 ENABLE_DG_CORE 接入,默认关。DG_LEGACY=true 精确复现旧(64%)行为做 A/B。
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
# Fact —— eval 的产物:一个值 + 一个可验证自检分 + 注入文本
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
# 每个组合子接受参数,输出 (value, confidence);confidence = 实例级确定性自检,而非死常数。
# =====================================================================================

# ---- unit / scope 构造子(COUNT 的参数,可组合)----
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
    """参数化计数器:一个组合子覆盖 页/词/元素/标题/参考文献/正则跨度。返回 (value, confidence, evidence)。
    confidence = 该 unit 的实例级确定性自检;value=None 表示该 unit 在本文档不可数。"""
    kind, params = unit
    skind, szone = scope
    if kind == "page":
        return m.page_count, 0.95, {}                      # 精确可数 -> 自检恒高
    if kind == "word":
        v = len(m.full_text.split())
        alt = _pypdf_wordcount(m.pdf_path)
        # 自检 = 跨源一致率(PyMuPDF vs pypdf):吻合->高;发散->低->弃权。数据集无关。
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
        conf = round(min(1.0, m.page_count / c), 2) if c else 0.0   # 合理度:被切碎则骤降->弃权
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
        return c, 0.4, {}                                  # 章节粒度口径不可观测 -> 低置信
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
            if it.get("text_level"):
                break
            t = str(it.get("text", "")).strip()
            if it.get("type") == "list":
                c += max(1, t.count("\n") + 1)
            elif re.match(r"^\[?\d+\]?\.?\s|\b\(\d{4}\)\b|\b\d{4}\.", t):
                c += 1
        if c < 3:
            return None, 0.0, {}
        return c, 0.45, {}
    if kind == "span":
        counts = [_flex_count(_span_text(m), v) for v in params]
        best = max(counts, default=0)
        return best, None, {"counts": counts}              # conf 交调用方(需 target 长度判假阳性)
    return None, 0.0, {}


def LOCATE(m: DocModel, target: str):
    """定位组合子:把 target 定位到【真正讨论它的页】(排除目录"提一嘴")。
    自检 = 命中章节标题(0.9)/ 否则出现最密集页的占比。"""
    tl = target.lower()
    cands = {tl}                                   # 健壮性:单复数变体("related works"->"related work")
    _w = tl.split()
    if _w and _w[-1].endswith("s") and len(_w[-1]) > 3:
        cands.add(" ".join(_w[:-1] + [_w[-1][:-1]]))
    toc = set()
    for it in m.elements:
        if it.get("text_level") and re.search(r"table of contents|^\s*contents\s*$",
                                              str(it.get("text", "")), re.I):
            pg = it.get("page_idx")
            if isinstance(pg, int):
                toc.add(pg + 1)
    heading_pages = []
    for it in m.elements:
        pg = it.get("page_idx")
        if (isinstance(pg, int) and it.get("text_level") and (pg + 1) not in toc
                and any(c in str(it.get("text", "")).lower() for c in cands)
                and (pg + 1) not in heading_pages):
            heading_pages.append(pg + 1)
    if heading_pages:
        pages, conf = sorted(heading_pages)[:2], 0.9
    else:
        hits = {}
        for i, t in enumerate(m.per_page_text):
            if (i + 1) in toc:
                continue
            c = max((_flex_count(t, cand) for cand in cands), default=0)
            if c > 0:
                hits[i + 1] = c
        if not hits:
            return None
        total = sum(hits.values())
        top = max(hits.values())
        pages = sorted([p for p, c in hits.items() if c == top])[:2]
        conf = round(top / total, 2)
    if m.page_map.confident or _dg_env("DG_PAGEMAP_REFRAME", False):
        framed = "; ".join(m.page_map.phys_frame(p) for p in pages)
        note = (f'[Programmatic locator: "{target}" appears on {framed}. '
                f"Page questions may expect either the printed or physical number.]")
    else:
        framed = ", ".join(f"page {p}" for p in pages)
        note = f'[Programmatic locator: "{target}" appears on {framed}.]'
    return Fact("locate", pages, conf, note, "content_list/pdf")


def EXTRACT(m: DocModel, phys: int, label: str, clean: bool):
    """取页组合子:返回某物理页文本。自检 = 页引用是否被唯一/可靠解析(clean)+ 文本非空。"""
    snip = m.page_text(phys)
    if not snip:
        return None
    conf = 0.85 if clean else 0.65
    return Fact("extract_page", phys, conf,
                f"[Content of {label}, extracted programmatically: {snip}]", "pdf")


def LOOKUP(m: DocModel, fieldname: str):
    """查域组合子:从结构化区读单值。field ∈ {title, authors, last_author, date, abbrev, topwords}。"""
    if fieldname == "title":
        if not m.has_elements:
            return None
        h1 = [it for it in m.elements if it.get("text_level") == 1 and str(it.get("text", "")).strip()]
        if not h1:
            return None
        title = " ".join(str(h1[0]["text"]).split())
        conf = 0.9 if len(h1) == 1 else 0.65          # 自检 = H1 唯一性
        return Fact("title", title, conf,
                    f'[Programmatically extracted document title: "{title}".]', "content_list")
    if fieldname in ("authors", "last_author"):
        block = _author_block(m)
        names = _author_names(block)
        if not names:
            return None
        clean = not any(re.search(r"\s+and\s+|,", t) for t in block)   # 自检 = 作者区是否干净
        conf = 0.8 if clean else 0.45
        if fieldname == "last_author":
            return Fact("last_author", names[-1], conf,
                        f'[Programmatically extracted from the front matter: the last author listed is '
                        f'"{names[-1]}".]', "content_list")
        return Fact("authors", len(names), conf,
                    f"[Programmatically extracted from the front matter: the paper lists {len(names)} "
                    f"authors ({', '.join(names[:8])}).]", "content_list")
    if fieldname == "date":
        head = " ".join(m.per_page_text[:2]) if m.per_page_text else ""
        uniq = sorted({d.title() for d in _RE_COVER_DATE.findall(head)})
        if len(uniq) != 1:                            # 自检 = 封面唯一 Month YYYY;否则弃权
            return None
        return Fact("date", uniq[0], 0.7,
                    f"[Programmatically extracted from the cover/front matter: the document date is "
                    f"{uniq[0]}.]", "pdf")
    if fieldname in ("abbrev", "topwords"):
        if text_stats is None:
            return None
        try:
            s = text_stats(m.full_text)
        except Exception:
            return None
        if fieldname == "abbrev":
            abv = s.get("top_abbrevs") or []
            if not abv:
                return None
            ab, cnt = abv[0]
            f2 = abv[1][1] if len(abv) > 1 else 0
            conf = round(1.0 - (f2 / cnt), 2) if cnt else 0.0   # 自检 = 头名领先度 1−f2/f1
            return Fact("abbrev", ab, conf,
                        f'[Programmatically computed: the most frequent abbreviation/acronym is "{ab}" '
                        f"({cnt} occurrences).]", "pdf")
        tw = s.get("top_words") or []
        if not tw:
            return None
        top3 = ", ".join(tw[:3])
        return Fact("topwords", top3, 0.7,
                    f"[Programmatically computed: the top-3 most frequent words are {top3}.]", "pdf")
    return None


# ---- front_matter 结构化解析(覆盖前页事实:作者/日期)。论文通用结构,与数据集无关。----
_AFFIL_PREFIX = "*∗†‡§¶"


def _author_block(m: DocModel):
    """页0:level-1 标题之后、首个 level-2 标题(Abstract/Introduction)之前的 text 行 = 作者区。
    去掉邮箱行/隶属(标记或数字开头)行/过长句子行。返回候选作者文本行。"""
    if not m.elements:
        return []
    started, lines = False, []
    for it in m.elements:
        if (it.get("page_idx") or 0) > 0:
            break
        lv = it.get("text_level")
        t = " ".join(str(it.get("text", "")).split())
        if not started:
            if lv == 1 and t:
                started = True
            continue
        if lv and lv >= 2:
            break
        if it.get("type") != "text" or not t:
            continue
        if "@" in t or t[0] in _AFFIL_PREFIX or t[0].isdigit():
            continue
        if len(t.split()) > 12:
            continue
        lines.append(t)
    return lines


def _author_names(lines):
    """从作者区行抽姓名:行内可能 'A and B and C' 或 'A, B, and C'。姓名 = 2-4 个首字母大写词。"""
    names = []
    for t in lines:
        for p in re.split(r"\s+and\s+|,\s*", t):
            p = p.strip().strip(_AFFIL_PREFIX + " ")
            toks = p.split()
            if 2 <= len(toks) <= 4 and sum(1 for w in toks if w[:1].isupper()) >= 2:
                names.append(p)
    return names


_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
_RE_COVER_DATE = re.compile(r"\b((?:" + _MONTHS + r")\s+\d{4})\b", re.I)


# =====================================================================================
# 解析文法:问题 q ──▶ 类型化查询 Q(零 LLM,确定性)。
# Q 的种类:Count / Locate / Extract / Lookup —— 对应 4 个组合子。
# =====================================================================================

@dataclass(frozen=True)
class Count:
    unit: tuple                 # ("page",) ("word",) ("element",..) ("ref",) ("heading",k) ("span",target)
    scope: tuple = S_WHOLE      # 或 ("page_printed", N) 表示"印刷第 N 页上的词数"


@dataclass(frozen=True)
class Locate:
    target: str


@dataclass(frozen=True)
class Extract:
    ref: tuple                  # ("first_sent",N)|("printed",N)|("last",)|("second_last",)|("front",)


@dataclass(frozen=True)
class Lookup:
    field: str                  # title|authors|last_author|date|abbrev|topwords


# ---- 意图正则(确定性语义解析器的词法)----
_RE_MENTION = re.compile(r"how many time", re.I)
_RE_WORDS = re.compile(r"how many words|number of words|word count|words (?:are|does|in|in total)", re.I)
_RE_PAGES_TOTAL = re.compile(r"how many pages", re.I)
_RE_ELEM = re.compile(r"how many\s+(figures?|images?|tables?|equations?|charts?|illustrations?)", re.I)
_RE_REFS = re.compile(r"how many\s+(references?|citations?|cited)", re.I)
_RE_SECTIONS = re.compile(r"how many\s+(sections?|chapters?|parts?|subsections?)", re.I)
_RE_FOOTNOTES = re.compile(r"how many\s+footnotes?", re.I)
_RE_LOCATE = re.compile(
    r"\b(on|at|from)\s+(which|what)\s+page\b"
    r"|\bwhich\s+page\b[^?]*\b(is|are|does|do|located|show|present|discuss|list|introduce|begin|start)\b"
    r"|\bwhat\s+page\b", re.I)
_RE_PAGE_REF = re.compile(r"\bpage\s+(\d{1,4})\b", re.I)
_RE_FIRST_SENT = re.compile(r"first sentence on page\s+(\d{1,4})", re.I)
_RE_REL_LAST = re.compile(r"\b(last|final)\s+page\b", re.I)
_RE_REL_2ND = re.compile(r"\bsecond[-\s]to[-\s]last\s+page\b", re.I)
_RE_REL_FRONT = re.compile(r"\b(front\s*page|first\s+page|cover\s+page|frontpage)\b", re.I)
_RE_PAGE_CONTENT = re.compile(
    r"(what|topic|content|focus|message|summary|purpose|talk about|conveyed|present)", re.I)
_RE_TITLE = re.compile(r"\b(document|paper|report|newspaper)\s+title\b|\btitle of the\b", re.I)
_RE_ABBR = re.compile(r"\b(?:most\s+(?:common|frequent)\s+)?(?:abbreviation|acronym)s?\b", re.I)
_RE_TOPWORDS = re.compile(r"\b(?:top\s*\d*\s*)?most\s+(?:common|frequent)\s+words?\b", re.I)
_RE_AUTHORS_CNT = re.compile(r"how many\s+authors?\b", re.I)
_RE_LAST_AUTHOR = re.compile(r"\b(?:last|final)\s+author\b", re.I)
_RE_DATE_Q = re.compile(
    r"\bwhen (?:was|is)\b|\bwhat (?:date|year)\b|\b(?:release|publication|published|issue)\s+date\b"
    r"|\bdate (?:of|was|the document)\b", re.I)


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
    inner = re.findall(r"\(([^)]+)\)", target)
    variants.update(x.strip() for x in inner if x.strip())
    return [v for v in variants if v]


_STRUCT_WORDS = ("table of contents", "related work", "future work", "appendix", "appendices",
                 "references", "bibliography", "introduction", "conclusion", "abstract",
                 "acknowledg", "methodology", "discussion")


def _locate_target(q: str):
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
            t = re.sub(r"^(?:about|the|a|an)\s+", "", t, flags=re.I).strip()   # 去引导词(治"about the related works")
            if 2 <= len(t) <= 60 and len(t.split()) <= 8 and not re.match(
                    r"^(it|them|this|that|these|those|the\s+(document|report|paper))\b", t, re.I):
                return t
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


# ---- 文法规则:每条 = 一个 builder(question) -> Query | None。优先级 = 列表顺序(具体->泛化)。----
def _g_mention(q):
    if not _RE_MENTION.search(q):
        return None
    t = _mention_target(q)
    return Count(("span", t)) if t else None


def _g_authors(q):
    if _dg_env("DG_LEGACY", False) or not _dg_env("DG_COVERAGE", True):
        return None
    if _RE_LAST_AUTHOR.search(q):
        return Lookup("last_author")
    if _RE_AUTHORS_CNT.search(q):
        return Lookup("authors")
    return None


def _g_date(q):
    if _dg_env("DG_LEGACY", False) or not _dg_env("DG_COVERAGE", True):
        return None
    return Lookup("date") if _RE_DATE_Q.search(q) else None


def _g_abbrev(q):
    if text_stats is None or not _dg_env("DG_META_STATS", True):
        return None
    is_abbr = bool(_RE_ABBR.search(q))
    is_tw = bool(_RE_TOPWORDS.search(q)) and not is_abbr
    if is_abbr:
        return Lookup("abbrev")
    if is_tw:
        return Lookup("topwords")
    return None


def _g_locate(q):
    if not _RE_LOCATE.search(q):
        return None
    t = _locate_target(q)
    return Locate(t) if t else None


def _g_extract(q):
    if re.search(r"how many", q, re.I):
        return None
    fs = _RE_FIRST_SENT.search(q)
    if fs:
        return Extract(("first_sent", int(fs.group(1))))
    if _RE_REL_2ND.search(q):
        return Extract(("second_last",))
    if _RE_REL_LAST.search(q):
        return Extract(("last",))
    if _RE_REL_FRONT.search(q):
        return Extract(("front",))
    pr = _RE_PAGE_REF.search(q)
    if pr and _RE_PAGE_CONTENT.search(q):
        return Extract(("printed", int(pr.group(1))))
    return None


def _g_title(q):
    return Lookup("title") if _RE_TITLE.search(q) else None


def _g_elements(q):
    mm = _RE_ELEM.search(q)
    if not mm:
        return None
    kind = mm.group(1).lower()
    combined = bool(re.search(r"tables?\s+and\s+(figures?|images?)|figures?\s+and\s+tables?", q, re.I))
    excluding = bool(re.search(r"excluding|without|except", q, re.I))
    return Count(("element", kind, combined, excluding))


def _g_references(q):
    return Count(("ref",)) if _RE_REFS.search(q) else None


def _g_sections(q):
    mm = _RE_SECTIONS.search(q)
    if not mm:
        return None
    unit = mm.group(1).lower()
    return Count(("heading", "subsection" if unit.startswith("subsection") else "section", unit))


def _g_pages(q):
    if not _RE_PAGES_TOTAL.search(q):
        return None
    if re.search(r"excluding|without|except", q, re.I):
        return None
    return Count(("page",))


def _g_words(q):
    if not _RE_WORDS.search(q):
        return None
    pr = _RE_PAGE_REF.search(q)
    if pr:
        return Count(("word",), scope=("page_printed", int(pr.group(1))))
    return Count(("word",))


# 文法(优先级 = 顺序),与旧 _RESOLVERS 一一对应,保证行为等价。footnotes 永远弃权 -> 无规则。
_GRAMMAR = [_g_mention, _g_authors, _g_date, _g_abbrev, _g_locate, _g_extract, _g_title,
            _g_elements, _g_references, _g_sections, _g_pages, _g_words]


def parse(q: str):
    """问题 -> 候选查询列表(按优先级)。多数题只匹配一条;门控再从中择优/弃权。"""
    out = []
    for rule in _GRAMMAR:
        try:
            query = rule(q)
        except Exception:
            query = None
        if query is not None:
            out.append(query)
    return out


# =====================================================================================
# 解释器:evaluate(Q, D) -> Fact | None。把类型化查询在文档模型上确定性执行。
# =====================================================================================
_ELEM_TYPES = {"figure": ("image", "chart"), "figures": ("image", "chart"),
               "image": ("image",), "images": ("image",),
               "table": ("table",), "tables": ("table",),
               "equation": ("equation",), "equations": ("equation",),
               "chart": ("chart",), "charts": ("chart",),
               "illustration": ("image",), "illustrations": ("image",)}


def _eval_count(m: DocModel, query: Count):
    head = query.unit[0]
    if head == "page":
        v, conf, _ = COUNT(m, U_PAGE())
        return Fact("pages", v, conf,
                    f"[Programmatically verified: the document has {v} physical pages.]", "pdf")
    if head == "word":
        if query.scope[0] == "page_printed":
            printed = query.scope[1]
            phys = m.page_map.to_physical(printed)
            if not phys:
                return None
            wc = len(m.page_text(phys, max_chars=10**9).split())
            return Fact("words_page", wc, 0.45,
                        f"[Approximate word count on {m.page_map.frame(printed)} "
                        f"(from extracted text) = {wc}.]", "pdf")
        wc, conf, _ = COUNT(m, U_WORD())
        return Fact("words", wc, conf,
                    f"[Programmatically counted: the document contains {wc} words "
                    f"(whitespace-delimited tokens over the extracted text).]", "pdf")
    if head == "span":
        target = query.unit[1]
        variants = _normalize_mention_variants(target)
        best, _, ev = COUNT(m, U_SPAN(variants))
        if best == 0:
            # 健壮性:正文未命中 -> 回退到含表格正文再数一次(治"短语只出现在表格里",如封面/财报表格)。
            alt = [_flex_count(m.element_text_with_tables(), v) for v in variants]
            best = max(alt, default=0)
            ev = {"counts": alt}
            if best == 0:
                return None
        counts = ev.get("counts") or [best]
        tot = sum(counts) or best
        agree = max(counts) / tot if tot else 1.0
        conf = round(0.85 * agree, 2)
        if len(target.replace(" ", "")) <= 2:        # 短目标易假阳性 -> 压低置信
            conf = min(conf, 0.5)
        return Fact("mention", best, conf,
                    f'[Programmatically verified: the phrase "{target}" appears {best} times '
                    f"in the document (including tables).]", "full_text+tables")
    if head == "element":
        if not m.has_elements or m.page_count > 30:    # 需 content_list;长报告图被切碎 -> 弃权
            return None
        _, kind, combined, excluding = query.unit
        ap = _appendix_page(m) if excluding else None
        scope = S_EXCLUDING(ap) if ap is not None else S_WHOLE
        if combined:
            figs, _, _ = COUNT(m, U_ELEMENT(("image", "chart")), scope)
            tbls, _, _ = COUNT(m, U_ELEMENT(("table",)), scope)
            total = figs + tbls
            if total < 1:
                return None
            conf = round(min(1.0, m.page_count / total), 2) if total else 0.0
            return Fact("elements_sum", total, conf,
                        f"[Programmatically counted from the parsed document: {figs} figures + "
                        f"{tbls} tables = {total} in total.]", "content_list")
        cnt, conf, _ = COUNT(m, U_ELEMENT(_ELEM_TYPES.get(kind, ("image",))), scope)
        if cnt < 1:
            return None
        return Fact("elements", cnt, conf,
                    f"[Programmatically counted from the parsed document: {cnt} {kind}.]", "content_list")
    if head == "ref":
        if not m.has_elements:
            return None
        cnt, conf, _ = COUNT(m, U_REF())
        if cnt is None:
            return None
        return Fact("references", cnt, conf,
                    f"[Approximate reference count from the parsed bibliography ≈ {cnt}.]", "content_list")
    if head == "heading":
        if not m.has_elements:
            return None
        hk, unitword = query.unit[1], query.unit[2]
        cnt, conf, _ = COUNT(m, U_HEADING(hk))
        if cnt < 1:
            return None
        return Fact("sections", cnt, conf,
                    f"[Approximate count of top-level {unitword} from parsed headings ≈ {cnt}; "
                    f"section granularity may differ from the reference.]", "content_list")
    return None


def _eval_extract(m: DocModel, query: Extract):
    ref = query.ref
    phys = None
    label = ""
    clean = True
    if ref[0] == "first_sent":
        printed = ref[1]
        phys = m.page_map.to_physical(printed)
        label = m.page_map.frame(printed)
        clean = m.page_map.confident
    elif ref[0] == "second_last":
        phys = m.page_count - 1 if m.page_count >= 2 else m.page_count
        label = f"second-to-last page ({m.page_map.phys_frame(phys)})"
    elif ref[0] == "last":
        phys = m.page_count
        label = f"last page ({m.page_map.phys_frame(phys)})"
    elif ref[0] == "front":
        phys = 1
        label = "front page (physical page 1)"
    elif ref[0] == "printed":
        printed = ref[1]
        phys = m.page_map.to_physical(printed)
        label = m.page_map.frame(printed)
        clean = m.page_map.confident
    if not phys:
        return None
    return EXTRACT(m, phys, label, clean)


def evaluate(m: DocModel, query):
    """⟦Q⟧(D):把类型化查询在文档上确定性执行,得 Fact(value, 自检置信) 或 None(不可算->弃权)。"""
    if isinstance(query, Count):
        return _eval_count(m, query)
    if isinstance(query, Locate):
        return LOCATE(m, query.target)
    if isinstance(query, Extract):
        return _eval_extract(m, query)
    if isinstance(query, Lookup):
        return LOOKUP(m, query.field)
    return None


# =====================================================================================
# Layer 3 —— 确定性可验证门控(gate),两条件:
#   默认门控(唯一一条):仅当该算子 kind 在留出 dev split 上 p_op > p_base(由 calibrate_v3 校准、
#     存 DG_CALIB_FILE)才注入,否则弃权回退基座 → 期望非回归。**无手设阈值、无魔法数字。**
#   实例自检 confidence(pages/words 跨源一致率/mention 集中度/locate 唯一性…)仅用于多命中时仲裁取优。
# 下面的 _THRESHOLD / _LEGACY_* 仅供 DG_LEGACY=true 的 A/B 消融复现旧行为,不在默认路径使用。
# =====================================================================================
_THRESHOLD = {
    "mention": 0.6, "extract_page": 0.6, "title": 0.6, "topwords": 0.6, "pages": 0.6,
    "authors": 0.6, "last_author": 0.6, "date": 0.6,
    "abbrev": 0.5,
    "locate": 0.4,
    "elements": 0.8, "elements_sum": 0.8,
    "references": 0.99, "sections": 0.99,
    "words": 0.5,
    "words_page": 0.99, "footnotes": 0.99, "meta_stats": 0.99,
}

# DG_LEGACY=true:恢复旧(64%)的死常数置信度 + first-over-threshold,用于精确 A/B 复现。
_LEGACY_CONF = {"words": 0.85, "mention": 0.8, "title": 0.7, "extract_page": 0.7,
                "pages": 0.9, "topwords": 0.7, "words_page": 0.45}
_LEGACY_THRESHOLD = dict(_THRESHOLD, words=0.6)


def _load_calib():
    """可选:DG_CALIB_FILE -> {'threshold': {kind: τ}, 'kinds': {kind: {'p_op':_, 'p_base':_}}}。
    threshold 覆盖实例阈值;kinds 给决策论门控(p_op vs p_base)。缺省空 = 固定阈值 + 全启用(当前默认)。
    由 calibrate_v3.py 在 dev split 上产出 -> 论文可写 'calibrated on a held-out dev split'。"""
    path = os.getenv("DG_CALIB_FILE")
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("threshold", {}) or {}, d.get("kinds", {}) or {}
    except Exception:
        return {}, {}


_PBASE_MARGIN = float(os.getenv("DG_PBASE_MARGIN", "0.0") or 0.0)


def _kind_enabled(kind, calib_kinds):
    """决策论门控(kind 粒度):仅当该算子在 dev 上 p_op > p_base + margin 才采纳,否则该类弃权。
    三态:① 无 calib(calib_kinds 空)-> 全开(=观测/校准跑本身,不改变结果);
          ② 有 calib 但该 kind 未被 dev 观测到(缺席)-> 保守弃权(非回归,杜绝裸奔注入);
          ③ 有该 kind 的 p_op/p_base(已 Laplace 平滑)-> p_op > p_base + margin 才采纳。"""
    if not calib_kinds:
        return True
    info = calib_kinds.get(kind)
    if info is None:
        return False
    return float(info.get("p_op", 1.0)) > float(info.get("p_base", 0.0)) + _PBASE_MARGIN


def ground(question: str, pdf_path: str, content_list_path: str = None,
           model: DocModel = None) -> Fact | None:
    """框架总入口:build DocModel ─ parse(q) ─ evaluate(Q,D) ─ gate ─ Fact 或 None(弃权)。
    门控(默认,唯一一条、无手设阈值):仅当该算子 kind 在留出 dev split 上答对率 p_op > 基座 p_base
        (calibrate_v3 校准、存 DG_CALIB_FILE)才注入,否则弃权回退基座 → 期望非回归。
    多命中时取实例自检最高者作仲裁(自检是原理比值,非门控阈值)。
    消融:DG_ABSTAIN=false 关门控(全注入);DG_LEGACY=true 复现旧死常数+手设阈值+first-over。"""
    m = model or build_doc_model(pdf_path, content_list_path)
    if m is None:
        return None
    no_abstain = not _dg_env("DG_ABSTAIN", True)
    legacy = _dg_env("DG_LEGACY", False)
    arbitrate = _dg_env("DG_ARBITRATE", True) and not legacy
    calib_kinds = {} if legacy else _load_calib()[1]

    candidates = []
    for query in parse(question):
        try:
            fact = evaluate(m, query)
        except Exception:
            fact = None
        if not (fact and fact.note):
            continue
        if legacy:
            # 旧消融(DG_LEGACY):死常数置信 + 手设阈值 + first-over,仅供 A/B 复现。
            if fact.kind in _LEGACY_CONF:
                fact = replace(fact, confidence=_LEGACY_CONF[fact.kind])
            gate_ok = fact.confidence >= _LEGACY_THRESHOLD.get(fact.kind, 0.6)
        else:
            # 默认门控(唯一一条、无手设阈值、无魔法数字):仅当该算子 kind 在留出 dev split 上
            # 答对率 p_op > 基座 p_base(由 calibrate_v3 校准、存 DG_CALIB_FILE)才注入,否则弃权。
            gate_ok = _kind_enabled(fact.kind, calib_kinds)
        if no_abstain or gate_ok:
            if not arbitrate:
                return fact
            candidates.append(fact)
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.confidence, reverse=True)   # 取自检最高;稳定排序保留优先级
    return candidates[0]


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
