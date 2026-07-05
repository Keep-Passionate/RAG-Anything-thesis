"""Backbone-agnostic multimodal grounding for document QA.

This module is intentionally independent from RAG-Anything.  It consumes only a
PDF path plus an optional MinerU/Docling-style ``content_list`` JSON file and
returns a small ``MMFact`` that can be appended to any RAG query.

The design mirrors ``dg_core``:

    parse(question) -> choose a multimodal evidence operator
    evaluate(query, MMDocModel) -> MMFact | None
    gate -> abstain when no stable evidence is found

Unlike ``dg_core``, this layer does not claim to deterministically answer visual
questions.  Its job is to locate the right multimodal evidence (table body,
figure/chart image, caption, page) and expose the original image path in a
standard ``Image Path: ...`` line so a host system with a VLM can inspect it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMElement:
    index: int
    kind: str
    page: Optional[int]
    caption: str = ""
    body: str = ""
    footnote: str = ""
    text: str = ""
    image_path: str = ""
    section_path: str = ""
    neighbor_text: str = ""
    labels: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_table(self) -> bool:
        return self.kind == "table"

    @property
    def is_visual(self) -> bool:
        return self.kind in {"image", "chart", "figure"}

    @property
    def search_text(self) -> str:
        return " ".join(
            x
            for x in (
                self.caption,
                self.body,
                self.footnote,
                self.text,
                self.section_path,
                self.neighbor_text,
            )
            if x
        )


@dataclass
class MMDocModel:
    pdf_path: str
    content_list_path: Optional[str] = None
    elements: List[MMElement] = field(default_factory=list)

    @property
    def tables(self) -> List[MMElement]:
        return [e for e in self.elements if e.is_table]

    @property
    def visuals(self) -> List[MMElement]:
        return [e for e in self.elements if e.is_visual]


@dataclass(frozen=True)
class MMFact:
    kind: str
    note: str
    confidence: float
    provenance: str
    requires_vlm: bool = False
    evidence_images: Tuple[str, ...] = field(default_factory=tuple)
    evidence_count: int = 0


# ---------------------------------------------------------------------------
# Content-list loading
# ---------------------------------------------------------------------------


def locate_content_list(pdf_path: str, explicit_path: str | None = None) -> Optional[Path]:
    """Locate a parser content-list file without depending on RAG-Anything.

    Lookup order:
      1. explicit_path
      2. sibling files beside the PDF
      3. PARSE_OUTPUT_DIR / OUTPUT_DIR / ./output recursively
    """

    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    pdf = Path(pdf_path)
    stem = pdf.stem
    sibling_candidates = [
        pdf.with_name(f"{stem}_content_list.json"),
        pdf.parent / "auto" / f"{stem}_content_list.json",
    ]
    for cand in sibling_candidates:
        if cand.exists():
            return cand

    roots = []
    for name in ("PARSE_OUTPUT_DIR", "OUTPUT_DIR"):
        v = os.getenv(name)
        if v:
            roots.append(Path(v))
    roots.append(Path("./output"))

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for cand in (
            root / stem / "auto" / f"{stem}_content_list.json",
            root / stem / f"{stem}_content_list.json",
        ):
            if cand.exists():
                return cand
        hits = sorted(root.glob(f"**/{stem}_content_list.json"))
        if hits:
            return hits[0]
    return None


def build_mm_doc_model(
    pdf_path: str, content_list_path: str | None = None
) -> MMDocModel:
    cl = locate_content_list(pdf_path, content_list_path)
    if not cl:
        return MMDocModel(pdf_path=pdf_path, content_list_path=None, elements=[])
    try:
        with open(cl, encoding="utf-8") as f:
            raw_items = json.load(f)
    except Exception:
        return MMDocModel(pdf_path=pdf_path, content_list_path=str(cl), elements=[])

    base_dir = cl.parent
    elements = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        elem = _element_from_item(idx, item, base_dir)
        if elem is not None:
            elements.append(elem)
    return MMDocModel(pdf_path=pdf_path, content_list_path=str(cl), elements=elements)


def _element_from_item(index: int, item: dict, base_dir: Path) -> Optional[MMElement]:
    raw_type = str(item.get("type", "") or "").lower()
    if raw_type not in {"image", "table", "chart", "figure"}:
        return None

    if raw_type == "table":
        kind = "table"
    elif raw_type == "chart":
        kind = "chart"
    else:
        kind = "image"

    caption = _first_text(
        item,
        "table_caption",
        "image_caption",
        "img_caption",
        "caption",
        "title",
    )
    body = _table_body(item)
    footnote = _first_text(item, "table_footnote", "image_footnote", "img_footnote")
    text = _first_text(item, "text")
    image_path = _resolve_image_path(
        _first_text(item, "img_path", "table_img_path", "image_path"), base_dir
    )
    page_idx = item.get("page_idx")
    page = page_idx + 1 if isinstance(page_idx, int) else None

    labels = _extract_labels(" ".join([caption, text]))
    section_path = _stringify(item.get("_section_path", ""))
    neighbor_text = _stringify(item.get("_neighbor_text", ""))

    return MMElement(
        index=index,
        kind=kind,
        page=page,
        caption=caption,
        body=body,
        footnote=footnote,
        text=text,
        image_path=image_path,
        section_path=section_path,
        neighbor_text=neighbor_text,
        labels=tuple(sorted(labels)),
    )


def _first_text(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        text = _stringify(value)
        if text:
            return text
    return ""


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(row, (list, tuple)) for row in value):
            return "\n".join(" | ".join(_stringify(cell) for cell in row) for row in value)
        return " ".join(_stringify(v) for v in value if _stringify(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value).split())


def _table_body(item: dict) -> str:
    for key in ("table_body", "table_data", "body", "data", "text"):
        body = _stringify(item.get(key))
        if body:
            return body
    return ""


def _resolve_image_path(path_text: str, base_dir: Path) -> str:
    if not path_text:
        return ""
    p = Path(path_text)
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve()) if p.exists() else str(p)


# ---------------------------------------------------------------------------
# Parsing and routing
# ---------------------------------------------------------------------------


_LABEL_RE = re.compile(
    r"\b(?P<head>fig(?:ure)?s?|tables?|tabs?|charts?|plots?)\.?\s*"
    r"(?:no\.?|number|#)?\s*(?P<label>[A-Za-z]?\d+(?:[.\-]\d+)?[A-Za-z]?)\b",
    re.IGNORECASE,
)

_TABLE_WORDS = {
    "table",
    "row",
    "column",
    "cell",
    "cells",
    "header",
    "value",
    "values",
    "score",
    "accuracy",
    "average",
    "mean",
    "maximum",
    "minimum",
    "highest",
    "lowest",
    "increase",
    "decrease",
    "表",
    "表格",
    "行",
    "列",
}

_VISUAL_WORDS = {
    "figure",
    "fig",
    "image",
    "picture",
    "photo",
    "chart",
    "plot",
    "graph",
    "diagram",
    "panel",
    "axis",
    "legend",
    "curve",
    "bar",
    "map",
    "color",
    "colour",
    "red",
    "yellow",
    "green",
    "blue",
    "black",
    "white",
    "font",
    "visible",
    "shown",
    "depicted",
    "图",
    "图片",
    "图像",
    "图表",
    "颜色",
    "地图",
    "曲线",
}

_PAGE_WORDS = {"page", "pages", "where", "located", "location", "which page", "what page", "第几页"}

_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "which",
    "what",
    "where",
    "when",
    "does",
    "from",
    "with",
    "into",
    "onto",
    "about",
    "according",
    "document",
    "paper",
    "report",
    "figure",
    "fig",
    "table",
    "chart",
    "image",
    "page",
    "pages",
    "shown",
    "show",
    "shows",
    "color",
    "colour",
    "value",
    "values",
    "there",
    "their",
    "have",
    "has",
}


@dataclass(frozen=True)
class MMQuery:
    intent: str
    ref_kind: str = ""
    ref_label: str = ""


def parse(question: str) -> Optional[MMQuery]:
    q = (question or "").strip()
    if not q:
        return None
    ref = _explicit_reference(q)
    ql = q.lower()

    if ref and _has_any(ql, _PAGE_WORDS):
        return MMQuery("locate_element", ref[0], ref[1])
    if ref and ref[0] == "table":
        return MMQuery("table_evidence", ref[0], ref[1])
    if ref:
        return MMQuery("visual_evidence", ref[0], ref[1])

    table_intent = _has_any(ql, _TABLE_WORDS)
    visual_intent = _has_any(ql, _VISUAL_WORDS)

    if table_intent and not _false_table_friend(ql):
        return MMQuery("table_evidence")
    if visual_intent:
        return MMQuery("visual_evidence")
    return None


def _explicit_reference(question: str) -> Optional[Tuple[str, str]]:
    m = _LABEL_RE.search(question or "")
    if not m:
        return None
    head = m.group("head").lower().rstrip("s").rstrip(".")
    if head in {"fig", "figure"}:
        kind = "figure"
    elif head in {"tab", "table"}:
        kind = "table"
    elif head in {"chart", "plot"}:
        kind = "chart"
    else:
        kind = "figure"
    return kind, m.group("label").lower()


def _extract_labels(text: str) -> set[str]:
    labels = set()
    for m in _LABEL_RE.finditer(text or ""):
        head = m.group("head").lower().rstrip("s").rstrip(".")
        label = m.group("label").lower()
        if head in {"fig", "figure"}:
            labels.add(f"figure:{label}")
        elif head in {"tab", "table"}:
            labels.add(f"table:{label}")
        elif head in {"chart", "plot"}:
            labels.add(f"chart:{label}")
    return labels


def _has_any(text: str, words: Iterable[str]) -> bool:
    return any(w in text for w in words)


def _false_table_friend(text: str) -> bool:
    return any(x in text for x in ("table of contents", "table of figures", "目录"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def ground(
    question: str,
    pdf_path: str,
    content_list_path: str | None = None,
    model: MMDocModel | None = None,
) -> Optional[MMFact]:
    q = parse(question)
    if q is None:
        return None
    m = model or build_mm_doc_model(pdf_path, content_list_path)
    if not m.elements:
        return None

    if q.intent == "locate_element":
        return _ground_locate(q, m)
    if q.intent == "table_evidence":
        return _ground_table(q, question, m)
    if q.intent == "visual_evidence":
        return _ground_visual(q, question, m)
    return None


def _ground_locate(query: MMQuery, model: MMDocModel) -> Optional[MMFact]:
    elem = _find_by_label(model, query.ref_kind, query.ref_label)
    if elem is None or elem.page is None:
        return None
    name = _display_name(query.ref_kind, query.ref_label)
    note = (
        f"[Multimodal grounding: {name} is on physical page {elem.page} "
        f"according to the parsed document layout."
    )
    if elem.caption:
        note += f"\nCaption: {elem.caption}"
    note += "]"
    return MMFact(
        kind="mm_locate",
        note=note,
        confidence=0.92,
        provenance="content_list",
        requires_vlm=False,
        evidence_count=1,
    )


def _ground_table(query: MMQuery, question: str, model: MMDocModel) -> Optional[MMFact]:
    elems: List[MMElement]
    confidence = 0.65
    if query.ref_label:
        elem = _find_by_label(model, "table", query.ref_label)
        elems = [elem] if elem else []
        confidence = 0.92
    else:
        elems = _rank_elements(question, model.tables, limit=_top_k("MM_TABLE_TOPK", 2))
        if elems:
            confidence = min(0.85, 0.5 + 0.1 * len(elems))

    elems = [e for e in elems if e is not None]
    if not elems:
        return None

    note = _format_evidence_note(
        "Multimodal table grounding",
        elems,
        include_body=True,
        include_image=True,
        max_chars=_max_chars("MM_TABLE_MAX_CHARS", 5000),
    )
    evidence_images = tuple(e.image_path for e in elems if e.image_path)
    return MMFact(
        kind="mm_table",
        note=note,
        confidence=confidence,
        provenance="content_list",
        requires_vlm=bool(evidence_images),
        evidence_images=evidence_images,
        evidence_count=len(elems),
    )


def _ground_visual(query: MMQuery, question: str, model: MMDocModel) -> Optional[MMFact]:
    elems: List[MMElement]
    confidence = 0.6
    if query.ref_label:
        elem = _find_by_label(model, query.ref_kind or "figure", query.ref_label)
        elems = [elem] if elem else []
        confidence = 0.92
    else:
        elems = _rank_elements(question, model.visuals, limit=_top_k("MM_VISUAL_TOPK", 3))
        if not elems and _env_on("MM_BROAD_VISUAL_FALLBACK", True):
            elems = [e for e in model.visuals if e.image_path][:_top_k("MM_VISUAL_TOPK", 3)]
            confidence = 0.35
        elif elems:
            confidence = min(0.82, 0.45 + 0.1 * len(elems))

    elems = [e for e in elems if e is not None and (e.image_path or e.caption)]
    if not elems:
        return None

    note = _format_evidence_note(
        "Multimodal visual grounding",
        elems,
        include_body=False,
        include_image=True,
        max_chars=_max_chars("MM_VISUAL_MAX_CHARS", 4000),
    )
    evidence_images = tuple(e.image_path for e in elems if e.image_path)
    return MMFact(
        kind="mm_visual",
        note=note,
        confidence=confidence,
        provenance="content_list",
        requires_vlm=bool(evidence_images),
        evidence_images=evidence_images,
        evidence_count=len(elems),
    )


def _find_by_label(model: MMDocModel, kind: str, label: str) -> Optional[MMElement]:
    if not label:
        return None
    norm_kind = "figure" if kind in {"fig", "figure", "image"} else kind
    candidates = []
    for e in model.elements:
        labels = set(e.labels)
        if norm_kind == "figure" and f"figure:{label}" in labels:
            candidates.append(e)
        elif norm_kind == "chart" and (f"chart:{label}" in labels or f"figure:{label}" in labels):
            candidates.append(e)
        elif norm_kind == "table" and f"table:{label}" in labels:
            candidates.append(e)
    if candidates:
        return candidates[0]
    return None


def _rank_elements(question: str, elements: Sequence[MMElement], limit: int) -> List[MMElement]:
    if limit <= 0:
        return []
    terms = _query_terms(question)
    scored = []
    for e in elements:
        text = e.search_text.lower()
        if not text:
            continue
        score = 0
        for term in terms:
            if term in text:
                score += 2 if len(term) > 4 else 1
        if e.caption:
            score += 1
        if e.image_path:
            score += 1
        if score > 0:
            scored.append((score, e.index, e))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [e for _, _, e in scored[:limit]]


def _query_terms(question: str) -> List[str]:
    q = (question or "").lower()
    terms = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", q) if t not in _STOPWORDS]
    quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]", question or "")
    for item in quoted:
        item = item.strip().lower()
        if item:
            terms.append(item)
    # Preserve order while deduplicating.
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _format_evidence_note(
    title: str,
    elements: Sequence[MMElement],
    include_body: bool,
    include_image: bool,
    max_chars: int,
) -> str:
    parts = [
        f"[{title}: the following parsed document elements are likely relevant. "
        "Use this evidence only when it directly answers the question."
    ]
    used = 0
    for i, elem in enumerate(elements, 1):
        block = [f"Evidence {i}: {elem.kind} element"]
        if elem.page is not None:
            block[0] += f" on physical page {elem.page}"
        block[0] += "."
        if include_image and elem.image_path:
            block.append(f"Image Path: {elem.image_path}")
        if elem.caption:
            block.append(f"Caption: {elem.caption}")
        if elem.footnote:
            block.append(f"Footnote: {elem.footnote}")
        if include_body and elem.body:
            block.append(f"Table body:\n{elem.body}")
        if elem.neighbor_text:
            block.append(f"Nearby text: {elem.neighbor_text[:800]}")
        text = "\n".join(block)
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining <= 120:
                break
            text = text[:remaining] + "..."
        parts.append(text)
        used += len(text)
        if used >= max_chars:
            break
    parts.append("]")
    return "\n".join(parts)


def _display_name(kind: str, label: str) -> str:
    if kind == "table":
        return f"Table {label}"
    if kind == "chart":
        return f"Chart {label}"
    return f"Figure {label}"


def _env_on(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _top_k(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except Exception:
        return default


def _max_chars(name: str, default: int) -> int:
    try:
        return max(500, int(os.getenv(name, str(default))))
    except Exception:
        return default


def summarize_model(model: MMDocModel) -> str:
    return (
        f"content_list={model.content_list_path or 'missing'}, "
        f"elements={len(model.elements)}, tables={len(model.tables)}, visuals={len(model.visuals)}"
    )


__all__ = [
    "MMDocModel",
    "MMElement",
    "MMFact",
    "MMQuery",
    "build_mm_doc_model",
    "ground",
    "locate_content_list",
    "parse",
    "summarize_model",
]
