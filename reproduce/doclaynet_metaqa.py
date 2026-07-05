"""Build a DocBench-style meta-data QA set from DocLayNet layout annotations.

DocLayNet is a published human-annotated document-layout benchmark, not a QA
benchmark. This script derives simple global-attribute QA items from its COCO
layout annotations while keeping the annotations as gold labels only.

Output layout:
  <out>/<doc_id>/<doc_id>.pdf
  <out>/<doc_id>/<doc_id>_qa.jsonl
  <out>/<doc_id>/meta.json
  <out>/manifest.json

The generated questions intentionally mirror DocBench meta-data questions and
stay within DG-RAG's deterministic Count operators:
  pages, tables, figures/pictures, equations/formulas, section headers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DOCLAYNET_CORE_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)

CLASS_NAMES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

QA_SPECS = [
    {
        "key": "pages",
        "question": "How many pages does this document have?",
        "operator": "Count",
        "gold_source": "DocLayNet page metadata",
    },
    {
        "key": "Table",
        "question": "How many tables are in this document?",
        "operator": "Count",
        "gold_source": "DocLayNet human layout annotation",
    },
    {
        "key": "Picture",
        "question": "How many figures are in this document?",
        "operator": "Count",
        "gold_source": "DocLayNet human layout annotation",
    },
    {
        "key": "Formula",
        "question": "How many equations are in this document?",
        "operator": "Count",
        "gold_source": "DocLayNet human layout annotation",
    },
    {
        "key": "Section-header",
        "question": "How many section headers are in this document?",
        "operator": "Count",
        "gold_source": "DocLayNet human layout annotation",
    },
]


def _safe_name(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "doc")[:max_len]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing archive: {dest}", flush=True)
        return
    print(f"Downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "DG-RAG DocLayNet-MetaQA builder"})
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        seen = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if total:
                pct = seen / total * 100
                print(f"  {seen / (1024 ** 2):.1f} MiB / {total / (1024 ** 2):.1f} MiB ({pct:.1f}%)", flush=True)


def _extract_core(archive: Path, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / ".doclaynet_core_extracted"
    if marker.exists():
        return data_dir
    print(f"Extracting {archive} -> {data_dir}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return data_dir


def _find_core_root(data_dir: Path) -> Path:
    for p in [data_dir, *data_dir.glob("*")]:
        if (p / "COCO").is_dir() and (p / "PNG").is_dir():
            return p
    raise FileNotFoundError(f"Could not find COCO/ and PNG/ under {data_dir}")


def _load_coco(core_root: Path, split: str):
    path = core_root / "COCO" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        by_image[ann["image_id"]].append(ann)
    return data.get("images", []), by_image


def _index_members(names) -> dict[str, str]:
    """Map useful suffixes (COCO/val.json, PNG/foo.png) to archive members."""
    idx = {}
    for name in names:
        clean = name.replace("\\", "/")
        parts = clean.split("/")
        if len(parts) >= 2:
            idx["/".join(parts[-2:])] = name
        idx[parts[-1]] = name
    return idx


def _load_coco_remote(url: str, split: str, timeout: float):
    try:
        from remotezip import RemoteZip
    except Exception as exc:
        raise RuntimeError(
            "Selective download requires remotezip: pip install remotezip"
        ) from exc

    kwargs = {"timeout": timeout} if timeout and timeout > 0 else {}
    print(f"Opening remote zip index: {url}", flush=True)
    rz = RemoteZip(url, **kwargs)
    member_index = _index_members(rz.namelist())
    key = f"COCO/{split}.json"
    member = member_index.get(key)
    if member is None:
        rz.close()
        raise FileNotFoundError(f"Could not find {key} in remote zip")
    print(f"Reading {member} from remote zip", flush=True)
    with rz.open(member) as f:
        data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        by_image[ann["image_id"]].append(ann)
    return rz, member_index, data.get("images", []), by_image


def _class_name(category_id: int) -> str | None:
    # DocLayNet COCO uses 1-based category ids in the raw annotations.
    idx = category_id - 1
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return None


def _group_documents(images, by_image):
    docs = defaultdict(list)
    for img in images:
        doc_name = img.get("doc_name") or img.get("file_name") or str(img.get("id"))
        counts = Counter()
        for ann in by_image.get(img["id"], []):
            cls = _class_name(int(ann.get("category_id", -1)))
            if cls:
                counts[cls] += 1
        item = {
            "image_id": img["id"],
            "file_name": img["file_name"],
            "page_no": int(img.get("page_no", 0)),
            "width": img.get("width"),
            "height": img.get("height"),
            "doc_category": img.get("doc_category", ""),
            "collection": img.get("collection", ""),
            "counts": dict(counts),
        }
        docs[doc_name].append(item)
    for pages in docs.values():
        pages.sort(key=lambda x: (x["page_no"], x["file_name"]))
    return docs


def _doc_counts(pages) -> Counter:
    counts = Counter()
    for page in pages:
        counts.update(page["counts"])
    counts["pages"] = len(pages)
    return counts


def _select_docs(docs, args):
    candidates = []
    required = [x.strip() for x in args.require_any.split(",") if x.strip()]
    for doc_name, pages in docs.items():
        counts = _doc_counts(pages)
        if len(pages) < args.min_pages or len(pages) > args.max_pages:
            continue
        if required and not any(counts.get(k, 0) > 0 for k in required):
            continue
        candidates.append((doc_name, pages, counts))
    candidates.sort(key=lambda x: (x[0], len(x[1])))
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    limit = args.candidate_docs or args.limit_docs
    return candidates[:limit] if limit else candidates


def _write_pdf(core_root: Path, pages, out_pdf: Path) -> bool:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to create PDFs: pip install pillow") from exc

    imgs = []
    for page in pages:
        p = core_root / "PNG" / page["file_name"]
        if not p.exists():
            print(f"Missing PNG: {p}", file=sys.stderr)
            return False
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        imgs.append(im)
    if not imgs:
        return False
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    first, rest = imgs[0], imgs[1:]
    first.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=rest)
    for im in imgs:
        im.close()
    return True


def _read_remote_member(remote_zip, member: str, retries: int, retry_sleep: float) -> bytes:
    last = None
    for attempt in range(1, retries + 1):
        try:
            with remote_zip.open(member) as f:
                return f.read()
        except Exception as exc:
            last = exc
            print(
                f"Remote read failed ({attempt}/{retries}) for {member}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise last


def _write_pdf_remote(
    remote_zip,
    member_index: dict[str, str],
    pages,
    out_pdf: Path,
    retries: int,
    retry_sleep: float,
) -> bool:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to create PDFs: pip install pillow") from exc

    imgs = []
    for page in pages:
        key = f"PNG/{page['file_name']}"
        member = member_index.get(key) or member_index.get(page["file_name"])
        if member is None:
            print(f"Missing PNG member in remote zip: {key}", file=sys.stderr)
            return False
        try:
            data = _read_remote_member(remote_zip, member, retries, retry_sleep)
        except Exception as exc:
            print(
                f"Giving up on document because {member} could not be read: {exc}",
                file=sys.stderr,
                flush=True,
            )
            for im in imgs:
                im.close()
            return False
        im = Image.open(io.BytesIO(data))
        if im.mode != "RGB":
            im = im.convert("RGB")
        imgs.append(im)
    if not imgs:
        return False
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    first, rest = imgs[0], imgs[1:]
    first.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=rest)
    for im in imgs:
        im.close()
    return True


def _qa_records(counts: Counter, include_zero: bool):
    records = []
    for spec in QA_SPECS:
        value = int(counts.get(spec["key"], 0))
        if value == 0 and not include_zero and spec["key"] != "pages":
            continue
        records.append(
            {
                "question": spec["question"],
                "answer": str(value),
                "type": "meta-data",
                "operator": spec["operator"],
                "gold_source": spec["gold_source"],
                "doclaynet_label": spec["key"],
            }
        )
    return records


def build(args) -> None:
    work_dir = Path(args.work_dir)
    out_root = Path(args.out)
    archive = Path(args.archive) if args.archive else work_dir / "DocLayNet_core.zip"
    remote_zip = None
    member_index = {}

    if args.selective_download:
        remote_zip, member_index, images, by_image = _load_coco_remote(
            args.url,
            args.split,
            args.remote_timeout,
        )
        print("Selective mode: no full DocLayNet zip will be downloaded.", flush=True)
    elif args.core_dir:
        core_root = _find_core_root(Path(args.core_dir))
        print(f"Core root: {core_root}", flush=True)
        images, by_image = _load_coco(core_root, args.split)
    else:
        _download(args.url, archive)
        core_root = _find_core_root(_extract_core(archive, work_dir / "DocLayNet_core"))
        print(f"Core root: {core_root}", flush=True)
        images, by_image = _load_coco(core_root, args.split)

    docs = _group_documents(images, by_image)
    selected = _select_docs(docs, args)
    print(f"Split={args.split}; documents={len(docs)}; selected={len(selected)}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    total_q = 0
    skipped_pdf = 0

    for idx, (doc_name, pages, counts) in enumerate(selected, 1):
        if args.limit_docs and len(manifest) >= args.limit_docs:
            break
        digest = hashlib.sha1(doc_name.encode("utf-8")).hexdigest()[:8]
        doc_id = f"dl_{args.split}_{idx:04d}_{digest}"
        doc_dir = out_root / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = doc_dir / f"{doc_id}.pdf"
        if not args.no_pdf and not pdf_path.exists():
            if args.selective_download:
                ok = _write_pdf_remote(
                    remote_zip,
                    member_index,
                    pages,
                    pdf_path,
                    retries=args.remote_retries,
                    retry_sleep=args.remote_retry_sleep,
                )
            else:
                ok = _write_pdf(core_root, pages, pdf_path)
            if not ok:
                skipped_pdf += 1
                continue

        records = _qa_records(counts, include_zero=args.include_zero)
        qa_path = doc_dir / f"{doc_id}_qa.jsonl"
        with qa_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        page_files = [p["file_name"] for p in pages]
        meta = {
            "doc_id": doc_id,
            "source": "DocLayNet",
            "source_url": args.url,
            "split": args.split,
            "doc_name": doc_name,
            "doc_category": pages[0].get("doc_category", "") if pages else "",
            "collection": pages[0].get("collection", "") if pages else "",
            "page_count": len(pages),
            "layout_counts": {k: int(v) for k, v in sorted(counts.items())},
            "page_files": page_files,
            "n_questions": len(records),
            "pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else "",
        }
        (doc_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(meta)
        total_q += len(records)

        print(
            f"  wrote {len(manifest)}/{args.limit_docs or len(selected)} docs "
            f"(candidate {idx}/{len(selected)}), questions={total_q}",
            flush=True,
        )

    summary = {
        "name": "DocLayNet-MetaQA",
        "description": (
            "DocBench-style meta-data QA derived from DocLayNet human layout annotations. "
            "DocLayNet annotations are used as gold labels only."
        ),
        "source": "DocLayNet",
        "split": args.split,
        "documents": len(manifest),
        "questions": total_q,
        "include_zero": args.include_zero,
        "qa_specs": QA_SPECS,
        "skipped_pdf": skipped_pdf,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "dataset_card.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if remote_zip is not None:
        remote_zip.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Build DocLayNet-derived DocBench-style meta QA.")
    ap.add_argument("--out", default="/root/autodl-tmp/DocLayNet_MetaQA")
    ap.add_argument("--work-dir", default="/root/autodl-tmp/doclaynet_work")
    ap.add_argument("--core-dir", default="", help="Existing extracted DocLayNet core directory.")
    ap.add_argument("--archive", default="", help="Existing or target DocLayNet_core.zip path.")
    ap.add_argument("--url", default=DOCLAYNET_CORE_URL)
    ap.add_argument(
        "--selective-download",
        action="store_true",
        help="Use HTTP range requests to read only COCO/<split>.json and selected PNG pages from the remote zip.",
    )
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit-docs", type=int, default=80)
    ap.add_argument(
        "--candidate-docs",
        type=int,
        default=0,
        help="Number of candidate documents to try before stopping at --limit-docs; useful if remote reads fail.",
    )
    ap.add_argument("--min-pages", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--remote-retries", type=int, default=5)
    ap.add_argument("--remote-retry-sleep", type=float, default=3.0)
    ap.add_argument(
        "--remote-timeout",
        type=float,
        default=30.0,
        help="Per HTTP range request timeout in seconds for selective download. Use 0 to disable.",
    )
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument(
        "--require-any",
        default="Table,Picture,Formula,Section-header",
        help="Comma-separated labels; keep docs with at least one of them. Empty disables.",
    )
    ap.add_argument("--include-zero", action="store_true", help="Include zero-count element questions.")
    ap.add_argument("--no-pdf", action="store_true", help="Only write QA/meta files, not page-image PDFs.")
    return ap.parse_args()


if __name__ == "__main__":
    build(parse_args())
