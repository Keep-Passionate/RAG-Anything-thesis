"""选 GraphRAG 跨基座迁移用的 meta-data 子集（纯本地、零 LLM 调用）。

遍历 DocBench 各文档的 <id>_qa.jsonl，统计每篇 type=="meta-data" 的题数，
按 meta 题数降序挑 TOP_N 篇，导出 manifest（含 PDF 真实文件名）。只读数据。

用法:
  python select_meta_subset.py [TOP_N]
环境变量:
  DOCBENCH_ROOT  DocBench 各 <id>/ 目录的父目录
                 本地默认 D:\\project\\RAG-Anything-main\\data\\DocBench_download
                 服务器请设 /root/autodl-tmp/DocBench_subset
  EXCLUDE_IDS    逗号分隔的 doc_id，从子集中剔除（如已知触发内容审查的敏感文档）

迁移实验只动"骨干"一个变量：子集、增强器都不变，故子集一次选定、两骨干共用。
"""
import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(os.getenv(
    "DOCBENCH_ROOT", r"D:\project\RAG-Anything-main\data\DocBench_download"))
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EXCLUDE = {s.strip() for s in os.getenv("EXCLUDE_IDS", "").split(",") if s.strip()}
OUT = Path(__file__).with_name("meta_subset_manifest.json")

rows = []
total_meta = 0
type_totals = {}
for d in sorted(DATA_ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
    if not d.is_dir():
        continue
    qa = d / f"{d.name}_qa.jsonl"
    if not qa.exists():
        continue
    meta_n = n = 0
    with open(qa, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("type", "")
            type_totals[t] = type_totals.get(t, 0) + 1
            n += 1
            if t == "meta-data":
                meta_n += 1
    total_meta += meta_n
    pdfs = [p.name for p in d.iterdir() if p.suffix.lower() == ".pdf"]
    rows.append({
        "doc_id": d.name,
        "pdf": pdfs[0] if pdfs else None,
        "meta_questions": meta_n,
        "total_questions": n,
    })

with_meta = [r for r in rows if r["meta_questions"] > 0 and r["doc_id"] not in EXCLUDE]
with_meta.sort(key=lambda r: (-r["meta_questions"], int(r["doc_id"])))
subset = with_meta[:TOP_N]

print(f"全库文档: {len(rows)}  含 meta 题的文档: {sum(1 for r in rows if r['meta_questions']>0)}  meta 题总数: {total_meta}")
print(f"剔除 EXCLUDE_IDS: {sorted(EXCLUDE) or '(无)'}")
print(f"按题型: {dict(sorted(type_totals.items(), key=lambda x: -x[1]))}")
print(f"TOP {TOP_N} 子集: {len(subset)} 篇, meta 题合计 {sum(r['meta_questions'] for r in subset)}")
for r in subset:
    print(f"  {r['doc_id']:>4} meta={r['meta_questions']} total={r['total_questions']}  {r['pdf']}")

OUT.write_text(json.dumps({
    "top_n": TOP_N, "exclude_ids": sorted(EXCLUDE),
    "total_docs": len(rows), "total_meta_questions": total_meta,
    "subset_meta_questions": sum(r["meta_questions"] for r in subset),
    "type_totals": type_totals, "subset": subset,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写 manifest: {OUT}")
