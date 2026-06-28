"""50 篇 meta 子集三方对比:paperbase vs ours vs dgcore(读已有 eval 结果,零 LLM)。
用法(服务器): /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/compare_subset.py

注意:eval 记录的 doc_id 是 PDF 文件名;且同一题文本会跨文档重复(如"document title")——
必须用 (doc_id, question) 联合键,并按 50 子集的 PDF 名过滤,否则会被跨文档碰撞污染。
"""
import collections
import glob
import json
import os

ROOT = "/root/autodl-tmp/DocBench_subset"
EVAL = "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"
IDS = ("0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 "
       "114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 "
       "162 163 164 165 166 167").split()


def norm(q):
    return " ".join((q or "").split())


# 允许的 PDF 名集合 + 金标((pdf名, 题) -> 题型)
allowed = set()
gold = {}
for did in IDS:
    pdfs = glob.glob(os.path.join(ROOT, did, "*.pdf"))
    if not pdfs:
        continue
    pdfname = os.path.basename(pdfs[0])
    allowed.add(pdfname)
    qa = os.path.join(ROOT, did, f"{did}_qa.jsonl")
    if not os.path.exists(qa):
        continue
    for line in open(qa, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        gold[(pdfname, norm(o.get("question")))] = o.get("type", "")

raw = json.load(open(EVAL, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw

acc = collections.defaultdict(lambda: [0, 0])   # method -> [correct, total]
verdict = {}                                     # (method, (pdf,q)) -> 0/1
for r in recs:
    did = r.get("doc_id")
    if did not in allowed:
        continue
    key = (did, norm(r.get("question")))
    if gold.get(key) != "meta-data":
        continue
    try:
        a = int(r.get("accuracy"))
    except Exception:
        continue
    m = r.get("method")
    acc[m][0] += a
    acc[m][1] += 1
    verdict[(m, key)] = a

print("==== 50 篇 meta 子集 · 三方准确率(按 doc_id+题 联合键)====")
for m in ("paperbase", "ours", "dgcore"):
    c, t = acc[m]
    print(f"  {m:10}: {c}/{t} = {c / t * 100:.1f}%" if t else f"  {m:10}: (无数据)")


def pair(a_m, b_m):
    keys = [k for (mm, k) in verdict if mm == a_m and (b_m, k) in verdict]
    win = sum(1 for k in keys if verdict[(a_m, k)] == 1 and verdict[(b_m, k)] == 0)
    lose = sum(1 for k in keys if verdict[(a_m, k)] == 0 and verdict[(b_m, k)] == 1)
    return win, lose, len(keys)


for b in ("ours", "paperbase"):
    w, l, n = pair("dgcore", b)
    print(f"  dgcore vs {b}: 赢 {w} / 输 {l} / 净 {w - l}  (配对 n={n})")
