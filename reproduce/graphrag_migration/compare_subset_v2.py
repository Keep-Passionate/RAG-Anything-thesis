"""50 篇 meta 子集多方对比:paperbase / ours / dgcore(旧 64%) / dgcoreV2(新)。
读已有 eval 结果(零 LLM)。核心看:dgcoreV2 是否【不低于】dgcore(非回归)且【高于】ours。
用法(服务器): /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/compare_subset_v2.py
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
METHODS = ("paperbase", "ours", "dgcore", "dgcoreV2")


def norm(q):
    return " ".join((q or "").split())


allowed, gold = set(), {}
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

acc = collections.defaultdict(lambda: [0, 0])
verdict = {}
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

print("==== 50 篇 meta 子集 · 各方准确率(按 doc_id+题 联合键)====")
for m in METHODS:
    c, t = acc[m]
    print(f"  {m:10}: {c}/{t} = {c / t * 100:.1f}%" if t else f"  {m:10}: (无数据)")


def pair(a_m, b_m):
    keys = [k for (mm, k) in verdict if mm == a_m and (b_m, k) in verdict]
    win = sum(1 for k in keys if verdict[(a_m, k)] == 1 and verdict[(b_m, k)] == 0)
    lose = sum(1 for k in keys if verdict[(a_m, k)] == 0 and verdict[(b_m, k)] == 1)
    return win, lose, len(keys)


print("\n==== 配对对比(dgcoreV2 = 新版)====")
for b in ("dgcore", "ours", "paperbase"):
    w, l, n = pair("dgcoreV2", b)
    tag = "  ← 非回归检查(须 净≥0)" if b == "dgcore" else ""
    print(f"  dgcoreV2 vs {b:10}: 赢 {w} / 输 {l} / 净 {w - l}  (配对 n={n}){tag}")

# 逐道列出 dgcoreV2 输给 dgcore 的题(若有=新版打坏了旧版对的题,需排查)
reg = [k for (mm, k) in verdict if mm == "dgcoreV2"
       and ("dgcore", k) in verdict
       and verdict[("dgcoreV2", k)] == 0 and verdict[("dgcore", k)] == 1]
if reg:
    print(f"\n⚠️ dgcoreV2 打坏了 dgcore 答对的 {len(reg)} 道(需排查):")
    for k in reg:
        print(f"    {k[0]} | {k[1][:80]}")
else:
    print("\n✅ 无回归:dgcoreV2 未打坏任何 dgcore 答对的题。")
