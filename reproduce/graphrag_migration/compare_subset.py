"""50 篇 meta 子集三方对比:paperbase vs ours vs dgcore(读已有 eval 结果,零 LLM)。
用法(服务器): /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/compare_subset.py
"""
import collections
import json
import os

ROOT = "/root/autodl-tmp/DocBench_subset"
EVAL = "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"
IDS = set("0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 "
          "114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 "
          "162 163 164 165 166 167".split())


def norm(q):
    return " ".join((q or "").split())


# 金标题型
gold = {}
for did in IDS:
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
        gold[norm(o.get("question"))] = o.get("type", "")

raw = json.load(open(EVAL, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw

acc = collections.defaultdict(lambda: [0, 0])   # method -> [correct, total]
verdict = {}                                     # (method, q) -> 0/1
for r in recs:
    qn = norm(r.get("question"))
    if gold.get(qn) != "meta-data":
        continue
    try:
        a = int(r.get("accuracy"))
    except Exception:
        continue
    m = r.get("method")
    acc[m][0] += a
    acc[m][1] += 1
    verdict[(m, qn)] = a

print("==== 50 篇 meta 子集 · 三方准确率 ====")
for m in ("paperbase", "ours", "dgcore"):
    c, t = acc[m]
    print(f"  {m:10}: {c}/{t} = {c / t * 100:.1f}%" if t else f"  {m:10}: (无数据)")

metaqs = [q for q, ty in gold.items() if ty == "meta-data"
          and ("dgcore", q) in verdict and ("ours", q) in verdict]
win = sum(1 for q in metaqs if verdict[("dgcore", q)] == 1 and verdict[("ours", q)] == 0)
lose = sum(1 for q in metaqs if verdict[("dgcore", q)] == 0 and verdict[("ours", q)] == 1)
print(f"\n  dgcore vs ours: 赢 {win} / 输 {lose} / 净 {win - lose}  (配对 n={len(metaqs)})")
metaqp = [q for q, ty in gold.items() if ty == "meta-data"
          and ("dgcore", q) in verdict and ("paperbase", q) in verdict]
wp = sum(1 for q in metaqp if verdict[("dgcore", q)] == 1 and verdict[("paperbase", q)] == 0)
lp = sum(1 for q in metaqp if verdict[("dgcore", q)] == 0 and verdict[("paperbase", q)] == 1)
print(f"  dgcore vs paperbase: 赢 {wp} / 输 {lp} / 净 {wp - lp}  (配对 n={len(metaqp)})")
