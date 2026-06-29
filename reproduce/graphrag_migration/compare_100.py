"""100 篇 meta 子集对比:paperbase / ours / dgcoreV2cov(V2+覆盖)。读已有 eval(零 LLM)。
含【注入致害检查】:列出 dgcoreV2cov 答错而 paperbase 答对的 meta 题(=潜在被注入带偏,需排查)。
用法: /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/compare_100.py
"""
import collections
import glob
import json
import os

ROOT = "/root/autodl-tmp/DocBench_subset"
EVAL = "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"
IDS = ("0 2 3 4 5 6 9 10 11 15 17 18 19 21 22 23 24 25 26 27 32 52 54 59 69 77 85 86 91 92 "
       "100 102 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 145 "
       "148 150 154 155 157 158 159 160 161 162 163 164 165 166 167 169 170 171 172 173 174 "
       "175 176 177 178 179 191 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 "
       "222 223 224 225 226 227 228").split()
METHODS = ("paperbase", "ours", "dgcoreV2cov")


def norm(q):
    return " ".join((q or "").split())


allowed, gold = set(), {}
for did in IDS:
    pdfs = glob.glob(os.path.join(ROOT, did, "*.pdf"))
    if not pdfs:
        continue
    pdf = os.path.basename(pdfs[0])
    allowed.add(pdf)
    qa = os.path.join(ROOT, did, f"{did}_qa.jsonl")
    if not os.path.exists(qa):
        continue
    for line in open(qa, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                o = json.loads(line)
                gold[(pdf, norm(o.get("question")))] = o.get("type", "")
            except Exception:
                pass

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
    acc[r.get("method")][0] += a
    acc[r.get("method")][1] += 1
    verdict[(r.get("method"), key)] = a

print("==== 100 篇 meta 子集 · 准确率 ====")
for m in METHODS:
    c, t = acc[m]
    print(f"  {m:12}: {c}/{t} = {c / t * 100:.1f}%" if t else f"  {m:12}: (无数据)")


def pair(a_m, b_m):
    keys = [k for (mm, k) in verdict if mm == a_m and (b_m, k) in verdict]
    win = sum(1 for k in keys if verdict[(a_m, k)] == 1 and verdict[(b_m, k)] == 0)
    lose = sum(1 for k in keys if verdict[(a_m, k)] == 0 and verdict[(b_m, k)] == 1)
    return win, lose, len(keys)


print("\n==== 配对对比 ====")
for b in ("ours", "paperbase"):
    w, l, n = pair("dgcoreV2cov", b)
    chi = (abs(w - l) - 1) ** 2 / (w + l) if (w + l) else 0
    print(f"  dgcoreV2cov vs {b:10}: 赢 {w} / 输 {l} / 净 {w - l}  (配对 n={n}, McNemar χ²≈{chi:.2f})")

harm = [k for (mm, k) in verdict if mm == "dgcoreV2cov"
        and ("paperbase", k) in verdict
        and verdict[("dgcoreV2cov", k)] == 0 and verdict[("paperbase", k)] == 1]
print(f"\n==== 注入致害检查:dgcoreV2cov 错而 paperbase 对的 meta 题 = {len(harm)} ====")
for k in harm:
    print(f"    {k[0][:26]} | {k[1][:62]}")
print("(这些不一定是注入带偏——多数是'两边都弃权、基座那次蒙对'的噪声;若某题确为我们注入了错值,才需修。)")
