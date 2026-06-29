"""零-LLM 预检:在 50 篇 meta 子集上对比【旧 dg_core(DG_LEGACY=true,=64%行为)】vs【新 dg_core(默认 V2)】
的注入/弃权决策。ground() 不调用 LLM,纯读 PDF/content_list,故可在花钱跑 query 前廉价验证:
  ① 新代码是否崩;② 新版相对旧版在哪些题改了"注入↔弃权/换算子";③ 有没有"旧注入对、新弃权"的回归风险。
用法(服务器,先 export PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists):
  /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/smoke_dgcore.py
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # 让 dg_core 可 import
import dg_core   # noqa: E402

ROOT = os.getenv("DATA_DIR", "/root/autodl-tmp/DocBench_subset")
IDS = ("0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 "
       "114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 "
       "162 163 164 165 166 167").split()


def run(q, pdf, model, legacy):
    os.environ["DG_LEGACY"] = "true" if legacy else "false"
    try:
        f = dg_core.ground(q, pdf, model=model)
    except Exception as e:   # noqa: BLE001
        return ("CRASH:" + repr(e)[:60], None, None)
    if f is None:
        return ("ABSTAIN", None, None)
    return (f.kind, round(f.confidence, 2), f.value)


stat = {"old": collections.Counter(), "new": collections.Counter()}
flips = []
new_abstains_old_injected = []   # 回归风险:旧注入、新弃权
docs_seen = 0
for did in IDS:
    pdfs = glob.glob(os.path.join(ROOT, did, "*.pdf"))
    if not pdfs:
        continue
    pdf = pdfs[0]
    try:
        m = dg_core.build_doc_model(pdf)
    except Exception as e:   # noqa: BLE001
        print(f"[BUILD FAIL] doc{did}: {e!r}")
        continue
    if m is None:
        print(f"[BUILD None] doc{did}")
        continue
    docs_seen += 1
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
        if o.get("type") != "meta-data":
            continue
        q = o.get("question", "")
        old = run(q, pdf, m, True)
        new = run(q, pdf, m, False)
        stat["old"][old[0] if not old[0].startswith("CRASH") else "CRASH"] += 1
        stat["new"][new[0] if not new[0].startswith("CRASH") else "CRASH"] += 1
        o_used = old[0] != "ABSTAIN" and not old[0].startswith("CRASH")
        n_used = new[0] != "ABSTAIN" and not new[0].startswith("CRASH")
        if (o_used != n_used) or (old[0] != new[0]):
            flips.append((did, q[:72], old, new))
        if o_used and not n_used:
            new_abstains_old_injected.append((did, q[:72], old, new))

print(f"\n==== 预检:{docs_seen} 篇文档 / 旧(64%) vs 新(V2),零 LLM ====")
for cond in ("old", "new"):
    used = sum(v for k, v in stat[cond].items() if k not in ("ABSTAIN", "CRASH"))
    ab = stat[cond]["ABSTAIN"]
    cr = stat[cond]["CRASH"]
    print(f"  {cond}: 注入={used} 弃权={ab} 崩溃={cr}  按算子={dict(stat[cond])}")

print(f"\n==== old≠new 的题:{len(flips)} 道 ====")
for did, q, o, n in flips:
    print(f"  doc{did} [{q}]\n      old={o}  ->  new={n}")

print(f"\n==== ⚠️ 回归风险(旧注入 / 新弃权):{len(new_abstains_old_injected)} 道 ====")
for did, q, o, n in new_abstains_old_injected:
    print(f"  doc{did} [{q}]  old={o}")
print("\n解读:崩溃必须为 0 才能跑 query;'旧注入/新弃权'若命中的是旧版【答对】的题=潜在掉分,"
      "需逐道看是否该弃权(如词数跨源发散=该弃权)。确认无误再启动 LLM 跑题。")
