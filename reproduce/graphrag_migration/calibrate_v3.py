"""V3 决策论门控的校准:对每个算子 kind,测它触发时【算子答对率 p_op】vs【基座答对率 p_base】。
决策论判据:仅当 p_op > p_base 才该注入(否则不如让基座自己答)。这把"该不该开口"从直觉换成数据。

- 数据源:eval_4cond(各 method 的 accuracy)+ 各 doc 的 qa_results_<METHOD>.json(dg_used/dg_kind)。
- 全程零 LLM(只读已评结果)。p_base 用 paperbase(裸基座)。
- 防过拟合:DG_SPLIT=hash 时按 doc_id 哈希分 dev/test,dev 上校准、test 上验证(论文口径)。
  DG_SPLIT=none(默认,用于小子集预览)时不分,全量统计。
- 输出:① 终端表(每 kind:n / p_op / p_base / 决策);② DG_CALIB_FILE 用的 calib json(pbase per kind)。
用法(服务器):
  DG_METHOD=dgcoreV2     /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/calibrate_v3.py
  DG_METHOD=dgcoreV2cov DG_SPLIT=hash  ...                            (100/229 跑完后)
"""
import collections
import glob
import hashlib
import json
import os

ROOT = "/root/autodl-tmp/DocBench_subset"
EVAL = "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"
METHOD = os.getenv("DG_METHOD", "dgcoreV2")
SPLIT = os.getenv("DG_SPLIT", "none")          # none | hash
OUT = os.getenv("DG_CALIB_OUT", "/root/autodl-tmp/dgcore_calib.json")
MARGIN = float(os.getenv("DG_PBASE_MARGIN", "0.0") or 0.0)


def norm(q):
    return " ".join((q or "").split())


def split_of(did):
    if SPLIT != "hash":
        return "dev"
    h = int(hashlib.md5(str(did).encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"


gold, fired = {}, {}
for d in os.listdir(ROOT):
    pdfs = glob.glob(os.path.join(ROOT, d, "*.pdf"))
    if not pdfs:
        continue
    pdf = os.path.basename(pdfs[0])
    qa = os.path.join(ROOT, d, f"{d}_qa.jsonl")
    if os.path.exists(qa):
        for line in open(qa, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    o = json.loads(line)
                    gold[(pdf, norm(o.get("question")))] = o.get("type", "")
                except Exception:
                    pass
    rf = os.path.join(ROOT, d, f"qa_results_{METHOD}.json")
    if os.path.exists(rf):
        try:
            for rec in json.load(open(rf, encoding="utf-8")):
                fired[(pdf, norm(rec.get("question")))] = (
                    bool(rec.get("dg_used")), rec.get("dg_kind", "") or "")
        except Exception:
            pass

raw = json.load(open(EVAL, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw
acc = {}
for r in recs:
    try:
        acc[(r.get("method"), r.get("doc_id"), norm(r.get("question")))] = int(r.get("accuracy"))
    except Exception:
        pass

# kind -> split -> [n, op_correct, base_correct]
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
for (pdf, q), ty in gold.items():
    if ty != "meta-data":
        continue
    used, kind = fired.get((pdf, q), (False, ""))
    if not used:
        continue
    a_op = acc.get((METHOD, pdf, q))
    a_base = acc.get(("paperbase", pdf, q))
    if a_op is None or a_base is None:
        continue
    cell = agg[kind][split_of(pdf)]
    cell[0] += 1
    cell[1] += a_op
    cell[2] += a_base

print(f"method={METHOD}  split={SPLIT}  margin={MARGIN}")
hdr = "%-13s %4s %6s %7s %9s" % ("kind", "Ndev", "p_op", "p_base", "decision")
if SPLIT == "hash":
    hdr += "   %4s %6s %7s" % ("Ntst", "p_op_t", "p_base_t")
print(hdr)
calib_kinds = {}
for kind in sorted(agg):
    dv = agg[kind]["dev"]
    pop = dv[1] / dv[0] if dv[0] else 0.0
    pb = dv[2] / dv[0] if dv[0] else 0.0
    dec = "INJECT" if pop > pb + MARGIN else "drop?"
    calib_kinds[kind] = {"p_op": round(pop, 3), "p_base": round(pb, 3)}
    row = "%-13s %4d %6.2f %7.2f %9s" % (kind, dv[0], pop, pb, dec)
    if SPLIT == "hash":
        ts = agg[kind]["test"]
        popt = ts[1] / ts[0] if ts[0] else 0.0
        pbt = ts[2] / ts[0] if ts[0] else 0.0
        row += "   %4d %6.2f %7.2f" % (ts[0], popt, pbt)
    print(row)

json.dump({"threshold": {}, "kinds": calib_kinds}, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nwrote {OUT}  (dg_core 用 DG_CALIB_FILE 读取;kinds 里 p_op<=p_base 的算子会被门控自动弃权)")
print("decision=INJECT 表示该 kind 算子答对率 > 基座(决策论:该注入);drop? 表示不如基座(应弃权或修)。")
print("注:小子集(50)按 kind 拆 dev/test 后样本很小,数字仅作机制演示;论文口径用 229 + DG_SPLIT=hash。")
