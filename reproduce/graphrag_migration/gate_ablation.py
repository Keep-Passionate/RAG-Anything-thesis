"""L3 门控消融 —— 零 LLM、零 query 预算。

目的:在不花钱、不面向测试集调优的前提下,从数据上选定 L3 门控(单门控 vs 两条件、是否 Laplace)。

原理(论文口径,确定性 mux):
  门控本质 = 在【算子答案】与【基座答案】之间做一次确定性选择。两者的逐题评测都已存在
  (method=dgcoreV2cov 是"全开/无门控"的算子原始答案;method=paperbase 是基座答案)。因此对【任意】
  候选门控策略,都能零 LLM 重算准确率:
    1) 对每道 meta 题确定性重跑 parse()+evaluate()(不调 LLM)拿候选 Fact(kind, 自检 conf);
    2) 按策略门控 -> 选定 kind(或弃权);
    3) 若选定 kind == all-fire 实际注入的 kind -> 用该题的算子准确率;弃权 -> 用基座准确率;
       若会选到与 all-fire 不同的 kind -> 标 ambiguous(无现成答案,A-vs-B 比较时应≈0)。

校准(每 kind 的 p_op/p_base)取自 all-fire 跑自身;DG_SPLIT=hash 时在 dev 半拟合、test 半报告(论文口径)。
Laplace(+1) 平滑:p=(correct+alpha)/(n+2*alpha),让小样本(n=1)不能凭一条幸运样本被采纳。

服务器用法:
  PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists DG_SPLIT=hash \
    /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/gate_ablation.py
"""
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
sys.path.insert(0, REPRO)
import dg_core  # noqa: E402

ROOT = os.getenv("DG_DATA", "/root/autodl-tmp/DocBench_subset")
EVAL = os.getenv("DG_EVAL", "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json")
OPM = os.getenv("DG_METHOD", "dgcoreV2cov")     # all-fire(无门控)算子原始答案
BASEM = os.getenv("DG_BASE", "paperbase")        # 基座答案
SPLIT = os.getenv("DG_SPLIT", "none")            # none | hash
ALPHA = float(os.getenv("DG_LAPLACE", "1") or 0)
MARGIN = float(os.getenv("DG_PBASE_MARGIN", "0") or 0)


def norm(q):
    return " ".join((q or "").split())


def split_of(did):
    if SPLIT != "hash":
        return "dev"
    return "dev" if int(hashlib.md5(str(did).encode()).hexdigest(), 16) % 2 == 0 else "test"


# ---- gold meta + pdf 路径(只取算子已跑过的文档) ----
gold = {}        # (pdf, normq) -> 原始 question
pdfpath = {}     # pdf basename -> 绝对路径
for d in sorted(os.listdir(ROOT)):
    pdfs = glob.glob(os.path.join(ROOT, d, "*.pdf"))
    if not pdfs:
        continue
    if not os.path.exists(os.path.join(ROOT, d, f"qa_results_{OPM}.json")):
        continue
    pdf = os.path.basename(pdfs[0])
    pdfpath[pdf] = pdfs[0]
    qa = os.path.join(ROOT, d, f"{d}_qa.jsonl")
    if os.path.exists(qa):
        for ln in open(qa, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
                if o.get("type") == "meta-data":
                    gold[(pdf, norm(o.get("question")))] = o.get("question")
            except Exception:
                pass

# ---- 逐题评测准确率 ----
raw = json.load(open(EVAL, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw
acc = {}
for r in recs:
    try:
        acc[(r.get("method"), r.get("doc_id"), norm(r.get("question")))] = int(r.get("accuracy"))
    except Exception:
        pass

# ---- all-fire 实际注入的 kind(算子原始答案对应哪个算子)----
opfire = {}      # (pdf, normq) -> (used, kind)
for d in sorted(os.listdir(ROOT)):
    rf = os.path.join(ROOT, d, f"qa_results_{OPM}.json")
    pdfs = glob.glob(os.path.join(ROOT, d, "*.pdf"))
    if not os.path.exists(rf) or not pdfs:
        continue
    pdf = os.path.basename(pdfs[0])
    try:
        for rec in json.load(open(rf, encoding="utf-8")):
            opfire[(pdf, norm(rec.get("question")))] = (bool(rec.get("dg_used")), rec.get("dg_kind") or "")
    except Exception:
        pass

# ---- 校准:从 all-fire 跑自身统计每 kind 的 [n, op_correct, base_correct](仅 dev) ----
agg = collections.defaultdict(lambda: [0, 0, 0])
for (pdf, nq) in gold:
    used, kind = opfire.get((pdf, nq), (False, ""))
    if not used or not kind or split_of(pdf) != "dev":
        continue
    ao = acc.get((OPM, pdf, nq))
    ab = acc.get((BASEM, pdf, nq))
    if ao is None or ab is None:
        continue
    c = agg[kind]
    c[0] += 1
    c[1] += ao
    c[2] += ab


def make_calib(alpha):
    out = {}
    for k, (n, oc, bc) in agg.items():
        denom = n + 2 * alpha
        pop = (oc + alpha) / denom if denom > 0 else 0.0
        pb = (bc + alpha) / denom if denom > 0 else 0.0
        out[k] = {"n": n, "p_op": round(pop, 3), "p_base": round(pb, 3)}
    return out


calib_raw = make_calib(0.0)
calib_lap = make_calib(ALPHA)


def kind_ok(kind, cal):
    info = cal.get(kind)
    if info is None:                      # dev 未观测到该 kind 触发 -> 保守弃权(非回归默认)
        return False
    return info["p_op"] > info["p_base"] + MARGIN


TAU = dg_core._THRESHOLD

# ---- 候选 Fact(确定性,不调 LLM);模型/事实缓存,避免重复建模 ----
modelcache = {}
factcache = {}


def facts_for(pdf, q):
    key = (pdf, norm(q))
    if key in factcache:
        return factcache[key]
    if pdf not in modelcache:
        modelcache[pdf] = dg_core.build_doc_model(pdfpath[pdf])
    m = modelcache[pdf]
    out = []
    if m is not None:
        for query in dg_core.parse(q):
            try:
                f = dg_core.evaluate(m, query)
            except Exception:
                f = None
            if f and f.note:
                out.append(f)
    factcache[key] = out
    return out


def choose(facts, keep):
    kept = [f for f in facts if keep(f)]
    if not kept:
        return None
    kept.sort(key=lambda f: f.confidence, reverse=True)   # 仲裁:自检最高者
    return kept[0]


POLICIES = {
    "all_fire":          lambda fs: choose(fs, lambda f: True),
    "self_check(tau)":   lambda fs: choose(fs, lambda f: f.confidence >= TAU.get(f.kind, 0.6)),
    "single(raw)":       lambda fs: choose(fs, lambda f: kind_ok(f.kind, calib_raw)),
    "single(laplace)":   lambda fs: choose(fs, lambda f: kind_ok(f.kind, calib_lap)),
    "two_cond(raw)":     lambda fs: choose(fs, lambda f: f.confidence >= TAU.get(f.kind, 0.6) and kind_ok(f.kind, calib_raw)),
    "two_cond(laplace)": lambda fs: choose(fs, lambda f: f.confidence >= TAU.get(f.kind, 0.6) and kind_ok(f.kind, calib_lap)),
}

# ---- 评分(零 LLM mux) ----
scored = [k for k in gold if acc.get((OPM,) + k) is not None and acc.get((BASEM,) + k) is not None]
results = {}
for name, pol in POLICIES.items():
    res = {"all": [0, 0], "test": [0, 0], "fire": collections.Counter(), "ambiguous": 0, "harm": []}
    for (pdf, nq) in scored:
        ao = acc[(OPM, pdf, nq)]
        ab = acc[(BASEM, pdf, nq)]
        used, opkind = opfire.get((pdf, nq), (False, ""))
        chosen = pol(facts_for(pdf, gold[(pdf, nq)]))
        if chosen is None:
            a = ab                                   # 弃权 -> 基座
        elif used and chosen.kind == opkind:
            a = ao                                   # 注入同一 kind -> 算子原始答案
            res["fire"][chosen.kind] += 1
        else:
            res["ambiguous"] += 1                    # 选到与 all-fire 不同的 kind:无现成答案
            a = ao if used else ab
            res["fire"][chosen.kind] += 1
        if chosen is not None and a == 0 and ab == 1:
            res["harm"].append((pdf, gold[(pdf, nq)], chosen.kind))
        sc = split_of(pdf)
        res["all"][0] += a
        res["all"][1] += 1
        if sc == "test":
            res["test"][0] += a
            res["test"][1] += 1
    results[name] = res

# ---- 打印 ----
print(f"data={ROOT}  op={OPM}  base={BASEM}  split={SPLIT}  laplace_alpha={ALPHA}  margin={MARGIN}")
print(f"meta 题数(可评分): {len(scored)}   文档数: {len(pdfpath)}")

print("\n-- 基线方法准确率(同一批 meta 题,零门控对照) --")
for bm in ("paperbase", "ours", OPM):
    ca = ta = ct = tt = 0
    for (pdf, nq) in scored:
        a = acc.get((bm, pdf, nq))
        if a is None:
            continue
        ca += a
        ta += 1
        if split_of(pdf) == "test":
            ct += a
            tt += 1
    line = "  %-14s acc(all)=%d/%d=%.1f%%" % (bm, ca, ta, 100 * ca / ta if ta else 0)
    if SPLIT == "hash":
        line += "   acc(test)=%d/%d=%.1f%%" % (ct, tt, 100 * ct / tt if tt else 0)
    print(line)

print("\n-- 校准(dev) raw vs laplace --")
print("%-13s %4s %8s %8s   %8s %8s" % ("kind", "n", "p_op", "p_base", "p_op_L", "p_base_L"))
for k in sorted(agg):
    r = calib_raw[k]
    l = calib_lap[k]
    dec_r = "INJECT" if kind_ok(k, calib_raw) else "drop"
    dec_l = "INJECT" if kind_ok(k, calib_lap) else "drop"
    print("%-13s %4d %8.3f %8.3f   %8.3f %8.3f   raw=%-6s lap=%-6s"
          % (k, r["n"], r["p_op"], r["p_base"], l["p_op"], l["p_base"], dec_r, dec_l))

print("\n-- 各门控策略准确率(零 LLM mux) --")
hdr = "%-18s %12s %7s %5s %5s" % ("policy", "acc(all)", "acc%", "fires", "harm")
if SPLIT == "hash":
    hdr += " %12s %7s" % ("acc(test)", "test%")
hdr += "  ambig"
print(hdr)
for name, res in results.items():
    c, t = res["all"]
    row = "%-18s %5d/%-4d %6.1f%% %5d %5d" % (name, c, t, 100 * c / t if t else 0, sum(res["fire"].values()), len(res["harm"]))
    if SPLIT == "hash":
        tc, tt = res["test"]
        row += " %5d/%-4d %6.1f%%" % (tc, tt, 100 * tc / tt if tt else 0)
    row += "   %d" % res["ambiguous"]
    print(row)

print("\n(acc(all)=dev+test 混合仅参考;acc(test)=留出测试集=论文口径。harm=门控后仍错而基座对的注入题。)")
print("ambig=策略选的 kind≠all-fire 实选 kind(无现成答案;A-vs-B 比较时应≈0,数大则需真跑核验)。")
print("一致性自检:all_fire 的 fires 分布应≈ qa_results_%s 的 dg_kind 分布。" % OPM)
