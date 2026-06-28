"""L3 阈值校准:测每个 resolver 触发时的真实准确率(开发集 calibration,非训练)。
读 qa_results_dgcore.json(dg_used/dg_kind)+ eval verdict,零 LLM。
用途:① 论文写"thresholds calibrated on dev split"有据;② 哪个 resolver 精度低就调高其阈值/弃权。
用法(服务器): /root/miniconda3/envs/rag/bin/python reproduce/graphrag_migration/calibrate_dgcore.py
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


# pdf 名 + 金标题型 + dgcore 触发情况(dg_kind)
allowed, gold, fired = set(), {}, {}
for did in IDS:
    pdfs = glob.glob(os.path.join(ROOT, did, "*.pdf"))
    if not pdfs:
        continue
    pdfname = os.path.basename(pdfs[0])
    allowed.add(pdfname)
    qa = os.path.join(ROOT, did, f"{did}_qa.jsonl")
    if os.path.exists(qa):
        for line in open(qa, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    o = json.loads(line)
                    gold[(pdfname, norm(o.get("question")))] = o.get("type", "")
                except Exception:
                    pass
    res = os.path.join(ROOT, did, "qa_results_dgcore.json")
    if os.path.exists(res):
        try:
            for rec in json.load(open(res, encoding="utf-8")):
                fired[(pdfname, norm(rec.get("question")))] = (
                    bool(rec.get("dg_used")), rec.get("dg_kind", "") or "")
        except Exception:
            pass

raw = json.load(open(EVAL, encoding="utf-8"))
recs = raw.get("results", raw) if isinstance(raw, dict) else raw
verdict = {}  # method -> {key: 0/1}
for r in recs:
    did = r.get("doc_id")
    if did not in allowed:
        continue
    try:
        a = int(r.get("accuracy"))
    except Exception:
        continue
    verdict.setdefault(r.get("method"), {})[(did, norm(r.get("question")))] = a

dg = verdict.get("dgcore", {})
base = verdict.get("paperbase", {})

# 仅看 meta 题
bykind = collections.defaultdict(lambda: [0, 0])    # dg_kind -> [correct, fired]
abst = [0, 0]                                        # 弃权的 meta:[baseline对, baseline错]
for key, ty in gold.items():
    if ty != "meta-data" or key not in dg:
        continue
    used, kind = fired.get(key, (False, ""))
    if used:
        bykind[kind][0] += dg[key]
        bykind[kind][1] += 1
    else:
        if key in base:
            abst[0 if base[key] == 1 else 1] += 1

print("==== 各 resolver 触发时的准确率(meta 题,dev 校准)====")
print(f"{'resolver(dg_kind)':>20} {'触发数':>6} {'答对':>5} {'准确率':>7}")
for k in sorted(bykind, key=lambda k: -bykind[k][1]):
    c, t = bykind[k]
    print(f"{k:>20} {t:>6} {c:>5} {c / t * 100:>6.0f}%")
fired_total = sum(t for _, t in bykind.values())
fired_correct = sum(c for c, _ in bykind.values())
print(f"\n触发合计: {fired_correct}/{fired_total} = {fired_correct / max(fired_total,1)*100:.0f}%")
print(f"弃权的 meta 题: {sum(abst)} 道(其中 baseline 本就答对 {abst[0]}、答错 {abst[1]})")
print("解读:某 resolver 准确率明显低于 baseline 总体→应调高其阈值(更倾向弃权);"
      "弃权题里 baseline 答错多→说明还有可救空间(可放开对应 resolver)。")
