"""meta-data 错题归因 join（在服务器 rag python 跑，零 LLM 调用——只读已有结果）。

把三份已有数据 join 起来，回答"ours 答错的 meta 题错在哪类(A/B/C)"，并 dump 小 JSON：
  1) 金标题型/答案     : DOCBENCH_ROOT/<id>/<id>_qa.jsonl  (type / answer)
  2) 逐题判分(对/错)   : EVAL_JSON  (method / question / accuracy / generated_answer)
  3) 算子是否触发      : DOCBENCH_ROOT/<id>/qa_results_<cond>.json (doc_meta_used / locate_used)

归因：
  B = 答错且本题路由【没触发】任何算子（ours==baseline，纯漏接）→ 改关键词/加算子
  A/C = 答错且【触发了】算子（A=算子算错值；C=本就非确定性）→ 看 generated/gold 人工判 A vs C
并对【没触发】的题报出 paperbase 是否也错——只有 baseline 本就错的才是真正值得做算子的目标。

用法（AutoDL 默认路径，按需用 env 覆盖）：
  /root/miniconda3/envs/rag/bin/python reproduce/diag_meta_errors.py
env:
  DOCBENCH_ROOT  默认 /root/autodl-tmp/DocBench_subset
  EVAL_JSON      默认 /root/autodl-tmp/eval_4cond/llm_evaluation_results.json
  OURS_METHOD / BASE_METHOD  eval 里的 method 字段名（脚本会先打印所有可用 method，
                             若默认猜错，看打印值后用 env 指定重跑）
  OURS_COND / BASE_COND      qa_results 文件名后缀，默认 ours / paperbase
  OUT_JSON       dump 路径，默认 ./diag_meta_errors_dump.json
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.getenv("DOCBENCH_ROOT", "/root/autodl-tmp/DocBench_subset"))
EVAL_JSON = Path(os.getenv("EVAL_JSON", "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"))
OURS_METHOD = os.getenv("OURS_METHOD", "ours")
BASE_METHOD = os.getenv("BASE_METHOD", "paperbase")
OURS_COND = os.getenv("OURS_COND", "ours")
BASE_COND = os.getenv("BASE_COND", "paperbase")
OUT_JSON = Path(os.getenv("OUT_JSON", "./diag_meta_errors_dump.json"))


def norm(q):
    return " ".join((q or "").strip().split())


def load_gold():
    """norm(question) -> {type, answer, doc_id}"""
    g = {}
    for d in sorted(ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
        qa = d / f"{d.name}_qa.jsonl"
        if not qa.exists():
            continue
        for line in open(qa, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            g[norm(o.get("question"))] = {
                "type": o.get("type", ""), "answer": o.get("answer"), "doc_id": d.name}
    return g


def load_eval():
    """(method, norm(question)) -> record"""
    raw = json.load(open(EVAL_JSON, encoding="utf-8"))
    recs = raw.get("results", raw) if isinstance(raw, dict) else raw
    by = {}
    methods = Counter()
    for r in recs:
        m = r.get("method", "")
        methods[m] += 1
        by[(m, norm(r.get("question")))] = r
    return by, methods


def load_fired(cond):
    """norm(question) -> {doc_meta_used, locate_used} (from per-doc qa_results_<cond>.json)"""
    fired = {}
    for d in sorted(ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
        f = d / f"qa_results_{cond}.json"
        if not f.exists():
            continue
        try:
            for rec in json.load(open(f, encoding="utf-8")):
                fired[norm(rec.get("question"))] = {
                    "doc_meta_used": rec.get("doc_meta_used", False),
                    "locate_used": rec.get("locate_used", False),
                    "answer": rec.get("answer"),
                }
        except Exception:
            continue
    return fired


def acc_of(rec):
    a = rec.get("accuracy") if rec else None
    try:
        return int(a) if a is not None else None
    except Exception:
        return None


def main():
    gold = load_gold()
    ev, methods = load_eval()
    print(f"=== eval 里可用的 method（若 OURS/BASE_METHOD 猜错，用 env 改）===")
    for m, c in methods.most_common():
        print(f"   method={m!r:30} 记录数={c}")
    print(f"   当前用 OURS_METHOD={OURS_METHOD!r}  BASE_METHOD={BASE_METHOD!r}\n")

    fired_ours = load_fired(OURS_COND)
    meta_qs = [q for q, v in gold.items() if v["type"] == "meta-data"]
    print(f"=== meta-data 题 n={len(meta_qs)}；fired_ours 命中 {sum(1 for q in meta_qs if q in fired_ours)} 题 ===\n")

    buckets = Counter()
    dump = {"B_notfired_baseline_wrong": [], "B_notfired_baseline_right": [],
            "A_or_C_fired_wrong": [], "ours_wrong_total": []}
    for q in meta_qs:
        our_rec = ev.get((OURS_METHOD, q))
        base_rec = ev.get((BASE_METHOD, q))
        our_acc = acc_of(our_rec)
        base_acc = acc_of(base_rec)
        if our_acc is None:
            buckets["NO_EVAL_RECORD"] += 1
            continue
        if our_acc == 1:
            buckets["ours_correct"] += 1
            continue
        # ours 答错
        info = gold[q]
        fr = fired_ours.get(q, {})
        is_fired = bool(fr.get("doc_meta_used") or fr.get("locate_used"))
        item = {
            "doc_id": info["doc_id"], "question": q,
            "gold": str(info["answer"])[:160],
            "ours_answer": str((our_rec or {}).get("generated_answer", fr.get("answer", "")))[:160],
            "doc_meta_used": fr.get("doc_meta_used"), "locate_used": fr.get("locate_used"),
            "baseline_acc": base_acc,
        }
        dump["ours_wrong_total"].append(item)
        if is_fired:
            buckets["A_or_C_fired_wrong"] += 1
            dump["A_or_C_fired_wrong"].append(item)
        else:
            if base_acc == 1:
                buckets["B_notfired_baseline_RIGHT(别动)"] += 1
                dump["B_notfired_baseline_right"].append(item)
            else:
                buckets["B_notfired_baseline_WRONG(可救目标)"] += 1
                dump["B_notfired_baseline_wrong"].append(item)

    print("=== ours 答错的 meta 题归因 ===")
    for k, v in buckets.most_common():
        print(f"   {k:>34}: {v}")

    print(f"\n---- B 类·真正可救目标（没触发 且 baseline 也错）{len(dump['B_notfired_baseline_wrong'])} 道 ----")
    for it in dump["B_notfired_baseline_wrong"]:
        print(f"[doc {it['doc_id']}] {it['question'][:95]}")
        print(f"      gold: {it['gold'][:75]}  | ours: {it['ours_answer'][:55]}")

    print(f"\n---- A/C 类·触发了但答错 {len(dump['A_or_C_fired_wrong'])} 道（人工判 A 算错 vs C 非确定）----")
    for it in dump["A_or_C_fired_wrong"]:
        flags = f"meta={it['doc_meta_used']},loc={it['locate_used']}"
        print(f"[doc {it['doc_id']}] {it['question'][:90]}  ({flags})")
        print(f"      gold: {it['gold'][:70]}  | ours: {it['ours_answer'][:55]}")

    OUT_JSON.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已 dump: {OUT_JSON.resolve()}  （把它拉回本地给我，我深挖归因+设计算子）")


if __name__ == "__main__":
    sys.exit(main())
