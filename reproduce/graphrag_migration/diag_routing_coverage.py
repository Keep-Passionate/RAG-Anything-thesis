"""路由覆盖诊断（纯本地、零 LLM、零 PDF 读取）。

复刻 query.py 的关键词路由决策（op 层关 = 论文口径），跑在全部 DocBench meta-data
题 + 全部 location 题上，回答："我们的路由对哪些题根本不触发"——这是错题归因里
的 B 类（路由漏判）下界，本地即可量化，不必等服务器逐题判分。

注意：这里只判"是否触发/触发哪个算子"（取决于纯检测函数），不判"触发后算得对不对"
（A 类，需 PDF/content_list/逐题判分，见服务器诊断脚本）。
"""
import json
import os
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO))

from doc_meta import (  # noqa: E402
    detect_meta_intent, detect_count_intent, detect_mention_count_intent,
    extract_mention_target, find_page_reference,
)
from doc_locate import detect_location_intent  # noqa: E402
from modality import detect_visual_intent  # noqa: E402

DATA_ROOT = Path(os.getenv(
    "DOCBENCH_ROOT", r"D:\project\RAG-Anything-main\data\DocBench_download"))


def route_meta(q: str):
    """复刻 query.py:330-355 的 op-off 关键词路由，返回触发的算子标签或 None。"""
    kw_visual, kw_table = detect_visual_intent(q)
    if detect_mention_count_intent(q) and extract_mention_target(q):
        return "mention_count"
    if detect_count_intent(q):
        return "count"
    if detect_meta_intent(q) and not (kw_visual or kw_table):
        return "meta_stats" + ("+page_ref" if find_page_reference(q) else "")
    # detect_meta_intent 命中但被图/表避让门挡下 → 记为"被避让"
    if detect_meta_intent(q):
        return "_SUPPRESSED_visual"
    return None


def main():
    meta_qs, all_qs = [], []
    for d in sorted(DATA_ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
        qa = d / f"{d.name}_qa.jsonl"
        if not qa.exists():
            continue
        for line in open(qa, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            o["doc_id"] = d.name
            all_qs.append(o)
            if o.get("type") == "meta-data":
                meta_qs.append(o)

    # ---- meta 题路由覆盖 ----
    from collections import Counter
    buckets = Counter()
    not_fired, suppressed = [], []
    for o in meta_qs:
        r = route_meta(o["question"])
        if r is None:
            buckets["NOT_FIRED"] += 1
            not_fired.append(o)
        elif r == "_SUPPRESSED_visual":
            buckets["SUPPRESSED_visual"] += 1
            suppressed.append(o)
        else:
            buckets[r.split("+")[0]] += 1

    print(f"==== meta-data 题路由覆盖 (n={len(meta_qs)}) ====")
    for k, v in buckets.most_common():
        print(f"  {k:>22}: {v:>3}  ({v/len(meta_qs)*100:.1f}%)")
    fired = len(meta_qs) - buckets["NOT_FIRED"] - buckets["SUPPRESSED_visual"]
    print(f"  {'实际触发算子合计':>22}: {fired:>3}  ({fired/len(meta_qs)*100:.1f}%)")

    print(f"\n---- 路由完全没触发的 meta 题 (B 类下界, {len(not_fired)} 道) ----")
    for o in not_fired:
        print(f"[doc {o['doc_id']}] {o['question'][:100]}")
        print(f"        金标: {str(o.get('answer'))[:80]}")

    if suppressed:
        print(f"\n---- 命中 meta 关键词但被图/表避让门挡下 ({len(suppressed)} 道) ----")
        for o in suppressed:
            print(f"[doc {o['doc_id']}] {o['question'][:100]}")

    # ---- location 意图覆盖（全题型）----
    loc_hits = [o for o in all_qs if detect_location_intent(o["question"])]
    from collections import Counter as C2
    by_type = C2(o.get("type", "") for o in loc_hits)
    print(f"\n==== location 意图命中 (全 {len(all_qs)} 题中 {len(loc_hits)} 道) ====")
    for k, v in by_type.most_common():
        print(f"  按金标题型 {k:>14}: {v}")
    # meta 题里有多少其实是"在第几页"类
    meta_loc = [o for o in meta_qs if detect_location_intent(o["question"])]
    print(f"  其中金标=meta-data 的定位题: {len(meta_loc)} 道")


if __name__ == "__main__":
    main()
