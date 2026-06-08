#!/usr/bin/env python
"""离线体检：从带 retrieved_context 的 qa_results 里算"证据命中率"（recall 代理）。

证据命中 = 标准答案的关键内容是否出现在该题检索到的上下文里。
**不调用任何模型、不花钱。** 需要先用 SAVE_CONTEXT=true（或 ENABLE_RETRIEVAL_REFLECT=true）
跑过答题，使 qa_results 里带 "retrieved_context" 字段。

读出来的是"检索类方法的提升上限"：命中率越低，说明越多题是"证据根本没检索到"，
这正是 R2/R3 检索类方法能救的部分；反之若命中率已很高、却仍答错，则瓶颈在"读数/表示"。

用法：
  python reproduce/diag_recall.py <data_dir> [--result-name qa_results_mix_mm.json]
  <data_dir> 下应有 <id>/<result_name>，以及 <id>/<id>_qa.jsonl（用于按题型拆分，可选）。

注意：命中判定是"粗略字符串/数字匹配"的代理，只用于看大方向，不是精确指标。
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def evidence_hit(gold: str, ctx: str):
    """粗略判断标准答案是否被检索到。返回 True / False / None(无法判定)。"""
    g, c = _norm(gold), _norm(ctx)
    if not g or not c:
        return None
    if len(g) < 60 and g in c:  # 短答案：整串命中
        return True
    # 数字类答案：标准答案里的数字串是否都出现在上下文（去千分位逗号后比对）
    nums = re.findall(r"\d[\d,.]*\d|\d", gold)
    cc = c.replace(",", "")
    if nums and all(n.replace(",", "") in cc for n in nums):
        return True
    # 其它：长度>2 的实词命中比例 >= 0.8
    toks = [t for t in re.findall(r"[a-z0-9]+", g) if len(t) > 2]
    if toks and sum(t in c for t in toks) / len(toks) >= 0.8:
        return True
    return False


def load_types(qa_path: Path):
    """问题文本 -> 题型（multimodal / text / meta-data / unanswerable ...）。"""
    m = {}
    if not qa_path.exists():
        return m
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m[_norm(d.get("question", ""))] = d.get("type", "unknown")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="含 <id>/ 子目录的数据根目录（如 DocBench_subset）")
    ap.add_argument("--result-name", default="qa_results_mix_mm.json")
    args = ap.parse_args()
    root = Path(args.data_dir)

    total = hit = miss = unknown = 0
    by_type = defaultdict(lambda: [0, 0])  # type -> [hit, judged]
    missed_examples = []

    for sub in sorted(root.iterdir()):
        rp = sub / args.result_name
        if not rp.exists():
            continue
        types = load_types(sub / f"{sub.name}_qa.jsonl")
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in data:
            if "retrieved_context" not in r:
                continue
            total += 1
            t = types.get(_norm(r.get("question", "")), "unknown")
            h = evidence_hit(r.get("correct_answer", ""), r.get("retrieved_context", ""))
            if h is None:
                unknown += 1
                continue
            by_type[t][1] += 1
            if h:
                hit += 1
                by_type[t][0] += 1
            else:
                miss += 1
                if len(missed_examples) < 15:
                    missed_examples.append(
                        (sub.name, r.get("question", "")[:70], (r.get("correct_answer", "") or "")[:40])
                    )

    judged = hit + miss
    print(f"\n样本: {total}  (可判定 {judged}, 无法判定/空答案 {unknown})")
    if not judged:
        print("没有可判定的样本。确认 qa_results 里有 retrieved_context（需 SAVE_CONTEXT=true 跑答题）。")
        return
    print(f"证据命中率 (recall 代理): {hit}/{judged} = {hit / judged * 100:.1f}%")
    print(f"  → 约 {miss / judged * 100:.1f}% 的题'证据没被检索到' = 检索类方法(R2/R3)的提升上限")
    print("\n按题型:")
    for t, (h, n) in sorted(by_type.items()):
        if n:
            print(f"  {t:>14}: {h}/{n} = {h / n * 100:.0f}%")
    print("\n部分'没检索到'的题（doc / 问题 / 标准答案）:")
    for doc, qtext, a in missed_examples:
        print(f"  [{doc}] {qtext}  ||  {a}")


if __name__ == "__main__":
    main()
