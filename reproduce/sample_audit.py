#!/usr/bin/env python
"""裁判可靠性抽检：从评测结果分层抽样导出 CSV，供人工复核 LLM 裁判的判分。

为什么需要：评测裁判是 qwen-plus（与答题同源），论文必须回应"模型自评偏差"的
质疑。标准做法是人工复核一个分层样本、报告"人判 vs 裁判"的一致率（如 50 题
一致 94% → 裁判可信）。本脚本把样本准备成可直接填写的 CSV。

分层方式：按 method × 裁判判分(0/1) 均衡抽样——错题和对题各占一半，避免只复核
对的题导致一致率虚高。随机种子固定，论文可复现。

用法（服务器）：
  python reproduce/sample_audit.py \
      --eval /root/autodl-tmp/eval_out/llm_evaluation_results.json \
      --qa-dir /root/autodl-tmp/DocBench_subset \
      --methods paperbase full4 --n 50 --out /root/autodl-tmp/audit_sample.csv

填写：human_judgment 列填 1(答对)/0(答错)，填完发回来即可算一致率。
"""
import argparse
import csv
import json
import random
from pathlib import Path

from diff_results import build_doc_map, load_eval, norm


def load_answers(qa_dir, doc_map, method, doc_id, q_normed):
    """从 qa_results_<method>.json 取这道题的模型答案与标准答案。"""
    folder = doc_map.get(doc_id, doc_id)
    p = Path(qa_dir) / folder / f"qa_results_{method}.json"
    if not p.exists():
        return "", ""
    try:
        with open(p, encoding="utf-8") as f:
            for rec in json.load(f):
                if norm(rec.get("question")) == q_normed:
                    return rec.get("answer", ""), rec.get("correct_answer", "")
    except Exception:
        pass
    return "", ""


def main():
    ap = argparse.ArgumentParser(description="裁判可靠性人工抽检样本导出")
    ap.add_argument("--eval", required=True, help="llm_evaluation_results.json 路径")
    ap.add_argument("--qa-dir", required=True, help="含 <id>/<id>_qa.jsonl 的目录")
    ap.add_argument("--methods", nargs="+", required=True, help="抽哪些 method")
    ap.add_argument("--n", type=int, default=50, help="总样本数（均分到 method×判分）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（固定可复现）")
    ap.add_argument("--out", default="audit_sample.csv", help="输出 CSV 路径")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    acc = load_eval(args.eval)
    doc_map = build_doc_map(args.qa_dir)

    # 分层池：(method, accuracy) -> [(doc_id, q_normed)]
    pools = {}
    for (m, d, q), v in acc.items():
        if m in args.methods:
            pools.setdefault((m, v), []).append((d, q))

    per_cell = max(1, args.n // max(1, len(pools)))
    rows = []
    for (m, v), pool in sorted(pools.items()):
        rng.shuffle(pool)
        for d, q in pool[:per_cell]:
            answer, gold = load_answers(args.qa_dir, doc_map, m, d, q)
            rows.append({
                "method": m,
                "doc_id": d,
                "llm_judge": v,
                "question": q,
                "model_answer": answer,
                "correct_answer": gold,
                "human_judgment": "",
            })
    rng.shuffle(rows)  # 打乱顺序，避免复核时按 method 形成定势

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # sig=Excel 兼容
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已导出 {len(rows)} 题 -> {args.out}")
    print("人工在 human_judgment 列填 1/0，填完算一致率：")
    print("  一致率 = (human_judgment == llm_judge 的行数) / 总行数")


if __name__ == "__main__":
    main()
