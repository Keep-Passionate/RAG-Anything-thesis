#!/usr/bin/env python
"""按问题类型拆分各 method 的准确率（multimodal / text-only / meta-data / unanswerable …）。

把评测结果 (eval_*/llm_evaluation_results.json) 与各文档 qa.jsonl 的 type 字段
按"问题文本"对齐，输出 method × type 的正确率表。
用途：判断 L2 是否在某类问题（尤其多模态）上有效——这是 L2 的设计初衷。

用法：
  python reproduce/eval_by_type.py \
    --eval /root/autodl-tmp/eval_all/llm_evaluation_results.json \
    --qa-dir /root/autodl-tmp/DocBench_subset
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def norm(q: str) -> str:
    """问题文本规范化，用于跨文件对齐。"""
    return " ".join(str(q).split()).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="llm_evaluation_results.json 路径")
    ap.add_argument("--qa-dir", required=True, help="含 <id>/<id>_qa.jsonl 的目录")
    args = ap.parse_args()

    # 1) 问题文本 -> type
    q2type = {}
    for qa in Path(args.qa_dir).glob("*/*_qa.jsonl"):
        with open(qa, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q2type[norm(d.get("question", ""))] = d.get("type", "unknown")

    # 2) 读评测结果
    with open(args.eval, encoding="utf-8") as f:
        results = json.load(f)

    # 3) (method, type) -> [correct, total]
    agg = defaultdict(lambda: [0, 0])
    methods, types = set(), set()
    unmatched = 0
    for r in results:
        if not r.get("success"):
            continue
        acc = r.get("accuracy")
        if acc is None:
            continue
        m = r.get("method", "?")
        t = q2type.get(norm(r.get("question", "")), "UNMATCHED")
        if t == "UNMATCHED":
            unmatched += 1
        methods.add(m)
        types.add(t)
        agg[(m, t)][0] += acc
        agg[(m, t)][1] += 1

    types = sorted(types)
    methods = sorted(methods)

    print(f"\n准确率 method × type（correct/total，百分比）  未匹配题:{unmatched}\n")
    head = f"{'method':<16}" + "".join(f"{t:>16}" for t in types)
    print(head)
    print("-" * len(head))
    for m in methods:
        row = f"{m:<16}"
        for t in types:
            c, n = agg[(m, t)]
            cell = f"{c}/{n} {100*c/n:.0f}%" if n else "-"
            row += f"{cell:>16}"
        print(row)


if __name__ == "__main__":
    main()
