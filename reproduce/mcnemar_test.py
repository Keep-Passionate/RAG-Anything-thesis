#!/usr/bin/env python
"""对两个 method 的逐题对错做 McNemar 配对显著性检验——零成本离线，不调任何 API。

McNemar 检验专门针对"配对二元结果"：只看两边结论【不一致】的题。
  b = A 错→B 对（B 赢回）
  c = A 对→B 错（B 搞砸）
连续校正卡方：chi2 = (|b-c|-1)^2 / (b+c)，自由度 1。
精确 p：双侧二项检验 Binomial(min(b,c); n=b+c, p=0.5)（小样本更稳）。

这就是论文里 McNemar 数字的可复现来源；b/c 与 diff_results.py 的"赢回/搞砸"一致。

用法（与 diff_results.py 相同的输入）：
  python reproduce/mcnemar_test.py \
      --eval /root/autodl-tmp/eval_out/llm_evaluation_results.json \
      --a base_t0 --b ours3
  # 只看某题型（如元数据子集），加 --qa-dir 与 --type：
  python reproduce/mcnemar_test.py --eval ... --a base_t0 --b ours3 \
      --qa-dir /root/autodl-tmp/DocBench_subset --type metadata
"""
import argparse
import json
from pathlib import Path


def norm(q: str) -> str:
    return " ".join(str(q).split()).strip().lower()


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    acc = {}
    for r in results:
        if not r.get("success") or r.get("accuracy") is None:
            continue
        acc[(r.get("method"), str(r.get("doc_id")), norm(r.get("question")))] = int(
            r["accuracy"]
        )
    return acc


def load_types(qa_dir):
    q2type = {}
    for qa in Path(qa_dir).glob("*/*_qa.jsonl"):
        with open(qa, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    q2type[norm(d.get("question", ""))] = d.get("type", "unknown")
    return q2type


def binom_two_sided(k, n):
    """双侧精确二项 p（p0=0.5），不依赖 scipy。"""
    from math import comb
    if n == 0:
        return 1.0
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    p_obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= p_obs + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--a", required=True, help="基准 method")
    ap.add_argument("--b", required=True, help="对比 method")
    ap.add_argument("--qa-dir", help="给了才能按 --type 过滤")
    ap.add_argument("--type", help="只统计该题型（如 metadata）")
    args = ap.parse_args()

    acc = load_eval(args.eval)
    a = {(d, q): v for (m, d, q), v in acc.items() if m == args.a}
    b = {(d, q): v for (m, d, q), v in acc.items() if m == args.b}
    common = sorted(set(a) & set(b))

    if args.type:
        if not args.qa_dir:
            ap.error("--type 需要同时给 --qa-dir")
        q2type = load_types(args.qa_dir)
        common = [k for k in common if q2type.get(k[1]) == args.type]

    b_win = c_lose = 0  # b=A错→B对, c=A对→B错
    for k in common:
        va, vb = a[k], b[k]
        if va == 0 and vb == 1:
            b_win += 1
        elif va == 1 and vb == 0:
            c_lose += 1

    n_disc = b_win + c_lose
    chi2 = ((abs(b_win - c_lose) - 1) ** 2 / n_disc) if n_disc else 0.0
    p_exact = binom_two_sided(min(b_win, c_lose), n_disc)

    scope = f"题型={args.type}" if args.type else "全部题型"
    print(f"\nMcNemar  {args.a} vs {args.b}   ({scope})")
    print(f"  共同题 N = {len(common)}")
    print(f"  b = A错→B对（赢回） = {b_win}")
    print(f"  c = A对→B错（搞砸） = {c_lose}")
    print(f"  不一致对 b+c = {n_disc}")
    print(f"  连续校正卡方 chi2 = {chi2:.2f}  (df=1)")
    print(f"  精确双侧二项 p   = {p_exact:.2e}")
    sig = "p<0.001" if p_exact < 1e-3 else ("p<0.05" if p_exact < 0.05 else "不显著")
    print(f"  结论：{sig}")


if __name__ == "__main__":
    main()
