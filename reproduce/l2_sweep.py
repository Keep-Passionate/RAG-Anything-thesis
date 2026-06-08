#!/usr/bin/env python
"""L2 调参干跑：对一批文档图统计不同 (tau, theta) 下的候选同义边数。

不修改任何图、零 API 成本，用于在真正跑 query+评测前缩小调参范围。

用法：
  python reproduce/l2_sweep.py <parent_dir> [--ids 13 45 ...]

<parent_dir> 下应有 <id>/{graph_chunk_entity_relation.graphml, vdb_entities.json}。
输出：每格为所有文档候选同义边数之和（行=tau 列=theta）+ 每篇明细。
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from raganything.graph_fusion.synonym_linker import (
    sweep_thresholds,
    _DEFAULT_TAUS,
    _DEFAULT_THETAS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dir", help="含 <id>/ 子目录的存储父目录")
    ap.add_argument("--ids", nargs="*", default=None, help="只看这些文档 id")
    args = ap.parse_args()
    parent = Path(args.parent_dir)

    if args.ids:
        ids = args.ids
    else:
        ids = sorted(
            [p.name for p in parent.iterdir()
             if (p / "vdb_entities.json").exists()],
            key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
        )

    thetas = _DEFAULT_THETAS
    agg = {(t, th): 0 for t in _DEFAULT_TAUS for th in thetas}
    n_docs = 0

    print("\n每篇候选同义边数（行=tau 列=theta）：")
    for sid in ids:
        res = sweep_thresholds(parent / sid)
        if not res:
            print(f"  doc{sid}: (无数据，跳过)")
            continue
        n_docs += 1
        for k, v in res.items():
            agg[k] += v
        # 打每篇的完整小表
        print(f"  doc{sid}:")
        print("    tau\\theta | " + " ".join(f"{th:>5}" for th in thetas))
        for t in _DEFAULT_TAUS:
            row = " ".join(f"{res[(t, th)]:>5}" for th in thetas)
            print(f"      {t:<6}| {row}")

    print(f"\n==== {n_docs} 篇求和（行=tau 列=theta）====")
    print("tau\\theta | " + " ".join(f"{th:>6}" for th in thetas))
    print("-" * 44)
    for t in _DEFAULT_TAUS:
        row = " ".join(f"{agg[(t, th)]:>6}" for th in thetas)
        print(f"  {t:<6}  | {row}")


if __name__ == "__main__":
    main()
