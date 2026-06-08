#!/usr/bin/env python
"""统计 LightRAG 存储目录下各文档图的节点/边数（含 L2 同义边数）。

用法：
  python reproduce/graph_stats.py <parent_dir> [--ids 13 45 ...]

<parent_dir> 下应有 <id>/graph_chunk_entity_relation.graphml 子目录。
不传 --ids 则自动扫描所有含 graphml 的子目录。

用途：
  - 对比 baseline vs L1（L1 合并重复实体名 → 节点数应下降或持平）
  - 查看 L2 给每篇加了多少同义边（synonym 列）
"""
import argparse
from pathlib import Path

import networkx as nx

SYN_SOURCE = "L2_synonym_linker"


def stats_one(graphml_path):
    G = nx.read_graphml(str(graphml_path))
    syn = sum(
        1 for _, _, d in G.edges(data=True) if d.get("source_id") == SYN_SOURCE
    )
    return G.number_of_nodes(), G.number_of_edges(), syn


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
             if (p / "graph_chunk_entity_relation.graphml").exists()],
            key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
        )

    print(f"\n{parent}")
    print(f"{'doc':>8} {'nodes':>8} {'edges':>8} {'synonym':>8}")
    print("-" * 36)
    tn = te = ts = 0
    for sid in ids:
        g = parent / sid / "graph_chunk_entity_relation.graphml"
        if not g.exists():
            print(f"{sid:>8} {'MISSING':>8}")
            continue
        n, e, s = stats_one(g)
        tn += n
        te += e
        ts += s
        print(f"{sid:>8} {n:>8} {e:>8} {s:>8}")
    print("-" * 36)
    print(f"{'TOTAL':>8} {tn:>8} {te:>8} {ts:>8}")


if __name__ == "__main__":
    main()
