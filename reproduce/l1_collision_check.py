#!/usr/bin/env python
"""诊断 L1 名称规范化的真实效果（隔离 LLM 抽取随机性）。

对一张已建好的图，统计规范化后会被合并的实体名数量：
    merged = 不同原始名数 - 不同规范化名数
并可列出实际发生合并的名字组。

为什么需要：baseline 与 L1 是两次独立的 LLM 抽取，节点数差异主要来自
LLM 随机性（每次抽出的实体略有不同），无法反映 L1 的真实作用。本脚本在
【同一张图】上直接量出"L1 会把多少个名字合并掉"，零 LLM 噪声。

用法：
  python reproduce/l1_collision_check.py <parent_dir> [--ids 13 45 ...] [--show]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

sys.path.append(str(Path(__file__).parent.parent))
from raganything.graph_fusion.canonicalizer import normalize_entity_name


def check_one(graphml_path):
    G = nx.read_graphml(str(graphml_path))
    groups = defaultdict(list)
    for n in G.nodes():
        groups[normalize_entity_name(n)].append(n)
    merged = {c: raws for c, raws in groups.items() if len(raws) > 1}
    n_raw = G.number_of_nodes()
    n_canon = len(groups)
    return n_raw, n_canon, n_raw - n_canon, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dir", help="含 <id>/ 子目录的存储父目录")
    ap.add_argument("--ids", nargs="*", default=None, help="只看这些文档 id")
    ap.add_argument("--show", action="store_true", help="列出被合并的名字组")
    args = ap.parse_args()
    parent = Path(args.parent_dir)

    ids = args.ids or sorted(
        [p.name for p in parent.iterdir()
         if (p / "graph_chunk_entity_relation.graphml").exists()],
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )

    print(f"\n{parent}")
    print(f"{'doc':>8} {'raw':>8} {'canon':>8} {'merged':>8}")
    print("-" * 36)
    total_merged = 0
    for sid in ids:
        g = parent / sid / "graph_chunk_entity_relation.graphml"
        if not g.exists():
            print(f"{sid:>8} {'MISSING':>8}")
            continue
        nr, nc, m, merged = check_one(g)
        total_merged += m
        print(f"{sid:>8} {nr:>8} {nc:>8} {m:>8}")
        if args.show and merged:
            for canon, raws in merged.items():
                print(f"          {raws!r} -> {canon!r}")
    print("-" * 36)
    print(f"{'TOTAL merged':>26} {total_merged:>8}")
    print("\n说明：merged=0 表示 L1 在该数据上几乎不触发（干净文本的预期），"
          "其零风险价值在于'万一有变体也能安全合并'；真正提升靠 L2。")


if __name__ == "__main__":
    main()
