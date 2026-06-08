#!/usr/bin/env python
"""列出某张图里 L2 添加的同义边（实体对 + cos + jaccard），供人工检查是否合理。

用法：
  python reproduce/show_synonyms.py <working_dir>
例：
  python reproduce/show_synonyms.py /root/autodl-tmp/.../rag_storage_L2_t090_th10/63
"""
import argparse
import sys
from pathlib import Path

import networkx as nx

sys.path.append(str(Path(__file__).parent.parent))
from raganything.graph_fusion.synonym_linker import _SYNONYM_SOURCE_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("working_dir", help="单篇文档的存储目录")
    args = ap.parse_args()

    gpath = Path(args.working_dir) / "graph_chunk_entity_relation.graphml"
    G = nx.read_graphml(str(gpath))
    syn = [
        (u, v, d.get("weight"), d.get("description", ""))
        for u, v, d in G.edges(data=True)
        if d.get("source_id") == _SYNONYM_SOURCE_ID
    ]
    print(f"\n{gpath}")
    print(f"共 {len(syn)} 条同义边（按 cos 降序）：\n")
    for u, v, w, desc in sorted(syn, key=lambda x: -(x[2] or 0)):
        print(f"  [{w}] {u!r}  <->  {v!r}")


if __name__ == "__main__":
    main()
