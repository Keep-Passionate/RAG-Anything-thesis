#!/usr/bin/env python
"""L2 接线诊断（完全离线，不调用任何 API）。

回答一个关键问题：L2 加的同义边到底有没有、以及怎样影响 LightRAG 的检索？

依据 LightRAG operate.py 的检索实现（实测确认）：
  1. 实体经 vdb_entities 向量召回为 top 实体；
  2. `_find_most_related_edges_from_entities` 用 `get_nodes_edges_batch` 从【图存储
     (graphml/networkx)】取这些实体的边 —— 所以同义边确实会被读到，作为"关系"进上下文；
  3. `_find_related_text_unit_from_relations` 对每条关系读 `source_id` 拆成 chunk id
     去 text_chunks_db 取真实文本块。

问题就在第 3 步：L2 同义边的 source_id = "L2_synonym_linker"（占位标记，非真 chunk
id）→ 取文本块时查空 → 同义边**带不进任何真实证据**，另一端实体 B 的内容进不了上下文。
即"通了电但没载货"。本脚本离线量化证实这一点，并估算"若把 source_id 设为两端真实
chunk 并集，能多带多少证据"。

用法：
  python reproduce/diag_l2_wiring.py <working_dir> [--keep]

<working_dir> 需含 graph_chunk_entity_relation.graphml / vdb_entities.json /
kv_store_text_chunks.json / vdb_relationships.json。
脚本流程：清除旧同义边 → 强制加一遍同义边（用当前 config 超参/守卫）→ 离线分析 →
（默认）再清除，恢复原图；加 --keep 则保留所加的边。
"""
import argparse
import json
import sys
from pathlib import Path

import networkx as nx

sys.path.append(str(Path(__file__).parent.parent))

from raganything.graph_fusion.synonym_linker import (  # noqa: E402
    _SYNONYM_SOURCE_ID,
    _is_synonym_edge,
    _node_source_chunks,
    add_synonym_edges,
    remove_synonym_edges,
)


def _load_chunk_ids(working_dir: Path) -> set:
    """text_chunks_db 的所有真实 chunk id（kv_store_text_chunks.json 的键）。"""
    p = working_dir / "kv_store_text_chunks.json"
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        return set(json.load(f).keys())


def _count_synonym_in_rel_vdb(working_dir: Path) -> int:
    """vdb_relationships.json 里有多少条同义边（关系向量检索能看到的）。"""
    p = working_dir / "vdb_relationships.json"
    if not p.exists():
        return -1
    with open(p, encoding="utf-8") as f:
        data = json.load(f).get("data", [])
    # 用 L2 描述的唯一签名 "(synonym, cos=" 精确匹配，避免把论文中真实提到
    # "synonym" 的关系误计入（L2 边其实从不写入 vdb_relationships，应恒为 0）。
    return sum(
        1 for d in data if "(synonym, cos=" in json.dumps(d, ensure_ascii=False)
    )


def diagnose(working_dir: str, keep: bool = False):
    wd = Path(working_dir)
    gpath = wd / "graph_chunk_entity_relation.graphml"
    if not gpath.exists():
        print(f"[ERR] 找不到 {gpath}")
        return

    chunk_ids = _load_chunk_ids(wd)
    print("=" * 64)
    print(f"L2 接线诊断: {wd}")
    print(f"text_chunks_db 真实 chunk 数: {len(chunk_ids)}")

    # 清干净再强制加一遍（用当前 config 的超参与守卫），保证有边可分析
    remove_synonym_edges(working_dir)
    n_added = add_synonym_edges(working_dir, force=True)
    print(f"本次强制新增同义边: {n_added}")
    if n_added == 0:
        print("没有同义边可分析（可能阈值太严或被守卫全拦）。结束。")
        return

    G = nx.read_graphml(str(gpath))
    syn_edges = [(u, v, d) for u, v, d in G.edges(data=True) if _is_synonym_edge(d)]

    # ① 关系向量库里能否看到同义边
    n_in_rel_vdb = _count_synonym_in_rel_vdb(wd)
    # ② 边是 inert(占位符 source_id) 还是 载货(真实 chunk)
    n_inert = sum(1 for _, _, d in syn_edges
                  if d.get("source_id") == _SYNONYM_SOURCE_ID)
    n_carry = len(syn_edges) - n_inert
    # ③ 若 source_id 设为两端真实 chunk 并集，能多带多少“新”证据
    #    （A 通常已是 top 实体、其 chunk 已在上下文，故新增≈B 端独有的 chunk）
    bridge_gain = []   # 每条边：两端 chunk 并集大小
    bridge_new = []    # 每条边：相对另一端的“净新增”近似（取较小端，保守）
    for u, v, _ in syn_edges:
        cu, cv = _node_source_chunks(G, u), _node_source_chunks(G, v)
        bridge_gain.append(len(cu | cv))
        bridge_new.append(min(len(cu - cv), len(cv - cu)))

    print("-" * 64)
    print(f"graphml 中同义边数            : {len(syn_edges)}")
    print(f"vdb_relationships 中同义边数  : {n_in_rel_vdb}   "
          f"{'<- 关系向量检索看不到它们' if n_in_rel_vdb == 0 else ''}")
    print(f"  其中 inert(占位符 source_id): {n_inert}   "
          f"{'<- 取文本块查空 → 0 真实证据' if n_inert else ''}")
    print(f"  其中 载货(真实 chunk)       : {n_carry}   "
          f"{'<- 已能带进对端证据' if n_carry else ''}")
    if bridge_gain:
        print(f"两端 chunk 并集大小  平均 {sum(bridge_gain)/len(bridge_gain):.1f} / "
              f"最大 {max(bridge_gain)}（这是“若修好”每条边的证据上限）")
        print(f"另一端净新增 chunk   平均 {sum(bridge_new)/len(bridge_new):.1f} / "
              f"最大 {max(bridge_new)}（保守估计的桥接增益）")

    print("-" * 64)
    print("样本同义边（前 8）：")
    for u, v, d in syn_edges[:8]:
        cu, cv = _node_source_chunks(G, u), _node_source_chunks(G, v)
        print(f"  {u!r} ~ {v!r} | desc={d.get('description','')[:38]} | "
              f"A.chunks={len(cu)} B.chunks={len(cv)}")

    print("-" * 64)
    print("结论：")
    print(" · 同义边会被检索读到（作为“关系”进上下文，operate.py 从 graphml 取边）；")
    if n_carry and not n_inert:
        print(" · 已载货：source_id 为两端真实 chunk → 检索到同义边时会拉进对端证据。")
        print("   ⚠️ 载货会放大真/假同义边，务必先用 show_synonyms 确认精度再上线。")
    else:
        print(" · 但其 source_id 是占位符、且不在关系向量库 → 关系→文本块这步查空，")
        print("   同义边带不进真实文本块 → 对端实体内容进不了上下文（“通电未载货”）；")
        print(" · 修复：SYNONYM_CARRY_CHUNKS=true（或 --carry-chunks）把 source_id 设为")
        print("   两端真实 chunk 并集，让检索到同义边时真正拉进对端证据。")

    if not keep:
        remove_synonym_edges(working_dir)
        print("\n已移除本次所加同义边，图恢复原状（如需保留请加 --keep）。")
    else:
        print("\n已保留本次所加同义边（--keep）。")


def main():
    ap = argparse.ArgumentParser(description="L2 接线诊断（离线，不调 API）")
    ap.add_argument("working_dir", help="LightRAG 存储目录")
    ap.add_argument("--keep", action="store_true", help="分析后保留所加同义边（默认移除）")
    args = ap.parse_args()
    diagnose(args.working_dir, keep=args.keep)


if __name__ == "__main__":
    main()
