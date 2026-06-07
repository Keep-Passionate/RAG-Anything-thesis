"""L2 同义边——建图后对相似实体对加 synonym 边（不合并节点）。

判定一对实体是"同义"需同时满足两个条件（缺一不可，互相把关）：
  1. 语义相似：embedding 余弦相似度 cos > tau
  2. 结构相似：图中邻居集合的 Jaccard 相似度 > theta
结构约束（Jaccard）用于排除"营收 vs 净利润"这类语义近但含义不同的假阳性——
真正同义的两个实体，在图里周围连接的节点（语境）大部分会重合。

设计（便于消融/调参，刻意解耦）：
  - find_synonym_pairs(...)  : 纯计算层，只吃 numpy 矩阵 + 邻居字典，无文件/无图依赖，可单测
  - add_synonym_edges(...)   : I/O 层，读 vdb + graphml，调纯计算层，写回 graphml
  - sweep_thresholds(...)    : 干跑调参，不改图，统计不同 (tau,theta) 下的候选数
  - remove_synonym_edges(...): 清除 L2 加的边（按 source_id 标记），可重新调参而无需重新建图

关键性质：Jaccard 一律对【原始图】的邻居快照计算，与候选处理顺序无关 → 结果确定、可复现。

开关/超参见 config.py（ENABLE_SYNONYM_EDGES / SYNONYM_TAU / SYNONYM_THETA，默认关）。
参考：Jaccard 相似度（Paul Jaccard, 1901）；邻居结构约束为本工作原创加法。
"""

import argparse
import json
import logging
import time
from pathlib import Path

import networkx as nx
import numpy as np

from raganything.graph_fusion.config import (
    get_synonym_tau,
    get_synonym_theta,
    is_synonym_edges_enabled,
)

logger = logging.getLogger(__name__)

# L2 所加边的标记：用于在 graphml 里识别/清除（区别于原版关系边）
_SYNONYM_SOURCE_ID = "L2_synonym_linker"
_SYNONYM_KEYWORD = "synonym"

# 调参干跑的默认网格
_DEFAULT_TAUS = (0.80, 0.85, 0.90, 0.95)
_DEFAULT_THETAS = (0.05, 0.10, 0.15, 0.20)


# ---------------------------------------------------------------------------
# 纯计算层（无文件 / 无图对象依赖，便于单测）
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(matrix: np.ndarray) -> np.ndarray:
    """对 (N, D) embedding 矩阵，计算 N×N 余弦相似度矩阵（值域 [-1, 1]）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)  # 避免除零
    normed = matrix / norms
    return normed @ normed.T


def _jaccard(a: set, b: set) -> float:
    """两个集合的 Jaccard 相似度 = |交| / |并|。并集为空返回 0。"""
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def find_synonym_pairs(names, matrix, neighbor_of, tau, theta, sim=None):
    """纯计算：返回合格的同义实体对，按余弦降序排列。

    Args:
        names       : list[str]，长度 N 的实体名（与 matrix 行一一对应）
        matrix      : np.ndarray (N, D)，实体 embedding
        neighbor_of : dict[str, set[str]]，实体名 -> 邻居实体名集合（图结构快照）
        tau         : 余弦阈值（> tau 才算语义相似）
        theta       : Jaccard 阈值（> theta 才算结构相似）
        sim         : 可选，预先算好的 N×N 余弦矩阵（调参 sweep 时复用以省时）

    Returns:
        list[(name_i, name_j, cos, jaccard)]，按 cos 降序。
        会跳过：同名对、已经互为邻居（已有边）的对。
    """
    n = len(names)
    if n < 2:
        return []
    if sim is None:
        sim = _cosine_sim_matrix(matrix)

    # 只看上三角（i<j），且 cos>tau
    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] > tau
    iu, ju = iu[mask], ju[mask]

    pairs = []
    for i, j in zip(iu.tolist(), ju.tolist()):
        ni, nj = names[i], names[j]
        if ni == nj:
            continue
        si = neighbor_of.get(ni, set())
        sj = neighbor_of.get(nj, set())
        if nj in si or ni in sj:
            continue  # 已直接相连，无需再加同义边
        jac = _jaccard(si, sj)
        if jac > theta:
            pairs.append((ni, nj, float(sim[i, j]), jac))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# 文件 / 图 I/O 层
# ---------------------------------------------------------------------------

def _paths(working_dir):
    wd = Path(working_dir)
    return wd, wd / "graph_chunk_entity_relation.graphml", wd / "vdb_entities.json"


def _load_entity_embeddings(vdb_path: Path):
    """从 nano-vectordb 的 JSON 加载实体名与 embedding 矩阵。

    返回 (names: list[str], matrix: np.ndarray (N, D))。
    """
    from nano_vectordb import NanoVectorDB

    with open(vdb_path, encoding="utf-8") as f:
        dim = json.load(f).get("embedding_dim", 1024)

    db = NanoVectorDB(dim, storage_file=str(vdb_path))
    # nano-vectordb 没有公开"取整表矩阵"的接口，访问其内部存储拿矩阵
    storage = db._NanoVectorDB__storage
    names = [it["entity_name"] for it in storage["data"]]
    matrix = np.asarray(storage["matrix"], dtype=np.float32)
    return names, matrix


def _load_graph_and_neighbors(graphml_path, vdb_path):
    """加载图 + 实体 embedding，并对齐到图节点空间。

    返回 (G, fnames, fmatrix, neighbor_of)：
      - 只保留同时存在于图节点的实体（LightRAG 里节点 ID == entity_name）
      - neighbor_of 是【全图】的邻居快照（实体名 -> 邻居名集合）
    """
    names, matrix = _load_entity_embeddings(vdb_path)
    G = nx.read_graphml(str(graphml_path))
    graph_names = set(G.nodes())

    keep = [k for k, nm in enumerate(names) if nm in graph_names]
    missing = len(names) - len(keep)
    if missing:
        logger.info(
            "L2: %d/%d vdb entities not present as graph nodes (ignored)",
            missing, len(names),
        )
    fnames = [names[k] for k in keep]
    fmatrix = matrix[keep] if keep else matrix[:0]
    neighbor_of = {nm: set(G.neighbors(nm)) for nm in graph_names}
    return G, fnames, fmatrix, neighbor_of


def add_synonym_edges(working_dir, tau=None, theta=None, force=False) -> int:
    """对 working_dir 下的知识图谱添加同义边（L2）。

    Args:
        working_dir : LightRAG 存储目录（含 graphml 与 vdb_entities.json）
        tau / theta : 阈值；None 时读 config（SYNONYM_TAU / SYNONYM_THETA）
        force       : True 时忽略 ENABLE_SYNONYM_EDGES 开关强制执行（供 CLI 用）

    Returns:
        新增同义边数量。开关关闭且非 force 时返回 0。
    """
    if not force and not is_synonym_edges_enabled():
        return 0

    tau = get_synonym_tau() if tau is None else tau
    theta = get_synonym_theta() if theta is None else theta

    wd, gpath, vpath = _paths(working_dir)
    if not gpath.exists() or not vpath.exists():
        logger.warning("L2: missing graphml or vdb under %s, skip", wd)
        return 0

    G, fnames, fmatrix, neighbor_of = _load_graph_and_neighbors(gpath, vpath)
    if len(fnames) < 2:
        return 0

    pairs = find_synonym_pairs(fnames, fmatrix, neighbor_of, tau, theta)

    ts = int(time.time())
    added = 0
    for ni, nj, cos, jac in pairs:
        if G.has_edge(ni, nj):
            continue
        G.add_edge(
            ni, nj,
            weight=round(cos, 4),
            description=f"Synonym (L2): cos={cos:.3f}, jaccard={jac:.3f}",
            keywords=_SYNONYM_KEYWORD,
            source_id=_SYNONYM_SOURCE_ID,
            file_path=_SYNONYM_SOURCE_ID,
            created_at=ts,
            truncate="",
        )
        added += 1

    if added:
        nx.write_graphml(G, str(gpath))
    logger.info(
        "L2: tau=%.2f theta=%.2f -> %d synonym edges%s",
        tau, theta, added, f" written to {gpath}" if added else "",
    )
    return added


def remove_synonym_edges(working_dir) -> int:
    """移除 L2 加的同义边（按 source_id 标记），恢复原始图。返回移除数量。

    用途：重新调参时先清除旧同义边再重加；或撤销误加。
    """
    wd, gpath, _ = _paths(working_dir)
    if not gpath.exists():
        return 0
    G = nx.read_graphml(str(gpath))
    to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("source_id") == _SYNONYM_SOURCE_ID
    ]
    if to_remove:
        G.remove_edges_from(to_remove)
        nx.write_graphml(G, str(gpath))
    logger.info("L2: removed %d synonym edges from %s", len(to_remove), gpath)
    return len(to_remove)


def sweep_thresholds(working_dir, taus=_DEFAULT_TAUS, thetas=_DEFAULT_THETAS):
    """干跑调参：不改图，统计不同 (tau, theta) 下的候选同义边数量。

    复用同一份余弦矩阵，对网格逐格统计。返回 dict[(tau, theta)] -> count。
    """
    wd, gpath, vpath = _paths(working_dir)
    if not gpath.exists() or not vpath.exists():
        logger.warning("L2: missing graphml or vdb under %s", wd)
        return {}
    G, fnames, fmatrix, neighbor_of = _load_graph_and_neighbors(gpath, vpath)
    if len(fnames) < 2:
        return {}
    sim = _cosine_sim_matrix(fmatrix)  # 只算一次，全网格复用
    out = {}
    for t in taus:
        for th in thetas:
            out[(t, th)] = len(
                find_synonym_pairs(fnames, fmatrix, neighbor_of, t, th, sim=sim)
            )
    return out


# ---------------------------------------------------------------------------
# CLI：调参 / 应用 / 清除（不必重新建图）
# ---------------------------------------------------------------------------

def _main():
    ap = argparse.ArgumentParser(description="L2 同义边：调参 / 应用 / 清除")
    ap.add_argument("working_dir", help="LightRAG 存储目录")
    ap.add_argument("--tau", type=float, default=None, help="余弦阈值（默认读 config）")
    ap.add_argument("--theta", type=float, default=None, help="Jaccard 阈值（默认读 config）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计不同阈值下的候选数（不改图），用于调参")
    ap.add_argument("--remove", action="store_true", help="移除已加的同义边")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.remove:
        print("removed synonym edges:", remove_synonym_edges(args.working_dir))
        return

    if args.dry_run:
        res = sweep_thresholds(args.working_dir)
        if not res:
            print("(no data)")
            return
        thetas = _DEFAULT_THETAS
        header = "tau\\theta | " + " ".join(f"{th:>6}" for th in thetas)
        print("\n候选同义边数量（行=tau 列=theta）：")
        print(header)
        print("-" * len(header))
        for t in _DEFAULT_TAUS:
            row = " ".join(f"{res[(t, th)]:>6}" for th in thetas)
            print(f"  {t:<7}| {row}")
        return

    # 应用模式（CLI 显式调用即视为有意为之，force=True 绕过开关）
    n = add_synonym_edges(args.working_dir, tau=args.tau, theta=args.theta, force=True)
    print("added synonym edges:", n)


if __name__ == "__main__":
    _main()
