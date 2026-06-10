"""L2 同义边——建图后对相似实体对加 synonym 边（不合并节点）。

判定一对实体是"同义"需同时满足两个条件（缺一不可，互相把关）：
  1. 语义相似：embedding 余弦相似度 cos > tau
  2. 结构相似：图中邻居集合的 Jaccard 相似度 > theta
结构约束（Jaccard）用于排除"营收 vs 净利润"这类语义近但含义不同的假阳性——
真正同义的两个实体，在图里周围连接的节点（语境）大部分会重合。

防过连接守卫（Step3，治"同义边连了太多不相关内容→效率降低"）：
  规则1 文档作用域（require_same_doc）：两实体若无任何共同来源文档则拒连，根除
         "跨文档同名/近义污染"。仅对多篇合并在同一张图的情形有效；每篇独立索引
         时图内实体同属一篇，此守卫为空操作（无副作用）。
  规则4 每节点预算（max_per_node）：每个实体最多新增 K 条同义边（按 cos 取最高的），
         封住通用词/hub 的过连接。单篇图里这是治过连接的主力旋钮。
  类型过滤（skip_types）：两端任一实体属于指定 entity_type（默认 person）则拒连。
         治本地解决人名假阳性（同姓不同人 cos+Jaccard 双高，阈值分不开）。

载货（carry_chunks，默认关）：诊断（diag_l2_wiring.py）发现默认实现里同义边 source_id
  是占位符，检索读到边却拉不进真实文本块（"通电未载货"）。开启后把 source_id 设为两端
  实体真实 chunk 并集，检索到同义边时真正带进对端证据。⚠️ 会放大真/假同义边，先确认精度。
  注：L2 边的移除/识别一律用 file_path==_SYNONYM_SOURCE_ID 标记（与 source_id 是否载货无关）。

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
import difflib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from raganything.graph_fusion.config import (
    get_synonym_max_per_node,
    get_synonym_skip_types,
    get_synonym_tau,
    get_synonym_theta,
    is_enum_filter_enabled,
    is_synonym_carry_chunks_enabled,
    is_synonym_edges_enabled,
    is_synonym_require_name_match_enabled,
    is_synonym_require_same_type_enabled,
    is_synonym_same_doc_enabled,
)

logger = logging.getLogger(__name__)

# L2 所加边的标记：用于在 graphml 里识别/清除（区别于原版关系边）
_SYNONYM_SOURCE_ID = "L2_synonym_linker"
_SYNONYM_KEYWORD = "synonym"

# LightRAG 在节点属性里用此分隔符连接多来源（多 chunk / 多文档）。
# 文档作用域守卫据此从节点 file_path 还原"该实体出现在哪些文档"。
GRAPH_FIELD_SEP = "<SEP>"

# 枚举判别守卫用的正则
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")        # 数字（含小数/千分位）
_ALNUM_RE = re.compile(r"[a-z0-9]+")            # 字母数字 token（已小写）
_ROMAN_RE = re.compile(r"^(?:i{1,3}|iv|v|vi{0,3}|ix|x{1,3})$")  # 罗马数字 i..xiii

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


def _split_docs(file_path_attr) -> set:
    """把节点的 file_path 属性（多文档用 GRAPH_FIELD_SEP 连接）拆成文档名集合。

    例："paper1.pdf<SEP>paper2.pdf" -> {"paper1.pdf", "paper2.pdf"}。
    属性缺失/为空返回空集（表示"文档归属未知"，由调用方决定是否放行）。
    """
    if not file_path_attr:
        return set()
    return {p.strip() for p in str(file_path_attr).split(GRAPH_FIELD_SEP) if p.strip()}


def _node_source_chunks(G, name) -> set:
    """实体节点的真实来源 chunk 集合（节点 source_id 按 GRAPH_FIELD_SEP 拆）。载货用。"""
    raw = G.nodes.get(name, {}).get("source_id", "")
    if not raw:
        return set()
    return {c for c in str(raw).split(GRAPH_FIELD_SEP) if c}


def _is_synonym_edge(edge_attrs) -> bool:
    """判定一条边是否为 L2 同义边。

    用 file_path==_SYNONYM_SOURCE_ID 作标记——它与边是否"载货"无关（载货时 source_id
    会变成真实 chunk，不能再当标记），所以识别/移除一律走 file_path。
    """
    return edge_attrs.get("file_path") == _SYNONYM_SOURCE_ID


def _is_enumeration_variant(a: str, b: str) -> bool:
    """判断 a、b 是否"仅靠数字/序号区分的不同条目"（枚举变体），是则应拒绝同义边。

    专治财报/法律文档的高频假阳性（字符串 90%+ 相同、上下文相同，cos+Jaccard 双高
    却是不同实体）：
      债券 '…Due 2027' vs '2029'、页码 102/103、金额 ¥29.7B/¥26.6B、章节 4.02(a)/(b)、
      子公司 Sacramento I/II、Exhibit 10.22/10.23、CIK、日期、Fiscal Year 2020/2019 …

    判据（任一成立即判为枚举变体）：
      规则1：两名所含【数字集合】不同 → 不同编号的条目。
      规则2：token 对称差异【仅由单/双字母或罗马数字构成】 → (a)/(b)、I/II 之类序号。

    注意：缩写/全称(SEC↔Securities and Exchange Commission)、单复数(Rating↔Ratings)、
    后缀(X↔X, Inc.)等真同义不会被误伤——它们的差异 token 较长或数字一致。
    """
    la, lb = a.lower(), b.lower()
    # 规则1：数字集合不同
    if sorted(_NUM_RE.findall(la)) != sorted(_NUM_RE.findall(lb)):
        return True
    # 规则2：token 对称差异仅为短序号
    diff = Counter(_ALNUM_RE.findall(la)) - Counter(_ALNUM_RE.findall(lb))
    diff += Counter(_ALNUM_RE.findall(lb)) - Counter(_ALNUM_RE.findall(la))
    toks = list(diff.elements())
    if toks and all(len(t) <= 2 or _ROMAN_RE.match(t) for t in toks):
        return True
    return False


# 缩写判定时忽略的虚词
_ACRONYM_STOP = {"and", "of", "the", "for", "a", "an", "to", "in", "on", "&"}
_NAME_SPLIT = re.compile(r"[\s\-/]+")


def _acronym_of(short: str, long_: str) -> bool:
    """short 是否是 long_ 的首字母缩写（忽略 and/of/the 等虚词）。

    例：'SEC' 是 'Securities and Exchange Commission' 的缩写。
    """
    words = [w for w in _NAME_SPLIT.split(long_.lower()) if w and w not in _ACRONYM_STOP]
    initials = "".join(w[0] for w in words)
    s = re.sub(r"[^a-z0-9]", "", short.lower())
    return len(s) >= 2 and len(words) >= 2 and s == initials


def _names_name_match(a: str, b: str) -> bool:
    """名字是否"沾边"（精度守卫 B）。接受四类真同义常见形态，其余拒绝：

      1. 一方包含另一方（'Hybrid Retrieval' ⊂ 'Hybrid Retrieval Mechanism'）；
      2. 互为首字母缩写（'SEC' ↔ 'Securities and Exchange Commission'）；
      3. 字符级高度相似 ratio≥0.85（单复数/拼写：'Rating'↔'Ratings'、'color'↔'colour'）。

    刻意【不】接受"仅共享一个词"——那会放行 'Net income' vs 'Operating income' 这类
    换词的不同概念（描述相似 → cos 高，但并非同义）。宁可漏连不可连错。
    """
    la, lb = a.lower().strip(), b.lower().strip()
    if not la or not lb:
        return False
    if la == lb or la in lb or lb in la:           # 1. 包含
        return True
    if _acronym_of(a, b) or _acronym_of(b, a):     # 2. 缩写
        return True
    if difflib.SequenceMatcher(None, la, lb).ratio() >= 0.85:  # 3. 字符高相似
        return True
    return False


def find_synonym_pairs(names, matrix, neighbor_of, tau, theta, sim=None,
                       exclude_enum=True, doc_of=None, require_same_doc=False,
                       max_per_node=0, type_of=None, skip_types=None,
                       require_same_type=False, require_name_match=False):
    """纯计算：返回合格的同义实体对，按余弦降序排列。

    Args:
        names          : list[str]，长度 N 的实体名（与 matrix 行一一对应）
        matrix         : np.ndarray (N, D)，实体 embedding
        neighbor_of    : dict[str, set[str]]，实体名 -> 邻居实体名集合（图结构快照）
        tau            : 余弦阈值（> tau 才算语义相似）
        theta          : Jaccard 阈值（> theta 才算结构相似）
        sim            : 可选，预先算好的 N×N 余弦矩阵（调参 sweep 时复用以省时）
        exclude_enum   : 是否启用枚举判别守卫（剔除"仅差数字/序号"的枚举假阳性），默认 True
        doc_of         : 可选，dict[str, set[str]]，实体名 -> 来源文档集合（文档作用域用）
        require_same_doc: True 时启用 Step3 规则1——两实体无任何共同来源文档则拒连
                          （仅当 doc_of 提供时生效；某一侧文档未知则放行，不因缺元数据误杀）
        max_per_node   : Step3 规则4——每个实体最多保留的同义边数（0=不限）。按 cos
                          降序贪心保留，超额者丢弃，用于抑制 hub 过连接。结果确定可复现。
        type_of        : 可选，dict[str, str]，实体名 -> entity_type（类型过滤/同类型用）
        skip_types     : 可选，set[str]，小写类型集合；两端任一实体属于其中则拒连
                          （仅当 type_of 提供时生效），治人名假阳性。
        require_same_type: 精度守卫 A——两端 entity_type 不同则拒连（需 type_of）。
        require_name_match: 精度守卫 B——名字不"沾边"（包含/缩写/字符高相似）则拒连。

    Returns:
        list[(name_i, name_j, cos, jaccard)]，按 cos 降序。
        会跳过：同名对、已互为邻居（已有边）的对、枚举变体对（exclude_enum=True 时）、
        指定类型对（skip_types 命中时）、跨文档对（require_same_doc=True 时）、
        超出每节点预算的对（max_per_node>0 时）。
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
        if jac <= theta:
            continue
        if exclude_enum and _is_enumeration_variant(ni, nj):
            continue  # 枚举变体（债券年份/页码/章节序号/子公司 I·II 等），拒绝
        # Step3 类型过滤：两端任一为指定类型（默认 person）-> 拒绝，治人名假阳性。
        if skip_types and type_of is not None:
            if (type_of.get(ni, "").lower() in skip_types
                    or type_of.get(nj, "").lower() in skip_types):
                continue
        # 精度守卫 A：要求同 entity_type（两端类型都已知且不同 -> 拒）。
        if require_same_type and type_of is not None:
            ti, tj = type_of.get(ni, ""), type_of.get(nj, "")
            if ti and tj and ti.lower() != tj.lower():
                continue
        # 精度守卫 B：名字须"沾边"（包含/缩写/字符高相似），否则拒。
        if require_name_match and not _names_name_match(ni, nj):
            continue
        # Step3 规则1：文档作用域。两侧文档都已知且不相交 -> 跨文档污染，拒绝。
        if require_same_doc and doc_of is not None:
            di, dj = doc_of.get(ni), doc_of.get(nj)
            if di and dj and di.isdisjoint(dj):
                continue
        pairs.append((ni, nj, float(sim[i, j]), jac))

    pairs.sort(key=lambda x: x[2], reverse=True)

    # Step3 规则4：每节点同义边预算。已按 cos 降序，贪心保留高分边，封住 hub。
    if max_per_node and max_per_node > 0:
        deg = {}
        kept = []
        for ni, nj, cos, jac in pairs:
            if deg.get(ni, 0) >= max_per_node or deg.get(nj, 0) >= max_per_node:
                continue
            kept.append((ni, nj, cos, jac))
            deg[ni] = deg.get(ni, 0) + 1
            deg[nj] = deg.get(nj, 0) + 1
        pairs = kept

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

    返回 (G, fnames, fmatrix, neighbor_of, doc_of, type_of)：
      - 只保留同时存在于图节点的实体（LightRAG 里节点 ID == entity_name）
      - neighbor_of 是【全图】的邻居快照（实体名 -> 邻居名集合）
      - doc_of 是每个实体的来源文档集合（从节点 file_path 属性按 GRAPH_FIELD_SEP 还原）
      - type_of 是每个实体的 entity_type（从节点属性读，供类型过滤用）
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
    doc_of = {
        nm: _split_docs(G.nodes[nm].get("file_path", "")) for nm in graph_names
    }
    type_of = {nm: str(G.nodes[nm].get("entity_type", "")) for nm in graph_names}
    return G, fnames, fmatrix, neighbor_of, doc_of, type_of


def add_synonym_edges(working_dir, tau=None, theta=None, force=False,
                      require_same_doc=None, max_per_node=None,
                      skip_types=None, carry_chunks=None,
                      require_same_type=None, require_name_match=None) -> int:
    """对 working_dir 下的知识图谱添加同义边（L2）。

    Args:
        working_dir     : LightRAG 存储目录（含 graphml 与 vdb_entities.json）
        tau / theta     : 阈值；None 时读 config（SYNONYM_TAU / SYNONYM_THETA）
        force           : True 时忽略 ENABLE_SYNONYM_EDGES 开关强制执行（供 CLI 用）
        require_same_doc: 文档作用域守卫；None 时读 config（SYNONYM_SAME_DOC）
        max_per_node    : 每节点同义边预算；None 时读 config（SYNONYM_MAX_PER_NODE）
        skip_types      : 类型过滤集合；None 时读 config（SYNONYM_SKIP_TYPES）
        carry_chunks    : 是否载货（source_id 设为两端真实 chunk 并集）；
                          None 时读 config（SYNONYM_CARRY_CHUNKS）

    Returns:
        新增同义边数量。开关关闭且非 force 时返回 0。
    """
    if not force and not is_synonym_edges_enabled():
        return 0

    tau = get_synonym_tau() if tau is None else tau
    theta = get_synonym_theta() if theta is None else theta
    if require_same_doc is None:
        require_same_doc = is_synonym_same_doc_enabled()
    if max_per_node is None:
        max_per_node = get_synonym_max_per_node()
    if skip_types is None:
        skip_types = get_synonym_skip_types()
    if carry_chunks is None:
        carry_chunks = is_synonym_carry_chunks_enabled()
    if require_same_type is None:
        require_same_type = is_synonym_require_same_type_enabled()
    if require_name_match is None:
        require_name_match = is_synonym_require_name_match_enabled()

    wd, gpath, vpath = _paths(working_dir)
    if not gpath.exists() or not vpath.exists():
        logger.warning("L2: missing graphml or vdb under %s, skip", wd)
        return 0

    G, fnames, fmatrix, neighbor_of, doc_of, type_of = _load_graph_and_neighbors(
        gpath, vpath
    )
    if len(fnames) < 2:
        return 0

    pairs = find_synonym_pairs(
        fnames, fmatrix, neighbor_of, tau, theta,
        exclude_enum=is_enum_filter_enabled(),
        doc_of=doc_of,
        require_same_doc=require_same_doc,
        max_per_node=max_per_node,
        type_of=type_of,
        skip_types=skip_types,
        require_same_type=require_same_type,
        require_name_match=require_name_match,
    )

    ts = int(time.time())
    added = 0
    for ni, nj, cos, jac in pairs:
        if G.has_edge(ni, nj):
            continue
        # 载货：source_id 设为两端真实 chunk 并集 → 检索到同义边时带进对端证据；
        # 关闭则用占位符（边 inert，仅作关系展示）。file_path 始终作移除/识别标记。
        if carry_chunks:
            chunks = _node_source_chunks(G, ni) | _node_source_chunks(G, nj)
            source_id = GRAPH_FIELD_SEP.join(sorted(chunks)) or _SYNONYM_SOURCE_ID
        else:
            source_id = _SYNONYM_SOURCE_ID
        G.add_edge(
            ni, nj,
            weight=round(cos, 4),
            # 有意义的关系描述（告诉 LLM 二者同指），优于纯 cos 元数据
            description=f"{ni} and {nj} refer to the same concept (synonym, cos={cos:.2f}).",
            keywords=_SYNONYM_KEYWORD,
            source_id=source_id,
            file_path=_SYNONYM_SOURCE_ID,
            created_at=ts,
            truncate="",
        )
        added += 1

    if added:
        nx.write_graphml(G, str(gpath))
    logger.info(
        "L2: tau=%.2f theta=%.2f same_doc=%s max_per_node=%d skip_types=%s carry=%s"
        " same_type=%s name_match=%s -> %d synonym edges%s",
        tau, theta, require_same_doc, max_per_node, sorted(skip_types) or "-",
        carry_chunks, require_same_type, require_name_match, added,
        f" written to {gpath}" if added else "",
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
        (u, v) for u, v, d in G.edges(data=True) if _is_synonym_edge(d)
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
    G, fnames, fmatrix, neighbor_of, doc_of, type_of = _load_graph_and_neighbors(
        gpath, vpath
    )
    if len(fnames) < 2:
        return {}
    sim = _cosine_sim_matrix(fmatrix)  # 只算一次，全网格复用
    enum = is_enum_filter_enabled()
    same_doc = is_synonym_same_doc_enabled()
    max_pn = get_synonym_max_per_node()
    skip_types = get_synonym_skip_types()
    same_type = is_synonym_require_same_type_enabled()
    name_match = is_synonym_require_name_match_enabled()
    out = {}
    for t in taus:
        for th in thetas:
            out[(t, th)] = len(
                find_synonym_pairs(
                    fnames, fmatrix, neighbor_of, t, th, sim=sim, exclude_enum=enum,
                    doc_of=doc_of, require_same_doc=same_doc, max_per_node=max_pn,
                    type_of=type_of, skip_types=skip_types,
                    require_same_type=same_type, require_name_match=name_match,
                )
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
    ap.add_argument("--cross-doc", action="store_true",
                    help="关闭文档作用域守卫，允许跨文档配对（用于消融）")
    ap.add_argument("--max-per-node", type=int, default=None,
                    help="每个实体最多新增的同义边数（0=不限；默认读 config）")
    ap.add_argument("--no-type-filter", action="store_true",
                    help="关闭类型过滤（默认跳过 person），允许人名参与配对（用于消融）")
    ap.add_argument("--carry-chunks", action="store_true",
                    help="载货：source_id 设为两端真实 chunk 并集，让同义边带进对端证据")
    ap.add_argument("--no-same-type", action="store_true",
                    help="关闭精度守卫A（同类型才连），用于消融")
    ap.add_argument("--no-name-match", action="store_true",
                    help="关闭精度守卫B（名字须沾边），用于消融")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.remove:
        print("removed synonym edges:", remove_synonym_edges(args.working_dir))
        return

    # 让 --cross-doc / --max-per-node / --no-type-filter 同样作用于 --dry-run
    # （sweep 经 config 读取守卫）
    if args.cross_doc:
        os.environ["SYNONYM_SAME_DOC"] = "false"
    if args.max_per_node is not None:
        os.environ["SYNONYM_MAX_PER_NODE"] = str(args.max_per_node)
    if args.no_type_filter:
        os.environ["SYNONYM_SKIP_TYPES"] = ""
    if args.no_same_type:
        os.environ["SYNONYM_REQUIRE_SAME_TYPE"] = "false"
    if args.no_name_match:
        os.environ["SYNONYM_REQUIRE_NAME_MATCH"] = "false"

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
    n = add_synonym_edges(
        args.working_dir, tau=args.tau, theta=args.theta, force=True,
        require_same_doc=(False if args.cross_doc else None),
        max_per_node=args.max_per_node,
        skip_types=(set() if args.no_type_filter else None),
        carry_chunks=(True if args.carry_chunks else None),
        require_same_type=(False if args.no_same_type else None),
        require_name_match=(False if args.no_name_match else None),
    )
    print("added synonym edges:", n)


if __name__ == "__main__":
    _main()
