"""图融合改进的开关与超参（用环境变量控制，便于消融，不改代码即可切换）。

开关：
- ENABLE_CANONICALIZATION：L1 名称规范化（默认 false = 原版行为）
- ENABLE_SYNONYM_EDGES：L2 同义边（默认 false）

L2 超参：
- SYNONYM_TAU：余弦相似度阈值（默认 0.85）
- SYNONYM_THETA：邻居 Jaccard 阈值（默认 0.10）

L2 防过连接守卫（Step3）：
- SYNONYM_SAME_DOC：文档作用域（默认 true）——两实体若无任何共同来源文档则拒连。
  仅对"多篇文档合并在同一张图"的情形有效；每篇独立索引时图内实体同属一文档，
  此守卫为空操作（无副作用）。设 false 可做"跨文档 vs 同文档"对照消融。
- SYNONYM_MAX_PER_NODE：每节点同义边预算（默认 0=不限）——每个实体最多新增 K 条
  同义边（按余弦取最高的 K 条），用于封住通用词/hub 的过连接。单篇图里治过连接
  主要靠它，建议用 reproduce/l2_sweep.py 配合扫 K。
- SYNONYM_SKIP_TYPES：按实体类型过滤（默认 "person"）——两端任一实体属于这些
  entity_type 则拒连。专治人名假阳性（同姓不同人 cos+Jaccard 双高），比调阈值治本
  （真同义与人名假阳性的 cos 同样高，阈值分不开）。置空可关闭做消融。
- SYNONYM_CARRY_CHUNKS：同义边是否"载货"（默认 false）。诊断发现：默认实现里同义边
  source_id 是占位符 → 检索读到边却拉不进真实文本块（"通电未载货"）。开启后把 source_id
  设为两端实体真实 chunk 并集，检索到同义边时真正带进对端证据。⚠️ 这会同时放大真/假
  同义边——务必先用 show_synonyms 确认精度（≥90%）再开，否则放大假阳性反而掉分。
  默认关 = 现有 inert 行为，便于 inert vs 载货 干净消融。

消融时只需在 .env / 环境变量里改这些值，跑出 baseline / +L1 / +L2 / full：
    baseline : ENABLE_CANONICALIZATION=false ENABLE_SYNONYM_EDGES=false
    +L1      : ENABLE_CANONICALIZATION=true  ENABLE_SYNONYM_EDGES=false
    +L2      : ENABLE_CANONICALIZATION=false ENABLE_SYNONYM_EDGES=true
    full     : ENABLE_CANONICALIZATION=true  ENABLE_SYNONYM_EDGES=true
"""
import os


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def is_canonicalization_enabled():
    """L1 是否开启（读环境变量 ENABLE_CANONICALIZATION，默认关）。"""
    return _truthy(os.getenv("ENABLE_CANONICALIZATION", "false"))


def is_synonym_edges_enabled():
    """L2 是否开启（读环境变量 ENABLE_SYNONYM_EDGES，默认关）。"""
    return _truthy(os.getenv("ENABLE_SYNONYM_EDGES", "false"))


def get_synonym_tau():
    """L2 余弦相似度阈值（读 SYNONYM_TAU，默认 0.85）。"""
    return float(os.getenv("SYNONYM_TAU", "0.85"))


def get_synonym_theta():
    """L2 邻居 Jaccard 阈值（读 SYNONYM_THETA，默认 0.10）。"""
    return float(os.getenv("SYNONYM_THETA", "0.10"))


def is_enum_filter_enabled():
    """L2 枚举判别守卫是否开启（读 SYNONYM_FILTER_ENUM，默认开）。

    开启后剔除"仅靠数字/序号区分的不同条目"假阳性（债券年份、页码、章节序号、
    子公司 I/II 等）。设为 false 可做"无守卫 vs 有守卫"的对照消融。
    """
    return _truthy(os.getenv("SYNONYM_FILTER_ENUM", "true"))


def is_synonym_same_doc_enabled():
    """L2 文档作用域守卫是否开启（读 SYNONYM_SAME_DOC，默认开）。

    开启后：两实体若没有任何共同来源文档（节点 file_path 集合不相交）则拒绝加同义边，
    根除"跨文档同名/近义污染"。注意——每篇独立索引时图内实体同属一篇，此守卫为空操作。
    设为 false 可恢复旧版 L2 行为，做"跨文档 vs 同文档"对照消融。
    """
    return _truthy(os.getenv("SYNONYM_SAME_DOC", "true"))


def get_synonym_max_per_node():
    """L2 每节点同义边预算（读 SYNONYM_MAX_PER_NODE，默认 0=不限）。

    >0 时每个实体最多新增该条数的同义边（按余弦降序贪心保留最高的几条），
    用于抑制通用词/hub 的过连接。单篇图里这是治"过连接→效率降低"的主力旋钮。
    """
    return int(os.getenv("SYNONYM_MAX_PER_NODE", "0"))


def get_synonym_skip_types():
    """L2 实体类型过滤（读 SYNONYM_SKIP_TYPES，默认 "person"）。

    返回小写 entity_type 集合；两端任一实体属于其中则拒绝加同义边。
    专治人名假阳性（'Weizhi Zhang'↔'Weizhi Chen' 同姓不同人）。置空字符串可关闭。
    """
    raw = os.getenv("SYNONYM_SKIP_TYPES", "person")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def is_synonym_carry_chunks_enabled():
    """L2 同义边是否"载货"（读 SYNONYM_CARRY_CHUNKS，默认关）。

    关：source_id 用占位符，边 inert（仅作关系展示，拉不进真实证据）= 现有行为。
    开：source_id 设为两端实体真实 chunk 并集，检索到同义边时带进对端证据。
    ⚠️ 会放大真/假同义边，开前务必先确认精度（见 reproduce/show_synonyms.py）。
    """
    return _truthy(os.getenv("SYNONYM_CARRY_CHUNKS", "false"))
