"""图融合改进的开关与超参（用环境变量控制，便于消融，不改代码即可切换）。

开关：
- ENABLE_CANONICALIZATION：L1 名称规范化（默认 false = 原版行为）
- ENABLE_SYNONYM_EDGES：L2 同义边（默认 false）

L2 超参：
- SYNONYM_TAU：余弦相似度阈值（默认 0.85）
- SYNONYM_THETA：邻居 Jaccard 阈值（默认 0.10）

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
