"""图融合改进的开关（用环境变量控制，便于消融，不改代码即可切换）。

- ENABLE_CANONICALIZATION：L1 名称规范化（默认 false = 原版行为）
- ENABLE_SYNONYM_EDGES：L2 同义边（默认 false，预留给 L2）

消融时只需在 .env / 环境变量里改这两个开关，跑出 baseline / +L1 / +L2 / full。
"""
import os


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def is_canonicalization_enabled():
    """L1 是否开启（读环境变量 ENABLE_CANONICALIZATION，默认关）。"""
    return _truthy(os.getenv("ENABLE_CANONICALIZATION", "false"))


def is_synonym_edges_enabled():
    """L2 是否开启（读环境变量 ENABLE_SYNONYM_EDGES，默认关）。预留。"""
    return _truthy(os.getenv("ENABLE_SYNONYM_EDGES", "false"))
