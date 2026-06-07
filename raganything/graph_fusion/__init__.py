"""graph_fusion: 双图融合改进模块。

- L1 名称规范化（canonicalizer）：合并前万无一失地统一实体名书写形式。
- L2 同义边（synonym_linker）：建图后对相似实体加同义边（不合并节点）。

开关见 config.py（默认全关 = 原版 RAG-Anything 行为，便于消融）。
"""
from raganything.graph_fusion.canonicalizer import (
    normalize_entity_name,
    normalize_chunk_results,
)
from raganything.graph_fusion.config import (
    is_canonicalization_enabled,
    is_synonym_edges_enabled,
    get_synonym_tau,
    get_synonym_theta,
)
from raganything.graph_fusion.synonym_linker import (
    add_synonym_edges,
    remove_synonym_edges,
    sweep_thresholds,
    find_synonym_pairs,
)

__all__ = [
    "normalize_entity_name",
    "normalize_chunk_results",
    "is_canonicalization_enabled",
    "is_synonym_edges_enabled",
    "get_synonym_tau",
    "get_synonym_theta",
    "add_synonym_edges",
    "remove_synonym_edges",
    "sweep_thresholds",
    "find_synonym_pairs",
]
