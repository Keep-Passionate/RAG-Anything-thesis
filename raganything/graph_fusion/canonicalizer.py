"""L1 实体名称规范化（万无一失 / 零风险版）。

只做"绝不会把两个不同实体误并"的表面归一化：
  1. Unicode NFC 规范化（同一字符的不同编码统一）
  2. 去首尾空白
  3. 合并内部连续空白为单个空格
  4. 去结尾句读 . , ; :（但保护 + # * 等有意义符号，如 C++ / C#）

有风险的操作（小写、缩写展开、去重音、去停用词等）一律不在此处，
留给 L2 用 embedding + 邻居结构约束安全处理（加同义边而非合并）。

纯函数，便于单元测试与复用。
"""
import re
import unicodedata

# 只去这几种"句末标点"，绝不碰 + # * 等有语义的符号（保护 C++ / C# / F# 等）
_TRAILING_PUNCT = re.compile(r"[.,;:]+$")
_WHITESPACE = re.compile(r"\s+")


def normalize_entity_name(name):
    """把实体名做万无一失的规范化。

    例：" Figure 2. " 和 "Figure  2" -> "Figure 2"；"C++" 原样保留。
    """
    if name is None:
        return name
    s = unicodedata.normalize("NFC", str(name))   # 1. Unicode NFC
    s = s.strip()                                  # 2. 去首尾空白
    s = _WHITESPACE.sub(" ", s)                    # 3. 合并内部空白
    s = _TRAILING_PUNCT.sub("", s).strip()         # 4. 去结尾句读（保护 +#*）
    return s


def normalize_chunk_results(chunk_results):
    """对 extract_entities 返回的 [(maybe_nodes, maybe_edges), ...] 做名称规范化。

    - maybe_nodes: {实体名: [实体信息dict,...]} -> 规范化 key 及 dict 内 entity_name
    - maybe_edges: {(src,tgt): [关系信息dict,...]} -> 规范化 key 及 dict 内 src_id/tgt_id
    返回新列表，不原地修改输入。
    """
    new_results = []
    for maybe_nodes, maybe_edges in chunk_results:
        new_nodes = {}
        for ename, items in maybe_nodes.items():
            cname = normalize_entity_name(ename)
            fixed = []
            for it in items:
                it = dict(it)
                if "entity_name" in it:
                    it["entity_name"] = normalize_entity_name(it["entity_name"])
                fixed.append(it)
            new_nodes.setdefault(cname, []).extend(fixed)

        new_edges = {}
        for (src, tgt), items in maybe_edges.items():
            key = (normalize_entity_name(src), normalize_entity_name(tgt))
            fixed = []
            for it in items:
                it = dict(it)
                if "src_id" in it:
                    it["src_id"] = normalize_entity_name(it["src_id"])
                if "tgt_id" in it:
                    it["tgt_id"] = normalize_entity_name(it["tgt_id"])
                fixed.append(it)
            new_edges.setdefault(key, []).extend(fixed)

        new_results.append((new_nodes, new_edges))
    return new_results
