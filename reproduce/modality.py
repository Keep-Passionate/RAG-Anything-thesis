"""模态路由：决定一道题该不该让 VLM 看图、以及重排时给视觉证据保位。

背景（为什么需要三级信号）：
  用户并不总知道答案藏在图里还是文字里。三种触发方式各有盲区：
  - ENABLE_VLM 全开：raganything 的 aquery_vlm_enhanced 内部本就"证据侧回退"
    （上下文无有效图→自动退回文本作答），所以全开的真实代价不是每题看图，而是
    **上下文里只要混进任何图，整题就换 VLM 模型作答**——纯文本题被换模型=噪声。
  - ENABLE_MODALITY_VLM 关键词（问题侧）：精度高零成本，但漏掉"题面不提图、
    答案却在图里"的题（如 "Which country showed the largest increase in stores"）。
  - ENABLE_AUTO_VLM 证据侧（本模块支持）：关键词没命中时，看检索回来的上下文里
    有没有图片证据（与 raganything 相同的 Image Path 标记），有才开 VLM。
    = 关键词(精度) ∪ 证据(召回) 的并集路由。

重排视觉保位（RERANK_VISUAL_GUARD）：
  实测 rerank 在表格题上持续 -2~3 题（mm-t 86%→81%）：重排器按"查询-文本相关度"
  打分，表格/图片块的文字形态（HTML/管道符/路径）天然吃亏，被挤出 chunk_top_k
  截断线。守卫逻辑：当题面有图/表意图、且截断后视觉/表格块不足 K 个时，把截断线
  下得分最高的视觉块提升进来（替换队尾的文本块）。纯文本题完全不受影响。

本模块只有纯函数（可单测）；与 LightRAG/raganything 的接线在 query.py。
"""

import re

# ---------------------------------------------------------------------------
# 问题侧：关键词意图（零成本）
# ---------------------------------------------------------------------------

_VISUAL_KWS = (
    "figure", "fig.", "fig ", "chart", "plot", "image", "photo", "picture",
    "diagram", "graph", "panel", "axis", "legend", "curve", "shown in",
    "图", "图表", "图中", "如图", "曲线", "示意", "照片", "插图",
)
_TABLE_KWS = (
    "table", "row", "column", "cell", "spreadsheet",
    "表", "表格", "表中", "单元格", "列", "行",
)


def detect_visual_intent(question: str):
    """纯关键词判断问题是否需要看图/表。返回 (wants_visual, wants_table)。"""
    q = (question or "").lower()
    wants_table = any(k in q for k in _TABLE_KWS)
    wants_visual = any(k in q for k in _VISUAL_KWS)
    return wants_visual, wants_table


# ---------------------------------------------------------------------------
# 证据侧：检索上下文里有没有视觉证据
# ---------------------------------------------------------------------------

# 与 raganything/query.py 的 _process_image_paths_for_vlm 同一标记格式：
# 上下文里的图片块带 "Image Path: xxx.jpg" 行，VLM 增强正是靠它取图。
_IMAGE_MARKER_RE = re.compile(
    r"Image Path:\s*[^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)", re.IGNORECASE
)


def count_image_evidence(context: str) -> int:
    """统计检索上下文里的图片路径标记数（=VLM 实际取得到的图数上限）。"""
    return len(_IMAGE_MARKER_RE.findall(context or ""))


def is_visual_chunk(text: str) -> bool:
    """判断一个候选块是否是视觉/表格证据（重排保位用的启发式）。

    图：带 Image Path 标记。表：HTML 表标签，或管道符密集（markdown 表），
    或带 Table Caption。误判代价低——只在"题面本来就问图/表"时才用它保位。
    """
    t = (text or "")
    if _IMAGE_MARKER_RE.search(t):
        return True
    tl = t.lower()
    if "<table" in tl or "<td" in tl or "table caption" in tl:
        return True
    return t.count("|") >= 6  # markdown 表至少两行×三列的管道符密度


# ---------------------------------------------------------------------------
# 重排视觉保位（纯函数；async 接线在 query.py）
# ---------------------------------------------------------------------------

def guard_rerank_results(results, documents, query, top_n, min_visual=2):
    """对重排打分结果施加"视觉保位"，返回截断到 top_n 的结果列表。

    Args:
        results   : 重排器返回的全量 [{"index": int, "relevance_score": float}]，按分降序
        documents : 与 index 对应的候选块文本列表
        query     : 用户问题（只有它带图/表意图时才保位）
        top_n     : 截断条数（None = 不截断，原样返回）
        min_visual: 截断后至少保留的视觉/表格块数

    逻辑：题面有图/表意图、且 top_n 内视觉块不足 min_visual 时，把截断线下
    得分最高的视觉块提进来，替换队尾（得分最低）的非视觉块；顺序其余不变。
    纯文本题原样返回——守卫对它们零影响。
    """
    if top_n is None or len(results) <= top_n:
        return results if top_n is None else results[:top_n]

    kw_visual, kw_table = detect_visual_intent(query)
    if not (kw_visual or kw_table):
        return results[:top_n]

    head, tail = list(results[:top_n]), results[top_n:]

    def _vis(r):
        i = r.get("index", -1)
        return 0 <= i < len(documents) and is_visual_chunk(documents[i])

    have = sum(1 for r in head if _vis(r))
    promos = [r for r in tail if _vis(r)][: max(0, min_visual - have)]
    if not promos:
        return head

    # 从队尾往前找非视觉块的位置换出（保住高分文本块，牺牲低分文本块）
    slots = [i for i in range(len(head) - 1, -1, -1) if not _vis(head[i])][: len(promos)]
    for slot, promo in zip(slots, promos):
        head[slot] = promo
    return head
