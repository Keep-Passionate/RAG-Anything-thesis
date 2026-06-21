"""doc_operators.py —— 确定性接地的【工具库】重构（与现有 doc_meta/doc_locate 并行存在）。

把散落在 query.py 里的 if/elif 路由 + doc_meta/doc_locate 的函数，重构成
【统一接口的算子注册表 + 一个调度器】，演示 L1/L2/L3 三层解耦：

    算子 Operator = (name, desc, detect[L1], run[L2+L3])
    注册表 OPERATORS（按优先级）
    调度器 dispatch(question, ctx) -> (注入文本, 命中算子名列表)

设计要点：
- 【不改变任何现有行为】：算子的实现【复用】doc_meta/doc_locate 的现成函数；
  query.py 暂不接它（保留原关键词路由）。这是"重建一版、并行存在、A/B 后择优"的并行实现。
- L1 路由现仍是关键词（每个算子自带 detect）；将来把 dispatch 换成【神经路由】
  （function-calling：把各算子的 .desc 喂给 LLM 选调哪个）即可，L2/L3 一行不动。
- 与 query.py 的等价性：mention/element/meta 三选一（互斥、优先级递减），locate 可叠加。
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import doc_locate as dl
import doc_meta as dm


@dataclass
class Ctx:
    """阶段A 每篇文档只算一次的缓存；kw_visual/kw_table 是【每题】的视觉意图，调度前由调用方刷新。"""
    pdf_path: str
    doc_stats: Optional[dict] = None   # dm.compute_doc_stats(pdf) 的结果（页/词/缩写/图表/_text）
    locate_index: str = ""             # dl.build_heading_page_index(pdf) 的结果
    kw_visual: bool = False
    kw_table: bool = False


@dataclass
class Operator:
    name: str
    desc: str                          # 给（未来）神经路由 / function-calling 看的自描述
    detect: Callable                   # L1: (question, ctx) -> bool
    run: Callable                      # L2+L3: (question, ctx) -> 注入文本 str（""=放弃）
    stackable: bool = False            # True=可与统计类叠加（locate）；False=统计类三选一


# ---------------------------------------------------------------------------
# 各算子的 L1 detect + L2/L3 run —— 全部【复用】 doc_meta / doc_locate，不重写逻辑
# ---------------------------------------------------------------------------

def _mention_detect(q, ctx):
    return bool(ctx.doc_stats) and dm.detect_mention_count_intent(q) \
        and dm.extract_mention_target(q) is not None


def _mention_run(q, ctx):
    target = dm.extract_mention_target(q)
    n = dm.count_mentions(ctx.doc_stats.get("_text", ""), target)
    return (f'[Programmatically verified: the phrase "{target}" appears '
            f"{n} times in the document text.]")


def _elemcount_detect(q, ctx):
    return bool(ctx.doc_stats) and dm.detect_count_intent(q)


def _meta_detect(q, ctx):
    return bool(ctx.doc_stats) and dm.detect_meta_intent(q) \
        and not (ctx.kw_visual or ctx.kw_table)


def _stats_run(q, ctx):
    """element_count 与 meta_stats 共用：拼统计量说明 + 点名页 + 相对页（末页/前N页，自门控）。"""
    note = dm.format_stats_note(ctx.doc_stats)
    page_no = dm.find_page_reference(q)
    if page_no:
        snip = dm.extract_page_text(ctx.pdf_path, page_no)
        if snip:
            note += (f"\n[Beginning of page {page_no}, extracted "
                     f"programmatically: {snip}]")
    rel = dm.relative_page_note(ctx.pdf_path, q, (ctx.doc_stats or {}).get("pages"))
    if rel:
        note += "\n" + rel
    return note


def _locate_detect(q, ctx):
    # 有标题索引就触发；内容定位器开时即便没标题索引也触发（让它去搜正文/图表标题）
    return dl.detect_location_intent(q) and (bool(ctx.locate_index) or dl._content_locate_on())


def _locate_run(q, ctx):
    note = dl.format_locate_note(ctx.locate_index)          # 标题→页码（+ENABLE_LOCATE_ELEMENTS 的表/图）
    extra = dl.content_locate_note(ctx.pdf_path, q)         # 内容定位器（自门控 ENABLE_LOCATE_CONTENT）
    if extra:
        note = (note + "\n" + extra) if note else extra
    return note


# ---------------------------------------------------------------------------
# 注册表（顺序 = 优先级，与 query.py 的 if/elif 一致）
# ---------------------------------------------------------------------------
OPERATORS: List[Operator] = [
    Operator("mention_count", "数某个词/短语在全文出现多少次", _mention_detect, _mention_run),
    Operator("element_count", "数文档里有几张图/表/公式（可排除附录）", _elemcount_detect, _stats_run),
    Operator("meta_stats", "页数/词数/最高频词/最常见缩写/某页内容等全局统计量",
             _meta_detect, _stats_run),
    Operator("locate", "某章节/内容在第几页（标题→页码索引）",
             _locate_detect, _locate_run, stackable=True),
]


def dispatch(question: str, ctx: Ctx, operators=None):
    """按优先级跑注册表，返回 (拼好的注入文本, 命中算子名列表)。

    统计类（非 stackable）三选一取第一个命中；stackable 的（locate）独立叠加。
    与 query.py 现有路由等价。operators 可传自定义算子集（默认全局 OPERATORS）。
    """
    ops = operators if operators is not None else OPERATORS
    notes, fired = [], []
    statistics_done = False
    for op in ops:
        if op.stackable:
            if op.detect(question, ctx):
                notes.append(op.run(question, ctx))
                fired.append(op.name)
        elif not statistics_done and op.detect(question, ctx):
            notes.append(op.run(question, ctx))
            fired.append(op.name)
            statistics_done = True
    return "\n\n".join(n for n in notes if n), fired


# 扩展算子（语料级，未实现，留作大论文按同一接口注册）：
#   Operator("extremum",  "跨文档最大/最小（哪份报告页数最多）", ...)
#   Operator("sort",      "按属性排序（按页数给报告排序）",       ...)
#   Operator("topk",      "取前 k（引用最高的 10 篇）",          ...)
# —— 与 GlobalRAG 的 Counting/Extremum/Sorting/Top-k 对接；加它们 = 往注册表再 append，框架不变。


class Augmenter:
    """系统无关的【问题增强器】：给一个文档(PDF) + 问题，返回"要拼进生成提示的已验证事实"。

    可插入【任意 RAG / 图 RAG】（HippoRAG / GraphRAG / LightRAG …）——**不碰它们的检索与建图**，
    只在它们"生成答案前"对问题做增强。用法：

        aug = Augmenter()
        fact, fired = aug.augment(question, pdf_path)          # fact 可能为 ""
        prompt = question + (("\\n\\n" + fact) if fact else "")
        answer = your_graph_rag.generate(prompt, ...)          # 交给对方系统照常生成

    每篇文档只算一次（缓存）。operators_spec() 返回自描述算子清单，供未来 function-calling 神经路由。
    """

    def __init__(self, operators=None):
        self.operators = operators if operators is not None else OPERATORS
        self._cache = {}  # pdf_path -> Ctx（每篇只算一次）

    def _ctx(self, pdf_path: str, kw_visual: bool = False, kw_table: bool = False) -> Ctx:
        if pdf_path not in self._cache:
            self._cache[pdf_path] = build_ctx(pdf_path)
        c = self._cache[pdf_path]
        c.kw_visual, c.kw_table = kw_visual, kw_table
        return c

    def augment(self, question: str, pdf_path: str,
                kw_visual: bool = False, kw_table: bool = False):
        """返回 (注入事实文本, 命中算子名列表)。注入文本为 '' 表示这题我们不处理（=原系统行为）。"""
        return dispatch(question, self._ctx(pdf_path, kw_visual, kw_table), self.operators)

    def operators_spec(self):
        """自描述算子清单（给神经路由 / function-calling）。"""
        return [{"name": op.name, "description": op.desc, "stackable": op.stackable}
                for op in self.operators]


def build_ctx(pdf_path: str, kw_visual: bool = False, kw_table: bool = False) -> Ctx:
    """阶段A：每篇文档只算一次（贵活）。kw_* 可调度前每题再刷新。"""
    return Ctx(
        pdf_path=pdf_path,
        doc_stats=dm.compute_doc_stats(pdf_path),
        locate_index=dl.build_heading_page_index(pdf_path),
        kw_visual=kw_visual,
        kw_table=kw_table,
    )


if __name__ == "__main__":
    print("确定性接地 · 算子注册表（L1 detect / L2+L3 run，复用 doc_meta/doc_locate）：")
    for op in OPERATORS:
        tag = "（可叠加）" if op.stackable else "（统计类三选一）"
        print(f"  - {op.name:14s}{tag}: {op.desc}")
    print("\n调度优先级：mention_count > element_count > meta_stats（三选一）；locate 独立叠加。")
    print("用法：ctx = build_ctx(pdf); ctx.kw_visual,ctx.kw_table = detect_visual_intent(q); "
          "note, fired = dispatch(q, ctx)")
    print("未来：把 dispatch 换成神经路由（function-calling，喂各算子 .desc），L2/L3 不动。")
