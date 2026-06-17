#!/usr/bin/env python
"""DocBench 查询脚本：读 <id>_qa.jsonl 逐题作答，结果存 RESULT_NAME 指定的 JSON。

用法：python reproduce/query.py <pdf> --working_dir <该篇文档自己的索引目录>
（索引必须已由 index.py 建好；本脚本只做查询，不烧建图的钱。）

查询期开关（默认全关 = 原版 baseline，逐个打开做消融）：
  ENABLE_VLM              : 每题都让 VLM 看检索到的图（全开，贵且对纯文本题加噪）
  ENABLE_MODALITY_VLM     : 只对"问图/表"的题开 VLM（关键词检测，零成本，精准投放）
  ENABLE_AUTO_VLM         : 【旧EMR,已证有害弃用】生成端证据路由,选择性看图反而掉分
                            (54篇实测 -6.3pp;全开VLM是强基线,省钱式选择性看图漏题)
  ENABLE_EMR              : 【新EMR】检索端模态增强——视觉意图题加大 chunk_top_k
                            (EMR_CHUNK_TOP_K默认40)多捞图/表证据,保持全开VLM不变,
                            治"对的图表没被检索到"(论文 A.5 text-centric bias)
  ENABLE_DOC_META         : meta题注入程序化统计量(页数/词数/高频词/缩写/图表数/
                            关键词计数v4"X出现几次"/某页内容,见 doc_meta.py)
  ENABLE_NCG              : 表格数值计算接地——计算类题让模型抽数+列式、程序执行
                            (Program-of-Thoughts,治财报算差值/百分比题,见 ncg.py)
  ENABLE_DOC_OUTLINE      : 文档结构接地——结构/前页题注入章节树/首页footer
                            (治"发表在哪/几节/某节讲什么",与DSG独立,见 doc_outline.py)
  ENABLE_DOC_ANCHOR       : 被点名元素接地——"问某张具体表/图"的题(according to Table 8)
                            按标号确定性定位该元素并注入其内容(与DSG独立,见 doc_anchor.py)
  ENABLE_DOC_LOCATE       : 页码定位接地(Localization)——"X在第几页"的题注入
                            "章节标题→页码"索引,把定位从猜变查表(与DSG独立,见 doc_locate.py)
  ENABLE_RERANK           : DashScope gte-rerank 重排检索结果
  RERANK_VISUAL_GUARD     : 重排视觉保位——问图/表的题保证截断后至少留
                            RERANK_GUARD_SLOTS(默认2) 个视觉/表格块（治 rerank 丢表题）
  ENABLE_RETRIEVAL_REFLECT: R2 检索后充分性自检，不够则补检索（只补证据、绝不改答案）
  SAVE_CONTEXT            : 把检索上下文一并存进结果，供 diag_recall.py 离线体检
模型/温度等共用配置见 common.py（LLM_TEMPERATURE 默认 0 = 可复现）。
模态路由/保位的纯逻辑在 modality.py（可单测）。
"""

import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from common import build_arg_parser, build_model_funcs, configure_logging, env_on  # noqa: E402
from doc_meta import (  # noqa: E402
    compute_doc_stats,
    count_mentions,
    detect_count_intent,
    detect_mention_count_intent,
    detect_meta_intent,
    extract_mention_target,
    extract_page_text,
    find_page_reference,
    format_stats_note,
)
from ncg import (  # noqa: E402
    build_extraction_prompt,
    detect_calc_intent,
    format_calc_note,
    load_doc_tables,
    parse_ncg_json,
    safe_eval,
)
from doc_outline import (  # noqa: E402
    detect_structure_intent,
    format_outline_note,
    load_outline,
)
from doc_anchor import (  # noqa: E402
    detect_element_reference,
    find_referenced_element,
    format_anchor_note,
    load_content_items,
)
from doc_locate import (  # noqa: E402
    build_heading_page_index,
    detect_location_intent,
    format_locate_note,
)
from modality import count_image_evidence, detect_visual_intent, guard_rerank_results  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from lightrag.utils import logger  # noqa: E402
from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

# ---------------------------------------------------------------------------
# R2：检索后充分性自检 + 触发补检索（唯一保留的反思机制）。只判"检索够不够答"，
# 不够就触发补检索（更大 top_k，缺视觉时再开 VLM 看图），绝不改写已生成的答案——
# 因此下行有底，不会把对的改坏。
# （已删除答案重写式自反思：方案A 实测暴跌-29，方案C answer-conditioned 易自我
#  强化错误，均无正向收益；负向发现已存结果 JSON 与论文，代码不再保留。）
# ---------------------------------------------------------------------------

RETRIEVAL_SUFFICIENCY_PROMPT = """You are judging ONLY whether the retrieved context already contains the facts needed to answer the question. Do NOT answer the question.

Question: {question}

Retrieved context:
{context}

Reply with EXACTLY one token:
- SUFFICIENT            (the needed fact is clearly present)
- INSUFFICIENT_VISUAL   (the needed fact is likely in a figure/chart/photo, not present here)
- INSUFFICIENT_TABLE    (the needed fact is likely in a numeric table, not present here)
- INSUFFICIENT_OTHER    (the needed fact is absent for another reason)

Be conservative: default to SUFFICIENT unless the specific fact the question asks for is clearly missing from the context above."""


async def retrieval_sufficiency_check(llm_func, question, context, max_context_chars=8000):
    """检索后充分性自检：只判"够不够答"，不答题。

    返回 (verdict, need_more, want_visual)：
      - need_more  : 是否需要补检索（verdict 以 INSUFFICIENT 开头）
      - want_visual: 缺的信息是否疑似在图/表里（触发补检索时决定要不要开 VLM）
    """
    ctx = (context or "")[:max_context_chars]
    verdict = (
        await llm_func(
            RETRIEVAL_SUFFICIENCY_PROMPT.format(question=question, context=ctx),
            temperature=0,  # 判定类调用固定温度，保证可复现
        )
        or ""
    ).strip().upper()
    need_more = verdict.startswith("INSUFFICIENT")
    want_visual = ("VISUAL" in verdict) or ("TABLE" in verdict)
    return verdict, need_more, want_visual


# ---------------------------------------------------------------------------
# 主流程（模态意图/证据检测已抽到 modality.py，可单测）
# ---------------------------------------------------------------------------

def _load_queries(file_path: str):
    """读取与 PDF 同目录的 <id>_qa.jsonl。返回 [{question, answer}]；缺文件返回 []。"""
    folder = os.path.dirname(file_path)
    qa_path = os.path.join(folder, f"{os.path.basename(folder)}_qa.jsonl")
    if not os.path.exists(qa_path):
        logger.warning(f"QA file not found: {qa_path}")
        return []
    queries = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                queries.append({"question": d["question"], "answer": d.get("answer", "")})
    return queries


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """对已建好索引的文档逐题作答并保存结果。"""
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        llm_model_func, vision_model_func, embedding_func = build_model_funcs(
            api_key, base_url
        )

        # Reranker：默认关（= baseline）。开启用 DashScope gte-rerank（百炼 key 复用）。
        # RERANK_VISUAL_GUARD：实测 rerank 在表格题上持续 -2~3 题（视觉/表格块的文字
        # 形态打分吃亏被挤出截断线）。守卫=问图/表的题保证截断后至少留 K 个视觉块；
        # 纯文本题不受影响。逻辑在 modality.guard_rerank_results（纯函数）。
        rerank_model_func = None
        if env_on("ENABLE_RERANK"):
            from lightrag.rerank import ali_rerank

            base_rerank = partial(
                ali_rerank,
                model=os.getenv("RERANK_MODEL", "gte-rerank-v2"),
                api_key=os.getenv("RERANK_BINDING_API_KEY") or api_key,
            )
            if env_on("RERANK_VISUAL_GUARD"):
                guard_slots = int(os.getenv("RERANK_GUARD_SLOTS", "2"))

                async def rerank_model_func(query, documents, top_n=None, **kw):
                    # 全量打分（不让 API 截断），守卫后自己截断到 top_n
                    scored = await base_rerank(
                        query=query, documents=documents, top_n=None, **kw
                    )
                    return guard_rerank_results(
                        scored, documents, query, top_n, min_visual=guard_slots
                    )

                logger.info(
                    "Reranker ENABLED: ali_rerank + visual guard (slots=%d)", guard_slots
                )
            else:
                rerank_model_func = base_rerank
                logger.info("Reranker ENABLED: ali_rerank / gte-rerank-v2")

        lightrag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            enable_llm_cache=False,
            rerank_model_func=rerank_model_func,
        )
        await lightrag.initialize_storages()

        rag = RAGAnything(
            config=config,
            lightrag=lightrag,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        queries = _load_queries(file_path)
        if not queries:
            return

        # ---- 查询期开关（默认全关 = baseline）----
        vlm_on = env_on("ENABLE_VLM")                      # 每题全开 VLM
        modality_vlm_on = env_on("ENABLE_MODALITY_VLM")    # 仅图/表意图题开 VLM
        auto_vlm_on = env_on("ENABLE_AUTO_VLM")            # 【旧EMR,已证有害弃用】生成端证据路由
        auto_vlm_min = int(os.getenv("AUTO_VLM_MIN_IMAGES", "1"))
        # 新 EMR（检索端模态增强）：旧 EMR(AUTO_VLM)在生成端"选择性看图"反而掉分(全开VLM
        # 是强基线)。新 EMR 不动看图、保持全开，改在【检索端】——视觉意图题加大 chunk_top_k
        # 多捞图/表证据，治"对的图表没被检索到"(论文 A.5 text-centric bias)。
        emr_on = env_on("ENABLE_EMR")
        emr_chunk_top_k = int(os.getenv("EMR_CHUNK_TOP_K", "40"))  # 默认 2×（chunk_top_k=20）
        ncg_on = env_on("ENABLE_NCG")                      # 表格数值计算接地
        save_ctx = env_on("SAVE_CONTEXT")                  # 存检索上下文供体检
        rr_on = env_on("ENABLE_RETRIEVAL_REFLECT")         # R2 补检索
        rr_top_k = int(os.getenv("RR_TOP_K", "80"))            # 默认 2×（top_k=40）
        rr_chunk_top_k = int(os.getenv("RR_CHUNK_TOP_K", "40"))  # 默认 2×（chunk_top_k=20）

        # 文档统计量（ENABLE_DOC_META）：整篇只算一次，meta 题注入"已验证统计量"。
        doc_stats = None
        if env_on("ENABLE_DOC_META"):
            doc_stats = compute_doc_stats(file_path)
            if doc_stats:
                logger.info(
                    "Doc meta ENABLED: pages=%s words~%s",
                    doc_stats["pages"], doc_stats["words"],
                )
            else:
                logger.warning("ENABLE_DOC_META: failed to read PDF stats, notes disabled")

        # 文档结构（ENABLE_DOC_OUTLINE，与 DSG 平行、独立）：整篇只读一次 content_list，
        # 结构/前页题注入"提取的结构信息"（章节树 / 首页 footer）。
        outline = None
        if env_on("ENABLE_DOC_OUTLINE"):
            outline = load_outline(file_path)   # 前页正文字符串（content_list 前1~2页）
            if outline:
                logger.info("Doc outline ENABLED: front matter %d chars", len(outline))
            else:
                logger.warning("ENABLE_DOC_OUTLINE: no content_list/front matter, notes disabled")

        # 被点名元素（ENABLE_DOC_ANCHOR，与 DSG/outline 平行、独立）：整篇只读一次
        # content_list，"问某张具体表/图"的题按标号定位该元素、把内容注入（见 doc_anchor.py）。
        anchor_items = None
        if env_on("ENABLE_DOC_ANCHOR"):
            anchor_items = load_content_items(file_path)
            if anchor_items:
                logger.info("Doc anchor ENABLED: %d content_list items", len(anchor_items))
            else:
                logger.warning("ENABLE_DOC_ANCHOR: no content_list, notes disabled")

        # 页码定位（ENABLE_DOC_LOCATE，Localization 类，独立模块）：整篇只读一次 content_list，
        # 构造"章节标题→页码"索引，"X 在第几页"的题注入该索引、把定位从猜变查表。
        locate_index = None
        if env_on("ENABLE_DOC_LOCATE"):
            locate_index = build_heading_page_index(file_path)
            if locate_index:
                logger.info("Doc locate ENABLED: heading-page index %d chars", len(locate_index))
            else:
                logger.warning("ENABLE_DOC_LOCATE: no content_list/headings, notes disabled")

        results = []
        for query in queries:
            q = query["question"]

            # 本题是否开 VLM：全开 > 关键词（问题侧，零成本）> 证据侧（AUTO_VLM，
            # 关键词没命中时再看检索上下文里有没有图片证据，见下方触发点）
            kw_visual, kw_table = detect_visual_intent(q)
            q_vlm = vlm_on or ((modality_vlm_on or auto_vlm_on) and (kw_visual or kw_table))
            vlm_trigger = "all" if vlm_on else ("keyword" if q_vlm else "")

            # 送给模型的问题（meta 题附加统计量；结果里的 question 保持原文供评测匹配）。
            # 避让规则（v2）：问图/表的题跳过注入——首轮实测表格题被注入后轻微掉分。
            # 计数题覆盖（v3）："how many figures/tables" 含图表词但恰是统计量能精确
            # 回答的，强制注入。页码引用（v3）：题面点名 "page N" 时把该页开头文本
            # 一并给模型——"第 N 页讲什么"从猜变成读；N 越界则只有 total pages 兜底。
            q_llm = q
            meta_used = False
            # v4 关键词计数（"X 出现多少次"）优先——程序精确数指定词，避让门不挡它
            mention_target = (
                extract_mention_target(q)
                if (doc_stats and detect_mention_count_intent(q)) else None
            )
            if mention_target:
                n = count_mentions(doc_stats.get("_text", ""), mention_target)
                q_llm = (
                    f'{q}\n\n[Programmatically verified: the phrase "{mention_target}" '
                    f"appears {n} times in the document text.]"
                )
                meta_used = True
            elif doc_stats and (
                detect_count_intent(q)
                or (detect_meta_intent(q) and not (kw_visual or kw_table))
            ):
                note = format_stats_note(doc_stats)
                page_no = find_page_reference(q)
                if page_no:
                    snippet = extract_page_text(file_path, page_no)
                    if snippet:
                        note += (
                            f"\n[Beginning of page {page_no}, extracted "
                            f"programmatically: {snippet}]"
                        )
                q_llm = f"{q}\n\n{note}"
                meta_used = True

            # doc_outline：结构/前页题注入"提取的结构信息"（章节树 / 首页 footer）。
            # 与 DSG 独立、可叠加；意图互斥时一般只有一个触发。
            outline_used = False
            # 前页接地：只对"问发表/批准/作者单位"等前页题注入 content_list 前页正文。
            if outline and detect_structure_intent(q):
                onote = format_outline_note(outline)
                if onote:
                    q_llm = f"{q_llm}\n\n{onote}"
                    outline_used = True

            # doc_anchor：题目点名某张具体表/图时，按标号确定性定位该元素并注入其内容。
            # 与 DSG/outline 独立、纯加法；定位不到则不注入（对 baseline 零风险）。
            anchor_used = False
            if anchor_items:
                ref = detect_element_reference(q)
                if ref:
                    content = find_referenced_element(anchor_items, *ref)
                    anote = format_anchor_note(ref[0], ref[1], content)
                    if anote:
                        q_llm = f"{q_llm}\n\n{anote}"
                        anchor_used = True

            # doc_locate：问"X 在第几页"的题，注入"章节标题→页码"索引（Localization）。
            # 与 DSG/outline/anchor 独立、纯加法；无标题索引则不注入（对 baseline 零风险）。
            locate_used = False
            if locate_index and detect_location_intent(q):
                lnote = format_locate_note(locate_index)
                if lnote:
                    q_llm = f"{q_llm}\n\n{lnote}"
                    locate_used = True

            # NCG：计算类题——取检索context → 让模型抽数+列式（不直接算）→ 程序执行 →
            # 注入"计算辅助"帮正常生成答对。多 1 次取 context + 1 次 LLM 抽数，仅计算题触发。
            ncg_used = False
            if ncg_on and detect_calc_intent(q):
                try:
                    ncg_ctx = await rag.aquery(
                        q, mode="mix", only_need_context=True, vlm_enhanced=False
                    )
                    # 给模型完整原始表格(content_list)而非检索碎片,提升抽数准确率
                    ncg_tables = load_doc_tables(file_path, q)
                    ext = await llm_model_func(
                        build_extraction_prompt(q, ncg_ctx, ncg_tables), temperature=0
                    )
                    parsed = parse_ncg_json(ext)
                    if parsed:
                        val = safe_eval(parsed["formula"], parsed["numbers"])
                        if val is not None:
                            q_llm = (
                                f"{q_llm}\n\n"
                                f"{format_calc_note(parsed['numbers'], parsed['formula'], val)}"
                            )
                            ncg_used = True
                except Exception as e:
                    logger.warning(f"NCG failed, skip: {e}")

            retrieved_context = None
            rr_triggered = False
            rr_verdict = ""

            if rr_on:
                try:
                    retrieved_context = await rag.aquery(
                        q, mode="mix", only_need_context=True, vlm_enhanced=False
                    )
                    # 证据侧触发（R2 路径顺带零成本：上下文反正已取）
                    if (auto_vlm_on and not q_vlm
                            and count_image_evidence(retrieved_context) >= auto_vlm_min):
                        q_vlm = True
                        vlm_trigger = "evidence"
                    rr_verdict, need_more, want_visual = await retrieval_sufficiency_check(
                        llm_model_func, q, retrieved_context
                    )
                    # 关键词先验兜底 LLM 漏判：问题明说图/表就当需要视觉
                    want_visual = want_visual or kw_visual or kw_table
                    if need_more:
                        rr_triggered = True
                        # 补检索：加大检索面捞回被挤掉的块；仅当缺的是视觉信息才开 VLM。
                        # 除 top_k / VLM 外，prompt 与正常分支保持一致，便于干净消融。
                        result = await rag.aquery(
                            q_llm,
                            mode="mix",
                            response_type="One Sentence",
                            vlm_enhanced=want_visual,
                            top_k=rr_top_k,
                            chunk_top_k=rr_chunk_top_k,
                        )
                    else:
                        result = await rag.aquery(
                            q_llm, mode="mix", response_type="One Sentence",
                            vlm_enhanced=q_vlm,
                        )
                except Exception as e:
                    logger.warning(f"Retrieval-reflect failed, fallback to normal: {e}")
                    result = await rag.aquery(
                        q_llm, mode="mix", response_type="One Sentence", vlm_enhanced=q_vlm
                    )
            else:
                # 证据侧触发需要先取一次检索上下文（SAVE_CONTEXT 也要，二者共用一次）
                if save_ctx or (auto_vlm_on and not q_vlm):
                    try:
                        retrieved_context = await rag.aquery(
                            q, mode="mix", only_need_context=True, vlm_enhanced=False
                        )
                    except Exception as e:
                        logger.warning(f"Context fetch failed: {e}")
                if (auto_vlm_on and not q_vlm and retrieved_context
                        and count_image_evidence(retrieved_context) >= auto_vlm_min):
                    q_vlm = True
                    vlm_trigger = "evidence"
                # 新 EMR：视觉意图题在检索端加大 chunk_top_k 多捞图/表证据（不改看图）
                emr_kw = {}
                if emr_on and (kw_visual or kw_table):
                    emr_kw["chunk_top_k"] = emr_chunk_top_k
                result = await rag.aquery(
                    q_llm, mode="mix", response_type="One Sentence",
                    vlm_enhanced=q_vlm, **emr_kw
                )

            rec = {
                "question": q,
                "answer": result,
                "correct_answer": query["answer"],
            }
            if modality_vlm_on or auto_vlm_on or vlm_on:
                rec["vlm_used"] = q_vlm          # 便于离线分析哪些题触发了 VLM
                rec["vlm_trigger"] = vlm_trigger  # all / keyword / evidence / ""
            if doc_stats is not None:
                rec["doc_meta_used"] = meta_used
            if outline is not None:
                rec["outline_used"] = outline_used
            if anchor_items is not None:
                rec["anchor_used"] = anchor_used
            if locate_index is not None:
                rec["locate_used"] = locate_used
            if ncg_on:
                rec["ncg_used"] = ncg_used
            if rr_on:
                rec["rr_triggered"] = rr_triggered
                rec["rr_verdict"] = rr_verdict
            if save_ctx or rr_on:
                rec["retrieved_context"] = (retrieved_context or "")[:8000]
            results.append(rec)
            logger.info(f"Query: {q}")
            logger.info(f"Answer: {result}")
            logger.info(f"Correct Answer: {query['answer']}")

        result_name = os.getenv("RESULT_NAME", "qa_results_mix_mm.json")
        output_file = os.path.join(os.path.dirname(file_path), result_name)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    args = build_arg_parser("DocBench QA runner (query only, no indexing)").parse_args()

    if not args.api_key:
        logger.error("Error: API key is required")
        logger.error("Set LLM_BINDING_API_KEY env var or use --api-key option")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    configure_logging("raganything_example_qa.log")

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
