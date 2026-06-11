#!/usr/bin/env python
"""DocBench 查询脚本：读 <id>_qa.jsonl 逐题作答，结果存 RESULT_NAME 指定的 JSON。

用法：python reproduce/query.py <pdf> --working_dir <该篇文档自己的索引目录>
（索引必须已由 index.py 建好；本脚本只做查询，不烧建图的钱。）

查询期开关（默认全关 = 原版 baseline，逐个打开做消融）：
  ENABLE_VLM              : 每题都让 VLM 看检索到的图（全开，贵且对纯文本题加噪）
  ENABLE_MODALITY_VLM     : 只对"问图/表"的题开 VLM（关键词检测，零成本，精准投放）
  ENABLE_DOC_META         : meta 题注入程序化统计量（页数/词数/高频词/缩写，见 doc_meta.py）
  ENABLE_RERANK           : DashScope gte-rerank 重排检索结果
  ENABLE_RETRIEVAL_REFLECT: R2 检索后充分性自检，不够则补检索（只补证据、绝不改答案）
  SAVE_CONTEXT            : 把检索上下文一并存进结果，供 diag_recall.py 离线体检
模型/温度等共用配置见 common.py（LLM_TEMPERATURE 默认 0 = 可复现）。
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
from doc_meta import compute_doc_stats, detect_meta_intent, format_stats_note  # noqa: E402
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
# 模态意图检测（纯关键词，零 LLM 成本）。
# 用途：把 VLM 只用在真正需要看图/表的题上——既正面打 DocBench 的多模态题，
# 又避免对纯文本题开 VLM 带来的成本与噪声（VLM 看错图会自信答错）。
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
# 主流程
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
        rerank_model_func = None
        if env_on("ENABLE_RERANK"):
            from lightrag.rerank import ali_rerank

            rerank_model_func = partial(
                ali_rerank,
                model=os.getenv("RERANK_MODEL", "gte-rerank-v2"),
                api_key=os.getenv("RERANK_BINDING_API_KEY") or api_key,
            )
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

        results = []
        for query in queries:
            q = query["question"]

            # 送给模型的问题（meta 题附加统计量；结果里的 question 保持原文供评测匹配）
            q_llm = q
            meta_used = False
            if doc_stats and detect_meta_intent(q):
                q_llm = f"{q}\n\n{format_stats_note(doc_stats)}"
                meta_used = True

            # 本题是否开 VLM：全开 > 模态感知（关键词，零成本）> 关
            kw_visual, kw_table = detect_visual_intent(q)
            q_vlm = vlm_on or (modality_vlm_on and (kw_visual or kw_table))

            retrieved_context = None
            rr_triggered = False
            rr_verdict = ""

            if rr_on:
                try:
                    retrieved_context = await rag.aquery(
                        q, mode="mix", only_need_context=True, vlm_enhanced=False
                    )
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
                result = await rag.aquery(
                    q_llm, mode="mix", response_type="One Sentence", vlm_enhanced=q_vlm
                )
                if save_ctx:
                    try:
                        retrieved_context = await rag.aquery(
                            q, mode="mix", only_need_context=True, vlm_enhanced=False
                        )
                    except Exception as e:
                        logger.warning(f"Save context failed: {e}")

            rec = {
                "question": q,
                "answer": result,
                "correct_answer": query["answer"],
            }
            if modality_vlm_on:
                rec["vlm_used"] = q_vlm     # 便于离线分析哪些题触发了 VLM
            if doc_stats is not None:
                rec["doc_meta_used"] = meta_used
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
