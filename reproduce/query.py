#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
from lightrag import LightRAG

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


# ---------------------------------------------------------------------------
# 方案 A：training-free 检索后自反思（critique-and-revise）
# 由环境变量 ENABLE_REFLECTION 开关控制（默认关=原版行为，便于消融）。
# 思路：答案生成后，用同一个 LLM 对照【检索到的上下文】批判答案；若发现
# 不被支持/答非所问/数字人名日期错误，则据批判重写一遍。纯提示词，不训练。
# ---------------------------------------------------------------------------

REFLECT_CRITIQUE_PROMPT = """You are verifying a RAG-generated answer strictly against the retrieved context.

Question: {question}

Retrieved context:
{context}

Proposed answer: {answer}

Check carefully:
1. Is every factual claim in the answer supported by the context?
2. Does it fully and directly answer the question?
3. Are all numbers, names, and dates correct according to the context?

If the answer is fully correct and grounded, reply with exactly:
VERDICT: OK
Otherwise reply:
VERDICT: REVISE
<one short line listing the concrete problems>"""

REFLECT_REVISE_PROMPT = """Improve the answer so it is fully grounded in the retrieved context and directly answers the question.

Question: {question}

Retrieved context:
{context}

Previous answer: {answer}

Problems found: {critique}

Write a concise corrected answer (one or two sentences). If the context does not contain the answer, reply exactly: The document does not provide this information."""


async def reflect_answer(llm_func, question, context, answer, max_context_chars=6000):
    """对单个答案做一次"批判→（必要时）修订"。返回 (最终答案, 是否修订过)。"""
    ctx = context or ""
    if len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars]
    critique = await llm_func(
        REFLECT_CRITIQUE_PROMPT.format(question=question, context=ctx, answer=answer)
    )
    if (critique or "").strip().upper().startswith("VERDICT: OK"):
        return answer, False
    revised = await llm_func(
        REFLECT_REVISE_PROMPT.format(
            question=question, context=ctx, answer=answer, critique=critique
        )
    )
    revised = (revised or "").strip()
    return (revised or answer), True


# ---------------------------------------------------------------------------
# R2：检索后多模态自反思（与上面"答案后自反思"本质不同——这里只判"检索够不够
# 答"，不够就触发补检索，绝不改写已生成的答案，因此下行有底，不会把对的改坏）。
# 由环境变量 ENABLE_RETRIEVAL_REFLECT 开关控制（默认关 = 原版行为）。
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
      - want_visual: 缺的信息是否疑似在图/表里（仅作记录；本 MVP 触发时统一开 VLM）
    """
    ctx = (context or "")[:max_context_chars]
    verdict = (
        await llm_func(
            RETRIEVAL_SUFFICIENCY_PROMPT.format(question=question, context=ctx)
        )
        or ""
    ).strip().upper()
    need_more = verdict.startswith("INSUFFICIENT")
    want_visual = ("VISUAL" in verdict) or ("TABLE" in verdict)
    return verdict, need_more, want_visual


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example_qa.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
        # 模型名从环境变量读取：默认 gpt（原版行为），设 LLM_MODEL=qwen-plus 即用百炼 Qwen
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                os.getenv("LLM_MODEL", "gpt-4o-mini"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return openai_complete_if_cache(
                    os.getenv("VISION_MODEL", "gpt-4o-mini"),
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return openai_complete_if_cache(
                    os.getenv("VISION_MODEL", "gpt-4o-mini"),
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                api_key=api_key,
                base_url=base_url,
            ),
        )
        # Reranker：默认关（=原版 baseline 行为，所有历史条件一致）。
        # 设 ENABLE_RERANK=true 时启用 DashScope 百炼 gte-rerank（ali_rerank，端点内置），
        # 用现有百炼 key（RERANK_BINDING_API_KEY 缺省回退到 LLM 的 api_key）。
        from functools import partial

        rerank_model_func = None
        if os.getenv("ENABLE_RERANK", "false").lower() in ("1", "true", "yes", "on"):
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

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            lightrag=lightrag,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        import json

        folder_name = os.path.basename(os.path.dirname(file_path))
        qa_file_path = os.path.join(
            os.path.dirname(file_path), f"{folder_name}_qa.jsonl"
        )
        queries = []
        if os.path.exists(qa_file_path):
            with open(qa_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        qa_data = json.loads(line)
                        queries.append(
                            {
                                "question": qa_data["question"],
                                "answer": qa_data.get("answer", ""),
                            }
                        )
        else:
            logger.warning(f"QA file not found: {qa_file_path}")
            return

        reflect_on = os.getenv("ENABLE_REFLECTION", "false").lower() in (
            "1", "true", "yes", "on"
        )
        # 反思模式：text=方案A(仅用问题检索的上下文)；graph=方案C(额外用"答案"反向
        # 检索图谱作为证据，抓"答案偏离检索/凭空捏造"的错误)。证据均为加分制：
        # reflect_answer 只在"上下文不支持"时才改，不因图缺失而否定正确答案。
        reflect_mode = os.getenv("REFLECT_MODE", "text").lower()
        # VLM 增强：默认关（=原版 baseline）。设 ENABLE_VLM=true 时，检索到的图片会以
        # base64 交给 qwen-vl-max 看图作答，针对多模态题（DocBench 多模态题占多数）。
        vlm_on = os.getenv("ENABLE_VLM", "false").lower() in ("1", "true", "yes", "on")
        # 体检（SAVE_CONTEXT）：把每题"检索到的上下文"也存进结果，供离线脚本
        # diag_recall.py 算证据命中率（recall 代理）。默认关，不影响历史实验。
        save_ctx = os.getenv("SAVE_CONTEXT", "false").lower() in ("1", "true", "yes", "on")
        # R2（ENABLE_RETRIEVAL_REFLECT）：检索后自反思 + 触发补检索。默认关。
        # 流程：取上下文 → 自检够不够答 → 不够就用更大的 top_k 重检 + 开 VLM 看图，
        # 再答一次；够就正常答。全程 3 次调用/题，只补证据、不改答案。
        rr_on = os.getenv("ENABLE_RETRIEVAL_REFLECT", "false").lower() in (
            "1", "true", "yes", "on"
        )
        rr_top_k = int(os.getenv("RR_TOP_K", "80"))             # 默认 2×（LightRAG 默认 top_k=40）
        rr_chunk_top_k = int(os.getenv("RR_CHUNK_TOP_K", "40"))  # 默认 2×（默认 chunk_top_k=20）

        results = []
        for query in queries:
            q = query["question"]
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
                    if need_more:
                        rr_triggered = True
                        # 补检索：加大检索面捞回被挤掉的图/表块，并开 VLM 看图，再答一次
                        result = await rag.aquery(
                            q,
                            mode="mix",
                            response_type="One Sentence",
                            vlm_enhanced=True,
                            top_k=rr_top_k,
                            chunk_top_k=rr_chunk_top_k,
                            system_prompt="Answer in one concise sentence.",
                        )
                    else:
                        result = await rag.aquery(
                            q, mode="mix", response_type="One Sentence",
                            vlm_enhanced=vlm_on,
                        )
                except Exception as e:
                    logger.warning(f"Retrieval-reflect failed, fallback to normal: {e}")
                    result = await rag.aquery(
                        q, mode="mix", response_type="One Sentence", vlm_enhanced=vlm_on
                    )
            else:
                result = await rag.aquery(
                    q, mode="mix", response_type="One Sentence", vlm_enhanced=vlm_on
                )
                if save_ctx:
                    try:
                        retrieved_context = await rag.aquery(
                            q, mode="mix", only_need_context=True, vlm_enhanced=False
                        )
                    except Exception as e:
                        logger.warning(f"Save context failed: {e}")

            reflected = False
            if reflect_on:
                try:
                    # 复用已取的上下文（没有则现取），避免重复检索
                    q_ctx = retrieved_context
                    if q_ctx is None:
                        q_ctx = await rag.aquery(
                            q, mode="mix", only_need_context=True, vlm_enhanced=False
                        )
                    evidence = q_ctx
                    if reflect_mode == "graph":
                        # 方案C：用"生成的答案"反向检索 L1+L2 图谱，核验答案自身的断言
                        a_ctx = await rag.aquery(
                            result, mode="mix", only_need_context=True, vlm_enhanced=False
                        )
                        evidence = (
                            "[Evidence retrieved for the QUESTION]\n"
                            f"{(q_ctx or '')[:3500]}\n\n"
                            "[Evidence retrieved for the ANSWER's own claims]\n"
                            f"{(a_ctx or '')[:3500]}"
                        )
                    result, reflected = await reflect_answer(
                        llm_model_func, q, evidence, result, max_context_chars=8000,
                    )
                except Exception as e:
                    logger.warning(f"Reflection failed, keep original answer: {e}")

            rec = {
                "question": q,
                "answer": result,
                "correct_answer": query["answer"],
                "reflected": reflected,
            }
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
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
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
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
