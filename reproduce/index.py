#!/usr/bin/env python
"""DocBench 索引脚本：解析 PDF（MinerU，GPU）→ 建双图 + 向量库（昂贵步骤，只跑一次）。

用法：python reproduce/index.py <pdf> --working_dir <每篇文档独立的存储目录>
开关：ENABLE_CANONICALIZATION（L1 名称规范化）、ENABLE_SYNONYM_EDGES（L2 同义边，
      建图完成后作为后处理执行）。模型/温度等共用配置见 common.py。
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from common import build_arg_parser, build_model_funcs, configure_logging  # noqa: E402
from lightrag.utils import logger  # noqa: E402
from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """解析文档并建立索引（图 + 向量库）。"""
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # mineru or docling
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        llm_model_func, vision_model_func, embedding_func = build_model_funcs(
            api_key, base_url
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto",
            source="modelscope",
            backend="pipeline",
            device="cuda:0",
        )

        # L2: 建图完成后添加同义边（ENABLE_SYNONYM_EDGES=true 时生效）
        from raganything.graph_fusion.synonym_linker import add_synonym_edges

        n_syn = add_synonym_edges(working_dir or "./rag_storage")
        if n_syn:
            logger.info("L2: added %d synonym edges", n_syn)

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    args = build_arg_parser("DocBench indexer (MinerU + RAGAnything)").parse_args()

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
    configure_logging("raganything_example.log")

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
