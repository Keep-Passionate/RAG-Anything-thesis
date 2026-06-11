"""reproduce 脚本共享的基础设施：模型函数 / 日志 / 命令行（index.py 与 query.py 复用）。

此前两个脚本各自维护一份几乎逐字相同的 llm/vision/embedding 函数与日志配置
（约 180 行重复）——改温度、换模型名等任何调整都要改两处、极易漏改一处。
统一收口到这里，行为与原来逐字节一致（同样的环境变量、同样的默认值）：

- env_on(name)                 : 布尔开关环境变量解析（1/true/yes/on 不分大小写）
- configure_logging(filename)  : 控制台 + 滚动文件日志
- build_model_funcs(key, url)  : (llm, vision, embedding) 三件套。模型名/维度/温度由
                                 LLM_MODEL / VISION_MODEL / EMBEDDING_MODEL / EMBEDDING_DIM /
                                 LLM_TEMPERATURE 控制；默认 gpt = 原版行为，温度默认 0 = 可复现。
- build_arg_parser(desc)       : 两脚本一致的命令行参数。
"""

import argparse
import logging
import logging.config
import os

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug


def env_on(name: str, default: str = "false") -> bool:
    """读布尔开关环境变量：1/true/yes/on（不分大小写）为真。"""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def configure_logging(log_filename: str):
    """配置控制台 + 滚动文件日志。log_filename 仅文件名（index/query 各用各的）。"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, log_filename))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def build_model_funcs(api_key: str, base_url: str = None):
    """构建 (llm_model_func, vision_model_func, embedding_func) 三件套。

    温度默认 0（贪心解码）：同一输入结果确定可复现——小评测集上若答案 run-to-run
    抖动，真实的 ±2% 涨点会被随机噪声淹没。可用 LLM_TEMPERATURE 调整。
    """

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.setdefault("temperature", float(os.getenv("LLM_TEMPERATURE", "0")))
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        kwargs.setdefault("temperature", float(os.getenv("LLM_TEMPERATURE", "0")))
        # 多模态 messages 直传（VLM 增强查询用这个形态）
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
        # 传统单图形态
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
        # 纯文本回退
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

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

    return llm_model_func, vision_model_func, embedding_func


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """index.py / query.py 共用的命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
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
        help="API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Document parser: mineru or docling",
    )
    return parser
