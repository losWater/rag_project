from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, load_config
from .generate import Answer, answer_query
from .logging_utils import write_query_log
from .providers import create_chat_client, create_embedding_client
from .query_rewrite import QueryForRetrieval, rewrite_for_retrieval
from .retrieve import SearchResult, search


@dataclass(frozen=True)
class RAGResponse:
    """一次完整 RAG 调用的返回值，供 CLI 和 Streamlit 共用。"""

    question: str
    planned_query: QueryForRetrieval
    contexts: list[SearchResult]
    answer: Answer
    log_path: Path | None


def ask_question(
    question: str,
    config_path: str = "configs/rag.yaml",
    top_k: int | None = None,
    log: bool = True,
) -> RAGResponse:
    """从配置文件开始执行一次问答，适合外部入口直接调用。"""
    config = load_config(config_path)
    return ask_question_with_config(config, question, top_k=top_k, log=log)


def ask_question_with_config(
    config: AppConfig,
    question: str,
    top_k: int | None = None,
    log: bool = True,
) -> RAGResponse:
    """执行完整 RAG 流程。

    顺序是：创建模型客户端 -> 规划检索词 -> 多查询检索 -> 生成回答 -> 可选写日志。
    CLI 和 Streamlit 都调用这里，避免两套入口出现行为不一致。
    """
    embedding_client = create_embedding_client(config.embedding)
    chat_client = create_chat_client(config.chat)
    effective_top_k = top_k or config.top_k

    planned_query = rewrite_for_retrieval(chat_client, question)
    contexts = search(config, embedding_client, planned_query.retrieval_queries, top_k=effective_top_k)
    answer = answer_query(chat_client, question, contexts)

    log_path = (
        write_query_log(
            config,
            question,
            answer,
            effective_top_k,
            retrieval_query=planned_query.retrieval_query,
            retrieval_queries=planned_query.retrieval_queries,
            query_rewritten=planned_query.rewritten,
        )
        if log
        else None
    )

    return RAGResponse(
        question=question,
        planned_query=planned_query,
        contexts=contexts,
        answer=answer,
        log_path=log_path,
    )
