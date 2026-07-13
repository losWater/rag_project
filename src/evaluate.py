from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_yaml, resolve_path
from .providers import ChatClient, EmbeddingClient
from .query_rewrite import rewrite_for_retrieval
from .retrieve import SearchResult, search


@dataclass(frozen=True)
class RetrievalCase:
    """一条检索评估用例。"""

    id: str
    question: str
    expected_sources: list[str]
    expected_pages: list[int]


@dataclass(frozen=True)
class RetrievalEvalResult:
    """单条检索评估结果。"""

    case: RetrievalCase
    retrieval_queries: list[str]
    results: list[SearchResult]
    source_hit: bool
    page_hit: bool


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """从 YAML 读取轻量检索回归用例。"""
    data = load_yaml(resolve_path(path))
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("retrieval cases YAML must contain a cases list")

    cases: list[RetrievalCase] = []
    for item in raw_cases:
        cases.append(
            RetrievalCase(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_sources=[str(value) for value in item.get("expected_sources", [])],
                expected_pages=[int(value) for value in item.get("expected_pages", [])],
            )
        )
    return cases


def evaluate_retrieval_case(
    config,
    embedding_client: EmbeddingClient,
    chat_client: ChatClient,
    case: RetrievalCase,
    top_k: int,
) -> RetrievalEvalResult:
    """执行单条检索评估，检查 top-k 是否命中预期文件和页码。"""
    planned = rewrite_for_retrieval(chat_client, case.question)
    results = search(config, embedding_client, planned.retrieval_queries, top_k=top_k)

    source_hit = any(result.metadata.get("source_name") in case.expected_sources for result in results)
    page_hit = any(
        result.metadata.get("source_name") in case.expected_sources
        and int(result.metadata.get("page", -1)) in case.expected_pages
        for result in results
    )
    return RetrievalEvalResult(
        case=case,
        retrieval_queries=planned.retrieval_queries,
        results=results,
        source_hit=source_hit,
        page_hit=page_hit,
    )


def summarize_results(results: list[RetrievalEvalResult]) -> dict[str, float]:
    """汇总 source/page recall，方便比较不同版本的检索效果。"""
    if not results:
        return {"cases": 0, "source_recall": 0.0, "page_recall": 0.0}
    return {
        "cases": float(len(results)),
        "source_recall": sum(1 for item in results if item.source_hit) / len(results),
        "page_recall": sum(1 for item in results if item.page_hit) / len(results),
    }
