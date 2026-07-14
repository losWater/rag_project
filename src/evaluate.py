from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
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
    retrieval_queries: list[str] = field(default_factory=list)
    relevant_pages: frozenset[tuple[str, int]] = field(default_factory=frozenset)

    @property
    def relevance_judgments(self) -> frozenset[tuple[str, int]]:
        """Return explicit judgments, falling back to the legacy source/page product."""
        if self.relevant_pages:
            return self.relevant_pages
        return frozenset((source, page) for source in self.expected_sources for page in self.expected_pages)


@dataclass(frozen=True)
class RetrievalEvalResult:
    """单条检索评估结果。"""

    case: RetrievalCase
    retrieval_queries: list[str]
    results: list[SearchResult]
    source_hit: bool
    page_hit: bool
    reciprocal_rank: float
    ndcg: float
    latency_ms: float = 0.0


def is_relevant(result: SearchResult, case: RetrievalCase) -> bool:
    """Treat a chunk as relevant when both its source and page match the labels."""
    key = (str(result.metadata.get("source_name", "")), int(result.metadata.get("page", -1)))
    return key in case.relevance_judgments


def ranking_metrics(results: list[SearchResult], case: RetrievalCase) -> tuple[float, float]:
    """Calculate reciprocal rank and binary nDCG for one ranked result list."""
    relevance = [1 if is_relevant(result, case) else 0 for result in results]
    first_relevant = next((rank for rank, value in enumerate(relevance, start=1) if value), None)
    reciprocal_rank = 1.0 / first_relevant if first_relevant else 0.0

    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))
    ideal_count = min(len(case.relevance_judgments), len(results))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return reciprocal_rank, ndcg


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """从 YAML 读取轻量检索回归用例。"""
    data = load_yaml(resolve_path(path))
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("retrieval cases YAML must contain a cases list")

    cases: list[RetrievalCase] = []
    for item in raw_cases:
        raw_judgments = item.get("relevant_pages", {})
        if raw_judgments and not isinstance(raw_judgments, dict):
            raise ValueError(f"relevant_pages for {item.get('id')} must be a mapping")
        relevant_pages = frozenset(
            (str(source), int(page))
            for source, pages in raw_judgments.items()
            for page in pages
        )
        cases.append(
            RetrievalCase(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_sources=[str(value) for value in item.get("expected_sources", [])],
                expected_pages=[int(value) for value in item.get("expected_pages", [])],
                retrieval_queries=[str(value) for value in item.get("retrieval_queries", [])],
                relevant_pages=relevant_pages,
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
    planned = rewrite_for_retrieval(chat_client, case.question) if not case.retrieval_queries else None
    retrieval_queries = case.retrieval_queries or planned.retrieval_queries
    started = time.perf_counter()
    results = search(config, embedding_client, retrieval_queries, top_k=top_k)
    latency_ms = (time.perf_counter() - started) * 1000

    relevant_sources = {source for source, _ in case.relevance_judgments}
    source_hit = any(result.metadata.get("source_name") in relevant_sources for result in results)
    page_hit = any(is_relevant(result, case) for result in results)
    reciprocal_rank, ndcg = ranking_metrics(results, case)
    return RetrievalEvalResult(
        case=case,
        retrieval_queries=retrieval_queries,
        results=results,
        source_hit=source_hit,
        page_hit=page_hit,
        reciprocal_rank=reciprocal_rank,
        ndcg=ndcg,
        latency_ms=latency_ms,
    )


def summarize_results(results: list[RetrievalEvalResult]) -> dict[str, float]:
    """汇总 source/page recall，方便比较不同版本的检索效果。"""
    if not results:
        return {
            "cases": 0,
            "source_recall": 0.0,
            "page_recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "hit_rate": 0.0,
        }
    ordered_latencies = sorted(item.latency_ms for item in results)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    return {
        "cases": float(len(results)),
        "source_recall": sum(1 for item in results if item.source_hit) / len(results),
        "page_recall": sum(1 for item in results if item.page_hit) / len(results),
        "mrr": sum(item.reciprocal_rank for item in results) / len(results),
        "ndcg": sum(item.ndcg for item in results) / len(results),
        "avg_latency_ms": sum(item.latency_ms for item in results) / len(results),
        "p95_latency_ms": ordered_latencies[p95_index],
        "hit_rate": sum(1 for item in results if item.page_hit) / len(results),
    }
