import math

import pytest

from src.evaluate import RetrievalCase, RetrievalEvalResult, load_retrieval_cases, ranking_metrics, summarize_results
from src.retrieve import SearchResult


def test_load_retrieval_cases():
    cases = load_retrieval_cases("data/eval/retrieval_cases.yaml")

    assert len(cases) >= 1
    assert cases[0].id == "cross_entropy_zh"
    assert cases[0].retrieval_queries[0] == "cross entropy loss"


def test_summarize_results():
    case = RetrievalCase(id="x", question="q", expected_sources=[], expected_pages=[])
    results = [
        RetrievalEvalResult(
            case=case, retrieval_queries=[], results=[], source_hit=True, page_hit=False, reciprocal_rank=0.0, ndcg=0.0
        ),
        RetrievalEvalResult(
            case=case, retrieval_queries=[], results=[], source_hit=True, page_hit=True, reciprocal_rank=0.5, ndcg=0.75
        ),
    ]

    summary = summarize_results(results)

    assert summary["cases"] == 2
    assert summary["source_recall"] == 1.0
    assert summary["page_recall"] == 0.5
    assert summary["mrr"] == 0.25
    assert summary["ndcg"] == 0.375
    assert summary["avg_latency_ms"] == 0.0
    assert summary["p95_latency_ms"] == 0.0
    assert summary["hit_rate"] == 0.5


def test_ranking_metrics_rewards_earlier_relevant_results():
    case = RetrievalCase(id="x", question="q", expected_sources=["right.pdf"], expected_pages=[3])
    results = [
        SearchResult("wrong", {"source_name": "wrong.pdf", "page": 1}, 0.1),
        SearchResult("right", {"source_name": "right.pdf", "page": 3}, 0.2),
    ]

    reciprocal_rank, ndcg = ranking_metrics(results, case)

    assert reciprocal_rank == 0.5
    assert ndcg == pytest.approx(1 / math.log2(3))


def test_explicit_relevance_judgments_do_not_cross_source_and_page():
    case = RetrievalCase(
        id="x",
        question="q",
        expected_sources=["a.pdf", "b.pdf"],
        expected_pages=[1, 2],
        relevant_pages=frozenset({("a.pdf", 1), ("b.pdf", 2)}),
    )
    crossed = SearchResult("wrong pair", {"source_name": "a.pdf", "page": 2}, 0.1)

    reciprocal_rank, ndcg = ranking_metrics([crossed], case)

    assert reciprocal_rank == 0.0
    assert ndcg == 0.0
