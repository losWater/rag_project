from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from .retrieve import SearchResult


class PairScorer(Protocol):
    """Minimal interface shared by the real cross-encoder and test doubles."""

    def predict(self, pairs: list[tuple[str, str]]): ...


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str):
    """Load a cross-encoder once per process; model weights are cached by Hugging Face."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_results(
    candidates: list[SearchResult],
    query: str,
    scorer: PairScorer,
    top_k: int,
) -> list[SearchResult]:
    """Score query-document pairs jointly and return the highest-scoring chunks."""
    if not candidates:
        return []
    scores = scorer.predict([(query, candidate.text) for candidate in candidates])
    reranked = [
        SearchResult(
            text=candidate.text,
            metadata=candidate.metadata,
            distance=candidate.distance,
            matched_queries=candidate.matched_queries,
            retrieval_sources=candidate.retrieval_sources,
            fusion_score=candidate.fusion_score,
            rerank_score=float(score),
        )
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
    return reranked[:top_k]
