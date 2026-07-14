from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import AppConfig
from .index import get_collection
from .providers import EmbeddingClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """一次检索命中的文本片段及其距离分数。"""

    text: str
    metadata: dict
    distance: float | None
    matched_queries: tuple[str, ...] = ()
    retrieval_sources: tuple[str, ...] = ("vector",)
    fusion_score: float | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        """把来源元数据格式化成回答中可展示的引用。"""
        source = self.metadata.get("source_name", "unknown")
        page = self.metadata.get("page", "?")
        chunk = self.metadata.get("chunk_index", "?")
        return f"{source} p.{page} chunk {chunk}"


def retrieve(config: AppConfig, embedding_client: EmbeddingClient, query: str, top_k: int | None = None) -> list[SearchResult]:
    """对单个检索词做向量搜索。"""
    collection = get_collection(config)
    query_embedding = embedding_client.embed(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or config.top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        SearchResult(text=doc, metadata=meta or {}, distance=float(distance))
        for doc, meta, distance in zip(documents, metadatas, distances)
    ]


def chunk_key(result: SearchResult) -> tuple:
    """用来源文件、页码和 chunk 序号判断两个检索结果是否是同一个片段。"""
    return (
        result.metadata.get("source_path"),
        result.metadata.get("page"),
        result.metadata.get("chunk_index"),
    )


def retrieve_multi(
    config: AppConfig,
    embedding_client: EmbeddingClient,
    queries: list[str],
    top_k: int | None = None,
    per_query_k: int | None = None,
) -> list[SearchResult]:
    """对多个检索词分别搜索，再合并去重。

    中文问题会被改写成多个英文检索词。这里会保留每个 chunk 被哪些 query 命中过，
    并优先排序“距离近且被多个 query 命中”的片段。
    """
    final_top_k = top_k or config.top_k
    query_top_k = per_query_k or final_top_k
    merged: dict[tuple, SearchResult] = {}
    matched: dict[tuple, list[str]] = {}

    for query in queries:
        for result in retrieve(config, embedding_client, query, top_k=query_top_k):
            key = chunk_key(result)
            matched.setdefault(key, []).append(query)
            previous = merged.get(key)
            if previous is None or result.distance < previous.distance:
                merged[key] = result

    ranked: list[SearchResult] = []
    for key, result in merged.items():
        query_hits = tuple(dict.fromkeys(matched.get(key, [])))
        ranked.append(
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                distance=result.distance,
                matched_queries=query_hits,
                retrieval_sources=("vector",),
            )
        )

    ranked.sort(key=lambda item: (item.distance / max(1, len(item.matched_queries)), item.distance))
    return ranked[:final_top_k]


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize English course text while retaining technical identifiers."""
    tokens = re.findall(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", text.lower())
    return [token[:-1] if len(token) > 4 and token.endswith("s") and not token.endswith("ss") else token for token in tokens]


def retrieve_bm25_multi(
    config: AppConfig,
    queries: list[str],
    top_k: int | None = None,
) -> list[SearchResult]:
    """Build a lightweight BM25 index over Chroma documents and search all queries."""
    collection = get_collection(config)
    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not documents:
        return []

    corpus_tokens = [tokenize_for_bm25(document) for document in documents]
    bm25 = BM25Okapi(corpus_tokens)
    merged: dict[tuple, tuple[SearchResult, float]] = {}
    matched: dict[tuple, list[str]] = {}

    for query in queries:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            continue
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        for index in ranked_indices[: top_k or config.top_k]:
            score = float(scores[index])
            if score <= 0:
                continue
            result = SearchResult(
                text=documents[index],
                metadata=metadatas[index] or {},
                distance=None,
                retrieval_sources=("bm25",),
            )
            key = chunk_key(result)
            matched.setdefault(key, []).append(query)
            previous = merged.get(key)
            if previous is None or score > previous[1]:
                merged[key] = (result, score)

    ranked: list[SearchResult] = []
    for key, (result, score) in merged.items():
        ranked.append(
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                distance=result.distance,
                matched_queries=tuple(dict.fromkeys(matched[key])),
                retrieval_sources=result.retrieval_sources,
                fusion_score=score,
            )
        )
    ranked.sort(key=lambda item: item.fusion_score or 0.0, reverse=True)
    return ranked[: top_k or config.top_k]


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse independently ranked result lists without comparing incompatible scores."""
    fused_scores: dict[tuple, float] = {}
    representative: dict[tuple, SearchResult] = {}
    sources: dict[tuple, list[str]] = {}
    queries: dict[tuple, list[str]] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = chunk_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            representative.setdefault(key, result)
            sources.setdefault(key, []).extend(result.retrieval_sources)
            queries.setdefault(key, []).extend(result.matched_queries)

    ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused: list[SearchResult] = []
    for key in ranked_keys[:top_k]:
        result = representative[key]
        fused.append(
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                distance=result.distance,
                matched_queries=tuple(dict.fromkeys(queries[key])),
                retrieval_sources=tuple(dict.fromkeys(sources[key])),
                fusion_score=fused_scores[key],
            )
        )
    return fused


def search(
    config: AppConfig,
    embedding_client: EmbeddingClient,
    queries: list[str],
    top_k: int | None = None,
) -> list[SearchResult]:
    """Run the configured vector, BM25, or hybrid retrieval strategy."""
    final_top_k = top_k or config.top_k
    mode = getattr(config, "retrieval_mode", "vector")
    if mode == "vector":
        return retrieve_multi(
            config,
            embedding_client,
            queries,
            top_k=final_top_k,
            per_query_k=getattr(config, "vector_candidates", final_top_k),
        )
    if mode == "bm25":
        return retrieve_bm25_multi(config, queries, top_k=final_top_k)

    candidate_count = getattr(config, "rerank_candidates", 20) if mode == "rerank" else final_top_k
    vector_results = retrieve_multi(
        config,
        embedding_client,
        queries,
        top_k=getattr(config, "vector_candidates", final_top_k),
        per_query_k=getattr(config, "vector_candidates", final_top_k),
    )
    keyword_results = retrieve_bm25_multi(
        config,
        queries,
        top_k=getattr(config, "keyword_candidates", final_top_k),
    )
    hybrid_results = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        top_k=candidate_count,
        rrf_k=getattr(config, "rrf_k", 60),
    )
    if mode == "hybrid":
        return hybrid_results
    return _rerank(config, queries[0], hybrid_results, final_top_k)


def _rerank(config: AppConfig, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Apply the configured cross-encoder, with an explicit fallback to RRF order."""
    from .rerank import get_cross_encoder, rerank_results

    model_name = str(config.reranker.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    try:
        return rerank_results(candidates, query, get_cross_encoder(model_name), top_k)
    except Exception as exc:
        if not config.reranker.get("fallback_on_error", True):
            raise
        logger.warning("Reranker failed; falling back to RRF order: %s", exc)
        return candidates[:top_k]
