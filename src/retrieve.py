from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .index import get_collection
from .providers import EmbeddingClient


@dataclass(frozen=True)
class SearchResult:
    """一次检索命中的文本片段及其距离分数。"""

    text: str
    metadata: dict
    distance: float
    matched_queries: tuple[str, ...] = ()

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
            )
        )

    ranked.sort(key=lambda item: (item.distance / max(1, len(item.matched_queries)), item.distance))
    return ranked[:final_top_k]
