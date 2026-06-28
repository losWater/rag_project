from __future__ import annotations

from dataclasses import dataclass

from .ingest import PageText


@dataclass(frozen=True)
class TextChunk:
    """写入向量库的最小文本单元。"""

    id: str
    text: str
    metadata: dict


def normalize_text(text: str) -> str:
    """清理 PDF 提取出的空行和多余缩进，减少无意义 token。"""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按固定长度切分文本，并保留 overlap 来减少跨 chunk 信息断裂。"""
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int, file_hash: str) -> list[TextChunk]:
    """把页面文本转换成带来源信息的 chunk 列表。"""
    chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = split_text(page.text, chunk_size, chunk_overlap)
        for index, text in enumerate(page_chunks, start=1):
            chunk_id = f"{file_hash[:12]}:{page.page}:{index}"
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=text,
                    metadata={
                        "source_path": page.source_path,
                        "source_name": page.source_name,
                        "title": page.title,
                        "document_type": page.document_type,
                        "tags": ",".join(page.tags),
                        "page": page.page,
                        "chunk_index": index,
                        "file_hash": file_hash,
                    },
                )
            )
    return chunks
