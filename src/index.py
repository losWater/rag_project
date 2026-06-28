from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rich.progress import track

from .chunk import TextChunk, chunk_pages
from .config import AppConfig, PROJECT_ROOT, load_manifest, resolve_path
from .ingest import load_pdf_pages, sha256_file
from .providers import EmbeddingClient


@dataclass(frozen=True)
class IndexResult:
    """一次索引任务的结果统计。"""

    indexed_files: int
    skipped_files: int
    chunks_added: int


def load_index_state(path: Path) -> dict:
    """读取增量索引状态；首次运行时返回空状态。"""
    if not path.exists():
        return {"files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_index_state(path: Path, state: dict) -> None:
    """保存每个已索引文件的 hash 和 chunk 数量。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_collection(config: AppConfig):
    """获取 Chroma collection，所有检索和写入都走同一个 collection。"""
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.vectorstore_dir))
    return client.get_or_create_collection(name=config.collection_name)


def add_chunks(collection, embedding_client: EmbeddingClient, chunks: list[TextChunk]) -> None:
    """为 chunk 生成 embedding，并批量写入 Chroma。"""
    if not chunks:
        return

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []

    for chunk in track(chunks, description="Embedding chunks"):
        ids.append(chunk.id)
        documents.append(chunk.text)
        metadatas.append(chunk.metadata)
        embeddings.append(embedding_client.embed(chunk.text))

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def index_documents(config: AppConfig, embedding_client: EmbeddingClient, force: bool = False) -> IndexResult:
    """索引 manifest 中的新文件或变更文件。

    核心逻辑是用 SHA256 比较当前 PDF 和上次索引记录：没变就跳过，变了就删除旧
    chunk 后重新写入。这样新课件可以随时添加，同时避免每次全量重建索引。
    """
    documents = load_manifest(config.manifest_path)
    state = load_index_state(config.index_state_path)
    state.setdefault("files", {})
    collection = get_collection(config)

    indexed_files = 0
    skipped_files = 0
    chunks_added = 0

    for document in documents:
        rel_path = str(Path(document["path"]))
        pdf_path = resolve_path(rel_path)
        if pdf_path.suffix.lower() != ".pdf":
            continue
        file_hash = sha256_file(pdf_path)
        previous = state["files"].get(rel_path)

        if previous and previous.get("sha256") == file_hash and not force:
            skipped_files += 1
            continue

        if previous:
            # 文件内容变更时，先删除旧 chunk，避免同一 PDF 的旧内容和新内容同时存在。
            collection.delete(where={"source_path": rel_path})

        pages = load_pdf_pages(document)
        chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap, file_hash)
        add_chunks(collection, embedding_client, chunks)

        state["files"][rel_path] = {
            "sha256": file_hash,
            "chunk_count": len(chunks),
            "title": document.get("title", pdf_path.stem),
        }
        indexed_files += 1
        chunks_added += len(chunks)

    save_index_state(config.index_state_path, state)
    return IndexResult(indexed_files=indexed_files, skipped_files=skipped_files, chunks_added=chunks_added)


def index_summary(config: AppConfig) -> dict:
    """返回当前向量库和索引状态，供 CLI / Streamlit 展示。"""
    collection = get_collection(config)
    state = load_index_state(config.index_state_path)
    return {
        "collection": config.collection_name,
        "vectorstore_dir": str(config.vectorstore_dir.relative_to(PROJECT_ROOT)),
        "collection_count": collection.count(),
        "indexed_files": len(state.get("files", {})),
    }
