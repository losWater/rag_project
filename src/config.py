from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# 项目根目录。所有相对路径都会基于这个目录解析，避免从不同工作目录运行时路径错乱。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    """应用运行时配置。

    这个对象把 YAML 配置、路径配置和模型配置统一收在一起，后续模块只依赖
    AppConfig，不直接关心配置文件在哪里。
    """

    raw_data_dir: Path
    vectorstore_dir: Path
    index_state_path: Path
    manifest_path: Path
    retrieval_rules_path: Path
    chunk_size: int
    chunk_overlap: int
    collection_name: str
    top_k: int
    retrieval_mode: str
    vector_candidates: int
    keyword_candidates: int
    rrf_k: int
    rerank_candidates: int
    reranker: dict[str, Any]
    chat: dict[str, Any]
    embedding: dict[str, Any]


def resolve_path(value: str | Path) -> Path:
    """把配置中的相对路径转换成项目内的绝对路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML，并确保顶层结构是 mapping。"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_config(config_path: str | Path = "configs/rag.yaml") -> AppConfig:
    """加载主配置和本地环境变量。

    `.env` 只在本地生效，不提交到 git；配置文件里只保存环境变量名。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    config_file = resolve_path(config_path)
    data = load_yaml(config_file)

    paths = data.get("paths", {})
    chunking = data.get("chunking", {})
    retrieval = data.get("retrieval", {})

    retrieval_mode = str(retrieval.get("mode", "vector")).lower()
    if retrieval_mode not in {"vector", "bm25", "hybrid", "rerank"}:
        raise ValueError("retrieval.mode must be one of: vector, bm25, hybrid, rerank")

    return AppConfig(
        raw_data_dir=resolve_path(paths.get("raw_data_dir", "data/raw/comp9444")),
        vectorstore_dir=resolve_path(paths.get("vectorstore_dir", "data/vectorstore/chroma")),
        index_state_path=resolve_path(paths.get("index_state_path", "data/vectorstore/index_state.json")),
        manifest_path=resolve_path(paths.get("manifest_path", "manifest.yaml")),
        retrieval_rules_path=resolve_path(paths.get("retrieval_rules_path", "retrieval_rules.yaml")),
        chunk_size=int(chunking.get("chunk_size", 1200)),
        chunk_overlap=int(chunking.get("chunk_overlap", 180)),
        collection_name=str(retrieval.get("collection_name", "comp9444_chunks")),
        top_k=int(retrieval.get("top_k", 5)),
        retrieval_mode=retrieval_mode,
        vector_candidates=int(retrieval.get("vector_candidates", 12)),
        keyword_candidates=int(retrieval.get("keyword_candidates", 12)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
        rerank_candidates=int(retrieval.get("rerank_candidates", 20)),
        reranker=dict(data.get("reranker", {})),
        chat=dict(data.get("chat", {})),
        embedding=dict(data.get("embedding", {})),
    )


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """读取需要被索引的资料清单。

    manifest 中 `index: false` 的文档会被过滤掉，方便临时保留但不索引。
    """
    data = load_yaml(path)
    documents = data.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("manifest.yaml must contain a documents list")
    return [doc for doc in documents if isinstance(doc, dict) and doc.get("index", True)]


def with_retrieval_mode(config: AppConfig, mode: str | None) -> AppConfig:
    """Return a config with a validated retrieval mode override for CLI experiments."""
    if mode is None:
        return config
    normalized = mode.lower()
    if normalized not in {"vector", "bm25", "hybrid", "rerank"}:
        raise ValueError("retrieval mode must be one of: vector, bm25, hybrid, rerank")
    return replace(config, retrieval_mode=normalized)
