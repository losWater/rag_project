from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import PROJECT_ROOT, AppConfig
from .generate import Answer


def write_query_log(
    config: AppConfig,
    question: str,
    answer: Answer,
    top_k: int,
    retrieval_query: str | None = None,
    retrieval_queries: list[str] | None = None,
    query_rewritten: bool = False,
    log_dir: Path | None = None,
) -> Path:
    """把一次问答的可调试信息写入 JSONL。

    日志只记录 provider 名称、模型环境变量名、检索片段和回答，不记录 API key。
    """
    now = datetime.now(ZoneInfo("Australia/Sydney"))
    target_dir = log_dir or (PROJECT_ROOT / "logs/queries")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{now.date().isoformat()}.jsonl"

    record = {
        "timestamp": now.isoformat(),
        "question": question,
        "retrieval_query": retrieval_query or question,
        "retrieval_queries": retrieval_queries or [retrieval_query or question],
        "query_rewritten": query_rewritten,
        "answer": answer.answer,
        "answer_mode": answer.answer_mode,
        "top_k": top_k,
        "citations": answer.citations,
        "retrieved": [
            {
                "source_name": item.metadata.get("source_name"),
                "source_path": item.metadata.get("source_path"),
                "title": item.metadata.get("title"),
                "page": item.metadata.get("page"),
                "chunk_index": item.metadata.get("chunk_index"),
                "distance": item.distance,
                "matched_queries": list(item.matched_queries),
                "text_preview": item.text[:300],
            }
            for item in answer.contexts
        ],
        "chat_provider": config.chat.get("provider"),
        "chat_model_env": config.chat.get("model_env"),
        "embedding_provider": config.embedding.get("provider"),
        "embedding_model_env": config.embedding.get("model_env"),
    }

    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return target_path
