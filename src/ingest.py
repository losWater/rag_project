from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from .config import PROJECT_ROOT, resolve_path


@dataclass(frozen=True)
class PageText:
    """PDF 单页文本及其来源元数据。

    后续 chunk 和 citation 都依赖这些字段，所以这里保留 source、title、page 等信息。
    """

    source_path: str
    source_name: str
    title: str
    document_type: str
    tags: list[str]
    page: int
    text: str


def load_pdf_pages(document: dict) -> list[PageText]:
    """从一个 PDF manifest 记录中提取可搜索的页面文本。"""
    path = resolve_path(document["path"])
    if not path.exists():
        raise FileNotFoundError(path)

    title = str(document.get("title") or path.stem)
    document_type = str(document.get("document_type") or "unknown")
    tags = [str(tag) for tag in document.get("tags", [])]
    rel_path = str(path.relative_to(PROJECT_ROOT))

    pages: list[PageText] = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            # 只处理能直接提取文本的页面；纯图片扫描页在第一阶段先跳过。
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                PageText(
                    source_path=rel_path,
                    source_name=path.name,
                    title=title,
                    document_type=document_type,
                    tags=tags,
                    page=index,
                    text=text,
                )
            )
    return pages


def sha256_file(path: str | Path) -> str:
    """计算文件 hash，用于判断 PDF 是否新增或修改过。"""
    import hashlib

    resolved = resolve_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
