from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_config, load_manifest, with_retrieval_mode


def test_load_config_defaults():
    config = load_config()

    assert config.raw_data_dir == PROJECT_ROOT / "data/raw/comp9444"
    assert config.collection_name == "comp9444_chunks"
    assert config.top_k > 0


def test_load_manifest_has_documents():
    documents = load_manifest(Path("manifest.yaml"))

    assert len(documents) >= 1
    assert all("path" in doc for doc in documents)


def test_with_retrieval_mode_validates_override():
    config = load_config()

    assert with_retrieval_mode(config, "BM25").retrieval_mode == "bm25"
    with pytest.raises(ValueError, match="retrieval mode"):
        with_retrieval_mode(config, "invalid")
