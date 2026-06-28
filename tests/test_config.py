from pathlib import Path

from src.config import PROJECT_ROOT, load_config, load_manifest


def test_load_config_defaults():
    config = load_config()

    assert config.raw_data_dir == PROJECT_ROOT / "data/raw/comp9444"
    assert config.collection_name == "comp9444_chunks"
    assert config.top_k > 0


def test_load_manifest_has_documents():
    documents = load_manifest(Path("manifest.yaml"))

    assert len(documents) >= 1
    assert all("path" in doc for doc in documents)

