from src.retrieve import reciprocal_rank_fusion, retrieve_bm25_multi, retrieve_multi, tokenize_for_bm25


class FakeEmbedding:
    def embed(self, text):
        return [float(len(text))]


class FakeCollection:
    def query(self, query_embeddings, n_results, include):
        marker = int(query_embeddings[0][0])
        if marker == len("cross entropy loss"):
            return {
                "documents": [["doc a", "doc b"]],
                "metadatas": [[
                    {"source_path": "a.pdf", "page": 1, "chunk_index": 1},
                    {"source_path": "b.pdf", "page": 1, "chunk_index": 1},
                ]],
                "distances": [[1.0, 5.0]],
            }
        return {
            "documents": [["doc a", "doc c"]],
            "metadatas": [[
                {"source_path": "a.pdf", "page": 1, "chunk_index": 1},
                {"source_path": "c.pdf", "page": 1, "chunk_index": 1},
            ]],
            "distances": [[2.0, 3.0]],
        }


def test_retrieve_multi_deduplicates_and_tracks_matched_queries(monkeypatch):
    import src.retrieve as retrieve_module

    monkeypatch.setattr(retrieve_module, "get_collection", lambda config: FakeCollection())

    class Config:
        top_k = 3

    results = retrieve_multi(Config(), FakeEmbedding(), ["cross entropy loss", "cross entropy"], top_k=3)

    assert len(results) == 3
    assert results[0].metadata["source_path"] == "a.pdf"
    assert results[0].matched_queries == ("cross entropy loss", "cross entropy")


class FakeKeywordCollection:
    def get(self, include):
        return {
            "documents": [
                "Adam optimizer uses adaptive learning rates",
                "Convolutional networks use learned filters",
                "ReLU is a common activation function",
            ],
            "metadatas": [
                {"source_path": "adam.pdf", "page": 1, "chunk_index": 0},
                {"source_path": "cnn.pdf", "page": 2, "chunk_index": 0},
                {"source_path": "relu.pdf", "page": 3, "chunk_index": 0},
            ],
        }


def test_retrieve_bm25_multi_matches_exact_term(monkeypatch):
    import src.retrieve as retrieve_module

    monkeypatch.setattr(retrieve_module, "get_collection", lambda config: FakeKeywordCollection())

    class Config:
        top_k = 2

    results = retrieve_bm25_multi(Config(), ["Adam optimizer"], top_k=2)

    assert results[0].metadata["source_path"] == "adam.pdf"
    assert results[0].retrieval_sources == ("bm25",)
    assert results[0].fusion_score is not None


def test_reciprocal_rank_fusion_combines_sources():
    from src.retrieve import SearchResult

    metadata = {"source_path": "shared.pdf", "page": 1, "chunk_index": 0}
    vector = SearchResult("shared", metadata, 0.2, retrieval_sources=("vector",))
    keyword = SearchResult("shared", metadata, None, retrieval_sources=("bm25",), fusion_score=3.0)
    keyword_only = SearchResult(
        "keyword only",
        {"source_path": "keyword.pdf", "page": 2, "chunk_index": 0},
        None,
        retrieval_sources=("bm25",),
        fusion_score=2.0,
    )

    results = reciprocal_rank_fusion([[vector], [keyword, keyword_only]], top_k=2, rrf_k=60)

    assert results[0].metadata["source_path"] == "shared.pdf"
    assert results[0].retrieval_sources == ("vector", "bm25")
    assert results[0].fusion_score == 2 / 61


def test_bm25_tokenizer_normalizes_simple_english_plurals():
    assert tokenize_for_bm25("assessment components and weights") == [
        "assessment",
        "component",
        "and",
        "weight",
    ]
    assert tokenize_for_bm25("cross entropy loss") == ["cross", "entropy", "loss"]
