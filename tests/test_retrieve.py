from src.retrieve import retrieve_multi


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

