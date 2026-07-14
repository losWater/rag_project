from src.rerank import rerank_results
from src.retrieve import SearchResult


class FakeScorer:
    def predict(self, pairs):
        return [0.2 if "weak" in document else 0.9 for _, document in pairs]


def test_rerank_results_orders_by_cross_encoder_score():
    candidates = [
        SearchResult("weak match", {"source_path": "a", "page": 1, "chunk_index": 0}, 0.1),
        SearchResult("strong match", {"source_path": "b", "page": 1, "chunk_index": 0}, 0.2),
    ]

    results = rerank_results(candidates, "query", FakeScorer(), top_k=2)

    assert results[0].text == "strong match"
    assert results[0].rerank_score == 0.9
    assert results[1].rerank_score == 0.2
