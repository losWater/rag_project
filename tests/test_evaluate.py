from src.evaluate import RetrievalCase, RetrievalEvalResult, load_retrieval_cases, summarize_results


def test_load_retrieval_cases():
    cases = load_retrieval_cases("data/eval/retrieval_cases.yaml")

    assert len(cases) >= 1
    assert cases[0].id == "cross_entropy_zh"


def test_summarize_results():
    case = RetrievalCase(id="x", question="q", expected_sources=[], expected_pages=[])
    results = [
        RetrievalEvalResult(case=case, retrieval_queries=[], results=[], source_hit=True, page_hit=False),
        RetrievalEvalResult(case=case, retrieval_queries=[], results=[], source_hit=True, page_hit=True),
    ]

    summary = summarize_results(results)

    assert summary["cases"] == 2
    assert summary["source_recall"] == 1.0
    assert summary["page_recall"] == 0.5

