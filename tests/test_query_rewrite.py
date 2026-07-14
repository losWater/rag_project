from src.query_rewrite import glossary_queries, glossary_query, is_mostly_english, parse_query_lines, rewrite_for_retrieval


def test_is_mostly_english_for_english_question():
    assert is_mostly_english("What is cross entropy?")


def test_is_mostly_english_for_chinese_question():
    assert not is_mostly_english("请问什么是交叉熵，用英文回答")


def test_is_mostly_english_for_mixed_technical_question():
    assert not is_mostly_english("请问 PyTorch 怎么实现 neural network")


def test_glossary_query_for_cross_entropy():
    assert glossary_query("请问什么是交叉熵，用英文回答") == "cross entropy loss"


def test_glossary_queries_expand_cross_entropy():
    assert glossary_queries("请问什么是交叉熵，用英文回答") == [
        "cross entropy loss",
        "cross entropy",
        "cross entropy derivation",
        "cross entropy KL divergence",
    ]


def test_parse_query_lines():
    assert parse_query_lines("1. cross entropy\n2. loss function\n- cross entropy") == [
        "cross entropy",
        "loss function",
    ]


def test_english_query_removes_course_name_retrieval_noise():
    result = rewrite_for_retrieval(None, "What are the assessment weights in COMP9444?")

    assert result.original_query == "What are the assessment weights in COMP9444?"
    assert result.retrieval_queries == ["What are the assessment weights in"]
    assert result.rewritten
