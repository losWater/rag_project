from src.chunk import split_text


def test_split_text_uses_overlap():
    chunks = split_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, chunk_overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_split_text_drops_blank_lines():
    chunks = split_text("a\n\n b \n", chunk_size=10, chunk_overlap=2)

    assert chunks == ["a\nb"]

