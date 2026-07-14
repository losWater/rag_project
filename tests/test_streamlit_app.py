from streamlit.testing.v1 import AppTest


def test_streamlit_exposes_all_retrieval_modes():
    app = AppTest.from_file("streamlit_app.py").run()

    assert not app.exception
    assert len(app.selectbox) == 1
    assert app.selectbox[0].options == ["rerank", "hybrid", "vector", "bm25"]
    assert app.selectbox[0].value == "rerank"
