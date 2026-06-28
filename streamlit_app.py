from __future__ import annotations

import streamlit as st

from src.config import load_config
from src.index import index_summary
from src.pipeline import ask_question_with_config


st.set_page_config(
    page_title="COMP9444 RAG Assistant",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def cached_config():
    """缓存配置读取，避免 Streamlit 每次重绘页面都重新解析配置。"""
    return load_config()


def citation_rows(contexts):
    """把检索结果转换成表格行，供页面展示引用和距离分数。"""
    return [
        {
            "#": index,
            "source": item.metadata.get("source_name", "unknown"),
            "page": item.metadata.get("page", "?"),
            "chunk": item.metadata.get("chunk_index", "?"),
            "distance": round(item.distance, 4),
            "matched_queries": ", ".join(item.matched_queries),
        }
        for index, item in enumerate(contexts, start=1)
    ]


config = cached_config()

st.title("COMP9444 RAG Assistant")
st.caption("Course-material QA with multi-query retrieval, citations, and retrieval trace inspection.")

with st.sidebar:
    st.header("Controls")
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=6, step=1)
    enable_log = st.toggle("Write query log", value=True)

    st.header("Index")
    try:
        summary = index_summary(config)
        st.metric("Indexed files", summary["indexed_files"])
        st.metric("Chunks", summary["collection_count"])
    except Exception as exc:
        st.error(f"Index status unavailable: {exc}")

    st.header("Models")
    st.text(f"Chat: {config.chat.get('provider')}")
    st.text(f"Embedding: {config.embedding.get('provider')}")

default_question = "请问什么是交叉熵，用英文回答"
question = st.text_area(
    "Question",
    value=default_question,
    height=110,
    placeholder="Ask about COMP9444 course materials...",
)

ask = st.button("Ask", type="primary", use_container_width=False)

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    # 页面入口只负责收集输入和展示输出，实际 RAG 流程统一交给 pipeline。
    with st.spinner("Retrieving course context and generating an answer..."):
        try:
            response = ask_question_with_config(config, question.strip(), top_k=top_k, log=enable_log)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(response.answer.answer or "_No answer returned._")

    st.subheader("Retrieval Queries")
    st.write(response.planned_query.retrieval_queries)

    st.subheader("Citations")
    st.dataframe(citation_rows(response.contexts), use_container_width=True, hide_index=True)

    with st.expander("Retrieved Contexts", expanded=False):
        for index, item in enumerate(response.contexts, start=1):
            st.markdown(f"**[{index}] {item.citation}**")
            st.caption(f"Distance: {item.distance:.4f}")
            if item.matched_queries:
                st.caption("Matched queries: " + ", ".join(item.matched_queries))
            st.text(item.text[:1500])

    if response.log_path:
        st.caption(f"Query log written to {response.log_path}")
