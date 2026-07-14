from __future__ import annotations

import streamlit as st

from src.config import load_config, with_retrieval_mode
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
    """把检索结果转换成表格行，供页面展示检索来源和分数。"""
    return [
        {
            "#": index,
            "source": item.metadata.get("source_name", "unknown"),
            "page": item.metadata.get("page", "?"),
            "chunk": item.metadata.get("chunk_index", "?"),
            "retrieved_by": "+".join(item.retrieval_sources),
            "score": round(item.rerank_score, 4) if item.rerank_score is not None else (
                round(item.fusion_score, 4) if item.fusion_score is not None else item.distance
            ),
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
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["rerank", "hybrid", "vector", "bm25"],
        index=["rerank", "hybrid", "vector", "bm25"].index(config.retrieval_mode),
    )
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
    if retrieval_mode == "rerank":
        st.text(f"Reranker: {config.reranker.get('model')}")

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
            effective_config = with_retrieval_mode(config, retrieval_mode)
            response = ask_question_with_config(effective_config, question.strip(), top_k=top_k, log=enable_log)
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
            st.caption("Retrieved by: " + "+".join(item.retrieval_sources))
            if item.rerank_score is not None:
                st.caption(f"Rerank score: {item.rerank_score:.4f}")
            elif item.fusion_score is not None:
                st.caption(f"Fusion score: {item.fusion_score:.4f}")
            elif item.distance is not None:
                st.caption(f"Distance: {item.distance:.4f}")
            if item.matched_queries:
                st.caption("Matched queries: " + ", ".join(item.matched_queries))
            st.text(item.text[:1500])

    if response.log_path:
        st.caption(f"Query log written to {response.log_path}")
