from __future__ import annotations

from dataclasses import dataclass

from .providers import ChatClient
from .retrieve import SearchResult


@dataclass(frozen=True)
class Answer:
    """生成阶段的结果，包括回答文本、引用和使用到的上下文。"""

    answer: str
    citations: list[str]
    contexts: list[SearchResult]
    answer_mode: str = "grounded"


def build_messages(query: str, contexts: list[SearchResult]) -> list[dict[str, str]]:
    """构造给 chat model 的 grounded prompt。

    这里明确要求模型只基于检索到的课件上下文回答，并按用户问题或显式要求控制语言。
    """
    context_blocks = []
    for i, item in enumerate(contexts, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{i}] {item.citation}",
                    item.text,
                ]
            )
        )
    joined_context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a RAG assistant for UNSW COMP9444 course materials. "
        "Answer only using the provided retrieved course context. "
        "If the context is insufficient, say that the provided course material is insufficient "
        "and remind the user that they may need to upload or add the relevant course material. "
        "Cite sources using the bracket numbers from the context, such as [1] or [2]. "
        "Use the same language as the user's question by default. "
        "If the user explicitly asks for a specific response language, answer in that requested language."
    )
    user = (
        "Retrieved course context:\n"
        f"{joined_context}\n\n"
        "User question:\n"
        f"{query}\n\n"
        "Follow the user's language preference: use the user's question language by default, "
        "or the explicitly requested answer language if one is specified. "
        "Keep the answer concise and include citations."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_query(chat_client: ChatClient, query: str, contexts: list[SearchResult]) -> Answer:
    """调用大模型生成最终回答。"""
    messages = build_messages(query, contexts)
    text = chat_client.answer(messages)
    citations = [item.citation for item in contexts]
    return Answer(answer=text.strip(), citations=citations, contexts=contexts)
