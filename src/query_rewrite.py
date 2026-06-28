from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .providers import ChatClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryForRetrieval:
    """用户问题经过检索规划后的结果。"""

    original_query: str
    retrieval_query: str
    retrieval_queries: list[str]
    rewritten: bool


def is_mostly_english(text: str) -> bool:
    # 课件是英文。如果用户问题中包含中文，就触发英文检索改写；
    # 即使问题里已经包含 PyTorch 这类英文技术词，也仍然需要保留中文意图。
    if re.search(r"[\u3400-\u9fff]", text):
        return False
    return True


TERM_TRANSLATIONS = {
    # 高频中文术语的确定性兜底。这样即使 LLM query planner 不可用，也能搜到核心概念。
    "交叉熵": "cross entropy loss",
    "感知机": "perceptron",
    "反向传播": "backpropagation",
    "过拟合": "overfitting",
    "欠拟合": "underfitting",
    "正则化": "regularization",
    "梯度下降": "gradient descent",
    "损失函数": "loss function",
    "激活函数": "activation function",
    "神经网络": "neural network",
}


def _unique(items: list[str]) -> list[str]:
    """清理、去重，并保持原始顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = clean_rewritten_query(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def glossary_queries(query: str) -> list[str]:
    """根据内置术语表生成检索词。"""
    matches = [english for chinese, english in TERM_TRANSLATIONS.items() if chinese in query]
    expanded: list[str] = []
    for match in matches:
        expanded.append(match)
        if match == "cross entropy loss":
            expanded.extend(["cross entropy", "cross entropy derivation", "cross entropy KL divergence"])
    return _unique(expanded)


def glossary_query(query: str) -> str | None:
    queries = glossary_queries(query)
    return queries[0] if queries else None


def clean_rewritten_query(text: str) -> str:
    """清理 LLM 生成的检索词，去掉课程名、回答语言要求等检索噪声。"""
    cleaned = text.strip().strip('"').strip("'")
    banned_phrases = [
        "COMP9444",
        "comp9444",
        "neural networks lecture notes",
        "lecture notes",
        "course materials",
        "course material",
        "answer in English",
        "respond in English",
    ]
    for phrase in banned_phrases:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    return cleaned


def parse_query_lines(text: str) -> list[str]:
    """解析 LLM 返回的多行检索词。"""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        if line:
            lines.append(line)
    if len(lines) <= 1 and "," in text:
        lines = [part.strip() for part in text.split(",") if part.strip()]
    return _unique(lines)


def rewrite_for_retrieval(chat_client: ChatClient, query: str, max_queries: int = 5) -> QueryForRetrieval:
    """把用户问题转换成一个或多个适合英文课件检索的 query。

    英文问题直接使用原句。中文问题先走术语表，如果检索词还不够，再让 LLM 生成
    3 到 5 个英文检索词。最终回答仍然使用原始用户问题，避免丢失语言要求和意图。
    """
    if is_mostly_english(query):
        return QueryForRetrieval(original_query=query, retrieval_query=query, retrieval_queries=[query], rewritten=False)

    planned_queries = glossary_queries(query)

    if len(planned_queries) < 3:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 3 to 5 concise English retrieval queries for searching English COMP9444 "
                    "course materials. Return one query per line. Preserve technical concepts and "
                    "document-intent clues such as assignment, rubric, submission, deadline, code, "
                    "implementation, tutorial, formula, or concept. Do not answer the question. "
                    "Do not include requested answer language."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            planned_queries.extend(parse_query_lines(chat_client.answer(messages)))
        except Exception as exc:
            logger.warning("Query planning failed; falling back to glossary/raw query: %s", exc)

    planned_queries = _unique(planned_queries)[:max_queries]
    if not planned_queries:
        planned_queries = [query]

    return QueryForRetrieval(
        original_query=query,
        retrieval_query=planned_queries[0],
        retrieval_queries=planned_queries,
        rewritten=planned_queries != [query],
    )
