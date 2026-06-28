from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import requests
from openai import OpenAI


class EmbeddingClient(Protocol):
    """Embedding provider 的统一接口。"""

    def embed(self, text: str) -> list[float]:
        ...


class ChatClient(Protocol):
    """Chat provider 的统一接口。"""

    def answer(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass
class OllamaEmbeddingClient:
    """通过本地 Ollama 服务生成 embedding。"""

    base_url: str
    model: str
    timeout: int = 60

    def embed(self, text: str) -> list[float]:
        url = self.base_url.rstrip("/") + "/api/embeddings"
        response = requests.post(url, json={"model": self.model, "prompt": text}, timeout=self.timeout)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response did not contain an embedding list")
        return [float(value) for value in embedding]


@dataclass
class OpenAIEmbeddingClient:
    """通过 OpenAI-compatible embedding API 生成 embedding。"""

    base_url: str
    api_key: str
    model: str

    def embed(self, text: str) -> list[float]:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=text)
        return [float(value) for value in response.data[0].embedding]


@dataclass
class OpenAICompatibleChatClient:
    """调用 DeepSeek / OpenAI-compatible chat completion API。"""

    base_url: str
    api_key: str
    model: str
    max_tokens: int = 800
    temperature: float = 0.2

    def answer(self, messages: list[dict[str, str]]) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


@dataclass
class OllamaChatClient:
    """通过本地 Ollama 服务调用 chat model。"""

    base_url: str
    model: str
    temperature: float = 0.2
    timeout: int = 120

    def answer(self, messages: list[dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        response = requests.post(
            url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json().get("message", {}).get("content", ""))


def _env_value(config: dict, key: str, default: str | None = None) -> str:
    """根据配置中的环境变量名读取真实值，避免把密钥写进 YAML。"""
    env_name = config.get(key)
    if env_name:
        value = os.getenv(str(env_name))
        if value:
            return value
    if default is not None:
        return default
    raise RuntimeError(f"Missing environment value for {key}: {env_name}")


def create_embedding_client(config: dict) -> EmbeddingClient:
    """根据配置创建 embedding client。"""
    provider = str(config.get("provider", "ollama"))
    if provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=_env_value(config, "base_url_env", "http://localhost:11434"),
            model=_env_value(config, "model_env", "nomic-embed-text"),
        )
    if provider == "openai_api":
        return OpenAIEmbeddingClient(
            base_url=_env_value(config, "base_url_env", "https://api.openai.com/v1"),
            api_key=_env_value(config, "api_key_env"),
            model=_env_value(config, "model_env", "text-embedding-3-small"),
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def create_chat_client(config: dict) -> ChatClient:
    """根据配置创建 chat client。"""
    provider = str(config.get("provider", "deepseek_api"))
    if provider in {"deepseek_api", "openai_api"}:
        return OpenAICompatibleChatClient(
            base_url=_env_value(config, "base_url_env", "https://api.deepseek.com"),
            api_key=_env_value(config, "api_key_env"),
            model=_env_value(config, "model_env", "deepseek-v4-flash"),
            max_tokens=int(config.get("max_tokens", 800)),
            temperature=float(config.get("temperature", 0.2)),
        )
    if provider == "ollama":
        return OllamaChatClient(
            base_url=_env_value(config, "base_url_env", "http://localhost:11434"),
            model=_env_value(config, "model_env", "llama3.2"),
            temperature=float(config.get("temperature", 0.2)),
        )
    raise ValueError(f"Unsupported chat provider: {provider}")
