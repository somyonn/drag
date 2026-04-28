from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests

from rag.schemas import RetrievedChunk


@dataclass
class GenerationResult:
    answer: str
    prompt: str


class LLMClient:
    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        if not context:
            return "I could not find relevant context in the index."
        snippets = " ".join(chunk.text[:120] for chunk in context[:2])
        return f"[MOCK ANSWER] Based on retrieved context: {snippets}"


class OpenAIChatClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout_s: int = 30) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "You answer using the retrieved context and cite uncertainty when needed."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class CloudThenMockLLMClient(LLMClient):
    """Try cloud LLM first, then fallback to mock."""

    def __init__(self) -> None:
        self._mock = MockLLMClient()

    def generate(self, prompt: str, context: list[RetrievedChunk]) -> str:
        try:
            return OpenAIChatClient().generate(prompt, context)
        except Exception:
            return self._mock.generate(prompt, context)


def build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {i}] source={item.source_uri} score={item.score:.4f}\n{item.text}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context."
    return (
        "You are a helpful assistant. Use the provided context to answer.\n\n"
        f"{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

