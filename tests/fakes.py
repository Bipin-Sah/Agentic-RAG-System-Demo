"""Offline test doubles for OpenAI-backed components."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder: shared words => higher cosine similarity."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        if norm == 1.0 and not any(vec):
            vec[0] = 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]


class ScriptedChatClient:
    """Returns pre-scripted ChatResults in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def complete(self, messages, tools=None, tool_choice=None):
        import copy

        self.requests.append(
            {"messages": copy.deepcopy(messages), "tools": tools, "tool_choice": tool_choice}
        )
        if not self.responses:
            raise AssertionError("ScriptedChatClient ran out of responses")
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response


class FakeStore:
    """In-memory stand-in for PgVectorStore.search, for agent tests without Postgres."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.queries: list[list[float]] = []

    def search(self, embedding, k=5):
        self.queries.append(embedding)

        def score(c):
            return sum(a * b for a, b in zip(embedding, c[1], strict=True))

        ranked = sorted(self.chunks, key=score, reverse=True)[:k]
        from dataclasses import replace

        return [replace(c[0], score=score(c)) for c in ranked]
