"""Offline test doubles for OpenAI-backed components."""

from __future__ import annotations

from rag.offline import HashingEmbedder

__all__ = ["FakeStore", "HashingEmbedder", "ScriptedChatClient"]


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
