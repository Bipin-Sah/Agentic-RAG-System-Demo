"""Embedding providers."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Embeds text with the OpenAI embeddings API, batching requests."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        batch_size: int = 128,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") or " " for t in texts[start : start + self.batch_size]]
            response = self.client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors
