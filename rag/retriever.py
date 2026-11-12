"""Embeds a query and searches the vector store."""

from __future__ import annotations

from rag.embeddings import Embedder
from rag.types import RetrievedChunk


class Retriever:
    def __init__(self, store, embedder: Embedder, top_k: int = 5) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, k or self.top_k)
