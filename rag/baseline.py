"""Naive single-shot RAG baseline: always retrieve top-k for the raw question, one LLM call."""

from __future__ import annotations

import time

from rag.answer import Answer, SearchStep
from rag.citations import (
    CITATION_RULES,
    SourceRegistry,
    format_sources,
    resolve_citations,
    strip_all_labels,
    strip_unknown_labels,
)
from rag.llm import ChatClient
from rag.retriever import Retriever

BASELINE_SYSTEM_PROMPT = (
    "You answer questions about the user's documents using only the sources provided. "
    + CITATION_RULES
)


class NaiveRAG:
    def __init__(self, chat: ChatClient, retriever: Retriever, top_k: int = 5) -> None:
        self.chat = chat
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        start = time.perf_counter()
        result = Answer(question=question, answer="")
        registry = SourceRegistry()

        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, self.top_k)
        result.retrieval_ms = (time.perf_counter() - t0) * 1000
        result.searches.append(SearchStep(question, len(chunks), result.retrieval_ms))
        sources = format_sources(registry.add(chunks))

        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            *[{**m, "content": strip_all_labels(m.get("content") or "")} for m in history or []],
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ]
        t0 = time.perf_counter()
        response = self.chat.complete(messages)
        result.llm_ms = (time.perf_counter() - t0) * 1000
        result.llm_calls = 1
        result.prompt_tokens = response.prompt_tokens
        result.completion_tokens = response.completion_tokens

        result.answer = strip_unknown_labels(response.content or "", registry)
        result.retrieved = registry.chunks
        result.citations = resolve_citations(result.answer, registry)
        result.total_ms = (time.perf_counter() - start) * 1000
        return result
