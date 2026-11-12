"""Result type returned by the agent and the baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.citations import Citation
from rag.types import RetrievedChunk


@dataclass
class SearchStep:
    query: str
    num_results: int
    ms: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    searches: list[SearchStep] = field(default_factory=list)
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def used_retrieval(self) -> bool:
        return bool(self.searches)

    @property
    def cited_sources(self) -> list[str]:
        out: list[str] = []
        for c in self.citations:
            if c.chunk.source not in out:
                out.append(c.chunk.source)
        return out
