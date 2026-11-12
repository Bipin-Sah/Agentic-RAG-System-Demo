"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    """Raw text extracted from one page (PDF) or the whole document (DOCX/text)."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    """A piece of a document ready to be embedded."""

    content: str
    chunk_index: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search."""

    chunk_id: int
    document_id: int
    source: str
    chunk_index: int
    content: str
    score: float
    page: int | None = None

    @property
    def location(self) -> str:
        loc = self.source
        if self.page is not None:
            loc += f", p.{self.page}"
        return f"{loc}, chunk {self.chunk_index}"
