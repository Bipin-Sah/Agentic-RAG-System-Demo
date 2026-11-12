"""Split document text into overlapping chunks along natural boundaries."""

from __future__ import annotations

import re

from rag.types import Chunk, PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str, max_len: int) -> list[str]:
    """Break text into paragraph/sentence/word units no longer than max_len."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_len:
            units.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            if len(sentence) <= max_len:
                units.append(sentence)
                continue
            words = sentence.split(" ")
            buf = ""
            for word in words:
                while len(word) > max_len:
                    if buf:
                        units.append(buf)
                        buf = ""
                    units.append(word[:max_len])
                    word = word[max_len:]
                candidate = f"{buf} {word}" if buf else word
                if len(candidate) > max_len:
                    units.append(buf)
                    buf = word
                else:
                    buf = candidate
            if buf:
                units.append(buf)
    return units


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit) > overlap:
            break
        tail.insert(0, unit)
        total += len(unit) + 1
    return tail


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Greedily pack units into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share up to ``chunk_overlap`` characters of whole units.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in _split_units(text, chunk_size):
        added = len(unit) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, chunk_overlap)
            length = len(" ".join(current))
            # Drop overlap if it would prevent the new unit from fitting.
            while current and length + len(unit) + 1 > chunk_size:
                current.pop(0)
                length = len(" ".join(current))
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[PageText], chunk_size: int = 800, chunk_overlap: int = 150
) -> list[Chunk]:
    """Chunk each page independently so every chunk maps to a single page."""
    chunks: list[Chunk] = []
    for page in pages:
        for content in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(content=content, chunk_index=len(chunks), page=page.page))
    return chunks
