"""Source labelling and citation parsing shared by the agent and the baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.types import RetrievedChunk

_CITATION_GROUP = re.compile(r"\[((?:S\d+)(?:\s*,\s*S\d+)*)\]")

CITATION_RULES = (
    "Cite every factual claim taken from the sources with its label in square brackets, "
    "e.g. [S1] or [S2][S4]. Only cite labels that appear in the sources. "
    "If the sources do not contain the answer, say you could not find it in the documents "
    "instead of guessing."
)


@dataclass(frozen=True)
class Citation:
    label: str
    chunk: RetrievedChunk


class SourceRegistry:
    """Assigns stable labels (S1, S2, ...) to retrieved chunks, de-duplicating by chunk id."""

    def __init__(self) -> None:
        self._by_label: dict[str, RetrievedChunk] = {}
        self._label_by_id: dict[int, str] = {}

    def add(self, chunks: list[RetrievedChunk]) -> list[tuple[str, RetrievedChunk]]:
        labelled = []
        for chunk in chunks:
            label = self._label_by_id.get(chunk.chunk_id)
            if label is None:
                label = f"S{len(self._by_label) + 1}"
                self._by_label[label] = chunk
                self._label_by_id[chunk.chunk_id] = label
            labelled.append((label, chunk))
        return labelled

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return list(self._by_label.values())

    def get(self, label: str) -> RetrievedChunk | None:
        return self._by_label.get(label)

    def __len__(self) -> int:
        return len(self._by_label)


def format_sources(labelled: list[tuple[str, RetrievedChunk]]) -> str:
    if not labelled:
        return "No matching passages found."
    return "\n\n".join(
        f"[{label}] ({chunk.location})\n{chunk.content}" for label, chunk in labelled
    )


def extract_labels(text: str) -> list[str]:
    """Return citation labels in order of first appearance, e.g. '[S2][S1, S3]' -> S2, S1, S3."""
    seen: list[str] = []
    for group in _CITATION_GROUP.findall(text or ""):
        for label in (p.strip() for p in group.split(",")):
            if label not in seen:
                seen.append(label)
    return seen


def strip_unknown_labels(text: str, registry: SourceRegistry) -> str:
    """Remove labels that don't refer to a retrieved chunk ('[S7]', or 'S7' in '[S1, S7]')."""

    def keep(match: re.Match) -> str:
        labels = [p.strip() for p in match.group(1).split(",")]
        known = [label for label in labels if registry.get(label) is not None]
        return f"[{', '.join(known)}]" if known else ""

    cleaned = _CITATION_GROUP.sub(keep, text or "")
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()


def strip_all_labels(text: str) -> str:
    """Remove every citation label (used to scrub prior turns so stale labels aren't reused)."""
    return strip_unknown_labels(text, SourceRegistry())


def resolve_citations(text: str, registry: SourceRegistry) -> list[Citation]:
    return [
        Citation(label, chunk)
        for label in extract_labels(text)
        if (chunk := registry.get(label)) is not None
    ]
