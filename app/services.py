"""UI-independent helpers for the Streamlit app (kept free of Streamlit so they are testable)."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from rag.answer import Answer
from rag.ingest import Ingestor, IngestResult
from rag.loaders import SUPPORTED_EXTENSIONS

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
HISTORY_TURNS = 6
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


class AnswerSystem(Protocol):
    def answer(self, question: str, history: list[dict] | None = None) -> Answer: ...


@dataclass
class Components:
    """Everything the UI talks to; built from Settings in production, from fakes in tests."""

    store: object  # PgVectorStore-like: list_documents, delete_document, count_chunks
    ingestor: Ingestor
    systems: dict[str, AnswerSystem] = field(default_factory=dict)


def safe_filename(name: str) -> str:
    """Basename only, with unusual characters replaced, so uploads can't escape upload_dir."""
    base = Path(name.replace("\\", "/")).name
    cleaned = _UNSAFE.sub("_", base).strip(" .")
    if not cleaned or Path(cleaned).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file {name!r}. Supported types: {', '.join(UPLOAD_TYPES)}")
    return cleaned


def save_upload(name: str, data: bytes, upload_dir: Path) -> Path:
    filename = safe_filename(name)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    path.write_bytes(data)
    return path


def ingest_upload(ingestor: Ingestor, name: str, data: bytes, upload_dir: Path) -> IngestResult:
    """Save and ingest an uploaded file; re-uploading a changed file replaces its old version."""
    path = save_upload(name, data, upload_dir)
    return ingestor.ingest_file(path, source=path.name)


def history_for_model(messages: list[dict], turns: int = HISTORY_TURNS) -> list[dict]:
    """Last `turns` user/assistant exchanges as plain chat messages for the model."""
    plain = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in {"user", "assistant"}
    ]
    return plain[-2 * turns :]


@dataclass
class LatencyRecord:
    turn: int
    mode: str
    question: str
    total_ms: float
    retrieval_ms: float
    llm_ms: float
    searches: int
    llm_calls: int
    tokens: int


def latency_record(turn: int, mode: str, answer: Answer) -> LatencyRecord:
    return LatencyRecord(
        turn=turn,
        mode=mode,
        question=answer.question,
        total_ms=round(answer.total_ms, 1),
        retrieval_ms=round(answer.retrieval_ms, 1),
        llm_ms=round(answer.llm_ms, 1),
        searches=len(answer.searches),
        llm_calls=answer.llm_calls,
        tokens=answer.prompt_tokens + answer.completion_tokens,
    )


def latency_rows(records: list[LatencyRecord]) -> list[dict]:
    return [asdict(r) for r in records]


def latency_summary(records: list[LatencyRecord]) -> dict[str, float]:
    if not records:
        return {}
    totals = sorted(r.total_ms for r in records)
    return {
        "count": len(totals),
        "last_ms": records[-1].total_ms,
        "mean_ms": statistics.fmean(totals),
        "p50_ms": statistics.median(totals),
        "p95_ms": totals[min(len(totals) - 1, round(0.95 * (len(totals) - 1)))],
    }


def latency_caption(answer: Answer) -> str:
    parts = [
        f"total {answer.total_ms / 1000:.2f}s",
        f"retrieval {answer.retrieval_ms / 1000:.2f}s",
        f"LLM {answer.llm_ms / 1000:.2f}s ({answer.llm_calls} call{'s' * (answer.llm_calls != 1)})",
        f"{len(answer.searches)} search{'es' * (len(answer.searches) != 1)}",
    ]
    tokens = answer.prompt_tokens + answer.completion_tokens
    if tokens:
        parts.append(f"{tokens:,} tokens")
    return " · ".join(parts)


def citation_rows(answer: Answer, snippet_chars: int = 300) -> list[dict]:
    rows = []
    for c in answer.citations:
        text = " ".join(c.chunk.content.split())
        if len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + "…"
        rows.append({"label": c.label, "location": c.chunk.location, "snippet": text})
    return rows
