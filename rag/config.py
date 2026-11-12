"""Runtime configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_DATABASE_URL = "postgresql://rag:rag@localhost:5433/rag"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    offline: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> Settings:
        if load_dotenv_file:
            load_dotenv()
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            chat_model=os.getenv("RAG_CHAT_MODEL", cls.chat_model),
            embed_model=os.getenv("RAG_EMBED_MODEL", cls.embed_model),
            embed_dim=_int_env("RAG_EMBED_DIM", cls.embed_dim),
            chunk_size=_int_env("RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int_env("RAG_TOP_K", cls.top_k),
            offline=os.getenv("RAG_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"},
        )
