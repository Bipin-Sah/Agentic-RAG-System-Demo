"""Build configured components from Settings."""

from __future__ import annotations

from rag.config import Settings
from rag.embeddings import OpenAIEmbedder
from rag.store import PgVectorStore


class MissingAPIKey(RuntimeError):
    pass


def require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise MissingAPIKey(
            "OPENAI_API_KEY is not set. Export it or add it to .env (see .env.example)."
        )
    return settings.openai_api_key


def build_store(settings: Settings) -> PgVectorStore:
    store = PgVectorStore(settings.database_url, settings.embed_dim)
    store.init_schema()
    return store


def build_embedder(settings: Settings) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        model=settings.embed_model, dim=settings.embed_dim, api_key=require_api_key(settings)
    )
