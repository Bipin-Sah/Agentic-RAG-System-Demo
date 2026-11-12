"""Build configured components from Settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config import Settings
from rag.embeddings import Embedder, OpenAIEmbedder
from rag.store import PgVectorStore

if TYPE_CHECKING:
    from rag.agent import RetrievalAgent
    from rag.baseline import NaiveRAG
    from rag.llm import ChatClient
    from rag.retriever import Retriever


class MissingAPIKey(RuntimeError):
    pass


def require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise MissingAPIKey(
            "OPENAI_API_KEY is not set. Export it or add it to .env (see .env.example)."
        )
    return settings.openai_api_key


OFFLINE_SCHEMA = "offline"


def build_store(settings: Settings, schema: str | None = None) -> PgVectorStore:
    """Offline mode gets its own schema so fake vectors never mix with real embeddings."""
    if schema is None:
        schema = OFFLINE_SCHEMA if settings.offline else "public"
    store = PgVectorStore(settings.database_url, settings.embed_dim, schema=schema)
    store.init_schema()
    return store


def build_embedder(settings: Settings) -> Embedder:
    if settings.offline:
        from rag.offline import HashingEmbedder

        return HashingEmbedder(dim=settings.embed_dim)
    return OpenAIEmbedder(
        model=settings.embed_model, dim=settings.embed_dim, api_key=require_api_key(settings)
    )


def build_chat(settings: Settings) -> ChatClient:
    if settings.offline:
        from rag.offline import RuleChatClient

        return RuleChatClient()
    from rag.llm import OpenAIChatClient

    return OpenAIChatClient(model=settings.chat_model, api_key=require_api_key(settings))


def build_retriever(settings: Settings, store: PgVectorStore | None = None) -> Retriever:
    from rag.retriever import Retriever

    return Retriever(store or build_store(settings), build_embedder(settings), settings.top_k)


def build_agent(settings: Settings, store: PgVectorStore | None = None) -> RetrievalAgent:
    from rag.agent import RetrievalAgent

    return RetrievalAgent(
        build_chat(settings), build_retriever(settings, store), default_k=settings.top_k
    )


def build_baseline(settings: Settings, store: PgVectorStore | None = None) -> NaiveRAG:
    from rag.baseline import NaiveRAG

    return NaiveRAG(build_chat(settings), build_retriever(settings, store), settings.top_k)
