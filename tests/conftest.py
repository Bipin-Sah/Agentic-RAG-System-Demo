import uuid

import psycopg
import pytest

from rag.config import Settings
from rag.store import PgVectorStore
from tests.fakes import HashingEmbedder

EMBED_DIM = 64


def _db_available(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def database_url() -> str:
    url = Settings.from_env().database_url
    if not _db_available(url):
        pytest.skip(f"pgvector database not reachable at {url} (run: docker compose up -d db)")
    return url


@pytest.fixture
def store(database_url):
    """A PgVectorStore isolated in a throwaway schema."""
    s = PgVectorStore(database_url, dim=EMBED_DIM, schema=f"test_{uuid.uuid4().hex[:12]}")
    s.init_schema()
    yield s
    s.drop_schema()
    s.close()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=EMBED_DIM)
