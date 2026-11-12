import pytest

from rag.config import DEFAULT_DATABASE_URL, Settings


def test_defaults_when_env_empty(monkeypatch):
    for name in ["DATABASE_URL", "OPENAI_API_KEY", "RAG_CHUNK_SIZE", "RAG_TOP_K"]:
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env(load_dotenv_file=False)
    assert s.database_url == DEFAULT_DATABASE_URL
    assert s.openai_api_key is None
    assert s.chunk_size == 800
    assert s.top_k == 5


def test_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "300")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-test")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.database_url == "postgresql://x/y"
    assert s.chunk_size == 300
    assert s.chunk_overlap == 50
    assert s.chat_model == "gpt-test"


def test_invalid_int_env(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "many")
    with pytest.raises(ValueError, match="RAG_TOP_K"):
        Settings.from_env(load_dotenv_file=False)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunking(size, overlap):
    with pytest.raises(ValueError):
        Settings(chunk_size=size, chunk_overlap=overlap)
