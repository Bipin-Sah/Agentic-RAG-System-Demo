import pytest

import rag.factory
from rag.factory import MissingAPIKey
from rag.ingest import main


def test_reset_does_not_clear_when_api_key_missing(monkeypatch, tmp_path):
    cleared = []

    class Store:
        def clear(self):
            cleared.append(True)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rag.config.load_dotenv", lambda: None)
    monkeypatch.setattr(rag.factory, "build_store", lambda settings: Store())
    (tmp_path / "a.txt").write_text("hello")
    with pytest.raises(MissingAPIKey):
        main(["--reset", str(tmp_path / "a.txt")])
    assert cleared == []
