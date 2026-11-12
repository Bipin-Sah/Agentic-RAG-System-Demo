import pytest

from app.services import (
    citation_rows,
    history_for_model,
    ingest_upload,
    latency_caption,
    latency_record,
    latency_summary,
    safe_filename,
    save_upload,
)
from rag.answer import Answer, SearchStep
from rag.citations import Citation
from rag.types import RetrievedChunk


class RecordingIngestor:
    def __init__(self):
        self.calls = []

    def ingest_file(self, path, source=None):
        self.calls.append((path, source, path.read_bytes()))
        return "result"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/notes.txt", "notes.txt"),
        ("C:\\Users\\me\\Spec Sheet.DOCX", "Spec Sheet.DOCX"),
        ("we$ird<name>.md", "we_ird_name_.md"),
        ("报告.txt", "报告.txt"),
        ("Café notes.md", "Café notes.md"),
    ],
)
def test_safe_filename(name, expected):
    assert safe_filename(name) == expected


@pytest.mark.parametrize("name", ["malware.exe", "noext", "..", ""])
def test_safe_filename_rejects(name):
    with pytest.raises(ValueError, match="Unsupported"):
        safe_filename(name)


def test_save_and_ingest_upload(tmp_path):
    path = save_upload("../a.txt", b"hello", tmp_path / "up")
    assert path == tmp_path / "up" / "a.txt" and path.read_bytes() == b"hello"

    ingestor = RecordingIngestor()
    assert ingest_upload(ingestor, "a.txt", b"v2", tmp_path / "up") == "result"
    [(ingested_path, source, data)] = ingestor.calls
    assert ingested_path.parent == tmp_path / "up" and ingested_path.suffix == ".txt"
    assert ingested_path.name != "a.txt" and (source, data) == ("a.txt", b"v2")
    assert path.read_bytes() == b"v2"
    assert sorted(p.name for p in (tmp_path / "up").iterdir()) == ["a.txt"]


def test_ingest_upload_cleans_up_temp_file_on_failure(tmp_path):
    class Failing:
        def ingest_file(self, path, source=None):
            raise ValueError("No extractable text")

    with pytest.raises(ValueError):
        ingest_upload(Failing(), "b.txt", b"", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_history_for_model_keeps_recent_plain_turns():
    msgs = []
    for i in range(10):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}", "answer": object(), "mode": "x"})
    hist = history_for_model(msgs, turns=2)
    assert hist == [
        {"role": "user", "content": "q8"},
        {"role": "assistant", "content": "a8"},
        {"role": "user", "content": "q9"},
        {"role": "assistant", "content": "a9"},
    ]


def make_answer(total=1500.0, searches=2, calls=3, content="x " * 400):
    chunk = RetrievedChunk(1, 1, "spec.docx", 0, content, 0.9, None)
    return Answer(
        question="q",
        answer="a [S1]",
        citations=[Citation("S1", chunk)],
        searches=[SearchStep("q", 3, 10.0)] * searches,
        total_ms=total,
        retrieval_ms=300.0,
        llm_ms=1200.0,
        llm_calls=calls,
        prompt_tokens=1000,
        completion_tokens=234,
    )


def test_latency_caption():
    cap = latency_caption(make_answer())
    assert cap == (
        "total 1.50s · retrieval 0.30s · LLM 1.20s (3 calls) · 2 searches · 1,234 tokens"
    )
    one = latency_caption(make_answer(searches=1, calls=1))
    assert "(1 call)" in one and "1 search ·" in one


def test_latency_record_and_summary():
    records = [
        latency_record(i + 1, "Agent", make_answer(total=t)) for i, t in enumerate([100, 300, 200])
    ]
    assert records[0].tokens == 1234 and records[0].searches == 2
    s = latency_summary(records)
    assert s == {"count": 3, "last_ms": 200, "mean_ms": 200, "p50_ms": 200, "p95_ms": 300}
    assert latency_summary([]) == {}


def test_citation_rows_truncate_snippets():
    rows = citation_rows(make_answer(), snippet_chars=20)
    assert rows[0]["label"] == "S1" and rows[0]["location"].startswith("spec.docx")
    assert rows[0]["snippet"].endswith("…") and len(rows[0]["snippet"]) <= 21
