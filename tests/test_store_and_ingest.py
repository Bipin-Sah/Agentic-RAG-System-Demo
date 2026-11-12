from pathlib import Path

import pytest

from rag.ingest import Ingestor, expand_paths, source_name
from rag.store import PgVectorStore
from rag.types import Chunk
from tests.docgen import write_docx, write_pdf

pytestmark = pytest.mark.db


def test_add_and_search(store, embedder):
    texts = ["cats purr and meow", "rockets launch into orbit", "dogs bark loudly"]
    chunks = [Chunk(content=t, chunk_index=i) for i, t in enumerate(texts)]
    doc_id = store.add_document("animals.txt", "sha-1", chunks, embedder.embed(texts))
    results = store.search(embedder.embed(["rockets orbit"])[0], k=2)
    assert len(results) == 2
    assert results[0].content == "rockets launch into orbit"
    assert results[0].document_id == doc_id
    assert results[0].source == "animals.txt"
    assert results[0].score > results[1].score
    assert results[0].score == pytest.approx(1.0, abs=0.35)


def test_list_delete_clear(store, embedder):
    c = [Chunk("one", 0)]
    a = store.add_document("a.txt", "sha-a", c, embedder.embed(["one"]))
    store.add_document("b.txt", "sha-b", c, embedder.embed(["one"]))
    docs = store.list_documents()
    assert [d.source for d in docs] == ["a.txt", "b.txt"]
    assert all(d.num_chunks == 1 for d in docs)
    store.delete_document(a)
    assert [d.source for d in store.list_documents()] == ["b.txt"]
    store.clear()
    assert store.list_documents() == []
    assert store.count_chunks() == 0


def test_dimension_mismatch_detected(store):
    other = PgVectorStore(store.database_url, dim=store.dim + 1, schema=store.schema)
    with pytest.raises(ValueError, match="vector"):
        other.init_schema()
    other.close()


def test_ingest_mixed_formats_and_dedupe(store, embedder, tmp_path):
    write_pdf(
        tmp_path / "manual.pdf",
        ["The reactor coolant is heavy water.", "Shutdown takes 9 minutes."],
    )
    write_docx(tmp_path / "policy.docx", ["Employees get 25 vacation days per year."])
    (tmp_path / "faq.txt").write_text("The cafeteria opens at 8am.")
    (tmp_path / "ignored.png").write_bytes(b"x")

    ingestor = Ingestor(store, embedder, chunk_size=200, chunk_overlap=20)
    results = ingestor.ingest_paths([tmp_path])
    names = ["faq.txt", "manual.pdf", "policy.docx"]
    assert sorted(r.source for r in results) == [(tmp_path / n).as_posix() for n in names]
    assert not any(r.skipped for r in results)
    assert store.count_chunks() == 4  # 2 pdf pages + docx + txt

    hits = store.search(embedder.embed(["how many vacation days"])[0], k=1)
    assert hits[0].source.endswith("/policy.docx")
    pdf_hits = store.search(embedder.embed(["shutdown minutes"])[0], k=1)
    assert pdf_hits[0].page == 2

    again = ingestor.ingest_file(tmp_path / "faq.txt")
    assert again.skipped
    assert store.count_chunks() == 4


def test_ingest_rejects_empty_file(store, embedder, tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text(" ")
    with pytest.raises(ValueError, match="No extractable text"):
        Ingestor(store, embedder).ingest_file(p)


def test_expand_paths_filters_unsupported_and_uses_relative_sources(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.bin").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("other a")
    single = tmp_path / "sub" / "a.txt"
    expected = [(tmp_path / "a.txt").as_posix(), single.as_posix()]
    assert [s for _, s in expand_paths([tmp_path])] == expected
    assert expand_paths([single]) == [(single, single.as_posix())]  # same name either way


def test_source_name_ignores_cwd_and_symlink_targets(tmp_path, monkeypatch):
    (tmp_path / "here").mkdir()
    (tmp_path / "v1.txt").write_text("v1")
    link = tmp_path / "current.txt"
    link.symlink_to(tmp_path / "v1.txt")
    expected = (tmp_path / "current.txt").as_posix()
    monkeypatch.chdir(tmp_path)
    assert source_name(Path("current.txt")) == expected
    monkeypatch.chdir(tmp_path / "here")
    assert source_name(Path("../current.txt")) == expected
    link.unlink()
    link.symlink_to(tmp_path / "here")  # retargeted symlink keeps its name
    assert source_name(link) == expected


def test_separate_roots_with_same_relative_path_kept_separate(store, embedder, tmp_path):
    for root in ["r1", "r2"]:
        (tmp_path / root / "docs").mkdir(parents=True)
        (tmp_path / root / "docs" / "notes.txt").write_text(f"notes from {root}")
    Ingestor(store, embedder).ingest_paths([tmp_path / "r1" / "docs", tmp_path / "r2" / "docs"])
    sources = sorted(d.source for d in store.list_documents())
    assert sources == [(tmp_path / r / "docs" / "notes.txt").as_posix() for r in ["r1", "r2"]]


def test_same_basename_in_different_dirs_kept_separate(store, embedder, tmp_path):
    for d in ["x", "y"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "notes.txt").write_text(f"notes for team {d}")
    Ingestor(store, embedder).ingest_paths([tmp_path])
    sources = sorted(d.source for d in store.list_documents())
    assert sources == [(tmp_path / d / "notes.txt").as_posix() for d in ["x", "y"]]


def test_changed_file_replaces_previous_version(store, embedder, tmp_path):
    p = tmp_path / "policy.txt"
    p.write_text("The office closes at 5pm.")
    ingestor = Ingestor(store, embedder)
    first = ingestor.ingest_file(p)
    p.write_text("The office closes at 7pm.")
    second = ingestor.ingest_file(p)
    assert not second.skipped and second.document_id != first.document_id
    docs = store.list_documents()
    assert [(d.id, d.source) for d in docs] == [(second.document_id, p.as_posix())]
    hits = store.search(embedder.embed(["office closes"])[0], k=5)
    assert [h.content for h in hits] == ["The office closes at 7pm."]
