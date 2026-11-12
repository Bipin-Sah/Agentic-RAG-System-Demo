import pytest

from rag.ingest import Ingestor, expand_paths
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
    assert sorted(r.source for r in results) == ["faq.txt", "manual.pdf", "policy.docx"]
    assert not any(r.skipped for r in results)
    assert store.count_chunks() == 4  # 2 pdf pages + docx + txt

    hits = store.search(embedder.embed(["how many vacation days"])[0], k=1)
    assert hits[0].source == "policy.docx"
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


def test_expand_paths_filters_unsupported(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.bin").write_text("b")
    assert [p.name for p in expand_paths([tmp_path])] == ["a.txt"]
