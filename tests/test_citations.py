from rag.citations import (
    SourceRegistry,
    extract_labels,
    format_sources,
    resolve_citations,
    strip_all_labels,
    strip_unknown_labels,
)
from rag.types import RetrievedChunk


def chunk(i, source="doc.txt", page=None):
    return RetrievedChunk(
        chunk_id=i,
        document_id=1,
        source=source,
        chunk_index=i,
        content=f"content {i}",
        score=0.9,
        page=page,
    )


def test_extract_labels_handles_groups_and_dedupes():
    text = "A [S2]. B [S1, S3]. C [S2][S4]. Not a cite: [5] [source]."
    assert extract_labels(text) == ["S2", "S1", "S3", "S4"]


def test_extract_labels_empty():
    assert extract_labels("") == []
    assert extract_labels(None) == []


def test_registry_assigns_stable_labels_and_dedupes():
    reg = SourceRegistry()
    first = reg.add([chunk(10), chunk(11)])
    second = reg.add([chunk(11), chunk(12)])
    assert [label for label, _ in first] == ["S1", "S2"]
    assert [label for label, _ in second] == ["S2", "S3"]
    assert len(reg) == 3


def test_resolve_citations_ignores_unknown_labels():
    reg = SourceRegistry()
    reg.add([chunk(1, "a.pdf", page=3), chunk(2, "b.txt")])
    cites = resolve_citations("x [S2] y [S9] z [S1]", reg)
    assert [(c.label, c.chunk.source) for c in cites] == [("S2", "b.txt"), ("S1", "a.pdf")]


def test_format_sources():
    text = format_sources([("S1", chunk(1, "a.pdf", page=3))])
    assert text.startswith("[S1] (a.pdf, p.3, chunk 1)")
    assert "content 1" in text
    assert format_sources([]) == "No matching passages found."


def test_strip_unknown_labels():
    reg = SourceRegistry()
    reg.add([chunk(1), chunk(2)])
    text = "A [S1]. B [S7]. C [S2, S9]. D [S8, S9]."
    assert strip_unknown_labels(text, reg) == "A [S1]. B. C [S2]. D."


def test_strip_all_labels():
    assert strip_all_labels("25 days [S1][S2, S3].") == "25 days."
    assert strip_all_labels("no labels here") == "no labels here"
