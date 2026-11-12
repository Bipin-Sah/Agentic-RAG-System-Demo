import pytest

from rag.chunking import chunk_pages, chunk_text
from rag.types import PageText


def test_short_text_single_chunk():
    assert chunk_text("A short sentence.", chunk_size=100, chunk_overlap=10) == [
        "A short sentence."
    ]


def test_empty_text_no_chunks():
    assert chunk_text("  \n\n ", 100, 10) == []


def test_chunks_respect_size_and_cover_text():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    for s in sentences:
        assert s in joined


def test_overlap_between_consecutive_chunks():
    text = " ".join(f"Fact {i} is important." for i in range(40))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.split(". ")[-1].rstrip(".")
        assert last_sentence in nxt


def test_no_overlap():
    text = " ".join(f"Item {i}." for i in range(50))
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=0)
    assert " ".join(chunks).split() == text.split()


def test_very_long_word_is_split():
    chunks = chunk_text("x" * 250, chunk_size=100, chunk_overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_paragraph_boundaries_preferred():
    text = ("alpha " * 10).strip() + "\n\n" + ("beta " * 10).strip()
    chunks = chunk_text(text, chunk_size=70, chunk_overlap=0)
    assert chunks == [("alpha " * 10).strip(), ("beta " * 10).strip()]


def test_chunk_pages_tracks_page_and_index():
    pages = [PageText("Page one text. " * 20, page=1), PageText("Page two text.", page=2)]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2
    assert all(c.page == 1 for c in chunks[:-1])


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)
