import pytest

from rag.loaders import UnsupportedFileType, load_document
from tests.docgen import write_docx, write_pdf


def test_load_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Hello world.\n\nSecond paragraph.")
    pages = load_document(p)
    assert len(pages) == 1
    assert pages[0].page is None
    assert "Second paragraph." in pages[0].text


def test_load_markdown(tmp_path):
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nBody text")
    assert "Body text" in load_document(p)[0].text


def test_load_empty_text_returns_nothing(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n")
    assert load_document(p) == []


def test_load_pdf_keeps_page_numbers(tmp_path):
    p = write_pdf(tmp_path / "doc.pdf", ["The first page mentions apples.", "Page two: bananas."])
    pages = load_document(p)
    assert [pg.page for pg in pages] == [1, 2]
    assert "apples" in pages[0].text
    assert "bananas" in pages[1].text


def test_load_docx_paragraphs_and_tables(tmp_path):
    p = write_docx(
        tmp_path / "doc.docx",
        ["Quarterly report.", "Revenue grew."],
        table=[["Region", "Sales"], ["EMEA", "42"]],
    )
    pages = load_document(p)
    assert len(pages) == 1
    text = pages[0].text
    assert "Quarterly report." in text
    assert "EMEA | 42" in text


def test_unsupported_extension(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileType):
        load_document(p)


def test_docx_table_stays_in_document_order(tmp_path):
    p = write_docx(
        tmp_path / "doc.docx",
        ["Intro.", "Sales table:"],
        table=[["EMEA", "42"]],
        after=["Closing remarks."],
    )
    text = load_document(p)[0].text
    assert text.index("Sales table:") < text.index("EMEA | 42") < text.index("Closing remarks.")
