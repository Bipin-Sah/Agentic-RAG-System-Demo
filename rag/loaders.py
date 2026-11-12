"""Document loaders for PDF, DOCX and plain-text files."""

from __future__ import annotations

from pathlib import Path

from rag.types import PageText

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx"} | TEXT_EXTENSIONS


class UnsupportedFileType(ValueError):
    pass


def load_pdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(text=text, page=i))
    return pages


def load_docx(path: Path) -> list[PageText]:
    import docx
    from docx.table import Table

    document = docx.Document(str(path))
    blocks: list[str] = []
    # Walk paragraphs and tables in document order so tables keep their surrounding context.
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            rows = []
            for row in item.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
        elif item.text.strip():
            blocks.append(item.text.strip())
    text = "\n\n".join(blocks)
    return [PageText(text=text)] if text else []


def load_text(path: Path) -> list[PageText]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [PageText(text=text)] if text else []


def load_document(path: str | Path) -> list[PageText]:
    """Load a file and return its text, split by page where the format has pages."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in TEXT_EXTENSIONS:
        return load_text(path)
    raise UnsupportedFileType(
        f"Unsupported file type {ext!r} for {path.name}; "
        f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
