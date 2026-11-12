"""Ingestion pipeline: load -> chunk -> embed -> store.

CLI: ``python -m rag.ingest path/to/file_or_dir [...]``
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import chunk_pages
from rag.embeddings import Embedder
from rag.loaders import SUPPORTED_EXTENSIONS, load_document


@dataclass(frozen=True)
class IngestResult:
    source: str
    document_id: int
    num_chunks: int
    skipped: bool = False
    seconds: float = 0.0


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


class Ingestor:
    def __init__(
        self, store, embedder: Embedder, chunk_size: int = 800, chunk_overlap: int = 150
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: str | Path, source: str | None = None) -> IngestResult:
        """Ingest one file.

        Files already in the store (same SHA-256) are skipped; a changed file with the same
        ``source`` name replaces its previous version.
        """
        start = time.perf_counter()
        path = Path(path)
        source = source or path.name
        sha = file_sha256(path)
        existing = self.store.find_document(sha)
        if existing is not None:
            return IngestResult(existing.source, existing.id, existing.num_chunks, skipped=True)

        pages = load_document(path)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError(f"No extractable text found in {source}")
        embeddings = self.embedder.embed([c.content for c in chunks])
        doc_id = self.store.add_document(source, sha, chunks, embeddings)
        return IngestResult(source, doc_id, len(chunks), seconds=time.perf_counter() - start)

    def ingest_paths(self, paths: list[str | Path]) -> list[IngestResult]:
        results = []
        for path, source in expand_paths(paths):
            results.append(self.ingest_file(path, source=source))
        return results


def expand_paths(paths: list[str | Path]) -> list[tuple[Path, str]]:
    """Expand inputs into (file, source name) pairs.

    Directories are searched recursively for supported files; their source name is the path
    relative to the directory (e.g. ``policies/hr.pdf``) so same-named files stay distinct.
    """
    files: list[tuple[Path, str]] = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            files.extend((f, f.relative_to(p).as_posix()) for f in found)
        else:
            files.append((p, p.name))
    return files


def main(argv: list[str] | None = None) -> int:
    from rag.config import Settings
    from rag.factory import build_embedder, build_store

    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("paths", nargs="+", help="Files or directories (pdf, docx, txt, md)")
    parser.add_argument("--reset", action="store_true", help="Delete all documents first")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    embedder = build_embedder(settings)  # fail fast (e.g. missing API key) before touching data
    store = build_store(settings)
    if args.reset:
        store.clear()
    ingestor = Ingestor(store, embedder, settings.chunk_size, settings.chunk_overlap)
    for r in ingestor.ingest_paths(args.paths):
        status = "skipped (already ingested)" if r.skipped else f"{r.seconds:.2f}s"
        print(f"{r.source}: {r.num_chunks} chunks, doc_id={r.document_id} [{status}]")
    print(f"Total chunks in store: {store.count_chunks()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
