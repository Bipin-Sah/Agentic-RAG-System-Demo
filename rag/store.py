"""Postgres + pgvector vector store."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag.types import Chunk, RetrievedChunk

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    source: str
    sha256: str
    num_chunks: int
    created_at: datetime


class PgVectorStore:
    """Stores documents and chunk embeddings in Postgres using the pgvector extension.

    ``schema`` lets tests isolate their data in a throwaway schema.
    """

    def __init__(self, database_url: str, dim: int, schema: str = "public") -> None:
        if not _IDENT.match(schema):
            raise ValueError(f"Invalid schema name: {schema!r}")
        self.database_url = database_url
        self.dim = dim
        self.schema = schema
        self._conn: psycopg.Connection | None = None

    # -- connection -----------------------------------------------------------------
    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            conn = psycopg.connect(self.database_url, autocommit=True, row_factory=dict_row)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # -- schema ---------------------------------------------------------------------
    def init_schema(self) -> None:
        s = self.schema
        with self.conn.transaction():
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s}.documents (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s}.chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES {s}.documents(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    page INT,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({self.dim}) NOT NULL
                )"""
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON {s}.chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )
        existing = self._column_dim()
        if existing is not None and existing != self.dim:
            raise ValueError(
                f"Existing table {s}.chunks uses vector({existing}) but RAG_EMBED_DIM={self.dim}. "
                "Reset the database (docker compose down -v) or set RAG_EMBED_DIM to match."
            )

    def _column_dim(self) -> int | None:
        row = self.conn.execute(
            """
            SELECT a.atttypmod AS dim FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = 'chunks' AND a.attname = 'embedding'
            """,
            (self.schema,),
        ).fetchone()
        return row["dim"] if row else None

    def drop_schema(self) -> None:
        self.conn.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

    # -- writes ---------------------------------------------------------------------
    def find_document(self, sha256: str) -> DocumentInfo | None:
        for doc in self.list_documents():
            if doc.sha256 == sha256:
                return doc
        return None

    def add_document(
        self,
        source: str,
        sha256: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        s = self.schema
        with self.conn.transaction():
            # A new version of a file replaces any previous version with the same source name.
            self.conn.execute(f"DELETE FROM {s}.documents WHERE source = %s", (source,))
            row = self.conn.execute(
                f"INSERT INTO {s}.documents (source, sha256) VALUES (%s, %s) RETURNING id",
                (source, sha256),
            ).fetchone()
            doc_id = row["id"]
            with self.conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {s}.chunks "
                    "(document_id, chunk_index, page, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (
                            doc_id,
                            c.chunk_index,
                            c.page,
                            c.content,
                            Jsonb(c.metadata),
                            np.asarray(e, dtype=np.float32),
                        )
                        for c, e in zip(chunks, embeddings, strict=True)
                    ],
                )
        return doc_id

    def delete_document(self, document_id: int) -> None:
        self.conn.execute(f"DELETE FROM {self.schema}.documents WHERE id = %s", (document_id,))

    def clear(self) -> None:
        self.conn.execute(f"TRUNCATE {self.schema}.documents, {self.schema}.chunks CASCADE")

    # -- reads ----------------------------------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        s = self.schema
        rows = self.conn.execute(
            f"""
            SELECT d.id, d.source, d.sha256, d.created_at, count(c.id) AS num_chunks
            FROM {s}.documents d LEFT JOIN {s}.chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.id
            """
        ).fetchall()
        return [DocumentInfo(**r) for r in rows]

    def count_chunks(self) -> int:
        return self.conn.execute(f"SELECT count(*) AS n FROM {self.schema}.chunks").fetchone()["n"]

    def search(self, embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        """Return the ``k`` chunks with the highest cosine similarity to ``embedding``."""
        s = self.schema
        query = np.asarray(embedding, dtype=np.float32)
        rows = self.conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.document_id, d.source, c.chunk_index, c.page, c.content,
                   1 - (c.embedding <=> %s) AS score
            FROM {s}.chunks c JOIN {s}.documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()
        return [RetrievedChunk(**{**r, "score": float(r["score"])}) for r in rows]
