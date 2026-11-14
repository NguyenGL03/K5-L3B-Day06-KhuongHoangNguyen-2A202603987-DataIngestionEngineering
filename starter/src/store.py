"""Store and search vectors using PostgreSQL + pgvector (exact cosine search)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

TABLE = "rag_chunks"
# Columns a caller may filter on; values are always passed as SQL parameters.
FILTERABLE = {"document_id", "source", "page", "section", "section_no"}


def _vec(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def content_hash(chunk: Chunk, model_name: str) -> str:
    """Hash of everything that determines a stored row's vector and metadata."""
    payload = json.dumps([model_name, chunk.text, chunk.page, chunk.metadata],
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is empty")
        if not 0 < dimensions <= 16000:
            raise ValueError(f"Unsupported vector dimension {dimensions}")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=5)

    def versions(self) -> dict[str, str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_setting('server_version'), "
                "(SELECT extversion FROM pg_extension WHERE extname='vector')").fetchone()
        return {"postgresql": row[0], "pgvector": row[1]}

    def create_schema(self) -> None:
        """One table for all strategies; (strategy, chunk_id) is the primary key."""
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy        text        NOT NULL,
                    chunk_id        text        NOT NULL,
                    document_id     text        NOT NULL,
                    source          text        NOT NULL,
                    page            integer     NOT NULL,
                    section         text,
                    section_no      integer,
                    text            text        NOT NULL,
                    metadata        jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
                    content_hash    text        NOT NULL,
                    embedding_model text        NOT NULL,
                    embedding       vector({self.dimensions}) NOT NULL,
                    updated_at      timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_filter_idx "
                         f"ON {TABLE} (strategy, section, page)")
            # Validate an existing table was created for the same dimension.
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,)).fetchone()
            if row and row[0] != self.dimensions:
                raise RuntimeError(f"{TABLE}.embedding has dimension {row[0]}, "
                                   f"model produces {self.dimensions}")

    def existing_hashes(self, strategy: str, document_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, content_hash FROM {TABLE} "
                "WHERE strategy = %s AND document_id = %s", (strategy, document_id)).fetchall()
        return dict(rows)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               model_name: str = "") -> int:
        """Insert or update rows keyed by (strategy, chunk_id); never duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"vector has {len(vector)} dims, expected {self.dimensions}")
        rows = [(
            strategy, c.chunk_id, c.document_id, c.source, c.page,
            c.metadata.get("section"), c.metadata.get("section_no"), c.text,
            Jsonb(c.metadata), content_hash(c, model_name), model_name, _vec(v),
        ) for c, v in zip(chunks, vectors)]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, section,
                                     section_no, text, metadata, content_hash,
                                     embedding_model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                    page = EXCLUDED.page, section = EXCLUDED.section,
                    section_no = EXCLUDED.section_no, text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding, updated_at = now()""", rows)
        return len(rows)

    def delete_obsolete(self, strategy: str, document_id: str, keep_ids: list[str]) -> int:
        """Remove chunks of a re-ingested document that no longer exist."""
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = %s "
                "AND NOT (chunk_id = ANY(%s))", (strategy, document_id, keep_ids))
            return cur.rowcount

    def counts(self) -> dict[str, dict[str, int]]:
        """Rows, distinct IDs and distinct texts per strategy (duplicate check)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT strategy, count(*), count(DISTINCT chunk_id), "
                f"count(DISTINCT (document_id, page, text)) FROM {TABLE} GROUP BY strategy "
                "ORDER BY strategy").fetchall()
        return {r[0]: {"rows": r[1], "distinct_ids": r[2], "distinct_texts": r[3]} for r in rows}

    def distinct_values(self, column: str) -> list[Any]:
        if column not in FILTERABLE:
            raise ValueError(f"{column} is not filterable")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {TABLE} WHERE {column} IS NOT NULL "
                f"ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter in SQL, rank by cosine distance (<=>), return top-k.

        Without an ANN index PostgreSQL scans all rows: exact search, so a filter
        never reduces the number of results below top_k when enough rows match.
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query has {len(query_vector)} dims, expected {self.dimensions}")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"filter on {key!r} not allowed")
            if isinstance(value, (list, tuple)):
                where.append(f"{key} = ANY(%s)")
                params.append(list(value))
            else:
                where.append(f"{key} = %s")
                params.append(value)
        vec = _vec(query_vector)
        sql = (f"SELECT chunk_id, document_id, source, page, section, text, metadata, "
               f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [vec, *params, vec, top_k]).fetchall()
        return [{
            "chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
            "section": r[4], "text": r[5], "metadata": r[6], "score": round(float(r[7]), 6),
        } for r in rows]
