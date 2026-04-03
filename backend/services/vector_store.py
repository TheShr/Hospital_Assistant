"""
Supabase Vector Store
Handles:
  - Schema initialisation (pgvector extension + tables)
  - Chunk upsert with embeddings
  - Dense similarity search (cosine)
  - Metadata filtering (by document_id, page_number)

SQL schema is applied once via the setup_schema() call.
"""

import logging
import json
from typing import Optional

from supabase import create_client, Client

from core.config import settings
from core.schemas import RetrievedChunk, ChunkMetadata
from services.ingestion import TextChunk

logger = logging.getLogger(__name__)


# ── SQL Schema ────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Enable pgvector extension (run once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents metadata table
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    total_pages     INTEGER NOT NULL,
    total_chunks    INTEGER NOT NULL,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    page_number     INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    char_start      INTEGER,
    char_end        INTEGER,
    total_pages     INTEGER,
    embedding       vector({dim})         -- pgvector column
);

-- IVFFlat index for fast approximate nearest-neighbour search
-- Tune lists = sqrt(total_rows) for best performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- Index for metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_page
    ON document_chunks (page_number);
""".format(dim=settings.EMBEDDING_DIMENSION)


MATCH_FUNCTION_SQL = """
-- RPC function for similarity search (callable from Supabase client)
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding  vector({dim}),
    match_threshold  FLOAT DEFAULT 0.3,
    match_count      INT   DEFAULT 5,
    filter_doc_id    TEXT  DEFAULT NULL
)
RETURNS TABLE (
    id              TEXT,
    document_id     TEXT,
    filename        TEXT,
    page_number     INTEGER,
    chunk_index     INTEGER,
    content         TEXT,
    total_pages     INTEGER,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.filename,
        dc.page_number,
        dc.chunk_index,
        dc.content,
        dc.total_pages,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        (filter_doc_id IS NULL OR dc.document_id = filter_doc_id)
        AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
""".format(dim=settings.EMBEDDING_DIMENSION)


# ── Store class ───────────────────────────────────────────────────────────────

class SupabaseVectorStore:
    """
    Wraps Supabase client with document + chunk CRUD and vector search.
    """

    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY,
        )
        logger.info("Supabase client initialised ✓")

    # ── Schema ────────────────────────────────────────────────────────────────

    def setup_schema(self):
        """
        Run once to create tables and indexes.
        In production, use Supabase migrations instead.
        Prints the SQL to run in the Supabase SQL editor.
        """
        logger.info("Schema SQL to execute in Supabase SQL editor:")
        logger.info("\n" + SCHEMA_SQL)
        logger.info("\n" + MATCH_FUNCTION_SQL)
        print("=" * 70)
        print("RUN THIS IN SUPABASE > SQL EDITOR:")
        print("=" * 70)
        print(SCHEMA_SQL)
        print(MATCH_FUNCTION_SQL)

    # ── Document CRUD ─────────────────────────────────────────────────────────

    def upsert_document(
        self,
        document_id: str,
        filename: str,
        total_pages: int,
        total_chunks: int,
    ):
        """Insert or update a document record."""
        data = {
            "id": document_id,
            "filename": filename,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
        }
        result = (
            self.client.table(settings.DOCUMENTS_TABLE)
            .upsert(data)
            .execute()
        )
        logger.info(f"Document upserted: {document_id}")
        return result

    def document_exists(self, document_id: str) -> bool:
        result = (
            self.client.table(settings.DOCUMENTS_TABLE)
            .select("id")
            .eq("id", document_id)
            .execute()
        )
        return len(result.data) > 0

    # ── Chunk CRUD ────────────────────────────────────────────────────────────

    def upsert_chunks_batch(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ):
        """
        Batch upsert chunks with their embeddings.
        Processes in batches to avoid payload size limits.
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must match"

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append({
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "total_pages": chunk.total_pages,
                "embedding": embedding,  # Supabase accepts Python list
            })

        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self.client.table(settings.CHUNKS_TABLE).upsert(batch).execute()
            logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Total chunks upserted: {len(rows)}")

    def delete_document_chunks(self, document_id: str):
        """Delete all chunks for a document (cascades from documents table too)."""
        self.client.table(settings.CHUNKS_TABLE).delete().eq(
            "document_id", document_id
        ).execute()

    # ── Similarity Search ─────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.3,
        document_id: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Cosine similarity search using pgvector + match_chunks RPC.
        Returns RetrievedChunk objects sorted by similarity (desc).
        """
        params = {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k,
            "filter_doc_id": document_id,
        }

        result = self.client.rpc("match_chunks", params).execute()

        retrieved = []
        for row in result.data:
            retrieved.append(RetrievedChunk(
                chunk_id=row["id"],
                content=row["content"],
                metadata=ChunkMetadata(
                    document_id=row["document_id"],
                    filename=row["filename"],
                    page_number=row["page_number"],
                    chunk_index=row["chunk_index"],
                    total_chunks_in_doc=0,  # not returned by RPC
                    char_start=0,
                    char_end=0,
                ),
                similarity_score=float(row["similarity"]),
            ))

        logger.debug(f"Similarity search returned {len(retrieved)} chunks")
        return retrieved

    def list_documents(self) -> list[dict]:
        result = (
            self.client.table(settings.DOCUMENTS_TABLE)
            .select("*")
            .order("uploaded_at", desc=True)
            .execute()
        )
        return result.data
