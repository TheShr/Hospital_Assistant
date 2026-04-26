"""
FastAPI API Routes
POST /upload  — upload and ingest a PDF
POST /query   — ask a question, get a RAG answer
GET  /documents — list all ingested documents
"""

import os
import time
import uuid
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from core.config import settings
from core.schemas import UploadResponse, QueryRequest, QueryResponse
from core.dependencies import (
    get_embedding_service,
    get_vector_store,
    get_hybrid_search,
    get_rag_pipeline,
)
from services.ingestion import PDFIngestionService
from services.embeddings import EmbeddingService
from services.vector_store import SupabaseVectorStore
from services.hybrid_search import HybridSearchService
from services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure upload dir exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(..., description="Hospital PDF document"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: SupabaseVectorStore = Depends(get_vector_store),
    hybrid_search: HybridSearchService = Depends(get_hybrid_search),
):
    """
    Upload and ingest a hospital PDF document.

    Pipeline:
    1. Validate file (PDF, size limit)
    2. Extract text + chunk (sentence-aware sliding window)
    3. Embed all chunks (BGE-small, batched)
    4. Upsert to Supabase pgvector
    5. Rebuild BM25 index
    """
    start = time.perf_counter()

    # ── Validation ────────────────────────────────────────────────
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # ── Save to disk temporarily ───────────────────────────────────
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    save_path = Path(settings.UPLOAD_DIR) / safe_filename

    try:
        with open(save_path, "wb") as f:
            f.write(content)

        # ── Ingestion ─────────────────────────────────────────────
        ingester = PDFIngestionService(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        parsed_doc = ingester.ingest(
            pdf_path=save_path,
            filename=file.filename,
            document_id=document_id,
        )

        if not parsed_doc.chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        # ── Embed ─────────────────────────────────────────────────
        texts = [c.content for c in parsed_doc.chunks]
        embeddings = embedding_service.embed_passages(texts)

        # ── Store in Supabase ─────────────────────────────────────
        vector_store.upsert_document(
            document_id=document_id,
            filename=file.filename,
            total_pages=parsed_doc.total_pages,
            total_chunks=len(parsed_doc.chunks),
        )
        vector_store.upsert_chunks_batch(
            chunks=parsed_doc.chunks,
            embeddings=embeddings,
        )

        # ── Rebuild BM25 index ────────────────────────────────────
        hybrid_search.rebuild_index()

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        logger.info(f"Upload complete: {file.filename} ({elapsed}ms)")

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_chunks=len(parsed_doc.chunks),
            total_pages=parsed_doc.total_pages,
            message="Document ingested successfully.",
            processing_time_ms=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ingestion failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        # Clean up temp file
        if save_path.exists():
            save_path.unlink()


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_document(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Ask a question about the hospital documents.

    The system:
    1. Embeds the question
    2. Performs hybrid search (dense + BM25)
    3. Passes top chunks to LLM with strict no-hallucination prompt
    4. Returns answer + source page citations
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = rag_pipeline.query(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        logger.exception(f"Query failed: {request.question}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get("/documents", tags=["Documents"])
async def list_documents(
    vector_store: SupabaseVectorStore = Depends(get_vector_store),
):
    """List all ingested documents."""
    try:
        docs = vector_store.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /setup-schema ─────────────────────────────────────────────────────────

@router.get("/setup-schema", tags=["Admin"])
async def print_schema(
    vector_store: SupabaseVectorStore = Depends(get_vector_store),
):
    """Prints the SQL schema to run in Supabase SQL editor."""
    from services.vector_store import SCHEMA_SQL, MATCH_FUNCTION_SQL
    return {
        "instructions": "Run these SQL statements in Supabase > SQL Editor",
        "schema_sql": SCHEMA_SQL,
        "function_sql": MATCH_FUNCTION_SQL,
    }
