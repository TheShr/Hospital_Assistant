"""
Pydantic schemas — request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    total_pages: int
    message: str
    processing_time_ms: float


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="What are OPD timings?")
    document_id: Optional[str] = Field(
        default=None,
        description="Filter retrieval to a specific document. If None, searches all."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    page: int
    chunk_index: int
    text_preview: str           # First 200 chars of the chunk
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]          # e.g. ["page 1", "page 3"]
    source_chunks: list[SourceChunk]
    document_id: Optional[str]
    latency_ms: float
    retrieval_method: str       # "hybrid" | "dense"


# ── Internal ──────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    total_chunks_in_doc: int
    char_start: int
    char_end: int


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    similarity_score: float
