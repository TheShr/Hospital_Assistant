"""
Hybrid Search Service
Combines dense vector search (embeddings) + sparse BM25 keyword search.
Uses Reciprocal Rank Fusion (RRF) for score combination.

Why hybrid?
- Dense search: great at semantic similarity ("heart doctor" → cardiologist)
- Sparse search: great at exact matches ("1066", "$600", "OPD")
- Together they dominate both cases.
"""

import logging
import math
from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi

from core.schemas import RetrievedChunk, ChunkMetadata
from core.config import settings

logger = logging.getLogger(__name__)

# RRF constant (standard: 60)
RRF_K = 60


class HybridSearchService:
    """
    Maintains an in-memory BM25 index over the chunks stored in Supabase.
    On startup (or after ingestion), the index is rebuilt from the DB.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._corpus: list[RetrievedChunk] = []
        self._bm25: Optional[BM25Okapi] = None

    # ── Index management ──────────────────────────────────────────────────────

    def rebuild_index(self, document_id: Optional[str] = None):
        """
        Reload all chunks from Supabase and rebuild BM25 index.
        Call after each document upload.
        """
        logger.info("Rebuilding BM25 index...")
        rows = self._fetch_all_chunks(document_id)
        self._corpus = rows

        tokenised = [self._tokenise(r.content) for r in rows]
        if tokenised:
            self._bm25 = BM25Okapi(tokenised)
            logger.info(f"BM25 index built: {len(rows)} chunks")
        else:
            self._bm25 = None
            logger.warning("No chunks found — BM25 index is empty")

    def _fetch_all_chunks(self, document_id: Optional[str] = None) -> list[RetrievedChunk]:
        """Fetch all chunk rows from Supabase (no embedding column needed)."""
        query = (
            self.vector_store.client
            .table("document_chunks")
            .select("id, document_id, filename, page_number, chunk_index, content, total_pages")
        )
        if document_id:
            query = query.eq("document_id", document_id)

        result = query.execute()
        chunks = []
        for row in result.data:
            chunks.append(RetrievedChunk(
                chunk_id=row["id"],
                content=row["content"],
                metadata=ChunkMetadata(
                    document_id=row["document_id"],
                    filename=row["filename"],
                    page_number=row["page_number"],
                    chunk_index=row["chunk_index"],
                    total_chunks_in_doc=0,
                    char_start=0,
                    char_end=0,
                ),
                similarity_score=0.0,
            ))
        return chunks

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.3,
        document_id: Optional[str] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid search and return top_k chunks via RRF.
        Falls back to dense-only if BM25 index is empty.
        """
        # Dense retrieval (vector similarity)
        dense_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 3,    # fetch more, re-rank will cull
            threshold=threshold,
            document_id=document_id,
        )

        # Sparse retrieval (BM25)
        sparse_results = self._bm25_search(query, top_k=top_k * 3)

        if not sparse_results:
            logger.debug("BM25 returned nothing — using dense only")
            return dense_results[:top_k]

        # Merge with Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(
            [dense_results, sparse_results], k=RRF_K
        )

        # Re-attach similarity scores from dense results for source display
        score_map = {r.chunk_id: r.similarity_score for r in dense_results}
        for chunk in merged:
            if chunk.chunk_id in score_map:
                chunk.similarity_score = score_map[chunk.chunk_id]

        return merged[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._bm25 or not self._corpus:
            return []

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair corpus items with scores and sort
        scored = sorted(
            zip(self._corpus, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[:top_k]:
            if score > 0:
                chunk_copy = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    similarity_score=float(score),
                )
                results.append(chunk_copy)
        return results

    # ── RRF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        rank_lists: list[list[RetrievedChunk]],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Standard RRF: score(d) = Σ 1 / (k + rank_i(d))
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank_list in rank_lists:
            for rank, chunk in enumerate(rank_list, start=1):
                rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
                chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids]

    # ── Tokeniser ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Simple whitespace + lowercase tokeniser for BM25."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
