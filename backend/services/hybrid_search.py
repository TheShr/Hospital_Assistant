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
        return self.vector_store.get_cached_chunks(document_id=document_id)

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
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Run hybrid search and return top_k chunks plus method information.
        Falls back to dense-only if BM25 index is empty.
        """
        dense_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            threshold=threshold,
            document_id=document_id,
        )

        sparse_results = self._bm25_search(query, top_k=top_k * 3)

        if not sparse_results:
            logger.debug("BM25 returned nothing — using dense only")
            return dense_results[:top_k], "dense"

        # Weighted Reciprocal Rank Fusion for hybrid precision
        merged = self._weighted_rrf(
            dense_results,
            sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=RRF_K,
        )

        # Preserve dense similarity scores for source display
        score_map = {r.chunk_id: r.similarity_score for r in dense_results}
        for chunk in merged:
            if chunk.chunk_id in score_map:
                chunk.similarity_score = score_map[chunk.chunk_id]

        return merged[:top_k], "hybrid"

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

    def _weighted_rrf(
        self,
        dense_list: list[RetrievedChunk],
        sparse_list: list[RetrievedChunk],
        dense_weight: float,
        sparse_weight: float,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Weighted RRF blends dense and sparse rankings by preference weights.
        """
        weights = [dense_weight, sparse_weight]
        rank_lists = [dense_list, sparse_list]

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for weight, rank_list in zip(weights, rank_lists):
            if weight <= 0:
                continue
            for rank, chunk in enumerate(rank_list, start=1):
                rrf_scores[chunk.chunk_id] += weight / (k + rank)
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
