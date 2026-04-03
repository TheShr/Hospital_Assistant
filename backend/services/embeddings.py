"""
Embedding Service
Uses sentence-transformers with BAAI/bge-small-en-v1.5 (FREE, 384-dim, SOTA).

BGE models are instruction-tuned — queries get a prefix for best accuracy:
  Query: "Represent this sentence for searching relevant passages: {query}"
  Passage: no prefix needed
"""

import logging
import numpy as np
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton wrapper around SentenceTransformer.
    Handles both passage embeddings (ingestion) and query embeddings (retrieval).
    """

    # BGE instruction prefix for queries (improves retrieval accuracy ~5-10%)
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        logger.info(f"EmbeddingService initialised (model will lazy-load): {model_name}")

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded ✓")
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed document chunks. No special prefix needed for BGE passages.
        Returns list of float vectors.
        """
        model = self._load_model()
        if not texts:
            return []

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a user query with BGE instruction prefix for best retrieval accuracy.
        """
        model = self._load_model()
        prefixed = self.BGE_QUERY_PREFIX + query

        embedding = model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch embed multiple queries."""
        model = self._load_model()
        prefixed = [self.BGE_QUERY_PREFIX + q for q in queries]

        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        dim_map = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "nomic-ai/nomic-embed-text-v1": 768,
        }
        return dim_map.get(self.model_name, 384)


# ── Module-level singleton ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingService:
    return EmbeddingService(model_name)
