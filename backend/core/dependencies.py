"""
Dependency Injection Container
All services are singletons, initialised once at startup.
FastAPI Depends() uses these getters.
"""

from functools import lru_cache

from core.config import settings
from services.embeddings import EmbeddingService
from services.vector_store import SupabaseVectorStore
from services.hybrid_search import HybridSearchService
from services.rag_pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_hybrid_search() -> HybridSearchService:
    vs = get_vector_store()
    hs = HybridSearchService(vector_store=vs)
    hs.rebuild_index()
    return hs


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedding_service=get_embedding_service(),
        hybrid_search=get_hybrid_search(),
        vector_store=get_vector_store(),
    )
