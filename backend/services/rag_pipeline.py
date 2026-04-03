"""
RAG Pipeline
Orchestrates: Query → Embedding → Hybrid Search → LLM Generation

Key design decisions:
1. Strict grounding prompt — model can ONLY use retrieved context
2. If no relevant chunks found → deterministic fallback message
3. Sources are cited in answers as [Page X]
4. Supports Anthropic (Claude), Groq, and OpenAI LLM backends
"""

import logging
import time
from typing import Optional

from core.config import settings
from core.schemas import QueryResponse, SourceChunk, RetrievedChunk
from services.embeddings import EmbeddingService
from services.hybrid_search import HybridSearchService
from services.vector_store import SupabaseVectorStore

logger = logging.getLogger(__name__)


# ── Prompt Template ───────────────────────────────────────────────────────────
# Engineered to:
#   1. Hard-block hallucination via explicit negative instruction
#   2. Force page-level citation in the answer
#   3. Specify exact fallback string when context is insufficient

SYSTEM_PROMPT = """You are a precise Hospital Information Assistant. Your ONLY job is to answer patient questions using the EXACT information from the provided hospital document context.

STRICT RULES — you must follow ALL of these:
1. Answer ONLY from the [CONTEXT] section below. Do NOT use any external knowledge.
2. If the answer is not in the context, reply EXACTLY: "I don't have that information in the provided document."
3. Always cite the page number(s) in your answer using the format [Page X].
4. Be concise and factual. Do not add opinions, suggestions, or elaborations beyond what is stated.
5. If multiple pages contain relevant info, cite all of them.
6. Numbers, timings, costs, and names must be copied EXACTLY from the context — do not paraphrase figures."""

USER_PROMPT_TEMPLATE = """[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
(Answer strictly from the context above. Include page citations like [Page X].)"""


NO_CONTEXT_FALLBACK = "I don't have that information in the provided document."


class RAGPipeline:
    """
    Full RAG pipeline orchestrator.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        hybrid_search: HybridSearchService,
        vector_store: SupabaseVectorStore,
    ):
        self.embedding_service = embedding_service
        self.hybrid_search = hybrid_search
        self.vector_store = vector_store
        self._llm_client = None
        logger.info(f"RAG pipeline initialised (LLM provider: {settings.LLM_PROVIDER})")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> QueryResponse:
        """
        End-to-end RAG query.
        """
        start = time.perf_counter()
        k = top_k or settings.TOP_K

        # 1. Embed the query
        query_embedding = self.embedding_service.embed_query(question)

        # 2. Hybrid retrieval
        retrieved_chunks = self.hybrid_search.search(
            query=question,
            query_embedding=query_embedding,
            top_k=k,
            threshold=settings.SIMILARITY_THRESHOLD,
            document_id=document_id,
        )

        # 3. Check if we have usable context
        if not retrieved_chunks:
            return QueryResponse(
                answer=NO_CONTEXT_FALLBACK,
                sources=[],
                source_chunks=[],
                document_id=document_id,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                retrieval_method="hybrid",
            )

        # 4. Build context string with page markers
        context_str = self._build_context(retrieved_chunks)

        # 5. Generate answer from LLM
        answer = self._generate(question=question, context=context_str)

        # 6. Build source metadata for response
        sources = self._extract_sources(retrieved_chunks)
        source_chunks = self._build_source_chunks(retrieved_chunks)

        latency = round((time.perf_counter() - start) * 1000, 1)
        logger.info(f"Query answered in {latency}ms — {len(retrieved_chunks)} chunks retrieved")

        return QueryResponse(
            answer=answer,
            sources=sources,
            source_chunks=source_chunks,
            document_id=document_id,
            latency_ms=latency,
            retrieval_method="hybrid",
        )

    # ── Context Builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """
        Build a context block with clear page markers for the LLM.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"--- Chunk {i} | Page {chunk.metadata.page_number} "
                f"(relevance: {chunk.similarity_score:.2f}) ---\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)

    # ── LLM Generation ────────────────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """Route to the configured LLM provider."""
        user_message = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        provider = settings.LLM_PROVIDER.lower()
        if provider == "anthropic":
            return self._generate_anthropic(user_message)
        elif provider == "groq":
            return self._generate_groq(user_message)
        elif provider == "openai":
            return self._generate_openai(user_message)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _generate_anthropic(self, user_message: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    def _generate_groq(self, user_message: str) -> str:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.0,   # Deterministic — critical for factual RAG
        )
        return response.choices[0].message.content.strip()

    def _generate_openai(self, user_message: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    # ── Source Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sources(chunks: list[RetrievedChunk]) -> list[str]:
        """Return deduplicated, sorted list of page references."""
        pages = sorted(set(c.metadata.page_number for c in chunks))
        return [f"page {p}" for p in pages]

    @staticmethod
    def _build_source_chunks(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
        return [
            SourceChunk(
                page=c.metadata.page_number,
                chunk_index=c.metadata.chunk_index,
                text_preview=c.content[:200],
                similarity_score=round(c.similarity_score, 4),
            )
            for c in chunks
        ]
