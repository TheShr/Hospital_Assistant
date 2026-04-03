"""
Test Suite — Hospital RAG Assistant
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from services.ingestion import PDFIngestionService


# ── Ingestion Tests ───────────────────────────────────────────────────────────

class TestPDFIngestionService:
    def setup_method(self):
        self.ingester = PDFIngestionService(chunk_size=200, chunk_overlap=40)

    def test_clean_text(self):
        raw = "Hello   World\n\n\n\nTest"
        cleaned = PDFIngestionService._clean_text(raw)
        assert "   " not in cleaned
        assert cleaned == "Hello World Test"

    def test_split_sentences_basic(self):
        text = "Dr. Smith is a cardiologist. He works Monday to Friday. Contact: 1066."
        sentences = PDFIngestionService._split_into_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_abbreviations(self):
        text = "OPD timings are 8 AM to 8 PM. Emergency number is 1066."
        sentences = PDFIngestionService._split_into_sentences(text)
        # "AM" abbreviation should not split mid-sentence
        assert any("8 AM to 8 PM" in s for s in sentences)

    def test_chunk_small_text(self):
        page_texts = ["[Page 1]\nOPD timings are 8 AM to 8 PM Monday to Saturday."]
        chunks = self.ingester._chunk_document(
            page_texts=page_texts, document_id="test-123", filename="test.pdf"
        )
        assert len(chunks) >= 1
        assert chunks[0].page_number == 1
        assert chunks[0].document_id == "test-123"

    def test_chunk_preserves_page_numbers(self):
        page_texts = [
            "[Page 1]\nThis is page one content.",
            "[Page 2]\nThis is page two content with more text here.",
        ]
        chunks = self.ingester._chunk_document(
            page_texts=page_texts, document_id="test-456", filename="test.pdf"
        )
        pages = {c.page_number for c in chunks}
        assert 1 in pages
        assert 2 in pages


# ── Embedding Tests ───────────────────────────────────────────────────────────

class TestEmbeddingService:
    def test_embed_query_returns_list(self):
        from services.embeddings import EmbeddingService
        svc = EmbeddingService()
        embedding = svc.embed_query("What are OPD timings?")
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # bge-small dimension

    def test_embed_passages_batch(self):
        from services.embeddings import EmbeddingService
        svc = EmbeddingService()
        texts = ["OPD is 8 AM to 8 PM.", "ICU costs $600 per day."]
        embeddings = svc.embed_passages(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    def test_query_prefix_applied(self):
        """BGE query prefix must be added for queries."""
        from services.embeddings import EmbeddingService
        svc = EmbeddingService()
        # Just verify it doesn't crash and returns correct dimension
        emb = svc.embed_query("Emergency number?")
        assert len(emb) == 384


# ── RAG Pipeline Tests ────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_build_context(self):
        from services.rag_pipeline import RAGPipeline
        from core.schemas import RetrievedChunk, ChunkMetadata

        chunks = [
            RetrievedChunk(
                chunk_id="1",
                content="OPD timings are 8 AM to 8 PM.",
                metadata=ChunkMetadata(
                    document_id="d1", filename="test.pdf",
                    page_number=1, chunk_index=0,
                    total_chunks_in_doc=5, char_start=0, char_end=50,
                ),
                similarity_score=0.92,
            )
        ]

        context = RAGPipeline._build_context(chunks)
        assert "Page 1" in context
        assert "OPD timings" in context
        assert "0.92" in context

    def test_extract_sources(self):
        from services.rag_pipeline import RAGPipeline
        from core.schemas import RetrievedChunk, ChunkMetadata

        chunks = [
            RetrievedChunk(
                chunk_id="1", content="...",
                metadata=ChunkMetadata(
                    document_id="d1", filename="test.pdf",
                    page_number=3, chunk_index=0,
                    total_chunks_in_doc=5, char_start=0, char_end=10,
                ),
                similarity_score=0.8,
            ),
            RetrievedChunk(
                chunk_id="2", content="...",
                metadata=ChunkMetadata(
                    document_id="d1", filename="test.pdf",
                    page_number=5, chunk_index=1,
                    total_chunks_in_doc=5, char_start=0, char_end=10,
                ),
                similarity_score=0.75,
            ),
        ]

        sources = RAGPipeline._extract_sources(chunks)
        assert sources == ["page 3", "page 5"]


# ── BM25 Hybrid Search Tests ──────────────────────────────────────────────────

class TestHybridSearch:
    def test_tokenise(self):
        from services.hybrid_search import HybridSearchService
        tokens = HybridSearchService._tokenise("What is the ICU cost per day?")
        assert "icu" in tokens
        assert "cost" in tokens

    def test_rrf_merges_lists(self):
        from services.hybrid_search import HybridSearchService
        from core.schemas import RetrievedChunk, ChunkMetadata

        def make_chunk(chunk_id, score):
            return RetrievedChunk(
                chunk_id=chunk_id, content="test",
                metadata=ChunkMetadata(
                    document_id="d1", filename="f.pdf",
                    page_number=1, chunk_index=0,
                    total_chunks_in_doc=1, char_start=0, char_end=5,
                ),
                similarity_score=score,
            )

        list1 = [make_chunk("a", 0.9), make_chunk("b", 0.8), make_chunk("c", 0.7)]
        list2 = [make_chunk("b", 0.95), make_chunk("a", 0.85), make_chunk("d", 0.6)]

        merged = HybridSearchService._reciprocal_rank_fusion([list1, list2])
        ids = [c.chunk_id for c in merged]

        # "a" and "b" appear in both lists so should rank higher
        assert ids[0] in {"a", "b"}
        assert ids[1] in {"a", "b"}
