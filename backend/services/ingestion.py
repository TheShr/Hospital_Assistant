"""
Document Ingestion Service
PDF → Text → Smart Chunking → Embeddings → Supabase

Chunking strategy:
- Sentence-aware splitting (no mid-sentence cuts)
- Configurable size + overlap
- Page-boundary metadata preserved
"""

import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF — faster and more accurate than pdfplumber for most PDFs

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    chunk_id: str
    content: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    document_id: str
    filename: str
    total_pages: int = 0


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    total_pages: int
    full_text: str
    page_texts: list[str]           # text per page (index = page_number - 1)
    chunks: list[TextChunk] = field(default_factory=list)


class PDFIngestionService:
    """
    Extracts text from PDFs with page-level granularity,
    then applies a sentence-aware sliding-window chunker.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str | Path, filename: str,
               document_id: Optional[str] = None) -> ParsedDocument:
        """
        Full pipeline: PDF → ParsedDocument with chunks.
        """
        doc_id = document_id or str(uuid.uuid4())
        pdf_path = Path(pdf_path)

        logger.info(f"Ingesting {filename} (doc_id={doc_id})")

        page_texts = self._extract_pages(pdf_path)
        full_text = "\n\n".join(page_texts)

        chunks = self._chunk_document(
            page_texts=page_texts,
            document_id=doc_id,
            filename=filename,
        )

        doc = ParsedDocument(
            document_id=doc_id,
            filename=filename,
            total_pages=len(page_texts),
            full_text=full_text,
            page_texts=page_texts,
            chunks=chunks,
        )

        # Back-fill total_pages onto chunks
        for c in doc.chunks:
            c.total_pages = doc.total_pages

        logger.info(
            f"Ingested {filename}: {doc.total_pages} pages → {len(doc.chunks)} chunks"
        )
        return doc

    # ── Step 1: PDF → pages ───────────────────────────────────────────────────

    def _extract_pages(self, pdf_path: Path) -> list[str]:
        """Returns list of cleaned text strings, one per page."""
        page_texts: list[str] = []

        with fitz.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                raw = page.get_text("text")
                cleaned = self._clean_text(raw)
                if cleaned:
                    # Prefix each page with a marker so chunks carry page context
                    page_texts.append(f"[Page {page_num}]\n{cleaned}")
                else:
                    page_texts.append(f"[Page {page_num}]\n(No readable text)")

        return page_texts

    # ── Step 2: Sentence-aware chunking ───────────────────────────────────────

    def _chunk_document(
        self,
        page_texts: list[str],
        document_id: str,
        filename: str,
    ) -> list[TextChunk]:
        """
        Sliding-window chunker that:
        1. Stays within page boundaries where possible
        2. Never cuts mid-sentence
        3. Adds configurable overlap
        """
        all_chunks: list[TextChunk] = []
        chunk_index = 0

        for page_idx, page_text in enumerate(page_texts):
            page_number = page_idx + 1
            sentences = self._split_into_sentences(page_text)

            current_chunk_sentences: list[str] = []
            current_len = 0
            char_cursor = 0

            for sentence in sentences:
                sentence_len = len(sentence)

                # If adding this sentence exceeds limit → emit chunk
                if current_len + sentence_len > self.chunk_size and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if chunk_text:
                        all_chunks.append(TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=chunk_text,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            char_start=char_cursor - current_len,
                            char_end=char_cursor,
                            document_id=document_id,
                            filename=filename,
                        ))
                        chunk_index += 1

                    # Keep overlap: retain last N chars worth of sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk_sentences
                    )
                    current_chunk_sentences = overlap_sentences
                    current_len = sum(len(s) for s in overlap_sentences)

                current_chunk_sentences.append(sentence)
                current_len += sentence_len
                char_cursor += sentence_len + 1  # +1 for space

            # Emit remaining sentences on this page
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    all_chunks.append(TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=chunk_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        char_start=max(0, char_cursor - current_len),
                        char_end=char_cursor,
                        document_id=document_id,
                        filename=filename,
                    ))
                    chunk_index += 1

        return all_chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Return trailing sentences that sum to ≈ chunk_overlap chars."""
        overlap: list[str] = []
        total = 0
        for s in reversed(sentences):
            if total + len(s) > self.chunk_overlap:
                break
            overlap.insert(0, s)
            total += len(s)
        return overlap

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove junk characters."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)  # keep printable ASCII
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Simple but robust sentence splitter.
        Handles abbreviations like 'Dr.', 'AM.', etc.
        """
        # Protect common abbreviations
        text = re.sub(r'\b(Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|e\.g|i\.e|AM|PM)\.',
                      r'\1<PROT>', text)
        # Split on sentence terminators followed by space+capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)
        # Restore abbreviations
        sentences = [s.replace('<PROT>', '.') for s in sentences]
        # Filter empty
        return [s.strip() for s in sentences if s.strip()]
