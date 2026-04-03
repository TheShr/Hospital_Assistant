# 🏥 RAG-based Hospital Patient Query Assistant

> An **AI-powered Patient Query Assistant** that answers questions **strictly from your hospital document** — zero hallucination, page-level citations, hybrid search.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase)](https://supabase.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup Guide](#-setup-guide)
- [Running the App](#-running-the-app)
- [API Reference](#-api-reference)
- [Sample Queries](#-sample-queries)
- [Design Decisions](#-design-decisions)
- [Tech Stack](#-tech-stack)

---

## 🧠 Problem Statement

Build an AI-powered Patient Query Assistant that:

1. **Ingests** hospital PDF documents
2. **Vectorises** content using free, SOTA embeddings
3. **Stores** vectors in Supabase (pgvector)
4. **Answers** questions strictly from the document — no hallucination allowed

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        INGESTION PIPELINE                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  POST /upload (PDF)                                                  ║
║       │                                                              ║
║       ▼                                                              ║
║  PyMuPDF  ──►  Page-aware text extraction                            ║
║       │                                                              ║
║       ▼                                                              ║
║  Sentence-aware Chunker  (size=400 chars, overlap=80)                ║
║       │         No mid-sentence cuts · Page metadata preserved       ║
║       ▼                                                              ║
║  BGE-small Embeddings  (384-dim · L2-normalised · FREE)              ║
║       │         Instruction-tuned: query prefix boosts retrieval     ║
║       ▼                                                              ║
║  Supabase pgvector  (IVFFlat cosine index · batched upsert)          ║
║       │                                                              ║
║       ▼                                                              ║
║  BM25 In-memory Index rebuilt  (rank-bm25)                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         QUERY PIPELINE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  POST /query  { "question": "..." }                                  ║
║       │                                                              ║
║       ├──► BGE Query Embedding  (with instruction prefix)            ║
║       │           │                                                  ║
║       │           ▼                                                  ║
║       │    Dense Search  (pgvector cosine, top-15)                   ║
║       │                                                              ║
║       └──► BM25 Sparse Search  (keyword match, top-15)               ║
║                   │                                                  ║
║                   ▼                                                  ║
║          Reciprocal Rank Fusion  (RRF, k=60)                         ║
║                   │                                                  ║
║                   ▼                                                  ║
║          Top-5 ranked chunks + page metadata                         ║
║                   │                                                  ║
║                   ▼                                                  ║
║          Grounded RAG Prompt  (strict no-hallucination system)       ║
║                   │                                                  ║
║                   ▼                                                  ║
║          LLM  (Claude / Groq / OpenAI — configurable)                ║
║                   │                                                  ║
║                   ▼                                                  ║
║  { "answer": "...", "sources": ["page 1", "page 3"] }                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
rag-hospital-assistant/
├── backend/
│   ├── main.py                    # FastAPI app + CORS + GZip middleware
│   ├── api/
│   │   └── routes.py              # POST /upload · POST /query · GET /documents
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (all config from .env)
│   │   ├── schemas.py             # Request / Response Pydantic models
│   │   └── dependencies.py        # Singleton DI container (lru_cache)
│   └── services/
│       ├── ingestion.py           # PDF extraction + sentence-aware chunker
│       ├── embeddings.py          # BGE-small via sentence-transformers
│       ├── vector_store.py        # Supabase pgvector CRUD + cosine search
│       ├── hybrid_search.py       # BM25 + dense + RRF fusion
│       └── rag_pipeline.py        # Orchestrator + prompt engineering + LLM
├── frontend/
│   └── app.py                     # Streamlit chat UI
├── tests/
│   └── test_rag.py                # pytest test suite
├── requirements.txt
├── .env.example                   # copy to .env and fill in your keys
└── README.md
```

---

## ⚙️ Setup Guide

### Prerequisites

- Python 3.10+
- A free [Supabase](https://supabase.com) account
- One LLM API key: [Anthropic](https://console.anthropic.com) **or** [Groq](https://console.groq.com) (both free tier)

---

### 1. Clone & Install

```bash
git clone https://github.com/your-username/rag-hospital-assistant.git
cd rag-hospital-assistant
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses CPU-only PyTorch (~250 MB). For GPU, replace the torch line with `torch==2.3.0` and remove the `--extra-index-url` line.

---

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your keys
```

Minimum required:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://postgres:[PASSWORD]@db.your-project.supabase.co:5432/postgres

LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

---

### 3. Set Up Supabase

Go to your Supabase project → **SQL Editor** → run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL,
    total_pages INTEGER NOT NULL, total_chunks INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
    char_start INTEGER, char_end INTEGER, total_pages INTEGER,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON document_chunks (page_number);

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384), match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5, filter_doc_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id TEXT, document_id TEXT, filename TEXT, page_number INTEGER,
    chunk_index INTEGER, content TEXT, total_pages INTEGER, similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT dc.id, dc.document_id, dc.filename, dc.page_number,
           dc.chunk_index, dc.content, dc.total_pages,
           1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE (filter_doc_id IS NULL OR dc.document_id = filter_doc_id)
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding LIMIT match_count;
END; $$;
```

---

## 🚀 Running the App

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API docs → [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```
UI → [http://localhost:8501](http://localhost:8501)

---

## 📡 API Reference

### `POST /api/v1/upload`

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@hospital_document.pdf"
```

**Response:**
```json
{
  "document_id": "3fa85f64-...",
  "filename": "hospital_document.pdf",
  "total_chunks": 47,
  "total_pages": 10,
  "message": "Document ingested successfully.",
  "processing_time_ms": 2340.5
}
```

---

### `POST /api/v1/query`

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are OPD timings?"}'
```

**Response:**
```json
{
  "answer": "The OPD timings are 08:00 AM to 08:00 PM, Monday to Saturday. A limited Sunday Clinic operates 09:00 AM to 01:00 PM. [Page 1]",
  "sources": ["page 1"],
  "source_chunks": [
    {
      "page": 1,
      "chunk_index": 2,
      "text_preview": "[Page 1] Operating Hours: Emergency Services: 24/7...",
      "similarity_score": 0.9124
    }
  ],
  "document_id": null,
  "latency_ms": 842.3,
  "retrieval_method": "hybrid"
}
```

**Fallback** (answer not in document):
```json
{
  "answer": "I don't have that information in the provided document.",
  "sources": []
}
```

---

### `GET /api/v1/documents`

```bash
curl http://localhost:8000/api/v1/documents
```

---

## 💬 Sample Queries

| Question | Expected Source |
|----------|----------------|
| What are OPD timings? | Page 1 |
| Who is the cardiologist? | Page 2 |
| What is the cost of MRI? | Page 3 |
| Can I cancel appointment within 24 hours? | Page 4 |
| What is ICU cost per day? | Page 5 |
| Emergency number? | Page 1 |
| What is a private room cost? | Page 5 |
| Does the hospital support Hindi? | Page 1, 10 |
| What is the wifi password? | → Fallback |

---

## 🧩 Design Decisions

### BGE-small vs OpenAI Embeddings

BGE-small is free, runs locally, and uses instruction-tuned query prefixes that boost retrieval accuracy by ~5–10% on factual documents. No API cost means no per-ingestion charges.

### Hybrid Search (Dense + BM25 + RRF)

- **Dense** excels at semantic matching: `"heart doctor"` → cardiologist
- **BM25** excels at exact tokens: `"1066"`, `"$600"`, `"ICU"`
- **RRF** fuses both ranked lists with no hyperparameter tuning needed

### temperature=0.0

Deterministic generation prevents LLM "creativity" from interpolating between medical facts. Every run of the same query returns the same answer.

### Sentence-aware chunking

Mid-sentence cuts destroy semantic completeness. The chunker uses abbreviation-aware regex to find sentence boundaries and never cuts mid-sentence even at the configured chunk size limit.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| PDF | PyMuPDF (fitz) |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector DB | Supabase PostgreSQL + pgvector |
| Sparse Search | rank-bm25 |
| LLM | Claude / Groq / OpenAI (configurable) |
| UI | Streamlit |
| Config | Pydantic Settings |

---

## ✅ Bonus Features Implemented

- [x] Streamlit UI with professional chat interface
- [x] Source text highlighting (expandable context chunks)
- [x] Multi-document support (filter by `document_id` or search all)
- [x] Chat history with full conversation memory
- [x] Hybrid search (BM25 + dense embeddings + RRF)
- [x] Page-level citations in every answer `[Page X]`
- [x] Re-ranking via Reciprocal Rank Fusion

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```
