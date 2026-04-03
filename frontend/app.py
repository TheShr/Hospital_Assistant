"""
Streamlit UI — Hospital RAG Assistant
Clean, professional UI with source highlighting.
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="🏥 Hospital Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f8fafc; }

    .header-box {
        background: linear-gradient(135deg, #1e40af 0%, #0ea5e9 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    .answer-box {
        background: white;
        border-left: 4px solid #0ea5e9;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .source-chip {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .chunk-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .fallback-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #92400e;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Hospital RAG Assistant")
    st.divider()

    # Document Upload
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Hospital PDF", type=["pdf"], key="pdf_uploader"
    )

    if uploaded_file and st.button("🚀 Ingest Document"):
        with st.spinner("Processing PDF..."):
            try:
                res = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.current_doc_id = data["document_id"]
                    st.success(
                        f"✅ Ingested!\n\n"
                        f"📄 **{data['filename']}**\n\n"
                        f"📊 {data['total_chunks']} chunks | {data['total_pages']} pages\n\n"
                        f"⏱️ {data['processing_time_ms']}ms"
                    )
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is the backend running?")

    st.divider()

    # Existing documents
    st.markdown("### 📁 Documents")
    if st.button("🔄 Refresh"):
        try:
            res = requests.get(f"{API_BASE}/documents", timeout=10)
            if res.status_code == 200:
                docs = res.json()["documents"]
                if docs:
                    for doc in docs:
                        if st.button(
                            f"📄 {doc['filename'][:30]}...",
                            key=f"doc_{doc['id']}",
                        ):
                            st.session_state.current_doc_id = doc["id"]
                else:
                    st.info("No documents yet.")
        except:
            st.error("Cannot fetch documents.")

    if st.session_state.current_doc_id:
        st.success(f"🎯 Active doc: `{st.session_state.current_doc_id[:8]}...`")

    st.divider()
    st.markdown("### ⚡ Sample Questions")
    sample_questions = [
        "What are OPD timings?",
        "Who is the cardiologist?",
        "What is the cost of MRI?",
        "Can I cancel appointment within 24 hours?",
        "What is ICU cost per day?",
        "Emergency number?",
        "What is the cost of a private room?",
        "What languages does the hospital support?",
    ]

    for q in sample_questions:
        if st.button(f"💬 {q}", key=f"sample_{q}"):
            st.session_state["prefill_question"] = q


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1 style="margin:0; font-size:2rem;">🏥 Medicare Hospital Assistant</h1>
    <p style="margin:0.5rem 0 0; opacity:0.9;">AI-powered patient query system — answers only from your hospital document</p>
</div>
""", unsafe_allow_html=True)

# Chat history display
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        is_fallback = "don't have that information" in entry["answer"].lower()
        if is_fallback:
            st.markdown(f'<div class="fallback-box">⚠️ {entry["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)

        if entry.get("sources"):
            chips = " ".join([f'<span class="source-chip">📄 {s}</span>' for s in entry["sources"]])
            st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)

        if entry.get("latency_ms"):
            st.caption(f"⏱️ {entry['latency_ms']}ms | {entry.get('retrieval_method', 'hybrid')} search")

# Query input
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input("Ask a question about the hospital...", key="chat_input")
question = question or prefill

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                payload = {"question": question}
                if st.session_state.current_doc_id:
                    payload["document_id"] = st.session_state.current_doc_id

                res = requests.post(
                    f"{API_BASE}/query",
                    json=payload,
                    timeout=60,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    source_chunks = data.get("source_chunks", [])
                    latency = data.get("latency_ms", 0)
                    method = data.get("retrieval_method", "hybrid")

                    # Display answer
                    is_fallback = "don't have that information" in answer.lower()
                    if is_fallback:
                        st.markdown(f'<div class="fallback-box">⚠️ {answer}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    # Sources
                    if sources:
                        chips = " ".join([f'<span class="source-chip">📄 {s}</span>' for s in sources])
                        st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("⏱️ Latency", f"{latency}ms")
                    col2.metric("📑 Chunks Retrieved", len(source_chunks))
                    col3.metric("🔍 Method", method)

                    # Source chunk details (expandable)
                    if source_chunks:
                        with st.expander("🔎 View Retrieved Context"):
                            for i, chunk in enumerate(source_chunks, 1):
                                st.markdown(f"""
                                <div class="chunk-card">
                                    <strong>Chunk {i} — Page {chunk['page']} 
                                    (score: {chunk['similarity_score']:.3f})</strong><br/>
                                    {chunk['text_preview']}...
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "latency_ms": latency,
                        "retrieval_method": method,
                    })

                else:
                    st.error(f"API Error {res.status_code}: {res.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server. Start it with: `uvicorn main:app --reload`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
