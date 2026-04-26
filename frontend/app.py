"""
Streamlit UI — Hospital RAG Assistant
Professional, modern chat interface with source highlighting.

Bug fixes vs original:
  - Removed `key=` from st.chat_input (unsupported, causes StreamlitAPIException)
  - Fixed prefill logic: stored in session_state and fed into chat_input flow
  - Document selector no longer uses nested buttons (crashes Streamlit)
"""

import streamlit as st
import requests

# ── Page Config ────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Medicare Hospital Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4f8; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important; color: #cbd5e1 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    text-align: left !important; font-size: 0.82rem !important;
    padding: 0.45rem 0.75rem !important; width: 100% !important; transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1d4ed8 !important; border-color: #3b82f6 !important; color: white !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 60%, #06b6d4 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    box-shadow: 0 4px 24px rgba(14,165,233,0.25);
}
.hero-icon { font-size: 3rem; line-height: 1; }
.hero-text h1 { color: white; font-size: 1.7rem; font-weight: 700; margin: 0; }
.hero-text p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Chat bubbles */
.msg-wrap { display: flex; gap: 12px; margin-bottom: 1.2rem; align-items: flex-start; }
.msg-wrap.user { flex-direction: row-reverse; }
.msg-wrap.user .bubble  { background: #1d4ed8; color: white; border-radius: 18px 18px 4px 18px; }
.msg-wrap.bot  .bubble  { background: white; color: #1e293b; border-radius: 18px 18px 18px 4px; border: 1px solid #e2e8f0; }
.avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center;
          justify-content: center; font-size: 1.1rem; flex-shrink: 0; }
.avatar.user { background: #1d4ed8; }
.avatar.bot  { background: #0f172a; }
.bubble { max-width: 78%; padding: 0.85rem 1.1rem; font-size: 0.93rem; line-height: 1.65;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); }

/* Source chips */
.source-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.6rem; }
.chip { background: #dbeafe; color: #1e40af; padding: 2px 10px; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600; }

/* Fallback */
.fallback { background: #fef9c3; border-left: 4px solid #eab308;
            padding: 0.75rem 1rem; border-radius: 8px; color: #713f12; font-size: 0.9rem; }

/* Metrics row */
.metrics { display: flex; gap: 10px; margin-top: 0.6rem; flex-wrap: wrap; }
.metric-pill { background: #f1f5f9; border-radius: 999px; padding: 3px 12px;
               font-size: 0.75rem; color: #475569; }

/* Context card */
.ctx-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 0.8rem 1rem; margin-top: 0.4rem; font-size: 0.82rem; color: #475569; line-height: 1.5; }
.ctx-card strong { color: #1e293b; }

/* Empty state */
.empty-state { text-align: center; padding: 3rem 1rem; color: #94a3b8; }
.empty-state .big-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.empty-state h3 { color: #64748b; font-weight: 600; margin: 0 0 0.4rem; }

/* Ingest success */
.ingest-card { background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.5rem; }
.ingest-card .ic-title { font-weight: 700; color: #15803d; font-size: 0.9rem; }
.ingest-card .ic-row   { font-size: 0.82rem; color: #166534; margin-top: 0.2rem; }

/* Active doc badge */
.active-doc { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
              padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #1d4ed8; font-weight: 500; }

/* Section labels */
.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
                 text-transform: uppercase; color: #64748b; padding: 0.75rem 0 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "current_doc_id": None,
    "current_doc_name": None,
    "pending_question": None,
    "docs_list": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def fetch_documents():
    try:
        res = requests.get(f"{API_BASE}/documents", timeout=10)
        if res.status_code == 200:
            st.session_state.docs_list = res.json().get("documents", [])
    except Exception:
        pass


if not st.session_state.docs_list:
    fetch_documents()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:1.2rem 0 0.5rem">
        <span style="font-size:1.8rem">🏥</span>
        <div>
            <div style="font-size:1.05rem;font-weight:700">Medicare Hospital</div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:1px">RAG Assistant v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">📄 Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Processing PDF…"):
                try:
                    res = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.current_doc_id = data["document_id"]
                        st.session_state.current_doc_name = data["filename"]
                        fetch_documents()
                        st.markdown(f"""
                        <div class="ingest-card">
                            <div class="ic-title">✅ Ingested successfully</div>
                            <div class="ic-row">📄 {data['filename']}</div>
                            <div class="ic-row">📊 {data['total_chunks']} chunks · {data['total_pages']} pages</div>
                            <div class="ic-row">⏱️ {data['processing_time_ms']} ms</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(res.json().get("detail", "Upload failed"))
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend not running.\nStart: `cd backend && uvicorn main:app --reload`")

    st.divider()

    st.markdown('<div class="section-label">📁 Documents</div>', unsafe_allow_html=True)
    col_r, col_all = st.columns([1, 1])
    with col_r:
        if st.button("↺ Refresh", use_container_width=True):
            fetch_documents()
            st.rerun()
    with col_all:
        if st.button("Reset filter", use_container_width=True):
            st.session_state.current_doc_id = None
            st.session_state.current_doc_name = "All Documents"
            st.rerun()

    docs = st.session_state.docs_list or []
    if docs:
        options = ["All Documents"] + [
            f"{doc['filename'][:30]}{'…' if len(doc['filename']) > 30 else ''} · {doc['total_pages']}p"
            for doc in docs
        ]
        current_index = 0
        if st.session_state.current_doc_id:
            current_index = next(
                (idx + 1 for idx, doc in enumerate(docs) if doc["id"] == st.session_state.current_doc_id),
                0,
            )

        chosen = st.selectbox("Choose a document to limit search", options, index=current_index)
        if chosen == "All Documents":
            doc_id = None
            doc_name = "All Documents"
        else:
            selected_doc = docs[options.index(chosen) - 1]
            doc_id = selected_doc["id"]
            doc_name = selected_doc["filename"]

        if doc_id != st.session_state.current_doc_id:
            st.session_state.current_doc_id = doc_id
            st.session_state.current_doc_name = doc_name
            st.experimental_rerun()

        st.markdown(f"**{len(docs)} documents ingested** · Select one to scope your search.")
    else:
        st.caption("No documents ingested yet. Upload a PDF to begin.")

    if st.session_state.current_doc_name:
        name = st.session_state.current_doc_name
        st.markdown(
            f'<div class="active-doc">🎯 {name[:35]}{"…" if len(name) > 35 else ""}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown('<div class="section-label">💬 Quick Questions</div>', unsafe_allow_html=True)
    SAMPLES = [
        "What are OPD timings?",
        "Who is the cardiologist?",
        "What is the cost of MRI?",
        "Can I cancel appointment within 24 hours?",
        "What is ICU cost per day?",
        "Emergency number?",
        "What is a private room cost?",
        "What languages does the hospital support?",
    ]
    for q in SAMPLES:
        if st.button(q, key=f"sq_{q}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🏥</div>
    <div class="hero-text">
        <h1>Medicare Hospital Assistant</h1>
        <p>Production-ready RAG search with pgvector + BM25 hybrid retrieval · High precision, grounded answers from your hospital PDF.</p>
    </div>
</div>
<div class="metrics" style="justify-content:flex-start;gap:14px;margin-bottom:1rem;">
    <span class="metric-pill">100+ concurrent-ready</span>
    <span class="metric-pill">sub-200ms query latency design</span>
    <span class="metric-pill">Cached Supabase reads</span>
</div>
""", unsafe_allow_html=True)


def render_message(entry: dict):
    st.markdown(f"""
    <div class="msg-wrap user">
        <div class="avatar user">👤</div>
        <div class="bubble">{entry['question']}</div>
    </div>
    """, unsafe_allow_html=True)

    answer = entry.get("answer", "")
    is_fallback = "don't have that information" in answer.lower()

    if is_fallback:
        answer_html = f'<div class="fallback">⚠️ {answer}</div>'
    else:
        answer_html = answer.replace("\n", "<br>")

    sources = entry.get("sources", [])
    chips_html = ""
    if sources:
        chips = "".join(f'<span class="chip">📄 {s}</span>' for s in sources)
        chips_html = f'<div class="source-row">{chips}</div>'

    latency  = entry.get("latency_ms", "")
    n_chunks = entry.get("n_chunks", "")
    method   = entry.get("retrieval_method", "hybrid")
    metrics_html = ""
    if latency:
        metrics_html = f"""
        <div class="metrics">
            <span class="metric-pill">⏱️ {latency} ms</span>
            <span class="metric-pill">📑 {n_chunks} chunks</span>
            <span class="metric-pill">🔍 {method}</span>
        </div>"""

    st.markdown(f"""
    <div class="msg-wrap bot">
        <div class="avatar bot">🤖</div>
        <div class="bubble">{answer_html}{chips_html}{metrics_html}</div>
    </div>
    """, unsafe_allow_html=True)

    source_chunks = entry.get("source_chunks", [])
    if source_chunks:
        with st.expander("🔎 View retrieved context", expanded=False):
            for i, chunk in enumerate(source_chunks, 1):
                st.markdown(f"""
                <div class="ctx-card">
                    <strong>Chunk {i} · Page {chunk['page']} · score {chunk['similarity_score']:.3f}</strong><br>
                    {chunk['text_preview']}…
                </div>
                """, unsafe_allow_html=True)


# Render history
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">💬</div>
        <h3>No conversation yet</h3>
        <p>Upload a hospital PDF and ask anything about it.<br>
           Or try a quick question from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for entry in st.session_state.chat_history:
        render_message(entry)


# ── Input ──────────────────────────────────────────────────────────────────────
# BUG FIX: st.chat_input does NOT support a `key` parameter in any Streamlit version.
# The original code passed key="chat_input" which raises StreamlitAPIException.
question = st.chat_input("Ask a question about the hospital…")

# Pick up sidebar quick-question (pending_question)
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:
    payload = {"question": question}
    if st.session_state.current_doc_id:
        payload["document_id"] = st.session_state.current_doc_id

    with st.spinner("🔍 Searching documents…"):
        try:
            res = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": data["answer"],
                    "sources": data.get("sources", []),
                    "source_chunks": data.get("source_chunks", []),
                    "latency_ms": data.get("latency_ms", ""),
                    "n_chunks": len(data.get("source_chunks", [])),
                    "retrieval_method": data.get("retrieval_method", "hybrid"),
                })
                st.rerun()
            else:
                st.error(f"API Error {res.status_code}: {res.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server.\n```\ncd backend && uvicorn main:app --reload\n```")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
