"""
app.py — Streamlit UI for "Chat With Your Docs"

A local RAG web app that lets users upload PDF documents and ask questions
about them, powered by Groq cloud inference.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_GROQ_MODEL,
    TOP_K,
    add_documents,
    check_groq_status,
    chunk_text,
    clear_collection,
    extract_text_from_pdf,
    get_chroma_collection,
    get_collection_stats,
    get_embedding_function,
    query_documents,
    rewrite_query_with_ai,
    generate_document_questions,
    stream_answer,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat With Your Docs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── BASE ─────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #E2E8F0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    .stApp {
        background: #0F1117 !important;
    }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer, [data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }

    .block-container {
        max-width: 860px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 6rem !important;
    }

    /* ── SIDEBAR ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0A0B0F !important;
        border-right: 1px solid #1A1D28 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem !important;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0 0 0.6rem 0;
        padding: 0;
    }

    .subtle-divider {
        border: none;
        border-top: 1px solid #1A1D28;
        margin: 1.25rem 0;
    }

    /* ── BUTTONS ──────────────────────────────────────────── */
    .stButton > button {
        background: #4F46E5 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.55rem 1.1rem !important;
        transition: background 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow: none !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        background: #4338CA !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    .stButton > button:active {
        background: #3730A3 !important;
    }

    /* Secondary / ghost buttons (clear actions) */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid #2D3142 !important;
        color: #94A3B8 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #4F46E5 !important;
        color: #E2E8F0 !important;
        box-shadow: none !important;
    }

    /* ── INPUTS ───────────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stTextInput"] input:focus {
        background: #13151F !important;
        border: 1px solid #2D3142 !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #6366F1 !important;
    }

    /* ── FILE UPLOADER ────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        width: 100% !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #13151F !important;
        border: 1.5px dashed #2D3142 !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
        min-height: 90px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366F1 !important;
        background: #14162A !important;
    }

    /* Style the Upload button inside dropzone */
    [data-testid="stFileUploaderDropzone"] button {
        background: #1E2235 !important;
        border: 1px solid #2D3142 !important;
        border-radius: 8px !important;
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.1rem !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #252A40 !important;
        border-color: #6366F1 !important;
        color: #E2E8F0 !important;
    }

    /* Hide "200MB per file • PDF" limit text */
    [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2),
    [data-testid="stFileUploaderDropzoneInstructions"] > span:nth-child(2),
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2),
    [data-testid="stFileUploaderDropzoneInstructions"] > div > div:nth-child(2),
    [data-testid="stFileDropzoneInstructions"] > div:nth-child(2),
    [data-testid="stFileDropzoneInstructions"] > span:nth-child(2),
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    /* ── CHAT MESSAGES ────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: #13151F !important;
        border-radius: 12px !important;
        border: 1px solid #1E2235 !important;
        padding: 1.1rem 1.4rem !important;
        margin-bottom: 1rem !important;
        box-shadow: none !important;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: #4F46E5 !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: #059669 !important;
        border-radius: 8px !important;
    }

    /* ── CHAT INPUT ───────────────────────────────────────── */
    [data-testid="stChatInputContainer"] {
        background: #13151F !important;
        border: 1px solid #2D3142 !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease !important;
    }

    [data-testid="stChatInputContainer"]:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }

    /* ── EXPANDERS ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: #13151F !important;
        border: 1px solid #1E2235 !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }

    /* ── SOURCE CARDS ─────────────────────────────────────── */
    .source-card {
        background: #0D0F18;
        border: 1px solid #1E2235;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .source-label {
        font-family: 'JetBrains Mono', monospace;
        color: #818CF8;
        font-weight: 500;
        font-size: 0.78rem;
    }

    .score-badge {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── STAT CARDS ───────────────────────────────────────── */
    .stats-row {
        display: flex;
        gap: 10px;
        margin: 0.5rem 0;
    }

    .stat-card {
        flex: 1;
        background: #13151F;
        border: 1px solid #1E2235;
        border-radius: 10px;
        padding: 12px 10px;
        text-align: center;
    }

    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #E2E8F0;
        line-height: 1;
        margin-bottom: 4px;
    }

    .stat-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── QUICK-QUESTION CHIPS ─────────────────────────────── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #13151F !important;
        border: 1px solid #2D3142 !important;
        color: #CBD5E1 !important;
        border-radius: 99px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
        white-space: normal !important;
        line-height: 1.4 !important;
        text-align: center !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #1E2235 !important;
        border-color: #6366F1 !important;
        color: #E2E8F0 !important;
        box-shadow: none !important;
    }

    /* ── ALERTS / WARNINGS ────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 0.88rem !important;
    }

    /* ── PROGRESS BAR ─────────────────────────────────────── */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5, #818CF8) !important;
    }

    /* ── SPINNER ──────────────────────────────────────────── */
    [data-testid="stSpinner"] {
        color: #6366F1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# JS: hide "200MB per file • PDF" using components.html (real iframe, scripts actually execute)
components.html(
    """
    <script>
    (function() {
        function sweep() {
            try {
                const doc = window.parent.document;
                const walker = doc.createTreeWalker(
                    doc.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                let node;
                while ((node = walker.nextNode())) {
                    if (node.nodeValue && node.nodeValue.includes('200MB')) {
                        const el = node.parentElement;
                        if (el) el.style.setProperty('display', 'none', 'important');
                    }
                }
            } catch(e) {}
        }
        sweep();
        const id = setInterval(sweep, 300);
        setTimeout(function() { clearInterval(id); }, 30000);
    })();
    </script>
    """,
    height=0,
    scrolling=False,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        "chat_history": [],
        "processed_files": set(),
        "total_chunks": 0,
        "groq_api_key": "",
        "model_name": DEFAULT_GROQ_MODEL,
        "embedding_model": None,
        "suggested_questions": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


@st.cache_resource(show_spinner=False)
def get_cached_embed_model():
    return get_embedding_function()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Authentication ─────────────────────────────────────────
    st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    st.session_state.model_name = DEFAULT_GROQ_MODEL


    # ── Upload ───────────────────────────────────────────────
    st.markdown('<p class="section-label">PDF Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Select one or more PDF files. Text must be selectable (not scanned images).",
    )

    st.markdown(
        '<p style="font-size:0.75rem;color:#374151;margin:-6px 0 10px 0;text-align:center;">'
        'PDF only &nbsp;·&nbsp; text must be selectable'
        '</p>',
        unsafe_allow_html=True,
    )

    if uploaded_files:
        process_btn = st.button("Process Documents", use_container_width=True, type="primary")


        if process_btn:
            with st.spinner("Loading AI model…"):
                embed_model = get_cached_embed_model()
                st.session_state.embedding_model = embed_model

            collection = get_chroma_collection()
            total_new_chunks = 0
            processed_count = 0
            skipped_files: list[str] = []

            progress_bar = st.progress(0, text="Reading documents…")

            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name

                progress_bar.progress(
                    idx / len(uploaded_files),
                    text=f"Processing: {file_name}…",
                )

                text = extract_text_from_pdf(uploaded_file)

                if not text.strip():
                    skipped_files.append(file_name)
                    st.warning(
                        f"**{file_name}** has no readable text. "
                        f"It may be a scanned document — try converting it to text first."
                    )
                    continue

                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

                if not chunks:
                    skipped_files.append(file_name)
                    continue

                num_added = add_documents(chunks, file_name, collection, embed_model)
                total_new_chunks += num_added
                processed_count += 1
                st.session_state.processed_files.add(file_name)

            progress_bar.progress(1.0, text="Done!")

            st.session_state.total_chunks += total_new_chunks

            if processed_count > 0:
                st.success(f"✓ {processed_count} file(s) ready to chat.")
                if st.session_state.groq_api_key:
                    with st.spinner("Generating starter questions…"):
                        try:
                            sample_chunks = collection.get(limit=10, include=["documents"]).get("documents", [])
                            if sample_chunks:
                                st.session_state.suggested_questions = generate_document_questions(
                                    sample_chunks,
                                    st.session_state.groq_api_key,
                                    st.session_state.model_name,
                                )
                        except Exception:
                            pass

            if skipped_files:
                st.info(f"Skipped {len(skipped_files)} file(s) with no readable text.")

    # ── Tips ─────────────────────────────────────────────────
    with st.expander("💡 Tips for best results", expanded=False):
        st.markdown(
            """
- **Selectable text:** You should be able to highlight words in your PDF with your mouse.
- **Keep it focused:** Files under 50 pages give faster, more accurate answers.
- **Be specific:** Use names and keywords from your document in your questions.
            """
        )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Stats ────────────────────────────────────────────────
    stats = get_collection_stats()
    num_docs = len(stats["sources"])
    num_chunks = stats["total_chunks"]

    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{num_docs}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{num_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stats["sources"]:
        with st.expander("📂 Loaded files", expanded=False):
            for src in stats["sources"]:
                st.markdown(f"<div style='font-size:0.82rem;color:#94A3B8;padding:2px 0;'>📄 {src}</div>", unsafe_allow_html=True)

    # ── Clear actions ─────────────────────────────────────────
    if num_chunks > 0 or st.session_state.chat_history:
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.chat_history:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        with col2:
            if num_chunks > 0:
                if st.button("Clear Docs", use_container_width=True):
                    clear_collection()
                    st.session_state.chat_history = []
                    st.session_state.processed_files = set()
                    st.session_state.total_chunks = 0
                    st.rerun()

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

stats = get_collection_stats()
has_documents = stats["total_chunks"] > 0
num_active_docs = len(stats["sources"])

# Header
st.markdown(
    """
    <div style="margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #1A1D28;">
        <h1 style="font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 800; color: #FFFFFF; margin: 0 0 4px 0; letter-spacing: -0.03em;">
            Chat With Your Docs
        </h1>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0; font-weight: 400;">
            Upload a PDF in the sidebar, then ask anything about it.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View sources", expanded=False):
                for source in message["sources"]:
                    similarity = (1 - source["distance"]) * 100
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span class="source-label">📎 {source['filename']} — chunk {source['chunk_index']}</span>
                                <span class="score-badge">{similarity:.0f}% match</span>
                            </div>
                            <div style="color: #94A3B8; font-size: 0.875rem; line-height: 1.65;">
                                {source['text'][:500]}{'…' if len(source['text']) > 500 else ''}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------

if not st.session_state.chat_history and not has_documents:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1.5rem; border: 1px solid #1A1D28; border-radius: 14px; background: #0D0F18; margin: 1rem 0 2rem 0;">
            <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #E2E8F0; margin: 0 0 6px 0;">
                No documents loaded
            </h3>
            <p style="color: #4B5563; font-size: 0.9rem; margin: 0;">
                Upload a PDF in the sidebar to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif not st.session_state.chat_history and has_documents:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2.5rem 1.5rem; border: 1px solid #1A1D28; border-radius: 14px; background: #0D0F18; margin: 1rem 0 2rem 0;">
            <div style="display: inline-block; background: rgba(79,70,229,0.12); border-radius: 8px; padding: 4px 14px; margin-bottom: 14px;">
                <span style="color: #818CF8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    {num_active_docs} file{'s' if num_active_docs != 1 else ''} ready
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
                Ask any question about your documents below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Chat input & quick chips
# ---------------------------------------------------------------------------

if not has_documents:
    st.chat_input(
        "Upload a PDF in the sidebar first…",
        disabled=True,
    )
else:
    selected_chip = None

    # Quick-question chips
    if (
        has_documents
        and st.session_state.suggested_questions
        and not st.session_state.chat_history
    ):
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:#4B5563;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px 0;">Try asking</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            with cols[i]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    selected_chip = q

    prompt = st.chat_input("Ask a question about your documents…") or selected_chip

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.groq_api_key:
                st.error("The app administrator has not configured the Groq API key.")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ The app administrator has not configured the Groq API key.",
                    "sources": [],
                })
                st.stop()

            ok, err = check_groq_status(st.session_state.groq_api_key)
            if not ok:
                st.error(f"{err}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ {err}",
                    "sources": [],
                })
                st.stop()

            with st.spinner("Searching your documents…"):
                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = get_cached_embed_model()

                collection = get_chroma_collection()
                search_query = rewrite_query_with_ai(
                    prompt,
                    st.session_state.chat_history[:-1],
                    st.session_state.groq_api_key,
                    st.session_state.model_name,
                )
                results = query_documents(
                    search_query,
                    collection,
                    st.session_state.embedding_model,
                    TOP_K,
                )

            documents = results["documents"][0] if results["documents"][0] else []
            metadatas = results["metadatas"][0] if results["metadatas"][0] else []
            distances = results["distances"][0] if results["distances"][0] else []

            sources: list[dict] = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                sources.append({
                    "text": doc,
                    "filename": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                })

            try:
                # Pass the chat history EXCLUDING the current user prompt (since it was just added to the end)
                stream_gen = stream_answer(
                    question=prompt,
                    context_chunks=documents,
                    model=st.session_state.model_name,
                    api_key=st.session_state.groq_api_key,
                    chat_history=st.session_state.chat_history[:-1],
                )
                answer = st.write_stream(stream_gen)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                st.error(answer)

            if sources:
                with st.expander("View sources", expanded=False):
                    for source in sources:
                        similarity = (1 - source["distance"]) * 100
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span class="source-label">📎 {source['filename']} — chunk {source['chunk_index']}</span>
                                    <span class="score-badge">{similarity:.0f}% match</span>
                                </div>
                                <div style="color: #94A3B8; font-size: 0.875rem; line-height: 1.65;">
                                    {source['text'][:500]}{'…' if len(source['text']) > 500 else ''}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
