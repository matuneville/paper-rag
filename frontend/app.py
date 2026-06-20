"""
Streamlit frontend for paper-rag.

Run (from project root):
    .venv/bin/streamlit run frontend/app.py --server.port 8501

Requires the FastAPI backend running on http://localhost:8000.
"""

import streamlit as st
from pathlib import Path

import api_client
import styles

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Paper RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles.inject()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources?}]

if "paper_filter" not in st.session_state:
    st.session_state.paper_filter = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # set on submit, cleared after AI responds


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo + title
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size:2.4rem;">📄</div>
            <div style="font-size:1.3rem; font-weight:700;
                        background: linear-gradient(135deg, #667eea, #a78bfa);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        margin-top: 4px;">
                Paper RAG
            </div>
            <div style="font-size:0.75rem; color:#6b7280; margin-top:2px;">
                Chat with your research papers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API health indicator
    health_data, health_err = api_client.health()
    if health_err:
        st.markdown('<p class="status-error">⚠ API offline</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p class="status-ok">● API online &nbsp;v{health_data.get("version", "")}</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Upload paper</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    paper_title_input = st.text_input(
        "Paper title (optional)",
        placeholder="Leave blank to use filename",
        label_visibility="collapsed",
    )

    if st.button("⬆ Ingest PDF", use_container_width=True):
        if not uploaded_file:
            st.warning("Select a PDF first.")
        else:
            # Default title = filename without extension
            filename_stem = Path(uploaded_file.name).stem
            effective_title = paper_title_input.strip() or filename_stem
            if not paper_title_input.strip():
                st.info(f"No title provided — using **{filename_stem}** as title.")
            with st.spinner("Ingesting…"):
                data, err = api_client.upload_paper(
                    filename=uploaded_file.name,
                    content=uploaded_file.getvalue(),
                    paper_title=effective_title,
                )
            if err:
                if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    st.error(
                        "⏳ LLM API rate limit reached. "
                        "Please try again later."
                    )
                else:
                    st.error(err)
            else:
                st.success(
                    f"✓ **{data['paper_title']}**  \n"
                    f"{data['pages']} pages · {data['chunks']} chunks"
                )
                st.rerun()

    st.divider()

    # ── Indexed papers ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Indexed papers</p>', unsafe_allow_html=True)

    papers_data, papers_err = api_client.list_papers()

    if papers_err:
        st.caption(f"Could not load papers: {papers_err}")
    elif not papers_data or not papers_data.get("papers"):
        st.caption("No papers indexed yet.")
        st.session_state.paper_filter = None
    else:
        papers = papers_data["papers"]

        # Scope selector
        filter_options = ["All papers"] + [p["title"] for p in papers]
        selected = st.selectbox("Query scope", filter_options, label_visibility="collapsed")
        st.session_state.paper_filter = None if selected == "All papers" else selected

        # Paper list with per-paper delete
        for paper in papers:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div class="paper-pill">'
                    f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{paper["title"]}</span>'
                    f'<span class="chunk-badge">{paper["chunk_count"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑", key=f"del_{paper['title']}", help=f"Delete {paper['title']}"):
                    _, err = api_client.delete_paper(paper["title"])
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Deleted '{paper['title']}'")
                        if st.session_state.paper_filter == paper["title"]:
                            st.session_state.paper_filter = None
                        st.rerun()

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main — chat area
# ---------------------------------------------------------------------------

scope_label = (
    f"**{st.session_state.paper_filter}**"
    if st.session_state.paper_filter
    else "all indexed papers"
)
st.markdown(
    f"""
    <div style="padding: 8px 0 20px 0;">
        <h1 style="font-size:1.6rem; font-weight:700; margin:0;
                   background: linear-gradient(135deg, #e0e0f0, #a78bfa);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Ask your papers
        </h1>
        <p style="font-size:0.85rem; color:#6b7280; margin:4px 0 0 0;">
            Querying {scope_label}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render message history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;justify-content:flex-end;margin:8px 0;">'
            f'<div class="user-bubble">{msg["content"]}</div>'
            f'<span style="font-size:1.5rem;margin-left:8px;flex-shrink:0;line-height:1.2;">👤</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin:8px 0;">'
            f'<span style="font-size:1.5rem;margin-right:8px;flex-shrink:0;line-height:1.2;">🤖</span>'
            f'<div class="assistant-bubble">{msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    page_str = f"page {src['page']}" if src.get("page") is not None else "page ?"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["paper_title"]}</strong> · {page_str}<br>'
                        f'<span style="color:#8892b0;">{src["excerpt"][:200]}…</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Empty state
if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 0;">
            <div style="font-size:3rem; margin-bottom:12px;">💬</div>
            <div style="font-size:1rem; color:#6b7280;">
                Upload a PDF in the sidebar, then ask anything about it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Thinking bubble (renders on the rerun between user submit and AI response) ─
if st.session_state.pending_question:
    st.markdown(
        '<div style="display:flex;align-items:flex-start;margin:8px 0;">'
        '<span style="font-size:1.5rem;margin-right:8px;flex-shrink:0;line-height:1.2;">🤖</span>'
        '<div class="assistant-bubble thinking">Thinking…</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    result, err = api_client.chat(
        question=st.session_state.pending_question,
        paper_filter=st.session_state.paper_filter,
    )
    if err:
        if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            content = "⏳ LLM API rate limit reached. Please try again later."
        else:
            content = f"⚠ {err}"
        st.session_state.messages.append({"role": "assistant", "content": content, "sources": []})
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            }
        )
    st.session_state.pending_question = None
    st.rerun()

# ── Input form (Enter submits; clear_on_submit wipes the box) ────────────────
st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([9, 1])
    with col_input:
        question = st.text_input(
            "question",
            placeholder="Ask anything about your papers…",
            label_visibility="collapsed",
        )
    with col_send:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and question.strip():
    q = question.strip()
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.pending_question = q
    st.rerun()
