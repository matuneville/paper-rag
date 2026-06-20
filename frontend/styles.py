"""
CSS styles for the paper-rag Streamlit frontend.

Keeping styles in one place makes visual iteration faster
without touching application logic.
"""

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Chat message bubbles */
.user-bubble {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 15%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.assistant-bubble {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e0e0f0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 15% 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
}

/* Source card */
.source-card {
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #a0a8c8;
}

.source-card strong {
    color: #8b9cf7;
}

/* Paper pill */
.paper-pill {
    background: rgba(102, 126, 234, 0.12);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.83rem;
    color: #b0b8d8;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.paper-pill .chunk-badge {
    background: rgba(102, 126, 234, 0.25);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.75rem;
    color: #8b9cf7;
}

/* Input area */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Status messages */
.status-ok    { color: #5eead4; font-size: 0.8rem; }
.status-error { color: #f87171; font-size: 0.8rem; }

/* Section header */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 18px 0 8px 0;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
"""


def inject() -> None:
    """Inject global CSS into the Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
