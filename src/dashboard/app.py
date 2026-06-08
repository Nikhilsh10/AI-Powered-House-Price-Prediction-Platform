# src/dashboard/app.py
"""Main entry point for the Streamlit dashboard.

Streamlit multipage apps automatically discover Python files inside the
``pages/`` subdirectory. This file:
  1. Ensures the repo root is on sys.path before any project imports.
  2. Configures page-level settings (must be first Streamlit call).
  3. Injects the custom CSS stylesheet.
  4. Renders the sidebar branding.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path BEFORE any project imports.
#   This file lives at:  <repo>/src/dashboard/app.py
#     parents[0] = <repo>/src/dashboard
#     parents[1] = <repo>/src
#     parents[2] = <repo>          ← repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar branding
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:1.25rem 0 1rem 0;'>
            <div style='font-size:2.2rem; margin-bottom:0.35rem;'>🏠</div>
            <div style='
                font-family: Space Grotesk, sans-serif;
                font-size: 0.95rem;
                font-weight: 700;
                letter-spacing: -0.01em;
                background: linear-gradient(135deg, #5eead4, #fbbf24);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.2rem;
            '>Price Intelligence</div>
            <div style='
                font-size: 0.72rem;
                color: #4f5668;
                font-weight: 500;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            '>Bengaluru Housing</div>
        </div>
        <div style='height:1px; background: rgba(255,255,255,0.06); margin:0.5rem 0 1rem 0;'></div>
        """,
        unsafe_allow_html=True,
    )
