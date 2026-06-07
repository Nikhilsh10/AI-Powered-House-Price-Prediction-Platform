# src/dashboard/app.py
"""Main entry point for the Streamlit dashboard.

Streamlit multipage apps automatically discover Python files inside the
`pages/` subdirectory.  This file configures the page, injects CSS,
and ensures the repo root is on sys.path for all downstream imports.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path BEFORE any project imports.
# This file lives at:  <repo>/src/dashboard/app.py
#   parents[0] = <repo>/src/dashboard
#   parents[1] = <repo>/src
#   parents[2] = <repo>                ← repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
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
        <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
            <span style='font-size:2rem;'>🏠</span>
            <h3 style='margin:0.25rem 0 0 0; font-weight:700; font-size:1rem;
                        background: linear-gradient(135deg,#58a6ff,#3fb9a8);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;'>
                Price Prediction
            </h3>
            <p style='color:#8b949e; font-size:0.75rem; margin:0.25rem 0 0 0;'>
                Bengaluru Housing Dataset
            </p>
        </div>
        <hr style='border:none; height:1px;
                    background:linear-gradient(90deg,transparent,rgba(139,148,158,0.2),transparent);
                    margin:0.75rem 0;'>
        """,
        unsafe_allow_html=True,
    )
