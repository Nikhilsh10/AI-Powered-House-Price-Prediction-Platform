# src/dashboard/app.py
"""Main entry point for the Streamlit dashboard.

The Streamlit multipage app automatically discovers Python files in the
`pages/` subdirectory whose names start with a numeric prefix, e.g.
`1_Home.py`.  Therefore this `app.py` only needs to configure the page and
inject custom CSS.
"""

import streamlit as st
import os, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to the Python path (required for Cloud)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Global page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS if present
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom CSS not found – using default Streamlit styling.")

# No explicit routing is required; Streamlit will render pages based on the
# numeric prefixes in the `pages/` folder.
