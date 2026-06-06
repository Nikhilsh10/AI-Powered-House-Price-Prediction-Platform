# src/dashboard/pages/3_Explainability.py
"""Explainability page – SHAP visualisations for the selected prediction.

This page relies on the artefacts produced by the training pipeline:
- artifacts/model.pkl (required)
- artifacts/preprocessor.pkl (required)
- artifacts/metadata.json (required)

Optionally, if a serialized SHAP explainer is available at
artifacts/shap_explainer.pkl, we will render SHAP summary and waterfall
plots. If it is missing we fall back to a simple textual recap.
"""

import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# Deterministic project root – this file is at <repo>/src/dashboard/pages/3_Explainability.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Required artefacts
# ---------------------------------------------------------------------------
artifacts_dir = PROJECT_ROOT / "artifacts"
model_path = artifacts_dir / "model.pkl"
preproc_path = artifacts_dir / "preprocessor.pkl"
metadata_path = artifacts_dir / "metadata.json"

missing_required = []
for p in (model_path, preproc_path, metadata_path):
    if not p.exists():
        missing_required.append(str(p))

if missing_required:
    st.error(f"Missing required artefacts:\n" + "\n".join(missing_required))
    st.stop()

# Optional SHAP explainer
shap_path = artifacts_dir / "shap_explainer.pkl"

# ---------------------------------------------------------------------------
# Session-state handling – ensure a prediction has been made on Home page
# ---------------------------------------------------------------------------
required_keys = ["latest_prediction", "latest_result"]
missing_keys = [k for k in required_keys if k not in st.session_state]
if missing_keys:
    st.info("Run a prediction on the Home page first.")
    st.stop()

input_data = st.session_state["latest_prediction"]
result = st.session_state["latest_result"]

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import joblib
    return joblib.load(model_path)

@st.cache_resource
def load_preprocessor():
    import joblib
    return joblib.load(preproc_path)

@st.cache_resource
def load_shap_explainer():
    if shap_path.exists():
        import joblib
        return joblib.load(shap_path)
    return None

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
st.title("🔍 Explainability")
st.markdown("---")

st.subheader("Prediction recap")
st.write(f"**Price:** {result['predicted_price']:.2f} Lakh₹")
st.write(
    f"**Confidence interval:** {result['lower_bound']:.2f} – {result['upper_bound']:.2f}"
)

# ---------------------------------------------------------------------------
# SHAP visualisations (if explainer available)
# ---------------------------------------------------------------------------
explainer = load_shap_explainer()
if explainer:
    model = load_model()
    preproc = load_preprocessor()

    # Prepare data for SHAP
    import pandas as pd
    df = pd.DataFrame([input_data])
    df_processed = preproc.transform(df)

    # Compute SHAP values – works for both TreeExplainer and KernelExplainer
    shap_values = explainer.shap_values(df_processed)

    # -------------------------------------------------------------------
    # Summary plot
    # -------------------------------------------------------------------
    st.subheader("SHAP Summary Plot")
    import shap
    fig = plt.figure()
    shap.summary_plot(shap_values, df_processed, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Waterfall plot for this single instance
    # -------------------------------------------------------------------
    st.subheader("SHAP Waterfall (Feature impact)")
    fig_wf = plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            data=df_processed[0],
            feature_names=df.columns.tolist(),
        )
    )
    st.pyplot(fig_wf)
    plt.close(fig_wf)

    # -------------------------------------------------------------------
    # Feature contribution table – sorted by absolute impact
    # -------------------------------------------------------------------
    st.subheader("Feature Contributions")
    import numpy as np
    feature_names = df.columns.tolist()
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_values[0],
    })
    # Add absolute impact for sorting
    contrib_df["abs_impact"] = contrib_df["impact"].abs()
    # Sort by absolute impact descending and keep top 15 features
    contrib_df = contrib_df.sort_values("abs_impact", ascending=False).head(15)
    st.dataframe(contrib_df[["feature", "impact", "abs_impact"]])
else:
    st.info("SHAP explainer not found – only the textual prediction recap is shown.")
