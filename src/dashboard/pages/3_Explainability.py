# src/dashboard/pages/3_Explainability.py
"""Explainability page — SHAP visualisations for the latest prediction.

Required artifacts:
  - artifacts/model.pkl
  - artifacts/preprocessor.pkl

Optional artifacts:
  - artifacts/shap_explainer.pkl  (if absent, a TreeExplainer is built on-demand)

The page reads the latest prediction from st.session_state, which is
populated by the Home page when a prediction is made.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import ARTIFACTS_DIR

# ---------------------------------------------------------------------------
# Required artifact guard
# ---------------------------------------------------------------------------
_model_path  = ARTIFACTS_DIR / "model.pkl"
_preproc_path = ARTIFACTS_DIR / "preprocessor.pkl"
_missing = [str(p) for p in (_model_path, _preproc_path) if not p.exists()]

if _missing:
    st.error("**Missing required artifacts:**\n\n" + "\n".join(f"- `{p}`" for p in _missing))
    st.stop()

# ---------------------------------------------------------------------------
# Session-state guard
# ---------------------------------------------------------------------------
if "latest_prediction" not in st.session_state or "latest_result" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding:4rem 1rem; color:#4f5668;'>
        <div style='font-size:2.5rem; margin-bottom:0.75rem; opacity:0.4;'>🔍</div>
        <div style='font-size:0.92rem; font-weight:500; color:#8a93a8;'>
            No prediction yet
        </div>
        <div style='font-size:0.82rem; margin-top:0.4rem;'>
            Go to <strong style='color:#5eead4;'>Home</strong> and make a prediction first.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

input_data = st.session_state["latest_prediction"]
result     = st.session_state["latest_result"]

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import joblib
    return joblib.load(_model_path)


@st.cache_resource
def load_preprocessor():
    import joblib
    return joblib.load(_preproc_path)


@st.cache_resource
def load_or_build_explainer():
    """Load serialised SHAP explainer if present, else build TreeExplainer."""
    import shap
    shap_path = ARTIFACTS_DIR / "shap_explainer.pkl"
    if shap_path.exists():
        import joblib
        return joblib.load(shap_path)
    model = load_model()
    if hasattr(model, "get_booster") or hasattr(model, "feature_name_"):
        return shap.TreeExplainer(model)
    return None  # KernelExplainer requires background data; skip for now


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<div class='page-enter'>
<h1 style='margin-bottom:0.2rem;'>
    Model <span style='color:var(--accent-blue);'>Explainability</span>
</h1>
<p style='color:var(--text-secondary); font-size:0.92rem; margin-top:0; margin-bottom:1.75rem;'>
    SHAP-based feature attribution for your latest prediction
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1 — Prediction Recap
# ---------------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:0.5rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:var(--text-muted);'>Prediction Recap</div>
    <div style='flex:1; height:1px; background:var(--border-medium);'></div>
</div>
""", unsafe_allow_html=True)

pred  = result["predicted_price"]
lower = result["lower_bound"]
upper = result["upper_bound"]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-teal'>
        <div class='kpi-label'>Predicted Price</div>
        <div class='kpi-value'>₹{pred:,.2f}L</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-indigo'>
        <div class='kpi-label'>Lower Bound</div>
        <div class='kpi-value'>{lower:,.2f}L</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-amber'>
        <div class='kpi-label'>Upper Bound</div>
        <div class='kpi-value'>{upper:,.2f}L</div>
    </div>""", unsafe_allow_html=True)

# Input summary
with st.expander("Input features used for this prediction", expanded=False):
    st.json(input_data)

# ---------------------------------------------------------------------------
# Section 2 — SHAP visualisations
# ---------------------------------------------------------------------------
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:0.5rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:var(--text-muted);'>SHAP Analysis</div>
    <div style='flex:1; height:1px; background:var(--border-medium);'></div>
</div>
""", unsafe_allow_html=True)

try:
    preproc  = load_preprocessor()
    explainer = load_or_build_explainer()

    # Transform the single input instance
    df_input     = pd.DataFrame([input_data])
    df_processed = preproc.transform(df_input)

    # Resolve feature names from the preprocessor (post-transform)
    try:
        feature_names = list(preproc.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(df_processed.shape[1])]

    if explainer is not None:
        import shap
        shap_values = explainer.shap_values(df_processed)

        # Waterfall plot (single instance)
        st.subheader("Feature Impact — Waterfall")
        fig_wf, ax_wf = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.35)))
        fig_wf.patch.set_facecolor("#F9F9F6")
        ax_wf.set_facecolor("#F9F9F6")

        expected_val = (
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        expl = shap.Explanation(
            values=sv,
            base_values=expected_val,
            data=df_processed[0],
            feature_names=feature_names,
        )
        shap.plots.waterfall(expl, show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

        # Feature contribution table
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.subheader("Feature Contributions Table")

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": sv,
            "Direction": ["↑ Increases price" if v > 0 else "↓ Decreases price" for v in sv],
        })
        contrib_df["Abs SHAP"] = contrib_df["SHAP Value"].abs()
        contrib_df = contrib_df.sort_values("Abs SHAP", ascending=False).head(15)

        st.dataframe(
            contrib_df[["Feature", "SHAP Value", "Direction"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "SHAP explainer could not be initialised for this model type. "
            "The textual prediction recap above is still accurate."
        )

except Exception as e:
    st.error(f"**SHAP computation failed:** {e}")
    st.caption("This is non-critical — your prediction on the Home page is unaffected.")

st.markdown("""
<div class='app-footer'>
    SHAP values computed using the TreeExplainer · Values in Lakh ₹
</div>
""", unsafe_allow_html=True)
