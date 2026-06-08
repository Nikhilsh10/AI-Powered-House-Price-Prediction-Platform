# src/dashboard/pages/1_Home.py
"""Home page — property price prediction.

Collects user inputs, calls the inference module, and displays the
predicted price with a confidence interval and visual confidence gauge.
"""

import streamlit as st
from src.config import ARTIFACTS_DIR
from src.inference.predict import predict_price

# ---------------------------------------------------------------------------
# Artifact guard — fail fast with a clear message
# ---------------------------------------------------------------------------
_model_path = ARTIFACTS_DIR / "model.pkl"
_preproc_path = ARTIFACTS_DIR / "preprocessor.pkl"
_missing = [str(p) for p in (_model_path, _preproc_path) if not p.exists()]

if _missing:
    st.error("**Missing required artifact files:**\n\n" + "\n".join(f"- `{p}`" for p in _missing))
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _confidence_gauge(lower: float, upper: float, pred: float) -> str:
    """Return an HTML confidence gauge block.

    Design note: the fill represents confidence (INVERSE of interval width).
    A tight interval → wide fill (high confidence).
    A wide interval → narrow fill (low confidence).
    """
    if pred == 0:
        return ""
    interval_ratio = (upper - lower) / pred  # 0.10 = 10% spread
    # Map 10% spread → 100% confidence, 30%+ spread → 0%
    confidence_pct = max(0.0, min(100.0, (1 - interval_ratio / 0.30) * 100))

    if confidence_pct >= 70:
        color = "#5eead4"   # teal — high
        label = "High"
        tag_cls = "tag-teal"
    elif confidence_pct >= 40:
        color = "#fbbf24"   # amber — medium
        label = "Medium"
        tag_cls = "tag-amber"
    else:
        color = "#fb7185"   # rose — low
        label = "Low"
        tag_cls = "tag-rose"

    return f"""
    <div class='confidence-wrap'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;'>
            <span style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#4f5668; font-weight:600;'>
                Prediction Confidence
            </span>
            <span class='tag {tag_cls}'>{label}</span>
        </div>
        <div class='confidence-track'>
            <div class='confidence-fill' style='width:{confidence_pct:.1f}%; background:{color};'></div>
        </div>
        <div style='font-size:0.75rem; color:#4f5668; margin-top:0.3rem; font-family:Space Mono,monospace;'>
            Range: {lower:.2f}L – {upper:.2f}L
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------
def render():
    # Page title
    st.markdown("""
    <div class='page-enter'>
    <h1 style='margin-bottom:0.2rem;'>
        House Price <span style='background:linear-gradient(135deg,#5eead4,#fbbf24);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;'>Prediction</span>
    </h1>
    <p style='color:#8a93a8; font-size:0.92rem; margin-top:0; margin-bottom:1.75rem;'>
        Enter property details below to get an instant price estimate
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Input Form --------------------------------------------------------
    with st.form(key="price_form", clear_on_submit=False):
        st.markdown("""
        <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1.25rem;'>
            <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                        text-transform:uppercase; color:#4f5668;'>Property Details</div>
            <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            location = st.text_input(
                "Location",
                placeholder="e.g. Indiranagar, Koramangala",
                help="Neighbourhood or locality name in Bengaluru",
            )
            total_sqft = st.number_input(
                "Total Area (sqft)",
                min_value=100.0,
                max_value=50000.0,
                value=1200.0,
                step=50.0,
                help="Total built-up area in square feet",
            )
        with col_b:
            bhk = st.selectbox(
                "BHK",
                options=list(range(1, 7)),
                index=2,
                help="Number of bedrooms",
            )
            bathrooms = st.selectbox(
                "Bathrooms",
                options=list(range(1, 7)),
                index=1,
                help="Number of bathrooms",
            )

        submitted = st.form_submit_button("Estimate Price →", use_container_width=True)

    # ---- Result Display ----------------------------------------------------
    if submitted:
        if not location.strip():
            st.warning("Please enter a location to continue.")
            return

        input_data = {
            "location": location.strip(),
            "size": total_sqft,
            "bhk": bhk,
            "bath": bathrooms,
        }

        with st.spinner("Running inference…"):
            try:
                result = predict_price(input_data)
            except ValueError as e:
                st.error(f"**Invalid Input:** {e}")
                return

        # Persist for Explainability page
        st.session_state["latest_prediction"] = input_data
        st.session_state["latest_result"] = result

        pred  = result["predicted_price"]
        lower = result["lower_bound"]
        upper = result["upper_bound"]

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        # Hero result card
        st.markdown(f"""
        <div class='result-hero'>
            <div class='result-label'>Estimated Price</div>
            <div class='result-price'>₹{pred:,.2f}L</div>
            {_confidence_gauge(lower, upper, pred)}
        </div>
        """, unsafe_allow_html=True)

        # Breakdown columns
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='kpi-card kpi-accent-teal'>
                <div class='kpi-label'>Predicted (Lakh ₹)</div>
                <div class='kpi-value'>{pred:,.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='kpi-card kpi-accent-indigo'>
                <div class='kpi-label'>Lower Bound</div>
                <div class='kpi-value'>{lower:,.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='kpi-card kpi-accent-amber'>
                <div class='kpi-label'>Upper Bound</div>
                <div class='kpi-value'>{upper:,.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.caption(
            "Confidence interval is approximated as ±5% of the predicted price. "
            "Navigate to **Explainability** to see SHAP feature contributions."
        )

    else:
        # Empty state — visual affordance
        st.markdown("""
        <div style='text-align:center; padding:3rem 1rem; color:#4f5668;'>
            <div style='font-size:2.5rem; margin-bottom:0.75rem; opacity:0.4;'>🏠</div>
            <div style='font-size:0.88rem; font-weight:500; letter-spacing:0.04em;'>
                Fill in the form and click <strong style='color:#8a93a8;'>Estimate Price →</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='app-footer'>
        Powered by XGBoost &amp; LightGBM · Bengaluru Housing Dataset
    </div>
    """, unsafe_allow_html=True)


render()
