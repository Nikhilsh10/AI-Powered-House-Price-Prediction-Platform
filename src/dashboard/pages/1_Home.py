# src/dashboard/pages/1_Home.py
"""Home page – property price prediction.

Collect user inputs, call the inference module, and display the predicted price
with a confidence interval and a visual gauge indicating prediction confidence.
"""

import streamlit as st
import json
from src.utils.paths import repo_root

PROJECT_ROOT = repo_root()

# Defensive checks for required artifacts
artifacts_dir = PROJECT_ROOT / "artifacts"
model_path = artifacts_dir / "model.pkl"
preprocessor_path = artifacts_dir / "preprocessor.pkl"
features_path = artifacts_dir / "feature_columns.json"
metadata_path = artifacts_dir / "metadata.json"

missing = []
if not model_path.exists():
    missing.append(str(model_path))
if not preprocessor_path.exists():
    missing.append(str(preprocessor_path))
if not features_path.exists():
    missing.append(str(features_path))
if not metadata_path.exists():
    missing.append(str(metadata_path))
if missing:
    st.error(f"Missing artifact files: {', '.join(missing)}")
    st.stop()


from inference.predict import predict_price

# ---------------------------------------------------------------------------
# Helper – confidence gauge based on interval width ratio
# ---------------------------------------------------------------------------
def _confidence_gauge(lower: float, upper: float, pred: float) -> str:
    """Return an HTML snippet for a gauge.

    The gauge width is proportional to the relative interval size.
    """
    width_percent = min(100, max(0, ((upper - lower) / pred) * 100))
    # Determine confidence level
    if width_percent <= 5:
        color = "#4caf50"  # green – high confidence
        label = "High"
    elif width_percent <= 12:
        color = "#ffeb3b"  # amber – medium
        label = "Medium"
    else:
        color = "#f44336"  # red – low
        label = "Low"
    gauge_html = f"""
    <div class='confidence-gauge'>
        <div style='width:{width_percent}%; background:{color};'></div>
    </div>
    <div style='text-align:center; margin-top:4px; color:{color}; font-weight:600'>Confidence: {label}</div>
    """
    return gauge_html

# ---------------------------------------------------------------------------
def render():
    st.title("🏠 House Price Prediction")
    st.markdown("---")

    with st.form(key="price_form"):
        st.subheader("Property Details")
        location = st.text_input("Location", placeholder="e.g. Indiranagar, Bangalore")
        total_sqft = st.number_input("Total Sqft", min_value=100.0, step=10.0)
        bhk = st.selectbox("BHK (Bedrooms)", options=[i for i in range(1, 7)])
        bathrooms = st.selectbox("Bathrooms", options=[i for i in range(1, 7)])
        submit = st.form_submit_button("Predict Price")

    if submit:
        # Build input dictionary matching training features
        input_data = {
            "location": location,
            "size": total_sqft,
            "bhk": bhk,
            "bath": bathrooms,
        }
        try:
            result = predict_price(input_data)
            # Store for explainability page
            st.session_state["latest_prediction"] = input_data
            st.session_state["latest_result"] = result

            # Display cards using custom CSS class "glass-card"
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='glass-card'><h3>Estimated Price</h3><p style='font-size:2rem; font-weight:600'>{result['predicted_price']:.2f} Lakh₹</p></div>", unsafe_allow_html=True)
            with col2:
                lower = result["lower_bound"]
                upper = result["upper_bound"]
                st.markdown(f"<div class='glass-card'><h3>Confidence Range</h3><p>{lower:.2f}L – {upper:.2f}L</p></div>", unsafe_allow_html=True)

            # Confidence gauge
            gauge_html = _confidence_gauge(lower, upper, result["predicted_price"])
            st.markdown(gauge_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption("*Prediction confidence gauge is based on the width of the ±5% interval.*")
