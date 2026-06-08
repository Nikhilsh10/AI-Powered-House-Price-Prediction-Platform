# src/dashboard/pages/2_Analytics.py
"""Analytics page – visualise KPIs, charts, and model metrics.

The page uses deterministic paths from src.config and displays:
* KPI cards for average price, number of listings, etc.
* Plotly visualisations of price vs size and distribution per location.
* Metrics table summarising model performance.
"""

import streamlit as st
import json
import plotly.express as px
import pandas as pd
from pathlib import Path

# Project root – three levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Load data & artifacts – defensive checks
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()
if not METRICS_PATH.exists():
    st.error(f"Metrics file not found: {METRICS_PATH}")
    st.stop()

# Load dataframe – cache for performance
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

df = load_data()

# Load metrics – cache as well
@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

metrics = load_metrics()

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("📊 Analytics Dashboard")
st.markdown("---")

# ---- KPI cards ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    avg_price = df["price"].mean()
    st.markdown(f"""
    <div class='glass-card'>
        <h3>Avg. Price (Lakh₹)</h3>
        <p style='font-size:1.8rem; font-weight:600'>{avg_price:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    total_listings = len(df)
    st.markdown(f"""
    <div class='glass-card'>
        <h3>Total Listings</h3>
        <p style='font-size:1.8rem; font-weight:600'>{total_listings:,}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    # Determine which size column exists
    if "size" in df.columns:
        size_col = "size"
    elif "total_sqft" in df.columns:
        size_col = "total_sqft"
    else:
        st.error("Dataset missing both 'size' and 'total_sqft' columns.")
        st.stop()
    avg_sqft = df[size_col].mean()
    st.markdown(f"""
    <div class='glass-card'>
        <h3>Avg. Size (sqft)</h3>
        <p style='font-size:1.8rem; font-weight:600'>{avg_sqft:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("\n")

# ---- Scatter: Price vs Size -----------------------------------------------
st.subheader("Price vs Size")
fig_price_size = px.scatter(
    df,
    x=size_col,
    y="price",
    color="location",
    trendline="lowess",
    hover_data=["bhk", "bath"],
    title="Price (Lakh₹) vs Square Feet",
    template="plotly_dark",
)
fig_price_size.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_price_size, use_container_width=True)

# ---- Bar: Avg Price per Location ------------------------------------------
st.subheader("Average Price by Location")
avg_by_loc = df.groupby("location")["price"].mean().reset_index().sort_values("price", ascending=False)
fig_loc = px.bar(
    avg_by_loc,
    x="location",
    y="price",
    color="price",
    color_continuous_scale=px.colors.sequential.Viridis,
    text_auto='.2f',
    title="Average Price per Location",
    template="plotly_dark",
)
fig_loc.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=30, b=60))
st.plotly_chart(fig_loc, use_container_width=True)

# ---- Model Metrics Table ---------------------------------------------------
st.subheader("Model Performance Metrics")
# Convert metrics dict to DataFrame for pretty display
rows = []
for model_name, vals in metrics.items():
    rows.append({
        "Model": model_name.title(),
        "MAE": f"{vals.get('mae', 'N/A'):.2f}",
        "RMSE": f"{vals.get('rmse', 'N/A'):.2f}",
        "MAPE": f"{vals.get('mape', 'N/A'):.2f}%",
    })
metrics_df = pd.DataFrame(rows)
styled_metrics = metrics_df.style.set_properties(
    **{
        "background": "var(--gradient-card)",
        "color": "var(--text-primary)"
    }
)
st.dataframe(styled_metrics)

st.markdown("---")
st.caption("All charts use a dark theme to match the app styling.")
