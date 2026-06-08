# src/dashboard/pages/2_Analytics.py
"""Analytics page — dataset KPIs, interactive charts, model metrics table.

All paths resolved via src.config. Handles both 'size' and 'total_sqft'
column names. Metrics NaN/None values are rendered gracefully.
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import ARTIFACTS_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH    = DATA_DIR / "processed" / "clean_data.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Guard: required files
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH}`")
    st.stop()
if not METRICS_PATH.exists():
    st.error(f"Metrics file not found: `{METRICS_PATH}`")
    st.stop()

# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


df      = load_data()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Detect size column
# ---------------------------------------------------------------------------
if "size" in df.columns:
    size_col = "size"
elif "total_sqft" in df.columns:
    size_col = "total_sqft"
else:
    st.error("Dataset missing both `size` and `total_sqft` columns.")
    st.stop()

# Ensure numeric
df[size_col] = pd.to_numeric(df[size_col], errors="coerce")
df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
df = df.dropna(subset=[size_col, "price"])

# ---------------------------------------------------------------------------
# Metric formatting helper (handles None / NaN gracefully)
# ---------------------------------------------------------------------------
def fmt(val, suffix="", precision=2):
    if val is None:
        return "—"
    try:
        f = float(val)
        if f != f:  # NaN check
            return "—"
        return f"{f:,.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<div class='page-enter'>
<h1 style='margin-bottom:0.2rem;'>
    Analytics <span style='background:linear-gradient(135deg,#5eead4,#fbbf24);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;'>Dashboard</span>
</h1>
<p style='color:#8a93a8; font-size:0.92rem; margin-top:0; margin-bottom:1.75rem;'>
    Dataset statistics, price distributions, and model performance
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1 — KPI Cards
# ---------------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:0.5rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:#4f5668;'>Key Metrics</div>
    <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
</div>
""", unsafe_allow_html=True)

avg_price   = df["price"].mean()
total_rows  = len(df)
avg_sqft    = df[size_col].mean()
n_locations = df["location"].nunique() if "location" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-teal'>
        <div class='kpi-label'>Avg Price (Lakh ₹)</div>
        <div class='kpi-value'>{avg_price:,.1f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-amber'>
        <div class='kpi-label'>Total Listings</div>
        <div class='kpi-value'>{total_rows:,}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-indigo'>
        <div class='kpi-label'>Avg Size (sqft)</div>
        <div class='kpi-value'>{avg_sqft:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='kpi-card kpi-accent-rose'>
        <div class='kpi-label'>Locations</div>
        <div class='kpi-value'>{n_locations:,}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Price vs Size scatter
# ---------------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:0.5rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:#4f5668;'>Price vs Area</div>
    <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
</div>
""", unsafe_allow_html=True)

hover_cols = [c for c in ["bhk", "bath", "location"] if c in df.columns]
# Use trendline only if dataset is large enough for LOWESS
use_trendline = len(df) >= 10

scatter_kwargs = dict(
    data_frame=df,
    x=size_col,
    y="price",
    title="Price (Lakh ₹) vs Area (sqft)",
    template="plotly_dark",
    labels={size_col: "Area (sqft)", "price": "Price (Lakh ₹)"},
)
if "location" in df.columns:
    scatter_kwargs["color"] = "location"
if hover_cols:
    scatter_kwargs["hover_data"] = hover_cols
if use_trendline:
    scatter_kwargs["trendline"] = "lowess"

fig_scatter = px.scatter(**scatter_kwargs)
fig_scatter.update_layout(
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=10)),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#8a93a8"),
    title_font=dict(size=13, color="#f0f2f7"),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Avg Price by Location (only if enough locations)
# ---------------------------------------------------------------------------
if "location" in df.columns and n_locations > 0:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.6rem; margin:1rem 0 1rem;'>
        <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                    text-transform:uppercase; color:#4f5668;'>Price by Location</div>
        <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
    </div>
    """, unsafe_allow_html=True)

    avg_by_loc = (
        df.groupby("location")["price"]
        .mean()
        .reset_index()
        .sort_values("price", ascending=False)
        .head(20)  # cap for readability
    )
    fig_bar = px.bar(
        avg_by_loc,
        x="location",
        y="price",
        color="price",
        color_continuous_scale=[[0, "#14b8a6"], [0.5, "#5eead4"], [1, "#fbbf24"]],
        title="Top Locations by Average Price",
        template="plotly_dark",
        labels={"location": "Location", "price": "Avg Price (Lakh ₹)"},
        text_auto=".1f",
    )
    fig_bar.update_layout(
        xaxis_tickangle=-40,
        margin=dict(l=10, r=10, t=40, b=80),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#8a93a8"),
        title_font=dict(size=13, color="#f0f2f7"),
    )
    fig_bar.update_traces(textfont_size=10)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Price distribution histogram
# ---------------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:1rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:#4f5668;'>Price Distribution</div>
    <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
</div>
""", unsafe_allow_html=True)

fig_hist = px.histogram(
    df,
    x="price",
    nbins=min(30, max(5, len(df) // 2)),
    title="Price Distribution (Lakh ₹)",
    template="plotly_dark",
    color_discrete_sequence=["#5eead4"],
    labels={"price": "Price (Lakh ₹)", "count": "Count"},
)
fig_hist.update_layout(
    bargap=0.05,
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#8a93a8"),
    title_font=dict(size=13, color="#f0f2f7"),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — Model Metrics Table
# ---------------------------------------------------------------------------
st.markdown("""
<div style='display:flex; align-items:center; gap:0.6rem; margin:1rem 0 1rem;'>
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:#4f5668;'>Model Performance</div>
    <div style='flex:1; height:1px; background:rgba(255,255,255,0.06);'></div>
</div>
""", unsafe_allow_html=True)

# Build table rows
rows = []
for model_name, vals in metrics.items():
    if not isinstance(vals, dict):
        continue
    rows.append({
        "Model":  model_name.title(),
        "R²":     fmt(vals.get("r2")),
        "MAE":    fmt(vals.get("mae"), " L"),
        "RMSE":   fmt(vals.get("rmse"), " L"),
        "MAPE":   fmt(vals.get("mape"), "%"),
    })

if rows:
    rows_html = ""
    for i, row in enumerate(rows):
        rows_html += f"""
        <tr>
            <td>{row['Model']}</td>
            <td>{row['R²']}</td>
            <td>{row['MAE']}</td>
            <td>{row['RMSE']}</td>
            <td>{row['MAPE']}</td>
        </tr>"""

    st.markdown(f"""
    <div style='background:linear-gradient(145deg,rgba(26,30,40,0.7),rgba(13,15,20,0.9));
                border:1px solid rgba(255,255,255,0.08); border-radius:14px;
                overflow:hidden; margin-bottom:1rem;'>
        <table class='model-table'>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>MAPE</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No model metrics available.")

st.markdown("""
<div class='app-footer'>
    Analytics powered by the Bengaluru Housing Dataset
</div>
""", unsafe_allow_html=True)
