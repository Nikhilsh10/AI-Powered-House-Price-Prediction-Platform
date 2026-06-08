---
title: House Price Prediction Platform
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: src/dashboard/app.py
pinned: false
---

# 🏠 AI-Powered House Price Prediction Platform

## Overview

An end‑to‑end machine learning platform that predicts residential property prices using advanced regression models and explainable AI techniques.

The project combines data preprocessing, feature engineering, automated model selection, SHAP‑based explainability, and an interactive Streamlit dashboard to deliver accurate and transparent house price predictions.

## Features

- 📈 **House price prediction** using XGBoost and LightGBM
- ⚙️ **Automated model evaluation and selection**
- 🧹 **Data cleaning and feature engineering pipeline**
- 📊 **Interactive analytics dashboard** with Plotly
- 🔍 **SHAP‑based model explainability**
- 🧠 **Feature contribution analysis**
- 🗂️ **Artifact management** for deployment
- ✅ **Regression tests** for deployment reliability
- ☁️ **Streamlit Cloud deployment** support

## Tech Stack

**Machine Learning**
- Scikit‑learn
- XGBoost
- LightGBM
- SHAP

**Data Processing**
- Pandas
- NumPy

**Visualization**
- Plotly
- Matplotlib

**Application Layer**
- Streamlit

**Testing**
- Pytest

## Project Architecture

```
Raw Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
XGBoost & LightGBM Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Artifact Storage
   ↓
Streamlit Dashboard
   ↓
SHAP Explainability
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Property feature input form, real‑time price prediction, confidence interval, production model inference |
| 📊 **Analytics** | Dataset exploration, KPI cards, interactive Plotly visualizations, model performance comparison, feature distribution analysis |
| 🔍 **Explainability** | SHAP Summary Plot, SHAP Waterfall Plot, Feature Contribution Table, local prediction interpretation |

## Model Evaluation Metrics
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### Model Progression & Preprocessing Impact

During development, we rigorously tracked the impact of various preprocessing techniques on our model's performance on the Bengaluru housing dataset. The dataset is notoriously noisy, featuring inconsistent area units and extreme price outliers that skew regression distributions.

Below is the evaluation progression for our XGBoost model evaluated on the **original (exponentiated) price scale**:

| Step | Modification | R² | MAE (Lakh ₹) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Baseline (Raw 13k Dataset) | `0.608` | 18.09 | 26.3% |
| **1** | Log-transform Target (`np.log1p`) | `0.585` | 17.68 | 23.2% |
| **2** | Drop `price_per_sqft` Outliers (3 Std Dev) | `0.577` | 18.20 | 23.9% |
| **3** | Group Sparse Locations (<10 to 'other') | `0.529` | 19.35 | 25.8% |

**Honest Commentary on Metrics:**
While one might expect R² to universally increase with data cleaning, calculating standard R² on inverse-transformed (expm1) predictions reveals a classic ML trade-off. By log-transforming the target (Step 1), the model optimized for the geometric mean rather than the arithmetic mean, actively ignoring massive high-end price outliers. Because the R² metric heavily penalizes large absolute squared errors on extreme outliers (which the log-model now ignores), the absolute R² score decreased. 

However, looking at the **MAPE**, the log-transformation successfully reduced the percentage error from **26.3% to 23.2%**. The model became significantly better at predicting the vast majority of standard homes, demonstrating the importance of selecting evaluation metrics (like MAE/MAPE) that align with the business use-case rather than blindly optimizing R² on noisy, heavy-tailed distributions. Steps 2 and 3 proved neutral-to-negative at their current thresholds, highlighting areas for future hyperparameter tuning and outlier threshold optimization.

## Testing

Automated regression tests verify:
- Project‑root detection
- Artifact availability
- Deployment readiness

Run tests:
```bash
python -m pytest -v tests/
```

## Project Structure
```
src/
├── data/
├── training/
├── inference/
├── explainability/
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_Home.py
│       ├── 2_Analytics.py
│       └── 3_Explainability.py

artifacts/
    ├── model.pkl
    ├── preprocessor.pkl
    ├── feature_columns.json
    ├── metadata.json
    └── metrics.json

data/processed/clean_data.csv

tests/
```

## Screenshots

**Home Page**

[Insert Screenshot]

**Analytics Dashboard**

[Insert Screenshot]

**Explainability Dashboard**

[Insert Screenshot]

**SHAP Feature Importance**

[Insert Screenshot]

## Deployment

**Streamlit Cloud Deployment**

Live Demo: [Add Deployment URL]

## Key Engineering Highlights
- Robust project‑root discovery for local and cloud deployments
- Automated artifact validation
- Explainable AI with SHAP
- Regression testing for deployment stability
- Modular project architecture
- Production‑ready Streamlit application

## Future Improvements
- Hyperparameter optimization with Optuna
- Model monitoring dashboard
- Batch prediction support
- Docker deployment
- CI/CD pipeline integration
- Drift detection and retraining workflow

---

*Resume bullet*:
> Built a production‑ready House Price Prediction Platform using XGBoost and LightGBM with automated model selection, SHAP‑based explainability, interactive Streamlit dashboards, regression testing, and deployment‑ready artifact management. Designed end‑to‑end ML workflows covering data preprocessing, model evaluation, prediction serving, and interpretable AI visualizations.
