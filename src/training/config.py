# src/training/config.py
"""Configuration constants for training scripts.
All paths are relative to the project root.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "clean_data.csv")
FEATURE_METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_metadata.json")

# Model output paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.json")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Training hyper‑parameters (tune as needed)
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}

LGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "n_jobs": -1,
    "random_state": 42,
}
