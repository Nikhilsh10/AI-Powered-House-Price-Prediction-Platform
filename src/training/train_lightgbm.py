# src/training/train_lightgbm.py
"""Training script for LightGBM model.
Mirrors the XGBoost pipeline: loads cleaned data, preprocesses, trains, evaluates, and saves.
"""

import json
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

from src.training.config import (
    CLEAN_DATA_PATH,
    FEATURE_METADATA_PATH,
    LGB_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    LGB_PARAMS,
)

def load_data():
    return pd.read_csv(CLEAN_DATA_PATH)

def load_feature_metadata():
    with open(FEATURE_METADATA_PATH, "r") as f:
        return json.load(f)

def get_target_column(df: pd.DataFrame) -> str:
    for col in ["price", "Price", "SalePrice"]:
        if col in df.columns:
            return col
    raise KeyError("Target column not found")

def preprocess(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y.values, preprocessor

def train_and_save():
    df = load_data()
    target_col = get_target_column(df)
    X_processed, y, preprocessor = preprocess(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    # Save model and preprocessor (reuse same preprocessor path)
    joblib.dump(model, LGB_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mape = np.mean(np.abs((y_test - preds) / np.where(y_test == 0, 1e-8, y_test))) * 100
    metrics = {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}
    # Store LightGBM specific metrics
    model_metrics_path = os.path.join(os.path.dirname(LGB_MODEL_PATH), "lightgbm_metrics.json")
    with open(model_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("LightGBM training complete. Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    train_and_save()
