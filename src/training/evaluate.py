import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
"""Evaluation script to compare XGBoost and LightGBM models.
It loads the cleaned dataset, applies the saved preprocessor, splits data
using a fixed random seed, computes metrics for each model, selects the
best model based on a composite score, and writes the results to JSON
artifacts under the `models/` directory.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import config paths
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from training.config import (
    CLEAN_DATA_PATH,
    PREPROCESSOR_PATH,
    XGB_MODEL_PATH,
    LGB_MODEL_PATH,
    METRICS_PATH,
    BEST_MODEL_PATH,
)

def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    return df

def get_target_column(df: pd.DataFrame) -> str:
    for col in ["price", "Price", "SalePrice"]:
        if col in df.columns:
            return col
    raise KeyError("Target column not found in dataset")

def prepare_test_set(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # Load the preprocessor (trained on the full training data)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_processed = preprocessor.transform(X)
    # Train‑test split – same split for both models
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    return X_test, y_test

def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_test - preds) / np.where(y_test == 0, 1e-8, y_test))) * 100
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

def composite_score(metrics: dict, max_rmse: float, max_mae: float):
    # Normalized errors (lower is better)
    norm_rmse = metrics["rmse"] / max_rmse if max_rmse else 0
    norm_mae = metrics["mae"] / max_mae if max_mae else 0
    score = 0.5 * metrics["r2"] - 0.25 * norm_rmse - 0.25 * norm_mae
    return score

def main():
    df = load_data()
    target_col = get_target_column(df)
    X_test, y_test = prepare_test_set(df, target_col)

    # Evaluate both models
    xgb_metrics = evaluate_model(XGB_MODEL_PATH, X_test, y_test)
    lgb_metrics = evaluate_model(LGB_MODEL_PATH, X_test, y_test)

    # Determine max values for normalization
    max_rmse = max(xgb_metrics["rmse"], lgb_metrics["rmse"])
    max_mae = max(xgb_metrics["mae"], lgb_metrics["mae"])

    xgb_score = composite_score(xgb_metrics, max_rmse, max_mae)
    lgb_score = composite_score(lgb_metrics, max_rmse, max_mae)

    if xgb_score >= lgb_score:
        best_model_name = "xgboost"
        selection_metric = "r2"
        best_metrics = xgb_metrics
    else:
        best_model_name = "lightgbm"
        selection_metric = "r2"
        best_metrics = lgb_metrics

    # Prepare output structures
    all_metrics = {
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics,
    }
    # Save metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    # Save best_model.json
    best_info = {
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "metrics": best_metrics,
    }
    with open(BEST_MODEL_PATH, "w") as f:
        json.dump(best_info, f, indent=2)
    print("Evaluation complete. Results saved to:")
    print(f"  {METRICS_PATH}")
    print(f"  {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
