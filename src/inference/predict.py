# src/inference/predict.py
"""Prediction module for the House Price Prediction platform.

Implements model loading, preprocessing, price prediction and confidence
interval. The functions are deliberately lightweight to be called from
the Streamlit UI.
"""
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.config import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


def load_model() -> Any:
    """Load the trained model. Raises FileNotFoundError if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_preprocessor() -> Any:
    """Load the preprocessing pipeline. Raises FileNotFoundError if missing."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    return joblib.load(PREPROCESSOR_PATH)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training.

    Expected input columns: location, size, bhk, bath.
    Derives total_rooms and handles size ranges like "1195 - 1440".
    """
    df = df.copy()
    if "size" in df.columns:
        def parse_size(val):
            if isinstance(val, str) and "-" in val:
                try:
                    parts = val.split("-")
                    return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
                except Exception:
                    return np.nan
            try:
                return float(val)
            except Exception:
                return np.nan
        df["size"] = df["size"].apply(parse_size)
    if "bhk" in df.columns and "bath" in df.columns:
        df["total_rooms"] = df["bhk"] + df["bath"]
    return df


def validate_input(input_data: Dict[str, Any]) -> None:
    """Validate raw input data before inference."""
    if not input_data.get("location") or not str(input_data["location"]).strip():
        raise ValueError("Location cannot be empty.")
    try:
        size = float(input_data.get("size", 0))
    except (ValueError, TypeError):
        size = 0.0
    if size <= 0:
        raise ValueError("Total Area must be a positive number.")
    if int(input_data.get("bhk", 0)) <= 0:
        raise ValueError("BHK must be at least 1.")
    if int(input_data.get("bath", 0)) <= 0:
        raise ValueError("Bathrooms must be at least 1.")

def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray:
    """Convert raw input dict from UI into a preprocessed feature array."""
    df = pd.DataFrame([input_data])
    df = _engineer_features(df)
    preprocessor = load_preprocessor()
    return preprocessor.transform(df)


def predict_price(input_data: Dict[str, Any]) -> Dict[str, float]:
    """Predict house price and return a +/-5% confidence interval.

    Returns a dict with keys: predicted_price, lower_bound, upper_bound.
    Raises ValueError on invalid input.
    """
    validate_input(input_data)
    model = load_model()
    X = preprocess_input(input_data)
    pred_log = float(model.predict(X)[0])
    pred = float(np.expm1(pred_log))
    return {
        "predicted_price": pred,
        "lower_bound": pred * 0.95,
        "upper_bound": pred * 1.05,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict house price from JSON input")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
    args = parser.parse_args()
    with open(args.json, "r") as f:
        data = json.load(f)
    result = predict_price(data)
    print(json.dumps(result, indent=2))
