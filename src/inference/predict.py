# src/inference/predict.py
"""Prediction module for the House Price Prediction platform.
Implements model loading, preprocessing, price prediction and confidence interval.
The functions are deliberately lightweight to be called from Streamlit UI.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# Paths – assume this file resides in src/inference
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")


def load_model() -> Any:
    """Load the trained XGBoost model.
    Returns the model object (XGBRegressor) or raises FileNotFoundError.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


def load_preprocessor() -> Any:
    """Load the preprocessing pipeline (e.g., StandardScaler, ColumnTransformer).
    Returns the preprocessor object.
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    preproc = joblib.load(PREPROCESSOR_PATH)
    return preproc


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training.
    Expected columns (may vary with dataset):
        - location (categorical)
        - size (numeric, total_sqft)
        - bhk (numeric, number of bedrooms)
        - bath (numeric, number of bathrooms)
        - other numeric columns present in training.
    Returns a DataFrame ready for the preprocessor.
    """
    df = df.copy()
    # Ensure numeric conversion for size – some rows contain ranges like "1195 - 1440"
    if "size" in df.columns:
        # Convert possible range strings to the average of the two numbers
        def parse_size(val):
            if isinstance(val, str) and "-" in val:
                try:
                    parts = val.split("-")
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return (low + high) / 2.0
                except Exception:
                    return np.nan
            try:
                return float(val)
            except Exception:
                return np.nan
        df["size"] = df["size"].apply(parse_size)
    # Derive total_rooms if not present
    if "bhk" in df.columns and "bath" in df.columns:
        df["total_rooms"] = df["bhk"] + df["bath"]
    # Example placeholder for price_per_sqft (not used for prediction)
    # One‑hot encode location if still categorical – the preprocessor will handle it.
    return df


def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray:
    """Convert raw input dictionary from UI into a preprocessed feature array.
    The keys must correspond to the training feature names (excluding the target).
    Returns a NumPy array ready for model.predict.
    """
    df = pd.DataFrame([input_data])
    df = _engineer_features(df)
    preprocessor = load_preprocessor()
    X_processed = preprocessor.transform(df)
    return X_processed


def predict_price(input_data: Dict[str, Any]) -> Dict[str, float]:
    """Predict house price and a simple confidence interval.
    The interval is approximated as ±5% of the predicted price –
    suitable for demonstration; replace with quantile regression for production.
    Returns a dictionary with keys: predicted_price, lower_bound, upper_bound.
    """
    model = load_model()
    X = preprocess_input(input_data)
    pred = model.predict(X)[0]
    # Simple +-5% interval
    lower = pred * 0.95
    upper = pred * 1.05
    return {
        "predicted_price": float(pred),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }

# Helper for command‑line testing
if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser(description="Predict house price from JSON input")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file with input fields")
    args = parser.parse_args()
    with open(args.json, "r") as f:
        data = json.load(f)
    result = predict_price(data)
    print(json.dumps(result, indent=2))
