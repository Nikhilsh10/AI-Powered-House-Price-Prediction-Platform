# train module

import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from .data_processing import prepare_pipeline

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


def train(csv_path: str, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Train XGBoost model on the processed dataset and save model + scaler."""
    X_train, X_test, y_train, y_test, scaler = prepare_pipeline(csv_path)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    # Save model and scaler
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    # Simple evaluation
    preds = model.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Training completed. R2: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, scaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train house price model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    train(args.data)
