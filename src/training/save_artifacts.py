# src/training/save_artifacts.py
"""Utility script to package the selected model, preprocessor, and metadata
into a single `artifacts/` directory for deployment.
The Streamlit app will only read from this directory, making the backend
agnostic to whether the chosen model is XGBoost or LightGBM.
"""
import os
import json
import shutil
from pathlib import Path
import joblib

# Import configuration paths from config.py
from src import config

# Use config variables
XGB_MODEL_PATH = config.XGB_MODEL_PATH
LGB_MODEL_PATH = config.LGB_MODEL_PATH
PREPROCESSOR_PATH = config.PREPROCESSOR_PATH
METRICS_PATH = config.METRICS_PATH
BEST_MODEL_PATH = config.BEST_MODEL_PATH
FEATURE_METADATA_PATH = config.FEATURE_METADATA_PATH

ARTIFACTS_DIR = config.ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ARTIFACT = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_ARTIFACT = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_ARTIFACT = ARTIFACTS_DIR / "feature_columns.json"
METRICS_ARTIFACT = ARTIFACTS_DIR / "metrics.json"
BEST_MODEL_ARTIFACT = ARTIFACTS_DIR / "best_model.json"
METADATA_ARTIFACT = ARTIFACTS_DIR / "metadata.json"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"
EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

def load_best_model_info():
    """Read `best_model.json` and return the identifier (e.g., "xgboost")."""
    with open(BEST_MODEL_PATH, "r") as f:
        info = json.load(f)
    return info.get("best_model")

def select_model_path(best_model_name: str) -> Path:
    """Map the model name to its stored file path."""
    if best_model_name == "xgboost":
        return Path(XGB_MODEL_PATH)
    elif best_model_name == "lightgbm":
        return Path(LGB_MODEL_PATH)
    else:
        raise ValueError(f"Unsupported model name: {best_model_name}")

def copy_model(src_path: Path, dst_path: Path):
    """Copy the selected model file to the unified artifact location."""
    shutil.copy(src_path, dst_path)
    print(f"Copied model from {src_path} to {dst_path}")

def copy_preprocessor(src_path: Path, dst_path: Path):
    shutil.copy(src_path, dst_path)
    print(f"Copied preprocessor from {src_path} to {dst_path}")

def save_feature_columns():
    """Create `feature_columns.json` containing numerical, categorical, and all input features.
    The information is extracted from `feature_metadata.json` generated during data prep.
    """
    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    # meta is expected to contain a dict of column: dtype strings
    numerical = [col for col, dtype in meta["columns"].items() if "float" in dtype or "int" in dtype]
    categorical = [col for col, dtype in meta["columns"].items() if "object" in dtype or "category" in dtype]
    # The model input features are all columns except the target (price)
    target_candidates = ["price", "Price", "SalePrice"]
    target = next((c for c in target_candidates if c in meta["columns"]), None)
    input_features = [col for col in meta["columns"].keys() if col != target]
    feature_info = {
        "numerical_features": numerical,
        "categorical_features": categorical,
        "model_input_features": input_features,
    }
    with open(FEATURE_COLUMNS_ARTIFACT, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"Saved feature columns metadata to {FEATURE_COLUMNS_ARTIFACT}")

def save_project_metadata(best_model_name: str):
    """Write a high‑level JSON description of the project and model.
    The training date is inferred from the file's modification time.
    """
    # Derive training date from the model file's modification timestamp
    model_path = select_model_path(best_model_name)
    training_timestamp = os.path.getmtime(model_path)
    training_date = Path(model_path).stat().st_mtime
    from datetime import datetime
    training_date_str = datetime.fromtimestamp(training_timestamp).strftime("%Y-%m-%d")
    # Load feature metadata to count features
    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    feature_count = len(meta["columns"]) - 1  # exclude target column
    metadata = {
        "project": "House Price Prediction Platform",
        "dataset": "Bengaluru Housing",
        "model_type": best_model_name,
        "feature_count": feature_count,
        "training_date": training_date_str,
        "version": "1.0.0",
    }
    with open(METADATA_ARTIFACT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved project metadata to {METADATA_ARTIFACT}")

def copy_evaluation_results():
    """Copy the evaluation JSON files into the artifacts folder."""
    shutil.copy(METRICS_PATH, METRICS_ARTIFACT)
    shutil.copy(BEST_MODEL_PATH, BEST_MODEL_ARTIFACT)
    print(f"Copied metrics and best model info to {ARTIFACTS_DIR}")

def main():
    # 1. Determine which model was selected
    best_model_name = load_best_model_info()
    selected_model_path = select_model_path(best_model_name)

    # 2. Copy model and preprocessor
    copy_model(selected_model_path, MODEL_ARTIFACT)
    copy_preprocessor(PREPROCESSOR_PATH, PREPROCESSOR_ARTIFACT)

    # 3. Export feature columns metadata
    save_feature_columns()

    # 4. Save project‑level metadata
    save_project_metadata(best_model_name)

    # 5. Copy evaluation artifacts
    copy_evaluation_results()

    # 6. Validation checks – fail fast if anything is missing
    assert MODEL_ARTIFACT.exists(), f"Model artifact missing: {MODEL_ARTIFACT}"
    assert PREPROCESSOR_ARTIFACT.exists(), f"Preprocessor artifact missing: {PREPROCESSOR_ARTIFACT}"
    assert FEATURE_COLUMNS_ARTIFACT.exists(), f"Feature columns artifact missing: {FEATURE_COLUMNS_ARTIFACT}"
    assert METADATA_ARTIFACT.exists(), f"Metadata artifact missing: {METADATA_ARTIFACT}"
    assert METRICS_ARTIFACT.exists(), f"Metrics artifact missing: {METRICS_ARTIFACT}"
    assert BEST_MODEL_ARTIFACT.exists(), f"Best model artifact missing: {BEST_MODEL_ARTIFACT}"
    print("All artifacts successfully created in", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
