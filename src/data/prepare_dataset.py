# src/data/prepare_dataset.py
"""Utility to download Bengaluru house price dataset, clean it, and save processed version.
It uses the data_processing module defined earlier.
"""
import os
import pandas as pd
import json
from ..data_processing import load_data, clean_data, feature_engineering

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

RAW_PATH = os.path.join(RAW_DIR, "Bengaluru_House_Data.csv")
PROCESSED_PATH = os.path.join(PROC_DIR, "clean_data.csv")
METADATA_PATH = os.path.join(PROC_DIR, "feature_metadata.json")

def main():
    # Load raw CSV (assumes it already exists)
    df_raw = pd.read_csv(RAW_PATH)
    df_clean = clean_data(df_raw)
    df_feat = feature_engineering(df_clean)
    # Save processed CSV
    df_feat.to_csv(PROCESSED_PATH, index=False)
    # Save simple metadata (column types)
    metadata = {
        "columns": {col: str(dtype) for col, dtype in df_feat.dtypes.items()},
        "row_count": len(df_feat),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Processed data saved to {PROCESSED_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()
