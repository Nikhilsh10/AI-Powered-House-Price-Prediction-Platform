# smoke_test.py
"""Simple integration smoke test for the Streamlit dashboard.

It does NOT launch the full Streamlit UI (which requires a browser), but it
verifies that the core backend components used by the pages work correctly:

1. Load model and preprocessor from `artifacts/`.
2. Run a prediction on a few sample property dictionaries.
3. Generate a global SHAP summary plot.
4. Generate a local SHAP waterfall plot for the last prediction.
5. Load analytics data and metrics JSON.

Any exception will cause the script to exit with a non‑zero status, which is
useful for CI pipelines.
"""

import os, sys
import json
import traceback

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
# Add the src folder so that "inference" and "explainability" packages are discoverable
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import core functions
from inference.predict import predict_price
from explainability.explain import global_explain, local_explain
import pandas as pd

def load_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)

def load_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_csv(data_path)

def main():
    # 1. Test prediction with several sample inputs
    samples = [
        {"location": "Indiranagar", "size": 1200, "bhk": 3, "bath": 2},
        {"location": "Whitefield", "size": 1500, "bhk": 4, "bath": 3},
        {"location": "Jayanagar", "size": 900, "bhk": 2, "bath": 1},
    ]
    for i, sample in enumerate(samples, 1):
        result = predict_price(sample)
        assert "predicted_price" in result and "lower_bound" in result and "upper_bound" in result
        assert result["lower_bound"] <= result["predicted_price"] <= result["upper_bound"]
        print(f"Sample {i} prediction ok: {result}")

    # 2. Global SHAP summary
    summary_path = global_explain()
    assert os.path.exists(summary_path), "Global SHAP summary image not created"
    print(f"Global SHAP summary generated at {summary_path}")

    # 3. Local SHAP for the last sample
    local_path = local_explain(samples[-1])
    assert os.path.exists(local_path), "Local SHAP waterfall image not created"
    print(f"Local SHAP waterfall generated at {local_path}")

    # 4. Analytics data load
    df = load_dataset()
    assert not df.empty, "Dataset is empty"
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # 5. Metrics load
    metrics = load_metrics()
    assert metrics, "Metrics JSON is empty"
    print(f"Metrics loaded: {json.dumps(metrics, indent=2)}")

    print("All smoke tests passed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Smoke test failed:")
        traceback.print_exc()
        sys.exit(1)
