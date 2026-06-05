# src/explainability/explain.py
"""
Explainability utilities using SHAP.

The dashboard expects two main entry points:

1. ``global_explain()`` – produces a SHAP summary plot for the entire test set.
2. ``local_explain(sample_dict)`` – produces a SHAP waterfall plot for a single
   prediction (the same visual used by many XAI demos).

Both functions return the *file path* of a PNG image that can be embedded in the
Streamlit UI (the UI will simply ``st.image`` the returned path).

The module is deliberately lightweight – it does **not** perform any training.
It reads the artefacts produced by ``src/training/save_artifacts.py``:

- ``artifacts/model.pkl`` – the selected production model (XGBoost or LightGBM).
- ``artifacts/preprocessor.pkl`` – the scikit‑learn preprocessing pipeline.
- ``artifacts/feature_columns.json`` – ordered list of feature names expected by
  the model.
- ``data/processed/clean_data.csv`` – the cleaned dataset that was used for
  training; the same file is used to generate a test split for global SHAP.

The implementation follows best‑practice SHAP usage:

* For tree‑based models we use ``shap.TreeExplainer``.
* The preprocessor is applied **before** the explainer sees the data.
* Global explanations are computed on a *representative* test subset (default
  2,000 rows) to keep runtime reasonable.
* Local explanations are computed on a single instance supplied as a ``dict``
  mapping feature names to values.

All heavy imports (shap, matplotlib) are performed lazily inside the functions so
that importing this module does not incur a long start‑up cost for the Streamlit
app when the user navigates to a different page.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helper – resolve absolute paths relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_data.csv")

# ---------------------------------------------------------------------------
# Load artefacts – these are tiny and can be loaded at import time.
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'. Run save_artifacts.py first."
    )
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessor not found at '{PREPROCESSOR_PATH}'. Run save_artifacts.py first."
    )
if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(
        f"Feature columns descriptor missing at '{FEATURE_COLUMNS_PATH}'."
    )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)

# ---------------------------------------------------------------------------
# Internal SHAP utilities – imported lazily to keep startup fast
# ---------------------------------------------------------------------------
def _get_shap_explainer():
    """Create a SHAP TreeExplainer for the loaded model.

    Tree‑based models (XGBoost, LightGBM) are fully supported. If the model does
    not expose a ``tree_`` attribute, we fall back to the generic ``KernelExplainer``
    which works for any sklearn‑compatible estimator.
    """
    import shap

    # XGBoost and LightGBM both expose a ``feature_importances_`` attribute –
    # the presence of ``model.get_booster`` is a reliable XGBoost check.
    if hasattr(model, "get_booster") or hasattr(model, "feature_name_"):
        explainer = shap.TreeExplainer(model)
    else:
        # For safety we provide a background set using a small random sample.
        background = _load_test_set(sample_size=200)
        explainer = shap.KernelExplainer(model.predict, background)
    return explainer

def _load_test_set(sample_size: int = 2000) -> np.ndarray:
    """Load the cleaned dataset and return a pre‑processed NumPy matrix.

    Parameters
    ----------
    sample_size: int
        Number of rows to sample for global explanations. If the dataset is
        smaller than this value the whole set is used.
    """
    df = pd.read_csv(CLEAN_DATA_PATH)
    # Drop the target column – the helper ``get_target_column`` mirrors the one in
    # ``src/training/evaluate.py`` but we keep it lightweight.
    target_col = next(
        (c for c in ["price", "Price", "SalePrice"] if c in df.columns), None
    )
    if target_col is None:
        raise KeyError("Target column not found in clean_data.csv")
    X = df.drop(columns=[target_col])
    X = X[FEATURE_COLUMNS]  # enforce column order expected by the model
    X_processed = preprocessor.transform(X)
    # Randomly sample to keep SHAP runtime manageable.
    if len(X_processed) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_processed), sample_size, replace=False)
        X_processed = X_processed[idx]
    return X_processed

# ---------------------------------------------------------------------------
# Public API – global and local explanations
# ---------------------------------------------------------------------------
def global_explain(output_dir: str = None, max_rows: int = 2000) -> str:
    """Generate a SHAP summary plot for the entire test set.

    Parameters
    ----------
    output_dir: str, optional
        Directory where the PNG image will be saved. If omitted a temporary file
        inside the ``artifacts`` folder is created.
    max_rows: int
        Upper bound on the number of rows used for the explanation. Larger
        values produce richer plots but increase runtime.

    Returns
    -------
    str
        Absolute path to the created PNG image.
    """
    import shap
    import matplotlib.pyplot as plt
    import uuid

    X_test = _load_test_set(sample_size=max_rows)
    explainer = _get_shap_explainer()
    shap_values = explainer.shap_values(X_test)

    # Ensure output directory exists.
    if output_dir is None:
        output_dir = os.path.join(ARTIFACTS_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"global_shap_{uuid.uuid4().hex[:8]}.png")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURE_COLUMNS,
        plot_type="dot",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename

def local_explain(instance: dict, output_dir: str = None) -> str:
    """Generate a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    instance: dict
        Mapping of feature name → value for the observation of interest. The keys
        must match the column names used during training.
    output_dir: str, optional
        Directory where the PNG image will be saved. Defaults to the same folder
        used for the global plot.

    Returns
    -------
    str
        Absolute path to the PNG image containing the waterfall plot.
    """
    import shap
    import matplotlib.pyplot as plt
    import uuid

    # Convert dict to DataFrame with a single row and enforce column ordering.
    df = pd.DataFrame([instance])
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features for local explanation: {missing}")
    df = df[FEATURE_COLUMNS]
    X_processed = preprocessor.transform(df)

    explainer = _get_shap_explainer()
    shap_values = explainer.shap_values(X_processed)

    if output_dir is None:
        output_dir = os.path.join(ARTIFACTS_DIR, "explainability")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"local_shap_{uuid.uuid4().hex[:8]}.png")

    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(values=shap_values[0], base_value=explainer.expected_value, data=X_processed[0]),
        feature_names=FEATURE_COLUMNS,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename

# ---------------------------------------------------------------------------
# Convenience wrapper used by the Streamlit dashboard
# ---------------------------------------------------------------------------
def explain(sample: dict = None) -> str:
    """Unified entry point.

    - If ``sample`` is ``None`` a global summary plot is generated.
    - Otherwise a local waterfall plot for the supplied sample is produced.
    The returned path can be fed directly to ``st.image``.
    """
    if sample is None:
        return global_explain()
    return local_explain(sample)

# ---------------------------------------------------------------------------
# When executed as a script we provide a quick demo for developers.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating global SHAP summary plot …")
    print(global_explain())
    # Example local explanation – the user should replace with a real instance.
    # demo_instance = {col: 0 for col in FEATURE_COLUMNS}
    # print("Generating local SHAP waterfall plot …")
    # print(local_explain(demo_instance))
