# tests/test_artifacts.py
"""Tests that required artefacts exist in the repository.
This guards against deployment failures caused by missing model or preprocessing files.
"""

from src.utils.paths import get_project_root


def test_required_artifacts_exist():
    root = get_project_root()
    required = [
        root / "artifacts" / "model.pkl",
        root / "artifacts" / "preprocessor.pkl",
        root / "artifacts" / "feature_columns.json",
        root / "artifacts" / "metadata.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    assert not missing, f"Missing artifacts: {missing}"
