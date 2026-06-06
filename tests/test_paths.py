# tests/test_paths.py
"""Tests for the project‑root utility.
Ensures that the repository root contains the expected layout.
"""

from src.utils.paths import get_project_root


def test_project_root():
    root = get_project_root()
    assert (root / "src").exists(), "src directory missing"
    assert (root / "data").exists(), "data directory missing"
    assert (root / "requirements.txt").exists(), "requirements.txt missing"
