# src/config.py
"""Central configuration – single source of truth for all project paths.

This file lives at  <repo>/src/config.py
    parents[0] = <repo>/src
    parents[1] = <repo>          ← PROJECT_ROOT

Every module that needs a project path should import from here:

    from src.config import PROJECT_ROOT, ARTIFACTS_DIR, DATA_DIR
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
