# src/utils/paths.py
"""Project path utilities.

Prefer importing paths directly from ``src.config``. This module
exists for backward compatibility with tests that call get_project_root().
"""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return the repository root directory.

    Identified by the co-presence of ``src/``, ``data/``, and
    ``requirements.txt`` — all three exist only at the project root.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (
            (parent / "src").exists()
            and (parent / "data").exists()
            and (parent / "requirements.txt").exists()
        ):
            return parent
    raise RuntimeError("Project root not found")
