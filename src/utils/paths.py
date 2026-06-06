from pathlib import Path

def get_project_root() -> Path:
    """Return the repository root directory.
    It is identified by the presence of the ``src`` folder, ``data`` folder,
    and the ``requirements.txt`` file – all of which exist only at the project
    root.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "src").exists() and (parent / "data").exists() and (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError("Project root not found")
