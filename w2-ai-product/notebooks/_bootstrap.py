from pathlib import Path
import sys

NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent


def bootstrap_notebook() -> Path:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return PROJECT_ROOT
