# pyright: reportMissingImports=false

import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1] / "w2-ai-product"
NOTEBOOKS_DIR = APP_ROOT / "notebooks"
NOTEBOOK_PATHS = sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
def test_notebook_compiles(notebook_path: Path) -> None:
    py_compile.compile(str(notebook_path), doraise=True)


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
def test_notebook_imports(notebook_path: Path) -> None:
    sys.path.insert(0, str(NOTEBOOKS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(notebook_path.stem, notebook_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "app")
    finally:
        sys.path.remove(str(NOTEBOOKS_DIR))
