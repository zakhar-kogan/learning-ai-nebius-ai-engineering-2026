from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_HTML_DIR = OUTPUTS_DIR / "html"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"

PRODUCTS_CSV_PATH = DATA_DIR / "products.csv"
ASSIGNMENT_XLSX_PATH = OUTPUTS_DIR / "assignment_01.xlsx"
MLFLOW_DB_PATH = OUTPUTS_DIR / "experiments.db"


def ensure_output_dirs() -> None:
    """Create the canonical generated-artifacts directories if they do not exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_HTML_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def html_export_path(task_number: int) -> Path:
    return OUTPUTS_HTML_DIR / f"task_{task_number:02d}.html"
