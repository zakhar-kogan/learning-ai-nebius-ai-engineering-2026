from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

from pathlib import Path

from dotenv import load_dotenv
import mlflow

from src.paths import MLFLOW_DB_PATH, PROJECT_ROOT, ensure_output_dirs

_DOTENV_LOADED = False
_LITELLM_AUTOLOG_ENABLED = False


def load_project_env() -> Path:
    """Load the project .env once and return its path."""
    global _DOTENV_LOADED

    env_path = PROJECT_ROOT / ".env"
    if not _DOTENV_LOADED:
        load_dotenv(env_path)
        _DOTENV_LOADED = True
    return env_path


def _restore_deleted_experiment(experiment_name: str) -> bool:
    """Restore a soft-deleted experiment so notebooks can keep using stable names."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        return True

    for experiment in client.search_experiments(view_type=ViewType.DELETED_ONLY):
        if experiment.name == experiment_name:
            client.restore_experiment(experiment.experiment_id)
            return True

    return False


def setup_mlflow(experiment_name: str) -> Path:
    """Configure MLflow to use the canonical local SQLite store."""
    global _LITELLM_AUTOLOG_ENABLED

    ensure_output_dirs()
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    _restore_deleted_experiment(experiment_name)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException as exc:
        if "deleted experiment" not in str(exc).lower() or not _restore_deleted_experiment(
            experiment_name
        ):
            raise
        mlflow.set_experiment(experiment_name)
    if not _LITELLM_AUTOLOG_ENABLED:
        mlflow.litellm.autolog()
        _LITELLM_AUTOLOG_ENABLED = True
    return MLFLOW_DB_PATH


def read_text(path: Path) -> str:
    return path.read_text()
