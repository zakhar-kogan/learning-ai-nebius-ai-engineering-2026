# pyright: reportMissingImports=false

from pathlib import Path

import src.runtime as runtime


def test_setup_mlflow_restores_deleted_experiment(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "experiments.db"
    monkeypatch.setattr(runtime, "MLFLOW_DB_PATH", db_path)
    monkeypatch.setattr(runtime, "_LITELLM_AUTOLOG_ENABLED", True)
    monkeypatch.setattr(runtime, "ensure_output_dirs", lambda: db_path.parent.mkdir(parents=True, exist_ok=True))

    runtime.mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_id = runtime.mlflow.create_experiment("improvement_cycle")
    client = runtime.mlflow.MlflowClient()
    client.delete_experiment(experiment_id)
    assert client.get_experiment_by_name("improvement_cycle").lifecycle_stage == "deleted"

    result = runtime.setup_mlflow("improvement_cycle")

    restored = runtime.mlflow.MlflowClient().get_experiment_by_name("improvement_cycle")
    assert restored is not None
    assert restored.lifecycle_stage == "active"
    assert result == Path(db_path)
