# pyright: reportMissingImports=false

from pathlib import Path

import pandas as pd

from src.artifacts import load_csv_artifact, load_excel_artifact


def test_load_csv_artifact_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_csv_artifact(tmp_path / "missing.csv", required_columns=["a"]) is None


def test_load_csv_artifact_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "artifact.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(csv_path, index=False)

    assert load_csv_artifact(csv_path, required_columns=["a"]) is not None
    assert load_csv_artifact(csv_path, required_columns=["missing"]) is None


def test_load_excel_artifact_validates_required_columns(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "artifact.xlsx"
    pd.DataFrame([{"a": 1, "b": 2}]).to_excel(xlsx_path, index=False)

    loaded_df = load_excel_artifact(xlsx_path, required_columns=["a", "b"])
    assert loaded_df is not None
    assert list(loaded_df.columns) == ["a", "b"]
    assert load_excel_artifact(xlsx_path, required_columns=["missing"]) is None
