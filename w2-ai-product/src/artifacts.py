from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def _has_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    required = tuple(required_columns)
    return all(column in df.columns for column in required)


def load_csv_artifact(
    path: Path,
    *,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if not _has_required_columns(df, required_columns):
        return None
    return df


def load_excel_artifact(
    path: Path,
    *,
    required_columns: Iterable[str] = (),
    sheet_name: str | int = 0,
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return None

    if not isinstance(df, pd.DataFrame):
        return None
    if not _has_required_columns(df, required_columns):
        return None
    return df
