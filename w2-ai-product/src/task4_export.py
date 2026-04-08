from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.paths import (
    ASSIGNMENT_XLSX_PATH,
    OUTPUTS_REPORTS_DIR,
    PRODUCTS_CSV_PATH,
    TASK_04_EXPERIMENTS_XLSX_PATH,
)
from src.rubric import (
    CRITERION_COLS,
    JUDGED_COLS,
    compute_final_score,
    rate_cost_usd,
    rate_latency_ms,
)

TASK4_ARTIFACTS_DIR = OUTPUTS_REPORTS_DIR / "task_04_experiments"
TASK4_EXPERIMENTS_SHEET = "experiments"
TASK4_SUMMARY_SHEET = "summary"
TASK4_EXPERIMENT_KEY_COL = "experiment_key"
TASK4_EXPERIMENT_LABEL_COL = "experiment_label"
TASK4_BASELINE_KEY = "baseline"
TASK4_BASELINE_LABEL = "Baseline"
TASK4_EXPERIMENT_SPECS: dict[str, dict[str, str]] = {
    "exp1": {
        "label": "Exp 1",
        "generator": "Llama-3.1-8B",
        "model_id": "nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt_or_policy": "generation_v3_voice",
        "run_name": "exp1_llama8b_prompt_v3_voice",
    },
    "exp2": {
        "label": "Exp 2",
        "generator": "Qwen3-30B",
        "model_id": "nebius/Qwen/Qwen3-30B-A3B",
        "prompt_or_policy": "generation_v3_voice",
        "run_name": "exp2_qwen30b_prompt_v3_voice",
    },
    "exp3": {
        "label": "Exp 3",
        "generator": "Llama-3.1-8B",
        "model_id": "nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt_or_policy": "generation_v4_scaffold",
        "run_name": "exp3_llama8b_prompt_v4_scaffold",
    },
    "exp4": {
        "label": "Exp 4",
        "generator": "Llama-3.1-8B + selector",
        "model_id": "nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt_or_policy": "generation_v4_scaffold + generation_selector_v1",
        "run_name": "exp4_llama8b_prompt_v4_best_of_2",
    },
}
TASK4_EXPERIMENT_ORDER = [TASK4_BASELINE_KEY, *TASK4_EXPERIMENT_SPECS.keys()]
TASK4_MANUAL_COLS = ["comparison_sample", *JUDGED_COLS, "final_score"]


def _product_key(value: object) -> str:
    return " ".join(str(value).split())


def _normalize_manual_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in TASK4_MANUAL_COLS:
        if col not in normalized.columns:
            normalized[col] = ""
    return normalized


def _annotate_experiment_metadata(df: pd.DataFrame, *, experiment_key: str, label: str, generator: str, prompt_or_policy: str, run_id: str, table_path: str, is_baseline: bool) -> pd.DataFrame:
    annotated = df.copy()
    annotated[TASK4_EXPERIMENT_KEY_COL] = experiment_key
    annotated[TASK4_EXPERIMENT_LABEL_COL] = label
    annotated["generator"] = generator
    annotated["prompt_or_policy"] = prompt_or_policy
    annotated["run_id"] = run_id
    annotated["table_path"] = table_path
    annotated["is_baseline"] = is_baseline
    return annotated


def _legacy_workbook_to_long(existing_sheets: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    if "baseline" in existing_sheets:
        frames.append(
            _annotate_experiment_metadata(
                existing_sheets["baseline"],
                experiment_key=TASK4_BASELINE_KEY,
                label=TASK4_BASELINE_LABEL,
                generator="Llama-3.1-8B",
                prompt_or_policy="generation_v1",
                run_id="",
                table_path=str(ASSIGNMENT_XLSX_PATH),
                is_baseline=True,
            )
        )
    for experiment_key, spec in TASK4_EXPERIMENT_SPECS.items():
        if experiment_key not in existing_sheets:
            continue
        frames.append(
            _annotate_experiment_metadata(
                existing_sheets[experiment_key],
                experiment_key=experiment_key,
                label=str(spec["label"]),
                generator=str(spec["generator"]),
                prompt_or_policy=str(spec["prompt_or_policy"]),
                run_id="",
                table_path="",
                is_baseline=False,
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_existing_experiments_df(workbook_path: Path) -> pd.DataFrame | None:
    if not workbook_path.exists():
        return None
    existing_sheets = pd.read_excel(workbook_path, sheet_name=None)
    if TASK4_EXPERIMENTS_SHEET in existing_sheets:
        existing = existing_sheets[TASK4_EXPERIMENTS_SHEET].copy()
    else:
        existing = _legacy_workbook_to_long(existing_sheets)
        if existing is None:
            return None
    if TASK4_EXPERIMENT_KEY_COL not in existing.columns or "product_name" not in existing.columns:
        return None
    return _normalize_manual_columns(existing)


def _preserve_manual_scores(df_new: pd.DataFrame, existing_experiments_df: pd.DataFrame | None) -> pd.DataFrame:
    if existing_experiments_df is None:
        return _normalize_manual_columns(df_new)

    keep_cols = [
        col
        for col in [TASK4_EXPERIMENT_KEY_COL, "product_name", *JUDGED_COLS]
        if col in existing_experiments_df.columns
    ]
    if len(keep_cols) < 2:
        return _normalize_manual_columns(df_new)

    existing_scores = existing_experiments_df[keep_cols].copy()
    existing_scores["__product_key"] = existing_scores["product_name"].map(_product_key)
    existing_scores = existing_scores.drop_duplicates(
        subset=[TASK4_EXPERIMENT_KEY_COL, "__product_key"], keep="last"
    )

    merged = df_new.copy()
    merged["__product_key"] = merged["product_name"].map(_product_key)
    merged = merged.drop(columns=[col for col in [*JUDGED_COLS, "final_score"] if col in merged.columns]).merge(
        existing_scores.drop(columns=["product_name"]),
        on=[TASK4_EXPERIMENT_KEY_COL, "__product_key"],
        how="left",
    )
    return _normalize_manual_columns(merged.drop(columns=["__product_key"]))


def _resolve_comparison_names(
    baseline_df: pd.DataFrame, existing_experiments_df: pd.DataFrame | None
) -> set[str]:
    if "comparison_sample" in baseline_df.columns:
        comparison_mask = baseline_df["comparison_sample"].fillna("").astype(str).str.strip().eq("yes")
        comparison_names = {
            _product_key(name) for name in baseline_df.loc[comparison_mask, "product_name"].tolist()
        }
        if comparison_names:
            return comparison_names

    if existing_experiments_df is not None:
        existing_baseline = existing_experiments_df[
            existing_experiments_df[TASK4_EXPERIMENT_KEY_COL] == TASK4_BASELINE_KEY
        ].copy()
        if not existing_baseline.empty:
            comparison_mask = existing_baseline["comparison_sample"].fillna("").astype(str).str.strip().eq("yes")
            comparison_names = {
                _product_key(name)
                for name in existing_baseline.loc[comparison_mask, "product_name"].tolist()
            }
            if comparison_names:
                return comparison_names

    scored_mask = baseline_df["fluency"].fillna("").astype(str).str.strip() != ""
    comparison_names = {
        _product_key(name) for name in baseline_df.loc[scored_mask, "product_name"].tolist()
    }
    if comparison_names:
        return comparison_names
    return {_product_key(name) for name in baseline_df.head(15)["product_name"].tolist()}


def _mark_comparison_sample(df_experiments: pd.DataFrame, comparison_names: set[str]) -> pd.DataFrame:
    marked = df_experiments.copy()
    marked["comparison_sample"] = marked["product_name"].map(
        lambda value: "yes" if _product_key(value) in comparison_names else ""
    )
    return marked


def _rate_programmatic_value(value: object, rater: Callable[[float], str]) -> str:
    if pd.isna(value):
        return ""
    return rater(float(str(value)))


def _recompute_final_scores(df_experiments: pd.DataFrame) -> pd.DataFrame:
    scored = df_experiments.copy()
    scored["latency"] = scored["latency_ms"].map(lambda value: _rate_programmatic_value(value, rate_latency_ms))
    scored["cost"] = scored["cost_usd"].map(lambda value: _rate_programmatic_value(value, rate_cost_usd))
    scored["final_score"] = scored.apply(
        lambda row: compute_final_score({col: row.get(col, "") for col in CRITERION_COLS}),
        axis=1,
    )
    return scored


def _sort_experiments(df_experiments: pd.DataFrame) -> pd.DataFrame:
    ordered = df_experiments.copy()
    ordered["__experiment_order"] = ordered[TASK4_EXPERIMENT_KEY_COL].map(
        {key: index for index, key in enumerate(TASK4_EXPERIMENT_ORDER)}
    )
    ordered["__product_key"] = ordered["product_name"].map(_product_key)
    ordered = ordered.sort_values(["__experiment_order", "__product_key"], kind="stable")
    return ordered.drop(columns=["__experiment_order", "__product_key"]).reset_index(drop=True)


def _pass_rate_summary(df_experiment: pd.DataFrame) -> dict[str, object]:
    comparison_mask = df_experiment["comparison_sample"].fillna("").astype(str).str.strip().eq("yes")
    scored_mask = df_experiment["fluency"].fillna("").astype(str).str.strip() != ""
    scored = df_experiment[comparison_mask & scored_mask].copy()
    first_row = df_experiment.iloc[0]
    pass_count = int((scored["final_score"] == "pass").sum()) if len(scored) else 0
    fail_count = int((scored["final_score"] == "fail").sum()) if len(scored) else 0
    pass_rate = (pass_count / len(scored)) if len(scored) else None
    return {
        "experiment_key": first_row[TASK4_EXPERIMENT_KEY_COL],
        "experiment": first_row[TASK4_EXPERIMENT_LABEL_COL],
        "generator": first_row["generator"],
        "prompt_or_policy": first_row["prompt_or_policy"],
        "rows": len(df_experiment),
        "comparison_rows": int(comparison_mask.sum()),
        "n_scored": len(scored),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "mean_latency_ms": round(df_experiment["latency_ms"].mean(), 1),
        "total_cost_usd": round(df_experiment["cost_usd"].sum(), 8),
        "run_id": first_row["run_id"],
        "table_path": first_row["table_path"],
    }


def build_task4_experiments_table(
    *,
    baseline_df: pd.DataFrame,
    experiment_outputs: dict[str, dict[str, object]],
    products: pd.DataFrame,
    existing_experiments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    baseline_rows = _annotate_experiment_metadata(
        baseline_df.copy(),
        experiment_key=TASK4_BASELINE_KEY,
        label=TASK4_BASELINE_LABEL,
        generator="Llama-3.1-8B",
        prompt_or_policy="generation_v1",
        run_id="",
        table_path=str(ASSIGNMENT_XLSX_PATH),
        is_baseline=True,
    )

    experiment_frames = [baseline_rows]
    for experiment_key, meta in experiment_outputs.items():
        df_result = meta["df"]
        if not isinstance(df_result, pd.DataFrame):
            raise TypeError(f"Task 4 experiment '{experiment_key}' is missing a DataFrame payload.")
        experiment_sheet = products.merge(df_result, on="product_name", how="left")
        experiment_frames.append(
            _annotate_experiment_metadata(
                experiment_sheet,
                experiment_key=experiment_key,
                label=str(meta["label"]),
                generator=str(meta["generator"]),
                prompt_or_policy=str(meta["prompt_or_policy"]),
                run_id=str(meta.get("run_id", "")),
                table_path=str(meta["table_path"]),
                is_baseline=False,
            )
        )

    experiments_df = pd.concat(experiment_frames, ignore_index=True, sort=False)
    experiments_df = _preserve_manual_scores(experiments_df, existing_experiments_df)
    comparison_names = _resolve_comparison_names(baseline_df, existing_experiments_df)
    experiments_df = _mark_comparison_sample(experiments_df, comparison_names)
    experiments_df = _recompute_final_scores(experiments_df)
    return _sort_experiments(experiments_df)


def build_task4_summary(experiments_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for experiment_key in TASK4_EXPERIMENT_ORDER:
        df_experiment = experiments_df[
            experiments_df[TASK4_EXPERIMENT_KEY_COL] == experiment_key
        ].copy()
        if df_experiment.empty:
            continue
        summary_rows.append(_pass_rate_summary(df_experiment))
    return pd.DataFrame(summary_rows)


def build_task4_analysis_tables(experiments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_cols = list(JUDGED_COLS)
    rating_points = {"good": 1.0, "ok": 0.5, "bad": 0.0}
    criterion_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for experiment_key in TASK4_EXPERIMENT_ORDER:
        df_experiment = experiments_df[
            experiments_df[TASK4_EXPERIMENT_KEY_COL] == experiment_key
        ].copy()
        if df_experiment.empty:
            continue

        comparison_mask = df_experiment["comparison_sample"].fillna("").astype(str).str.strip().eq("yes")
        scored_mask = df_experiment["fluency"].fillna("").astype(str).str.strip() != ""
        sample_df = df_experiment[comparison_mask & scored_mask].copy()
        experiment_label = str(df_experiment[TASK4_EXPERIMENT_LABEL_COL].iloc[0])

        criterion_row: dict[str, object] = {"Experiment": experiment_label}
        for criterion in CRITERION_COLS:
            criterion_row[criterion.title()] = sample_df[criterion].map(rating_points).mean()
        criterion_row["Pass rate"] = sample_df["final_score"].eq("pass").mean()
        criterion_rows.append(criterion_row)

        failure_rows.append(
            {
                "Experiment": experiment_label,
                "Scored sample": f"{len(sample_df)}/{int(comparison_mask.sum())}",
                "Latency bad": int(sample_df["latency"].eq("bad").sum()),
                "Grounding bad": int(sample_df["grounding"].eq("bad").sum()),
                "Any manual bad": int(sample_df[manual_cols].eq("bad").any(axis=1).sum()),
                "Final fails": int(sample_df["final_score"].eq("fail").sum()),
            }
        )

    return pd.DataFrame(criterion_rows), pd.DataFrame(failure_rows)


def load_task4_experiment_outputs(
    artifacts_dir: Path = TASK4_ARTIFACTS_DIR,
) -> dict[str, dict[str, object]]:
    experiment_outputs: dict[str, dict[str, object]] = {}
    for key, spec in TASK4_EXPERIMENT_SPECS.items():
        table_path = artifacts_dir / f"{spec['run_name']}.csv"
        if not table_path.exists():
            raise FileNotFoundError(
                f"Missing persisted Task 4 output: {table_path}. Re-run that experiment cell first."
            )
        experiment_outputs[key] = {
            **spec,
            "df": pd.read_csv(table_path),
            "run_id": "",
            "table_path": str(table_path),
        }
    return experiment_outputs


def load_task4_experiments_table(
    workbook_path: Path = TASK_04_EXPERIMENTS_XLSX_PATH,
 ) -> pd.DataFrame:
    workbook_sheets = pd.read_excel(workbook_path, sheet_name=None)
    if TASK4_EXPERIMENTS_SHEET not in workbook_sheets:
        raise KeyError(
            f"Task 4 workbook {workbook_path} is missing sheet '{TASK4_EXPERIMENTS_SHEET}'."
        )
    return workbook_sheets[TASK4_EXPERIMENTS_SHEET].copy()


def export_task4_workbook(
    *,
    baseline_df: pd.DataFrame,
    experiment_outputs: dict[str, dict[str, object]],
    products: pd.DataFrame,
    workbook_path: Path = TASK_04_EXPERIMENTS_XLSX_PATH,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    existing_experiments_df = _read_existing_experiments_df(workbook_path)
    experiments_df = build_task4_experiments_table(
        baseline_df=baseline_df,
        experiment_outputs=experiment_outputs,
        products=products,
        existing_experiments_df=existing_experiments_df,
    )
    summary_df = build_task4_summary(experiments_df)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path) as writer:
        experiments_df.to_excel(writer, sheet_name=TASK4_EXPERIMENTS_SHEET, index=False)
        summary_df.to_excel(writer, sheet_name=TASK4_SUMMARY_SHEET, index=False)
    return workbook_path, experiments_df, summary_df


def rebuild_task4_workbook(
    *,
    assignment_path: Path = ASSIGNMENT_XLSX_PATH,
    products_path: Path = PRODUCTS_CSV_PATH,
    artifacts_dir: Path = TASK4_ARTIFACTS_DIR,
    workbook_path: Path = TASK_04_EXPERIMENTS_XLSX_PATH,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    baseline_df = pd.read_excel(assignment_path)
    products = pd.read_csv(products_path)
    experiment_outputs = load_task4_experiment_outputs(artifacts_dir)
    return export_task4_workbook(
        baseline_df=baseline_df,
        experiment_outputs=experiment_outputs,
        products=products,
        workbook_path=workbook_path,
    )
