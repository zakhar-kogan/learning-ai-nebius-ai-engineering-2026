# pyright: reportMissingImports=false

from pathlib import Path

import pandas as pd

from src.task4_export import (
    TASK4_EXPERIMENTS_SHEET,
    TASK4_SUMMARY_SHEET,
    TASK4_EXPERIMENT_SPECS,
    build_task4_analysis_tables,
    export_task4_workbook,
    load_task4_experiment_outputs,
)


def _product_row(product_name: str) -> dict[str, object]:
    return {
        "product_name": product_name,
        "Product_attribute_list": "Vacuum insulated",
        "material": "stainless steel",
        "warranty": "limited lifetime",
    }


def _result_row(product_name: str, latency_ms: float, cost_usd: float) -> dict[str, object]:
    return {
        "product_name": product_name,
        "generated_description": "Grounded description.",
        "latency_ms": latency_ms,
        "input_tokens": 100,
        "output_tokens": 60,
        "cost_usd": cost_usd,
    }


def test_load_task4_experiment_outputs_reads_persisted_csvs(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "task_04_experiments"
    artifacts_dir.mkdir()

    for spec in TASK4_EXPERIMENT_SPECS.values():
        pd.DataFrame([_result_row("Widget", 1800.0, 0.0001)]).to_csv(
            artifacts_dir / f"{spec['run_name']}.csv",
            index=False,
        )

    outputs = load_task4_experiment_outputs(artifacts_dir)

    assert sorted(outputs) == ["exp1", "exp2", "exp3", "exp4"]
    assert outputs["exp1"]["label"] == "Exp 1"
    assert list(outputs["exp1"]["df"].columns) == [
        "product_name",
        "generated_description",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    ]


def test_export_task4_workbook_migrates_legacy_scores_to_long_sheet(tmp_path: Path) -> None:
    workbook_path = tmp_path / "task_04_experiments.xlsx"

    baseline_df = pd.DataFrame(
        [
            {
                **_product_row("Hydro Flask 32 oz Wide Mouth"),
                **_result_row("Hydro Flask 32 oz Wide Mouth", 2100.0, 0.0001),
                "fluency": "good",
                "grammar": "good",
                "tone": "good",
                "length": "good",
                "grounding": "good",
            },
            {
                **_product_row("Other Item"),
                **_result_row("Other Item", 1800.0, 0.0001),
                "fluency": "",
                "grammar": "",
                "tone": "",
                "length": "",
                "grounding": "",
            },
        ]
    )

    products = pd.DataFrame(
        [
            _product_row("Hydro Flask 32 oz Wide Mouth"),
            _product_row("Other Item"),
        ]
    )

    legacy_baseline = pd.DataFrame(
        [
            {
                "product_name": "Hydro Flask 32 oz Wide Mouth",
                "comparison_sample": "yes",
                "fluency": "good",
                "grammar": "good",
                "tone": "good",
                "length": "good",
                "grounding": "good",
                "latency": "",
                "cost": "",
                "final_score": "",
            },
            {
                "product_name": "Other Item",
                "comparison_sample": "",
                "fluency": "",
                "grammar": "",
                "tone": "",
                "length": "",
                "grounding": "",
                "latency": "",
                "cost": "",
                "final_score": "",
            },
        ]
    )
    legacy_exp1 = pd.DataFrame(
        [
            {
                "product_name": "Hydro Flask 32 oz Wide Mouth",
                "comparison_sample": "yes",
                "fluency": "ok",
                "grammar": "good",
                "tone": "good",
                "length": "good",
                "grounding": "good",
                "latency": "",
                "cost": "",
                "final_score": "",
            },
            {
                "product_name": "Other Item",
                "comparison_sample": "",
                "fluency": "",
                "grammar": "",
                "tone": "",
                "length": "",
                "grounding": "",
                "latency": "",
                "cost": "",
                "final_score": "",
            },
        ]
    )
    with pd.ExcelWriter(workbook_path) as writer:
        legacy_baseline.to_excel(writer, sheet_name="baseline", index=False)
        legacy_exp1.to_excel(writer, sheet_name="exp1", index=False)

    experiment_outputs = {
        "exp1": {
            "label": "Exp 1",
            "generator": "Llama-3.1-8B",
            "prompt_or_policy": "generation_v3_voice",
            "run_id": "run-123",
            "table_path": "exp1.csv",
            "df": pd.DataFrame(
                [
                    _result_row(
                        "Hydro Flask 32 oz Wide Mouth",
                        2500.0,
                        0.0002,
                    ),
                    _result_row("Other Item", 1500.0, 0.0001),
                ]
            ),
        }
    }

    result_path, experiments_df, summary_df = export_task4_workbook(
        baseline_df=baseline_df,
        experiment_outputs=experiment_outputs,
        products=products,
        workbook_path=workbook_path,
    )

    workbook_sheets = pd.read_excel(workbook_path, sheet_name=None)
    assert result_path == workbook_path
    assert list(workbook_sheets) == [TASK4_EXPERIMENTS_SHEET, TASK4_SUMMARY_SHEET]

    exported_df = workbook_sheets[TASK4_EXPERIMENTS_SHEET]
    assert len(exported_df) == 4

    hydro_exp1 = exported_df.loc[
        (exported_df["experiment_key"] == "exp1")
        & (exported_df["product_name"] == products.loc[0, "product_name"])
    ].iloc[0]
    assert hydro_exp1["comparison_sample"] == "yes"
    assert hydro_exp1["fluency"] == "ok"
    assert hydro_exp1["grammar"] == "good"
    assert hydro_exp1["tone"] == "good"
    assert hydro_exp1["length"] == "good"
    assert hydro_exp1["grounding"] == "good"
    assert hydro_exp1["latency"] == "ok"
    assert hydro_exp1["cost"] == "good"
    assert hydro_exp1["final_score"] == "pass"

    exp1_summary = summary_df.loc[summary_df["experiment_key"] == "exp1"].iloc[0]
    assert exp1_summary["comparison_rows"] == 1
    assert exp1_summary["n_scored"] == 1
    assert exp1_summary["pass_count"] == 1
    assert exp1_summary["run_id"] == "run-123"

    criterion_score_df, failure_attribution_df = build_task4_analysis_tables(experiments_df)
    exp1_criteria = criterion_score_df.loc[criterion_score_df["Experiment"] == "Exp 1"].iloc[0]
    assert exp1_criteria["Fluency"] == 0.5
    assert exp1_criteria["Pass rate"] == 1.0

    exp1_failures = failure_attribution_df.loc[
        failure_attribution_df["Experiment"] == "Exp 1"
    ].iloc[0]
    assert exp1_failures["Scored sample"] == "1/1"
    assert exp1_failures["Final fails"] == 0
