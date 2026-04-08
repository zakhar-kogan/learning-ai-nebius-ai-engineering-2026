# pyright: reportMissingImports=false

from pathlib import Path

import pandas as pd

from src.judge_runtime import build_all_criteria_prompt
from src.task4_export import TASK4_EXPERIMENTS_SHEET, load_task4_experiments_table


def test_build_all_criteria_prompt_renders_current_rubric() -> None:
    prompt = build_all_criteria_prompt("## Evaluation Rubric\n\n{rubric_block}\n")

    assert "### Fluency — Natural, easy-to-read sentences" in prompt
    assert "### Grounding — Sticks to information provided (no hallucination)" in prompt
    assert "Note: count the words in the description carefully before deciding." in prompt
    assert "Note: for grounding, compare the description against the ORIGINAL PRODUCT DATA carefully. Only accept claims that appear in the data." in prompt
    assert "### Latency" not in prompt
    assert "### Cost" not in prompt


def test_load_task4_experiments_table_reads_experiments_sheet(tmp_path: Path) -> None:
    workbook_path = tmp_path / "task_04_experiments.xlsx"
    experiments_df = pd.DataFrame(
        [{"experiment_key": "baseline", "product_name": "Widget", "generated_description": "desc"}]
    )
    summary_df = pd.DataFrame([{"experiment": "Baseline", "rows": 1}])

    with pd.ExcelWriter(workbook_path) as writer:
        experiments_df.to_excel(writer, sheet_name=TASK4_EXPERIMENTS_SHEET, index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    loaded_df = load_task4_experiments_table(workbook_path)

    assert loaded_df.equals(experiments_df)
