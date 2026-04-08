# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent
APP_ROOT = WORKSPACE_ROOT / "w2-ai-product"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.paths import (  # noqa: E402
    ASSIGNMENT_XLSX_PATH,
    OUTPUTS_DIR,
    PRODUCTS_CSV_PATH,
    html_export_path,
)
from src.task4_export import rebuild_task4_workbook  # noqa: E402

TASK_NOTEBOOKS = {
    "task-01": APP_ROOT / "notebooks" / "01_rubric.py",
    "task-02": APP_ROOT / "notebooks" / "02_generate.py",
    "task-03": APP_ROOT / "notebooks" / "03_human_eval.py",
    "task-04": APP_ROOT / "notebooks" / "04_improve.py",
    "task-05": APP_ROOT / "notebooks" / "05_judge.py",
    "task-06": APP_ROOT / "notebooks" / "06_analysis.py",
}

HUMAN_CHECKPOINTS = [
    "After Task 3 writes outputs/assignment_01.xlsx, manually score 10–15 rows for fluency/grammar/tone/length/grounding.",
    "After Task 5 sanity-check runs, inspect the 5 sample judge outputs before launching the full judge run.",
    "After Task 6 runs, replace the placeholder analysis text with your final write-up.",
]


def dataset_size() -> int:
    with PRODUCTS_CSV_PATH.open(newline="") as handle:
        return max(sum(1 for _ in csv.DictReader(handle)), 0)


def marimo_edit_command(task_key: str) -> str:
    notebook = TASK_NOTEBOOKS[task_key]
    return f"marimo edit {notebook.relative_to(APP_ROOT)}"


def export_commands() -> list[str]:
    commands = []
    for index, notebook_path in enumerate(TASK_NOTEBOOKS.values(), start=1):
        output_path = html_export_path(index)
        commands.append(
            f"uv run marimo export html {notebook_path.relative_to(APP_ROOT)} -o {output_path.relative_to(APP_ROOT)}"
        )
    return commands


def print_plan() -> None:
    product_count = dataset_size()
    print("Assignment execution plan")
    print("=========================")
    print(f"Project root: {APP_ROOT}")
    print(f"Products in dataset: {product_count}")
    print(f"Workbook output: {ASSIGNMENT_XLSX_PATH.relative_to(APP_ROOT)}")
    print(f"All generated artifacts root: {OUTPUTS_DIR.relative_to(APP_ROOT)}")
    print()
    print("Execution order")
    for task_key, notebook_path in TASK_NOTEBOOKS.items():
        print(f"- {task_key}: {notebook_path.relative_to(APP_ROOT)}")
    print()
    print("Expected model-call volumes")
    print(f"- Baseline generation (Task 2): {product_count} calls")
    print("- Judge sanity check (Task 5): 5 calls")
    print(f"- Judge full run (Task 5): {product_count} calls")
    print(f"- Per-criterion judge (Task 6): {product_count * 5} calls")
    print()
    print("Human checkpoints")
    for checkpoint in HUMAN_CHECKPOINTS:
        print(f"- {checkpoint}")
    print()
    print("HTML export targets")
    for command in export_commands():
        print(f"- {command}")


def run_command(command: str) -> int:
    print(command)
    return subprocess.run(command, shell=True, cwd=APP_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Helpers for the Nebius AI Engineering assignment repo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="Print the execution plan and expected call volumes."
    )
    plan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for symmetry; plan is always read-only.",
    )

    open_parser = subparsers.add_parser(
        "open", help="Print or launch the marimo command for a task notebook."
    )
    open_parser.add_argument("task", choices=sorted(TASK_NOTEBOOKS.keys()))
    open_parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the marimo command instead of only printing it.",
    )

    export_parser = subparsers.add_parser(
        "export-html", help="Print or run HTML export commands for all notebooks."
    )
    export_parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the export commands instead of only printing them.",
    )
    export_parser.add_argument(
        "--dry-run", action="store_true", help="Only print the export commands."
    )

    subparsers.add_parser(
        "rebuild-task4",
        help="Rebuild the Task 4 workbook from persisted experiment CSVs without rerunning experiments.",
    )
    args = parser.parse_args()

    if args.command == "plan":
        print_plan()
        return 0

    if args.command == "open":
        command = marimo_edit_command(args.task)
        if args.execute:
            return run_command(command)
        print(command)
        return 0

    if args.command == "export-html":
        commands = export_commands()
        if not args.execute or args.dry_run:
            for command in commands:
                print(command)
            return 0
        for command in commands:
            exit_code = run_command(command)
            if exit_code != 0:
                return exit_code
        return 0

    if args.command == "rebuild-task4":
        workbook_path, _, summary_df = rebuild_task4_workbook()
        print(f"Task 4 workbook rebuilt: {workbook_path.relative_to(APP_ROOT)}")
        print(summary_df.to_string(index=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
