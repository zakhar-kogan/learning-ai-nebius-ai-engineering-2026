# Nebius AI Engineering Assignment Repo

This repository contains a notebook-first workflow for the Week 2 AI product assignment in `w2-ai-product/`.

## Stack
- Python 3.11+
- `uv` for environment management
- `marimo` notebooks for the assignment workflow
- `litellm` for model calls
- `pydantic` for judge schemas
- `pandas` / `openpyxl` for CSV/XLSX handling
- `mlflow` with a local SQLite store
- `tenacity` for judge retry logic

## Project layout
- `w2-ai-product/notebooks/01_rubric.py` — rubric definition
- `w2-ai-product/notebooks/02_generate.py` — baseline generation
- `w2-ai-product/notebooks/03_human_eval.py` — human labeling + baseline analysis
- `w2-ai-product/notebooks/04_improve.py` — prompt/model improvement cycle
- `w2-ai-product/notebooks/05_judge.py` — all-at-once LLM judge
- `w2-ai-product/notebooks/06_analysis.py` — judge analysis + per-criterion judging
- `w2-ai-product/src/` — shared helpers, scoring, paths, runtime config
- `w2-ai-product/prompts/` — committed prompt versions
- `w2-ai-product/data/products.csv` — source dataset
- `w2-ai-product/outputs/` — generated artifacts

## Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Create your env file:
   ```bash
   cp w2-ai-product/.env.example w2-ai-product/.env
   ```
3. Add the API keys you plan to use.

## Dry-run the workflow first
Use the helper CLI to see the full execution order, expected outputs, and model-call volumes before running anything:

```bash
uv run python assignment_cli.py plan --dry-run
```

## Exact execution order
1. `01_rubric.py` — inspect and confirm the rubric
2. `02_generate.py` — generate baseline descriptions and create the workbook
3. `03_human_eval.py` — add programmatic latency/cost ratings, then manually score 10–15 rows
4. `04_improve.py` — try prompt/model improvements against the baseline
5. `05_judge.py` — run the judge sanity check, then the full judge pass
6. `06_analysis.py` — compare human vs judge behavior and finish the write-up

To open a notebook command without memorizing the file name:

```bash
uv run python assignment_cli.py open task-03
```

To actually launch it:

```bash
uv run python assignment_cli.py open task-03 --execute
```

## Human checkpoints
These steps are intentionally manual and should not be automated away:
- After Task 3 writes `w2-ai-product/outputs/assignment_01.xlsx`, manually label 10–15 rows for `fluency`, `grammar`, `tone`, `length`, and `grounding`.
- After Task 5 produces the 5-row sanity check, inspect the judge outputs before running the full judge pass.
- After Task 6, replace placeholder notebook text with your real analysis.

## Generated artifacts
All generated files should live under `w2-ai-product/outputs/`.

Expected paths:
- `w2-ai-product/outputs/assignment_01.xlsx`
- `w2-ai-product/outputs/experiments.db`
- `w2-ai-product/outputs/html/task_01.html` ... `task_06.html`

## Export HTML
Preview the export commands:

```bash
uv run python assignment_cli.py export-html --dry-run
```

Run them:

```bash
uv run python assignment_cli.py export-html --execute
```

## Notes on cost accounting
This repo treats Nebius-compatible pricing as the explicit normalization policy for equal models. If LiteLLM does not return a concrete response cost, the fallback calculation still uses Nebius token rates by design.
