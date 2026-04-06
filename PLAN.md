# Second-pass refactor plan for `w2-ai-product`

## Goal
Make the assignment repo easier to run, easier to trust, and easier to evolve before the first real execution, without changing the underlying assignment methodology.

## Session decisions
- Keep the current notebook-first workflow; do not replace marimo with a pure CLI pipeline.
- Introduce a canonical generated-artifacts directory: `w2-ai-product/outputs/`.
- Keep Nebius-rate cost accounting as the explicit policy, even when another provider is used for compatible models.
- Commit prompt versions as the exact strings the model sees. For `generation_v2`, prefer a committed final prompt file over runtime-only prompt assembly.
- Separate execution failures from rubric values: verdict columns stay in the `good|ok|bad|""` domain, while status/error details move to dedicated columns.
- Add minimal automation only: tests, smoke checks, README, small CLI helpers, and CI. Do not automate manual-grading or prompt-review steps that require a human.

## Why these changes
Current evidence in the repo:
- Notebook bootstrap is repeated across all notebooks (`sys.path.insert`, dotenv loading, MLflow setup).
- `03_human_eval.py` duplicates executable latency/cost thresholds instead of reusing shared scoring helpers.
- `04_improve.py` creates `prompts/generation_v2.txt` at runtime instead of treating the prompt as a committed experiment artifact.
- `05_judge.py` and `06_analysis.py` write the string `"error"` into verdict columns, mixing transport failures with rubric labels.
- There is no root README, no tests, and no CI workflow.

## Scope
### In scope
1. Shared bootstrap/runtime helpers for notebooks
2. Shared paths/config for models, prompts, outputs, MLflow, dotenv
3. Shared executable scoring helpers for latency/cost/final score
4. Prompt version cleanup
5. Judge failure-status cleanup
6. Minimal CLI with dry-run
7. Tests and notebook smoke checks
8. Root README and generated-artifact documentation
9. Minimal CI for lint + tests

### Out of scope
- Running live model calls
- Completing the assignment write-ups or manual scoring
- Changing the rubric methodology or model choices beyond configuration cleanup
- Replacing marimo notebooks with a different UX

## Proposed design

### 1. Introduce a small shared runtime layer
Add a shared module under `w2-ai-product/src/` for project/runtime concerns, for example:
- `paths.py`
  - project root
  - prompts dir
  - data dir
  - outputs dir
  - xlsx/db/html output paths
- `runtime.py`
  - load project dotenv once
  - configure MLflow experiment
  - helper to ensure output directories exist
- `config.py`
  - provider/model defaults
  - prompt-version names and file paths
  - explicit Nebius-rate cost-policy comment

Because the notebooks currently live outside an importable package boundary, add a tiny notebook-local helper:
- `w2-ai-product/notebooks/_bootstrap.py`

That file will be the only place that handles notebook-relative import setup. The actual logic lives in `src/`, so the duplication disappears without needing a large package-layout rewrite.

### 2. Centralize executable scoring rules
Keep rubric definitions in `src/rubric.py`, but also move the machine-executable thresholds there (or into a dedicated scoring module) so the notebooks stop duplicating them.

Shared helpers to add:
- `rate_latency_ms(ms) -> RatingValue`
- `rate_cost_usd(cost) -> RatingValue`
- existing `compute_final_score()` remains the canonical pass/fail gate

This answers the current gap: the prose rubric is centralized, but the executable latency/cost thresholds are still duplicated in `03_human_eval.py`.

### 3. Make prompt versions explicit and reproducible
Recommended approach:
- Commit `prompts/generation_v2.txt` as the canonical full prompt string for experiment v2.
- If prompt composition becomes useful later, keep composition as an implementation detail, but the committed final prompt text remains the source of truth for the experiment version.

Reason: the experiment is defined by the exact prompt the model saw, not by the code that happened to assemble it.

### 4. Make outputs explicit
Create and use:
- `w2-ai-product/outputs/assignment_01.xlsx`
- `w2-ai-product/outputs/experiments.db`
- `w2-ai-product/outputs/html/task_01.html` ... `task_06.html`
- optionally `w2-ai-product/outputs/reports/` for dry-run summaries or manifests

All notebooks and the new CLI should resolve paths from the same helper.

### 5. Stop mixing verdicts and failures
Current problem:
- Judge verdict columns are logically categorical (`good|ok|bad`), but failure cases write `"error"` into the same columns.
- This pollutes downstream analysis and forces every consumer to treat domain values and transport failures as the same thing.

Plan:
- Keep verdict columns blank/NA on failures.
- Add dedicated status/error columns, e.g.:
  - `judge_status`
  - `judge_error`
  - `single_judge_status` or per-row failure summary
- Update metrics to compute error rates from status columns, not verdict columns.

### 6. Fix `run_async_batch()` contract
Current contract says “Returns results in order”, but `asyncio.as_completed()` returns completion order.

Plan:
- Either change implementation to preserve input order, or rename the helper to make completion-order explicit.
- Preferred: preserve input order and keep the current function name.

### 7. Add a minimal CLI with dry-run
Add a small script/entry point that helps run the assignment intentionally rather than by memory.

Suggested commands:
- `assignment-cli plan --dry-run`
  - print execution order
  - show data size
  - show expected output locations
  - show estimated API call volumes for baseline/judge/per-criterion judge
  - highlight human-intervention checkpoints
- `assignment-cli open task-01|task-02|...`
  - print or launch the correct marimo command
- `assignment-cli export-html`
  - print or run the export commands targeting `outputs/html/`

Dry-run must not make any API calls.

### 8. Add tests
Add `tests/` with at least:
- `test_rubric.py`
  - incomplete rows stay unscored
  - pass/fail threshold behavior
  - go/no-go behavior
  - latency/cost rating helpers
- `test_utils.py`
  - product formatting
  - judge-input formatting
  - cost extraction fallback policy
  - async batch ordering behavior
- `test_notebooks_smoke.py`
  - compile each notebook
  - import each notebook module without executing `app.run()`

### 9. Add minimal CI and linting
- Add GitHub Actions workflow running:
  - `uv sync --dev`
  - tests
  - lint
- Add minimal Ruff config only if needed to keep notebook linting practical.
- Keep lint scope pragmatic: prefer the shared code, tests, and notebook files actually touched by the refactor.

## Implementation phases

### Phase 1 — Runtime and path cleanup
Files likely touched:
- `pyproject.toml`
- `w2-ai-product/notebooks/01_rubric.py`
- `w2-ai-product/notebooks/02_generate.py`
- `w2-ai-product/notebooks/03_human_eval.py`
- `w2-ai-product/notebooks/04_improve.py`
- `w2-ai-product/notebooks/05_judge.py`
- `w2-ai-product/notebooks/06_analysis.py`
- new `w2-ai-product/notebooks/_bootstrap.py`
- new `w2-ai-product/src/{paths.py,runtime.py,config.py}`

Outcome:
- less repeated setup
- shared output paths
- shared MLflow/dotenv/config wiring

### Phase 2 — Scoring/prompt/judge correctness cleanup
Files likely touched:
- `w2-ai-product/src/rubric.py`
- `w2-ai-product/src/utils.py`
- `w2-ai-product/notebooks/03_human_eval.py`
- `w2-ai-product/notebooks/04_improve.py`
- `w2-ai-product/notebooks/05_judge.py`
- `w2-ai-product/notebooks/06_analysis.py`
- new `w2-ai-product/prompts/generation_v2.txt`

Outcome:
- no duplicated executable thresholds
- explicit Nebius-rate cost policy
- committed prompt v2
- clean separation of verdicts vs execution failures
- truthful async helper contract

### Phase 3 — Docs, CLI, tests, and CI
Files likely touched/added:
- new `README.md`
- new CLI script/module under `w2-ai-product/`
- new `tests/test_rubric.py`
- new `tests/test_utils.py`
- new `tests/test_notebooks_smoke.py`
- new `.github/workflows/ci.yml`
- maybe `pyproject.toml` test/lint config

Outcome:
- explicit run order
- dry-run command
- regression checks for core logic
- notebook smoke coverage
- automated CI verification

## Verification plan
No live API execution in this pass unless explicitly requested.

Verification will rely on:
- targeted unit tests
- notebook compile/import smoke tests
- CLI dry-run output
- lint on changed files
- reviewing final path changes so every notebook points at `outputs/`

## Risks and mitigations
- Risk: moving outputs may break notebook assumptions.
  - Mitigation: path helper used everywhere; smoke tests cover imports/compile; README and CLI use the same helpers.
- Risk: notebook bootstrap changes may break marimo execution.
  - Mitigation: keep `_bootstrap.py` tiny and notebook-local; avoid a large package-layout migration.
- Risk: status/error-column cleanup could break current analysis tables.
  - Mitigation: update analysis to read verdict columns plus status columns explicitly.
- Risk: linting notebooks may create noisy work.
  - Mitigation: keep Ruff config minimal and practical.

## Recommended execution order after this refactor
1. Implement shared runtime/path/bootstrap helpers
2. Move outputs to `outputs/` and update notebooks
3. Centralize executable scoring thresholds
4. Commit prompt v2 and remove runtime prompt generation
5. Clean up judge failure representation
6. Fix async helper ordering contract
7. Add CLI with dry-run
8. Add README
9. Add tests
10. Add CI/lint

## Acceptance criteria
- Notebook setup duplication is materially reduced
- All generated artifacts resolve under `w2-ai-product/outputs/`
- `generation_v2.txt` exists as a committed prompt version
- No judge verdict column stores `"error"`
- Shared tests cover rubric and utils behavior
- Smoke test proves notebooks compile/import cleanly
- README explains exact execution order and output locations
- CLI dry-run prints steps and expected call volumes without making API calls
- CI runs automatically on pushes/PRs and checks the new test/lint baseline
