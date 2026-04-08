import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Task 5 — Judge Model

    We build an automated judge: an LLM that grades descriptions using the same
    rubric defined in Task 1.

    ## Model Choice

    - **Judge model**: `google/gemma-2-9b-it` (model NOT used for generation)
    - If Gemma struggles with structured output, we fall back to `Qwen/Qwen3-30B-A3B-Instruct-2507`
    - **Temperature**: 0.0 — deterministic judging for reproducibility

    ## Why `explanation` Before `verdict` in the Schema

    LLMs generate tokens left-to-right. If `verdict` came first, the model would build all the answer around that score - a form of priming/anchoring bias.

    We're doing it the other way: first prompting for explanation, then summarizing that information to verdict.
    This is the LLM equivalent of chain-of-thought prompting embedded in the schema.

    ## Why use Pydantic at all?

    Pydantic gives us a strict contract for judge output: every criterion must be present,
    every verdict must be one of `good/ok/bad`. We also use tenacity to retry on fails.
    """)
    return


@app.cell
def _():
    from _bootstrap import bootstrap_notebook
    bootstrap_notebook()

    import pandas as pd
    from tqdm import tqdm

    from src.artifacts import load_csv_artifact
    from src.config import PROMPT_JUDGE_ALL, get_force_rerun, get_judge_config, prompt_path
    from src.judge_runtime import build_all_criteria_prompt, create_all_at_once_judge
    from src.paths import ASSIGNMENT_XLSX_PATH, TASK_05_JUDGE_SANITY_CSV_PATH
    from src.runtime import load_project_env, read_text, setup_mlflow
    from src.utils import format_judge_input

    load_project_env()
    setup_mlflow("judge_runs")
    return (
        ASSIGNMENT_XLSX_PATH,
        PROMPT_JUDGE_ALL,
        TASK_05_JUDGE_SANITY_CSV_PATH,
        build_all_criteria_prompt,
        create_all_at_once_judge,
        format_judge_input,
        get_force_rerun,
        get_judge_config,
        load_csv_artifact,
        pd,
        prompt_path,
        read_text,
        tqdm,
    )


@app.cell
def _(
    PROMPT_JUDGE_ALL,
    build_all_criteria_prompt,
    get_force_rerun,
    get_judge_config,
    prompt_path,
    read_text,
 ):
    FORCE_RERUN = get_force_rerun()
    judge_config = get_judge_config()
    JUDGE_MODEL = judge_config.model
    prompt_template = read_text(prompt_path(PROMPT_JUDGE_ALL))
    JUDGE_PROMPT = build_all_criteria_prompt(prompt_template)
    print(f"Judge model: {JUDGE_MODEL}")
    if FORCE_RERUN:
        print("FORCE_RERUN=1 — ignoring existing Task 5 sanity artifact.")
    print("\n--- Judge Prompt ---")
    print(JUDGE_PROMPT)
    return FORCE_RERUN, JUDGE_MODEL, JUDGE_PROMPT


@app.cell
def _(mo):
    mo.md(r"""
    ## Judge Function with Retry Logic
    """)
    return


@app.cell
def _(JUDGE_MODEL, JUDGE_PROMPT, create_all_at_once_judge, format_judge_input):
    run_judge = create_all_at_once_judge(
        model=JUDGE_MODEL,
        prompt=JUDGE_PROMPT,
        format_judge_input=format_judge_input,
    )
    return (run_judge,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Sanity Check — 5 Products

    Run the judge on 5 products and review manually before the full run.
    Check: does the judge apply the rubric correctly? Are explanations coherent?
    """)
    return


@app.cell
def _(
    ASSIGNMENT_XLSX_PATH,
    FORCE_RERUN,
    TASK_05_JUDGE_SANITY_CSV_PATH,
    load_csv_artifact,
    pd,
    run_judge,
    tqdm,
 ):
    required_columns = [
        "product_name",
        "description",
        "judge_status",
        "judge_error",
    ]
    existing_df = None
    if not FORCE_RERUN:
        existing_df = load_csv_artifact(
            TASK_05_JUDGE_SANITY_CSV_PATH,
            required_columns=required_columns,
        )

    if existing_df is not None:
        print(f"Source: loaded existing artifact {TASK_05_JUDGE_SANITY_CSV_PATH}")
        sanity_results = existing_df.to_dict("records")
    else:
        print(
            f"Source: artifact missing or invalid, running live sanity check -> {TASK_05_JUDGE_SANITY_CSV_PATH}"
        )
        df_main = pd.read_excel(ASSIGNMENT_XLSX_PATH)
        sanity_sample = df_main.head(5)

        sanity_results = []
        for _, row in tqdm(sanity_sample.iterrows(), total=5, desc="Sanity check"):
            try:
                result = run_judge(row.to_dict(), str(row["generated_description"]))
                ratings = result.to_ratings()
                sanity_results.append({
                    "product_name": row["product_name"],
                    "description": row["generated_description"],
                    "judge_status": "ok",
                    "judge_error": "",
                    **{f"{k}_verdict": v for k, v in ratings.items()},
                    **{f"{k}_explanation": getattr(result, k).explanation for k in ratings},
                })
            except Exception as e:
                print(f"Failed on {row['product_name']}: {e}")
                sanity_results.append({
                    "product_name": row["product_name"],
                    "description": row["generated_description"],
                    "judge_status": "error",
                    "judge_error": str(e),
                })

        pd.DataFrame(sanity_results).to_csv(TASK_05_JUDGE_SANITY_CSV_PATH, index=False)
        print(f"Saved sanity results to {TASK_05_JUDGE_SANITY_CSV_PATH}")

    return (sanity_results,)


@app.cell
def _(mo, sanity_results):
    # Display each sanity result with explanations
    _md = "## Sanity Check Results\n\n"
    for _r in sanity_results:
        if _r.get("judge_status") != "ok":
            _md += f"**{_r['product_name']}**: ERROR — {_r['judge_error']}\n\n"
        else:
            _md += f"### {_r['product_name']}\n"
            for _c in ["fluency", "grammar", "tone", "length", "grounding"]:
                _md += f"- **{_c}** [{_r[f'{_c}_verdict']}]: {_r[f'{_c}_explanation']}\n"
            _md += "\n"
    mo.md(_md)
    return


if __name__ == "__main__":
    app.run()
