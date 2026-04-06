import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Task 5 — Judge Model

        We build an automated judge: an LLM that grades descriptions using the same
        rubric defined in Task 1.

        ## Model Choice

        - **Judge model**: `google/gemma-2-9b-it` (model NOT used for generation)
        - If Gemma struggles with structured output, we fall back to `Qwen/Qwen3-30B-A3B-Instruct-2507`
        - **Temperature**: 0.0 — deterministic judging for reproducibility

        ## Why `explanation` Before `verdict` in the Schema

        The Pydantic schema places `explanation` (string) before `verdict` (enum).
        This ordering is intentional and consequential:

        When an LLM generates tokens left-to-right, the order of fields in a structured
        output schema determines the order of generation. If `verdict` came first, the model
        would commit to a rating ("good") and then generate an explanation that justifies
        that pre-committed answer — a form of **anchoring bias** and confirmation bias.

        By forcing `explanation` first, we require the model to reason through the
        evidence before reaching a conclusion. This mirrors how humans write good
        evaluations: "The description mentions X, Y, Z from the product data, but
        also claims W which is not in the attributes → verdict: bad."

        This is the LLM equivalent of chain-of-thought prompting embedded in the schema itself.
        """
    )
    return


@app.cell
def _():
    from _bootstrap import bootstrap_notebook
    bootstrap_notebook()

    import json
    import litellm
    import mlflow
    import pandas as pd
    from pydantic import ValidationError
    from tenacity import retry, retry_if_exception_type, stop_after_attempt
    from tqdm import tqdm

    from src.config import PROMPT_JUDGE_ALL, get_judge_config, prompt_path
    from src.paths import ASSIGNMENT_XLSX_PATH
    from src.runtime import load_project_env, read_text, setup_mlflow
    from src.schemas import JudgeOutput
    from src.rubric import compute_final_score
    from src.utils import format_judge_input

    load_project_env()
    mlflow_db_path = setup_mlflow("judge_runs")
    return (
        ASSIGNMENT_XLSX_PATH,
        PROMPT_JUDGE_ALL,
        JudgeOutput,
        ValidationError,
        compute_final_score,
        format_judge_input,
        get_judge_config,
        json,
        litellm,
        mlflow,
        mlflow_db_path,
        pd,
        prompt_path,
        read_text,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        tqdm,
    )


@app.cell
def _(PROMPT_JUDGE_ALL, get_judge_config, prompt_path, read_text):
    judge_config = get_judge_config()
    JUDGE_MODEL = judge_config.model
    JUDGE_PROMPT = read_text(prompt_path(PROMPT_JUDGE_ALL))
    print(f"Judge model: {JUDGE_MODEL}")
    print("\n--- Judge Prompt ---")
    print(JUDGE_PROMPT)
    return JUDGE_MODEL, JUDGE_PROMPT


@app.cell
def _(mo):
    mo.md(r"""## Judge Function with Retry Logic""")
    return


@app.cell
def _(
    JUDGE_MODEL,
    JUDGE_PROMPT,
    JudgeOutput,
    ValidationError,
    format_judge_input,
    json,
    litellm,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
):
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, KeyError)),
    )
    def run_judge(product: dict, description: str) -> JudgeOutput:
        """
        Call the judge model to evaluate a description against all 5 criteria at once.
        Uses response_format with json_schema (Pydantic) for structured output.
        Retries up to 3 times on parse failures.
        """
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",   "content": format_judge_input(product, description)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "JudgeOutput",
                    "schema": JudgeOutput.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=0.0,
            max_tokens=1024,
        )
        return JudgeOutput.model_validate_json(response.choices[0].message.content)
    return (run_judge,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sanity Check — 5 Products

        Run the judge on 5 products and review manually before the full run.
        Check: does the judge apply the rubric correctly? Are explanations coherent?
        """
    )
    return


@app.cell
def _(ASSIGNMENT_XLSX_PATH, pd, run_judge, tqdm):
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

    pd.DataFrame(sanity_results)
    return df_main, sanity_results, sanity_sample


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


@app.cell
def _(mo):
    mo.md(
        r"""
        *(After reviewing sanity results, adjust `prompts/judge_all.txt` if needed, then proceed.)*

        ## Full Run — All Products
        """
    )
    return


@app.cell
def _(ASSIGNMENT_XLSX_PATH, compute_final_score, df_main, mlflow, pd, run_judge, tqdm):
    criteria = ["fluency", "grammar", "tone", "length", "grounding"]
    judge_rows = []
    with mlflow.start_run(run_name="judge_all_criteria_gemma9b"):
        mlflow.log_params({
            "judge_model": "google/gemma-2-9b-it",
            "mode": "all_criteria_at_once",
            "temperature": 0.0,
        })

        for row_index, row in tqdm(df_main.iterrows(), total=len(df_main), desc="Judge (all)"):
            try:
                result = run_judge(row.to_dict(), str(row["generated_description"]))
                ratings = result.to_ratings()
                all_ratings = {
                    **ratings,
                    "latency": str(row.get("latency", "")).strip().lower(),
                    "cost": str(row.get("cost", "")).strip().lower(),
                }
                final = compute_final_score(all_ratings)
                judge_rows.append({
                    **{f"judge_{key}": value for key, value in ratings.items()},
                    **{f"judge_{key}_explanation": getattr(result, key).explanation for key in ratings},
                    "judge_final_score": final,
                    "judge_status": "ok",
                    "judge_error": "",
                })
            except Exception as e:
                print(f"Error on row {row_index}: {e}")
                judge_rows.append({
                    **{f"judge_{criterion}": "" for criterion in criteria},
                    **{f"judge_{criterion}_explanation": "" for criterion in criteria},
                    "judge_final_score": "",
                    "judge_status": "error",
                    "judge_error": str(e),
                })

        df_judge = pd.DataFrame(judge_rows)
        successful_rows = df_judge["judge_status"] == "ok"
        pass_rate = (
            (df_judge.loc[successful_rows, "judge_final_score"] == "pass").mean()
            if successful_rows.any()
            else 0.0
        )
        error_rate = (df_judge["judge_status"] == "error").mean()
        mlflow.log_metrics({
            "pass_rate": pass_rate,
            "error_rate": error_rate,
        })

    for _col in df_judge.columns:
        df_main[_col] = df_judge[_col].values

    df_main.to_excel(ASSIGNMENT_XLSX_PATH, index=False)
    print(
        f"Saved. Judge pass rate on successful rows: {pass_rate:.0%} | "
        f"Errors: {(df_judge['judge_status'] == 'error').sum()}"
    )
    return df_judge, judge_rows, ASSIGNMENT_XLSX_PATH


if __name__ == "__main__":
    app.run()
