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
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), ".."))

    import json
    import litellm
    import pandas as pd
    from tqdm import tqdm
    from tenacity import retry, stop_after_attempt, retry_if_exception_type
    from pydantic import ValidationError
    from dotenv import load_dotenv
    import mlflow

    from src.schemas import JudgeOutput, CriterionJudgment
    from src.rubric import compute_final_score, CRITERION_COLS
    from src.utils import format_judge_input, get_model_string

    load_dotenv(os.path.join(os.getcwd(), "..", ".env"))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), '..', 'experiments.db')}")
    mlflow.set_experiment("judge_runs")
    mlflow.litellm.autolog()
    return (
        CriterionJudgment,
        CRITERION_COLS,
        JudgeOutput,
        ValidationError,
        compute_final_score,
        format_judge_input,
        get_model_string,
        json,
        litellm,
        load_dotenv,
        mlflow,
        os,
        pd,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        sys,
        tqdm,
    )


@app.cell
def _(get_model_string, os):
    JUDGE_MODEL = get_model_string("nebius", "google/gemma-2-9b-it")
    JUDGE_PROMPT = open(os.path.join(os.getcwd(), "..", "prompts", "judge_all.txt")).read()
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
def _(os, pd, run_judge, tqdm):
    df_main = pd.read_excel(os.path.join(os.getcwd(), "..", "assignment_01.xlsx"))
    sanity_sample = df_main.head(5)

    sanity_results = []
    for _, row in tqdm(sanity_sample.iterrows(), total=5, desc="Sanity check"):
        try:
            result = run_judge(row.to_dict(), str(row["generated_description"]))
            ratings = result.to_ratings()
            sanity_results.append({
                "product_name": row["product_name"],
                "description":  row["generated_description"],
                **{f"{k}_verdict":     v               for k, v in ratings.items()},
                **{f"{k}_explanation": getattr(result, k).explanation for k in ratings},
            })
        except Exception as e:
            print(f"Failed on {row['product_name']}: {e}")
            sanity_results.append({"product_name": row["product_name"], "error": str(e)})

    pd.DataFrame(sanity_results)
    return df_main, sanity_results, sanity_sample


@app.cell
def _(mo, sanity_results):
    # Display each sanity result with explanations
    _md = "## Sanity Check Results\n\n"
    for _r in sanity_results:
        if "error" in _r:
            _md += f"**{_r['product_name']}**: ERROR — {_r['error']}\n\n"
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

        ## Full Run — All 51 Products
        """
    )
    return


@app.cell
def _(CRITERION_COLS, compute_final_score, df_main, mlflow, os, pd, run_judge, tqdm):
    judge_rows = []
    with mlflow.start_run(run_name="judge_all_criteria_gemma9b"):
        mlflow.log_params({
            "judge_model": "google/gemma-2-9b-it",
            "mode": "all_criteria_at_once",
            "temperature": 0.0,
        })

        for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="Judge (all)"):
            try:
                result = run_judge(row.to_dict(), str(row["generated_description"]))
                ratings = result.to_ratings()
                # Add latency/cost from original run for final_score computation
                all_ratings = {**ratings,
                               "latency": str(row.get("latency", "")),
                               "cost":    str(row.get("cost", ""))}
                final = compute_final_score(all_ratings)
                judge_rows.append({
                    **ratings,
                    "judge_final_score": final,
                    **{f"{k}_explanation": getattr(result, k).explanation for k in ratings},
                })
            except Exception as e:
                judge_rows.append({c: "error" for c in ["fluency","grammar","tone","length","grounding"]})
                judge_rows[-1]["judge_final_score"] = "error"
                print(f"Error on row {_}: {e}")

        df_judge = pd.DataFrame(judge_rows)
        mlflow.log_metrics({
            "pass_rate":   (df_judge["judge_final_score"] == "pass").mean(),
            "error_rate":  (df_judge["judge_final_score"] == "error").mean(),
        })

    # Merge back
    for _col in ["fluency","grammar","tone","length","grounding","judge_final_score"]:
        df_main[f"judge_{_col}"] = df_judge[_col].values if _col in df_judge else df_judge[f"judge_{_col}"].values

    for _col in ["fluency","grammar","tone","length","grounding"]:
        df_main[f"judge_{_col}"] = df_judge[_col].values
    df_main["judge_final_score"] = df_judge["judge_final_score"].values

    xlsx_out = os.path.join(os.getcwd(), "..", "assignment_01.xlsx")
    df_main.to_excel(xlsx_out, index=False)
    print(f"Saved. Judge pass rate: {(df_judge['judge_final_score'] == 'pass').mean():.0%}")
    return df_judge, judge_rows, xlsx_out


if __name__ == "__main__":
    app.run()
