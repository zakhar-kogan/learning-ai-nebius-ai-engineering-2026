import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Task 6 — Run and Analyze the Judge""")
    return


@app.cell
def _():
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), ".."))

    import json
    import litellm
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from tenacity import retry, stop_after_attempt, retry_if_exception_type
    from pydantic import ValidationError
    from dotenv import load_dotenv
    import mlflow

    from src.schemas import CriterionJudgment
    from src.rubric import RUBRIC, JUDGED_CRITERIA, JUDGED_COLS, compute_final_score
    from src.utils import format_judge_input, get_model_string

    load_dotenv(os.path.join(os.getcwd(), "..", ".env"))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), '..', 'experiments.db')}")
    mlflow.set_experiment("judge_runs")
    mlflow.litellm.autolog()
    return (
        CriterionJudgment,
        JUDGED_COLS,
        JUDGED_CRITERIA,
        RUBRIC,
        ValidationError,
        compute_final_score,
        format_judge_input,
        get_model_string,
        json,
        litellm,
        load_dotenv,
        mlflow,
        np,
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
    SINGLE_PROMPT_TEMPLATE = open(os.path.join(os.getcwd(), "..", "prompts", "judge_single.txt")).read()
    return JUDGE_MODEL, SINGLE_PROMPT_TEMPLATE


@app.cell
def _(os, pd):
    df = pd.read_excel(os.path.join(os.getcwd(), "..", "assignment_01.xlsx"))
    print(f"Loaded {len(df)} rows")
    df.head(2)
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Task 6.3 — Agreement Rate: Human vs Judge (All-at-Once)

        Compare judge verdicts against manual scores for the 10–15 manually scored rows.
        """
    )
    return


@app.cell
def _(JUDGED_COLS, df, mo, pd):
    # Rows that were manually scored
    _scored_mask = df["fluency"].notna() & (df["fluency"].str.strip() != "")
    _scored = df[_scored_mask].copy()

    if len(_scored) == 0:
        mo.md("**No manual scores found. Complete Task 3 first.**")
    else:
        _agreement_rows = []
        for _col in JUDGED_COLS:
            _human_col = _col
            _judge_col = f"judge_{_col}"
            if _judge_col not in _scored.columns:
                continue
            _both = _scored[_scored[_judge_col].notna() & (_scored[_judge_col] != "")]
            if len(_both) == 0:
                continue
            _agree = (_both[_human_col].str.strip() == _both[_judge_col].str.strip()).mean()
            _agreement_rows.append({
                "Criterion": _col.capitalize(),
                "N compared": len(_both),
                "Agreement %": f"{_agree:.0%}",
                "Agreement (raw)": _agree,
            })

        _agreement_df = pd.DataFrame(_agreement_rows).sort_values("Agreement (raw)", ascending=False)
        mo.ui.table(_agreement_df.drop(columns=["Agreement (raw)"]), label="Human vs Judge Agreement (all-at-once)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Analysis — Where do they agree/diverge?

        *(Fill in after running.)*

        **High agreement expected on:** Length (objective, countable) and Grammar (clear right/wrong).

        **Low agreement expected on:** Grounding and Tone — these require judgment calls about
        what counts as "grounded" or "appropriate tone." The judge may apply the rubric more
        literally than a human would.

        **Likely divergence patterns:**
        - Judge may rate Grounding more strictly (flags any claim not word-for-word in attributes)
        - Humans may unconsciously apply background knowledge about products (e.g. knowing an iPhone
          has certain features), which the judge doesn't do
        - Judge may miss subtle tone mismatches that a human reader notices immediately
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Task 6.4 — Per-Criterion Judging

        Instead of judging all 5 criteria in one call, we send a separate API call per criterion.
        This isolates each judgment, removing cross-criterion contamination.

        **Expected effects:**
        - More focused reasoning per criterion (no cognitive load of evaluating 5 things at once)
        - Potentially higher agreement with human scores on complex criteria like Grounding
        - 5× API cost and 5× latency for the judging phase
        """
    )
    return


@app.cell
def _(
    CriterionJudgment,
    JUDGE_MODEL,
    JUDGED_CRITERIA,
    SINGLE_PROMPT_TEMPLATE,
    ValidationError,
    format_judge_input,
    json,
    litellm,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
):
    def _make_single_judge(criterion):
        extra = ""
        if criterion.name == "Grounding":
            extra = "Note: compare the description against the ORIGINAL PRODUCT DATA carefully. Only accept claims traceable to the provided fields."
        elif criterion.name == "Length":
            extra = "Note: count the words in the description carefully before deciding."

        prompt = SINGLE_PROMPT_TEMPLATE.format(
            criterion_name=criterion.name,
            criterion_description=criterion.description,
            good=criterion.good,
            ok=criterion.ok,
            bad=criterion.bad,
            extra_instructions=extra,
        )

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, KeyError)),
        )
        def _judge_single(product: dict, description: str) -> CriterionJudgment:
            response = litellm.completion(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": format_judge_input(product, description)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CriterionJudgment",
                        "schema": CriterionJudgment.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=512,
            )
            return CriterionJudgment.model_validate_json(response.choices[0].message.content)

        return _judge_single

    single_judges = {c.name.lower(): _make_single_judge(c) for c in JUDGED_CRITERIA}
    return (single_judges,)


@app.cell
def _(JUDGED_CRITERIA, compute_final_score, df, mlflow, os, pd, single_judges, tqdm):
    single_rows = []
    with mlflow.start_run(run_name="judge_per_criterion_gemma9b"):
        mlflow.log_params({
            "judge_model": "google/gemma-2-9b-it",
            "mode": "per_criterion_separate_calls",
            "temperature": 0.0,
            "total_calls": len(df) * len(JUDGED_CRITERIA),
        })

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Per-criterion judge"):
            row_result = {}
            for crit in JUDGED_CRITERIA:
                cname = crit.name.lower()
                try:
                    j = single_judges[cname](row.to_dict(), str(row["generated_description"]))
                    row_result[f"single_{cname}"] = j.verdict.value
                    row_result[f"single_{cname}_explanation"] = j.explanation
                except Exception as e:
                    row_result[f"single_{cname}"] = "error"
                    row_result[f"single_{cname}_explanation"] = str(e)
            single_rows.append(row_result)

        df_single = pd.DataFrame(single_rows)

        # Compute final score from per-criterion results using the programmatic
        # latency/cost ratings already written into assignment_01.xlsx.
        def _single_final(row_s):
            ratings = {c.name.lower(): row_s.get(f"single_{c.name.lower()}", "") for c in JUDGED_CRITERIA}
            ratings["latency"] = (
                str(df.loc[row_s.name, "latency"]).strip().lower()
                if "latency" in df.columns and pd.notna(df.loc[row_s.name, "latency"])
                else ""
            )
            ratings["cost"] = (
                str(df.loc[row_s.name, "cost"]).strip().lower()
                if "cost" in df.columns and pd.notna(df.loc[row_s.name, "cost"])
                else ""
            )
            return compute_final_score(ratings)

        df_single["single_final_score"] = df_single.apply(_single_final, axis=1)
        mlflow.log_metrics({
            "pass_rate": (df_single["single_final_score"] == "pass").mean(),
        })

    # Merge back into main df
    for _c in df_single.columns:
        df[_c] = df_single[_c].values

    xlsx_out = os.path.join(os.getcwd(), "..", "assignment_01.xlsx")
    df.to_excel(xlsx_out, index=False)
    print(f"Per-criterion done. Pass rate: {(df_single['single_final_score'] == 'pass').mean():.0%}")
    return df_single, single_rows, xlsx_out


@app.cell
def _(JUDGED_COLS, df, df_single, mo, pd):
    # Agreement: human vs per-criterion judge
    _scored_mask = df["fluency"].notna() & (df["fluency"].str.strip() != "")
    _scored = df[_scored_mask].copy()

    if len(_scored) > 0:
        _rows = []
        for _col in JUDGED_COLS:
            _human_col = _col
            _judge_all_col  = f"judge_{_col}"
            _judge_sing_col = f"single_{_col}"
            _both = _scored[
                _scored[_judge_all_col].notna() &
                _scored[_judge_sing_col].notna()
            ]
            if len(_both) == 0:
                continue
            _agree_all  = (_both[_human_col].str.strip() == _both[_judge_all_col].str.strip()).mean()
            _agree_sing = (_both[_human_col].str.strip() == _both[_judge_sing_col].str.strip()).mean()
            _rows.append({
                "Criterion":           _col.capitalize(),
                "All-at-Once %":       f"{_agree_all:.0%}",
                "Per-Criterion %":     f"{_agree_sing:.0%}",
                "Delta":               f"{(_agree_sing - _agree_all):+.0%}",
            })
        mo.ui.table(pd.DataFrame(_rows), label="Human agreement: All-at-Once vs Per-Criterion")
    else:
        mo.md("No manual scores to compare against.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why might per-criterion judging lead to different outcomes?

        When the judge evaluates all criteria simultaneously, it must allocate attention across
        multiple rubric dimensions. This creates **interference**: the model's assessment of
        Tone may be influenced by its concurrent evaluation of Grammar, or a strong Grounding
        failure may colour the Fluency rating.

        Per-criterion judging forces the model to focus entirely on one dimension at a time,
        with a tailored prompt that includes only the relevant rubric definition. This reduces
        cognitive overload and produces more isolated, focused reasoning.

        However, isolation also has downsides: some criteria are naturally interdependent.
        A description with bad Fluency often also has bad Grammar — the judge may give
        inconsistent ratings when these are evaluated in separate calls without the context
        of the overall description quality.

        ---

        ## Task 6.5 — Analysis

        ### Trade-offs: Human Evaluation vs LLM-as-a-Judge

        | Dimension | Human Evaluation | LLM-as-a-Judge |
        |-----------|-----------------|----------------|
        | **Cost** | High labour cost (~5 min/product × $50/hr rate) | Low API cost (~$0.001/product) |
        | **Scale** | Doesn't scale — bottleneck at evaluator bandwidth | Scales linearly with API calls |
        | **Consistency** | Humans drift over time; inter-rater agreement ~70–80% on subjective criteria | Deterministic at temp=0; same prompt always gives same output |
        | **Accuracy (objective)** | High on countable criteria (Length, Grammar) | High on countable criteria; deterministic counting |
        | **Accuracy (subjective)** | High — humans catch subtle tone, cultural nuance, brand fit | Moderate — may miss nuance, applies rubric too literally |
        | **Grounding detection** | Good — humans apply background product knowledge | Variable — judge only knows what's in the prompt context |
        | **Bias** | Fatigue, ordering effects, individual taste | Prompt-induced bias, anchoring on early criteria |
        | **Explainability** | Implicit — hard to audit why a human rated something | Explicit — explanation field provides auditable reasoning |

        ### Recommendation for Production

        For a system generating **thousands of descriptions daily**, I recommend a **hybrid approach**:

        1. **LLM-as-a-Judge for routine evaluation** (all daily outputs): Fast, cheap, consistent.
           Use the judge to flag failures and compute pass rates automatically.

        2. **Human evaluation as a calibration layer** (~1–2% sample, weekly or per model update):
           Humans review a random sample to detect judge drift, prompt degradation, or
           systematic errors (e.g. the judge approving hallucinations for well-known brands).

        3. **Human-in-the-loop for edge cases**: Any description the judge rates with a narrow
           margin (e.g. borderline ok/bad on Grounding) should be queued for human review.

        This mirrors standard ML quality assurance: automated metrics for scale, human audits
        for calibration and trust.

        **The judge is not a replacement for human judgment — it is a force multiplier for it.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
