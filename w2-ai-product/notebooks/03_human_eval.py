import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Task 3 — Manual (Human) Evaluation

    ## Workflow
    1. This notebook adds programmatic ratings (latency, cost) to the workbook
    2. You manually open `outputs/assignment_01.xlsx` and rate 10–15 products for the 5 judged criteria
    3. Re-run this notebook to read back your scores and compute final_score + baseline analysis
    """)
    return


@app.cell
def _():
    from _bootstrap import bootstrap_notebook

    bootstrap_notebook()

    import pandas as pd
    from src.paths import ASSIGNMENT_XLSX_PATH
    from src.rubric import (
        CRITERION_COLS,
        JUDGED_COLS,
        compute_final_score,
        rate_cost_usd,
        rate_latency_ms,
    )

    return (
        ASSIGNMENT_XLSX_PATH,
        CRITERION_COLS,
        JUDGED_COLS,
        compute_final_score,
        pd,
        rate_cost_usd,
        rate_latency_ms,
    )


@app.cell
def _(ASSIGNMENT_XLSX_PATH, pd):
    xlsx_path = ASSIGNMENT_XLSX_PATH
    df = pd.read_excel(xlsx_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df, xlsx_path


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — Programmatic Ratings

    Latency and Cost are computed from API response metadata, not judged.
    """)
    return


@app.cell
def _(df, rate_cost_usd, rate_latency_ms, xlsx_path):
    df["latency"] = df["latency_ms"].apply(rate_latency_ms)
    df["cost"] = df["cost_usd"].apply(rate_cost_usd)
    df.to_excel(xlsx_path, index=False)
    print("Programmatic ratings written. Latency distribution:")
    print(df["latency"].value_counts().to_string())
    print("\nCost distribution:")
    print(df["cost"].value_counts().to_string())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — Manual Evaluation Instructions

    Open `outputs/assignment_01.xlsx` now. Choose 10–15 rows and fill in:

    | Column | Valid values |
    |--------|-------------|
    | `fluency` | good / ok / bad |
    | `grammar` | good / ok / bad |
    | `tone` | good / ok / bad |
    | `length` | good / ok / bad |
    | `grounding` | good / ok / bad |

    Leave other rows blank. Save and re-run this cell.

    **Grounding tip**: compare every claim in the description against the
    `Product_attribute_list`, `material`, and `warranty` columns. Any claim
    not traceable to those fields is a hallucination.
    """)
    return


@app.cell
def _(CRITERION_COLS, compute_final_score, pd, xlsx_path):
    # Re-read after manual scoring
    df2 = pd.read_excel(xlsx_path)

    # Compute final_score for manually scored rows
    def _apply_score(row):
        ratings = {col: str(row[col]).strip().lower() for col in CRITERION_COLS}
        return compute_final_score(ratings)

    df2["final_score"] = df2.apply(_apply_score, axis=1)
    df2.to_excel(xlsx_path, index=False)

    scored = df2[df2["fluency"].notna() & (df2["fluency"] != "")]
    print(f"Manually scored rows: {len(scored)}")
    print(
        f"Pass: {(scored['final_score'] == 'pass').sum()} | Fail: {(scored['final_score'] == 'fail').sum()}"
    )
    return (scored,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — Baseline Analysis
    """)
    return


@app.cell
def _(JUDGED_COLS, mo, pd, scored):
    if len(scored) == 0:
        summary_output = mo.md("**No manual scores yet — fill in the xlsx first.**")
    else:
        _rows = []
        for _col in JUDGED_COLS:
            _vals = scored[_col].value_counts()
            _rows.append(
                {
                    "Criterion": _col.capitalize(),
                    "good": _vals.get("good", 0),
                    "ok": _vals.get("ok", 0),
                    "bad": _vals.get("bad", 0),
                    "good %": f"{100 * _vals.get('good', 0) / len(scored):.0f}%",
                }
            )
        _summary = pd.DataFrame(_rows).sort_values("good %", ascending=False)
        summary_output = mo.ui.table(
            _summary,
            label="Criterion performance (best → worst)",
        )
    summary_output
    return


@app.cell
def _(mo, scored):
    scored_count = len(scored)
    pass_count = int((scored["final_score"] == "pass").sum()) if scored_count else 0
    pass_rate_label = f"{pass_count / scored_count:.0%}" if scored_count else "–"
    mo.md(
        f"""
        ## Step 4 — Baseline Analysis Write-Up

        **Best-performing criteria:**
        Length is strongest at 16/16 good. Grammar and tone follow at 15/16 good each, so the baseline already writes fluent, polished copy on most manually reviewed rows.

        **Worst-performing criteria:**
        Grounding is the clear weak point: 0/16 rows earned a `good`, with 12 `ok` and 4 `bad`. The main failure mode is small but unsupported embellishment rather than broken grammar.

        **Improvement strategy for Task 4:**
        Keep the strong style behavior, but add stricter anti-hallucination instructions and grounded examples so the model stays closer to the source data.

        **Pass rate:** {pass_count if scored_count > 0 else "–"} /
        {scored_count} manually evaluated ({pass_rate_label})
        """
    )
    return


if __name__ == "__main__":
    app.run()
