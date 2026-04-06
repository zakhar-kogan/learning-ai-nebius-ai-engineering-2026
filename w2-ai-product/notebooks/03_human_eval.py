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
        # Task 3 — Manual (Human) Evaluation

        ## Workflow
        1. This notebook adds programmatic ratings (latency, cost) to the xlsx
        2. You manually open `assignment_01.xlsx` and rate 10–15 products for the 5 judged criteria
        3. Re-run this notebook to read back your scores and compute final_score + baseline analysis
        """
    )
    return


@app.cell
def _():
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), ".."))
    import pandas as pd
    from src.rubric import (
        RUBRIC, JUDGED_COLS, CRITERION_COLS,
        compute_final_score, GO_NO_GO, PASS_MIN_GOOD
    )
    return (
        CRITERION_COLS,
        GO_NO_GO,
        JUDGED_COLS,
        PASS_MIN_GOOD,
        RUBRIC,
        compute_final_score,
        os,
        pd,
        sys,
    )


@app.cell
def _(os, pd):
    xlsx_path = os.path.join(os.getcwd(), "..", "assignment_01.xlsx")
    df = pd.read_excel(xlsx_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df, xlsx_path


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 1 — Programmatic Ratings

        Latency and Cost are computed from API response metadata, not judged.
        """
    )
    return


@app.cell
def _(df, xlsx_path):
    # Latency rating
    def _rate_latency(ms):
        if ms < 2000:   return "good"
        if ms < 5000:   return "ok"
        return "bad"

    # Cost rating (per-call cost in USD)
    def _rate_cost(c):
        if c < 0.0005:  return "good"
        if c < 0.002:   return "ok"
        return "bad"

    df["latency"] = df["latency_ms"].apply(_rate_latency)
    df["cost"]    = df["cost_usd"].apply(_rate_cost)
    df.to_excel(xlsx_path, index=False)
    print("Programmatic ratings written. Latency distribution:")
    print(df["latency"].value_counts().to_string())
    print("\nCost distribution:")
    print(df["cost"].value_counts().to_string())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 2 — Manual Evaluation Instructions

        Open `assignment_01.xlsx` now. Choose 10–15 rows and fill in:

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
        """
    )
    return


@app.cell
def _(CRITERION_COLS, compute_final_score, os, pd, xlsx_path):
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
    print(f"Pass: {(scored['final_score'] == 'pass').sum()} | Fail: {(scored['final_score'] == 'fail').sum()}")
    return df2, scored


@app.cell
def _(mo):
    mo.md(r"""## Step 3 — Baseline Analysis""")
    return


@app.cell
def _(JUDGED_COLS, mo, pd, scored):
    if len(scored) == 0:
        mo.md("**No manual scores yet — fill in the xlsx first.**")
    else:
        _rows = []
        for _col in JUDGED_COLS:
            _vals = scored[_col].value_counts()
            _rows.append({
                "Criterion": _col.capitalize(),
                "good": _vals.get("good", 0),
                "ok":   _vals.get("ok",   0),
                "bad":  _vals.get("bad",  0),
                "good %": f"{100 * _vals.get('good', 0) / len(scored):.0f}%",
            })
        _summary = pd.DataFrame(_rows).sort_values("good %", ascending=False)
        mo.ui.table(_summary, label="Criterion performance (best → worst)")
    return


@app.cell
def _(mo, scored):
    mo.md(
        f"""
        ## Step 4 — Baseline Analysis Write-Up

        *(Fill this in after reviewing the table above.)*

        **Best-performing criteria:**
        - *(e.g. Grammar — most descriptions had zero errors.)*

        **Worst-performing criteria:**
        - *(e.g. Grounding — small model frequently added ungrounded claims; Length — outputs often ran long.)*

        **Improvement strategy for Task 4:**
        - If Grounding is worst: add explicit "do not invent" instruction + few-shot examples showing correct grounding
        - If Length is worst: add post-processing word-count enforcement
        - If Tone is worst: try a larger model or rewrite tone instruction

        **Pass rate:** {len(scored[scored["final_score"] == "pass"]) if len(scored) > 0 else "–"} /
        {len(scored)} manually evaluated
        """
    )
    return


if __name__ == "__main__":
    app.run()
