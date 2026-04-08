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
    # Task 1 — Evaluation Rubric

    Before generating or evaluating anything, we define a clear, repeatable scoring
    framework. Every evaluation decision — whether made manually or by a judge model —
    follows these rules.

    ---

    ## Criteria Overview

    | Criterion | Description | Evaluated by |
    |-----------|-------------|--------------|
    | Fluency | Natural, easy-to-read sentences | Human + Judge |
    | Grammar | Correct spelling & punctuation | Human + Judge |
    | Tone | Matches friendly, credible sales voice | Human + Judge |
    | Length | 50–90 words | Human + Judge |
    | Grounding | Sticks to information provided | Human + Judge |
    | Latency | Avg. time per API call | Programmatic |
    | Cost | Avg. price per API call | Programmatic |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Criterion Definitions

    ### Fluency — Natural, easy-to-read sentences

    | Rating | Definition |
    |--------|-----------|
    | **good** | Natural, easy-to-read flow; no awkward phrasing or unnatural constructions |
    | **ok** | Mostly natural but 1–2 awkward phrases or choppy transitions |
    | **bad** | Stilted, robotic, or hard to follow; multiple unnatural constructions |

    ### Grammar — Correct spelling and punctuation

    | Rating | Definition |
    |--------|-----------|
    | **good** | Zero spelling or punctuation errors |
    | **ok** | 1–2 minor errors that don't impede understanding |
    | **bad** | 3+ errors, or any error that changes meaning or is immediately visible |

    ### Tone — Matches friendly, credible sales voice

    | Rating | Definition |
    |--------|-----------|
    | **good** | Friendly, credible sales voice; persuasive without being pushy or hyperbolic |
    | **ok** | Mostly appropriate but occasionally too formal, too casual, or generic |
    | **bad** | Inappropriate: overly aggressive, robotic, or clearly mismatched to product |

    ### Length — 50–90 words

    | Rating | Definition |
    |--------|-----------|
    | **good** | 50–90 words (inclusive) |
    | **ok** | 40–49 words or 91–110 words |
    | **bad** | Fewer than 40 words or more than 110 words |

    ### Grounding — Sticks to information provided (no hallucination)

    | Rating | Definition |
    |--------|-----------|
    | **good** | Every claim traceable to provided product name, attributes, material, or warranty; no invented features |
    | **ok** | Minor embellishment reasonable for the product category but not explicitly in the data (e.g. "premium feel") |
    | **bad** | Fabricated specs, invented features, or claims that contradict the provided data |

    ### Latency — Avg. time per API call (programmatic)

    | Rating | Definition |
    |--------|-----------|
    | **good** | < 2 000 ms |
    | **ok** | 2 000–5 000 ms |
    | **bad** | > 5 000 ms |

    ### Cost — Avg. price per API call in USD (programmatic)

    | Rating | Definition |
    |--------|-----------|
    | **good** | < $0.0005 |
    | **ok** | $0.0005–$0.002 |
    | **bad** | > $0.002 |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pass / Fail Definition

    ### Cumulative Pass Bar

    A description **passes** if **all** of the following hold:
    - At least **4 out of 7** criteria are rated "good"
    - **Zero** criteria are rated "bad"

    ### Go / No-Go Rules

    Two criteria trigger **automatic failure** regardless of all other scores:

    | Criterion | Reason |
    |-----------|--------|
    | **Grounding** | Fabricated or contradictory product information is unacceptable for e-commerce — it misleads customers and creates legal/brand risk |
    | **Length** | A description far outside 40–110 words means the prompt fundamentally failed; such output is not usable without human rewriting |

    ### Formula

    ```
    final_score = "fail"  if grounding == "bad"  (go/no-go)
    final_score = "fail"  if length == "bad"      (go/no-go)
    final_score = "pass"  if count(good) >= 4 AND count(bad) == 0
    final_score = "fail"  otherwise
    ```

    ### Rationale for These Thresholds

    - **4/7 good, 0 bad**: Requires a majority of criteria to be excellent while being tolerant of 2–3 "ok" ratings (e.g. slightly too long, minor tone quirk). An all-"ok" description would score only 0 good → fail, preventing mediocrity from passing.
    - **Grounding as go/no-go**: A description inventing features is not a minor flaw — it's a factual error. Better to fail and regenerate than publish.
    - **Length as go/no-go**: A 25-word stub or 200-word wall is structurally wrong. Unlike tone issues, length is objectively verifiable and fixable by simply enforcing the constraint.
    """)
    return


@app.cell
def _():
    from _bootstrap import bootstrap_notebook
    bootstrap_notebook()

    import pandas as pd
    from src.rubric import RUBRIC, render_rubric_for_prompt

    return RUBRIC, pd, render_rubric_for_prompt


@app.cell
def _(RUBRIC, mo, pd):
    _df = pd.DataFrame([
        {
            "Criterion": c.name,
            "Description": c.description,
            "Good": c.good,
            "OK": c.ok,
            "Bad": c.bad,
            "Evaluated by": "Human + Judge" if c.judged else "Programmatic",
        }
        for c in RUBRIC
    ])
    mo.ui.table(_df, label="Rubric summary")
    return


@app.cell
def _(mo, render_rubric_for_prompt):
    mo.md(
        f"""
        ## Rubric as rendered in prompts

        ```
        {render_rubric_for_prompt(judged_only=True)}
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
