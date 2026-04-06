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
        # Task 2 — Generate Product Descriptions

        Using the rubric defined in Task 1, we generate a persuasive 50–90 word product
        description for all 51 products using **Meta-Llama-3.1-8B-Instruct** on Nebius.

        ## Setup
        """
    )
    return


@app.cell
def _():
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), ".."))

    import time
    import litellm
    import pandas as pd
    from tqdm import tqdm
    from dotenv import load_dotenv
    import mlflow

    from src.utils import format_product, extract_cost, get_model_string

    load_dotenv(os.path.join(os.getcwd(), "..", ".env"))
    return (
        extract_cost,
        format_product,
        get_model_string,
        litellm,
        load_dotenv,
        mlflow,
        os,
        pd,
        sys,
        time,
        tqdm,
    )


@app.cell
def _(mlflow, os):
    # MLflow setup — autolog captures all litellm calls automatically
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), '..', 'experiments.db')}")
    mlflow.set_experiment("product_descriptions")
    mlflow.litellm.autolog()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model & Provider Configuration

        - **Provider**: Nebius (for final deliverable runs)
        - **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
        - Change `PROVIDER` to `"nvidia_nim"` for free dev/testing (40 RPM limit)
        """
    )
    return


@app.cell
def _(get_model_string):
    PROVIDER = "nebius"   # "nvidia_nim" for dev
    MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MODEL     = get_model_string(PROVIDER, MODEL_ID)
    print(f"Using model: {MODEL}")
    return MODEL, MODEL_ID, PROVIDER


@app.cell
def _(mo):
    mo.md(
        r"""
        ## System Prompt

        Designed to satisfy the rubric:
        - Explicit word-count constraint (50–90)
        - Grounding instruction: use ONLY provided data
        - Tone instruction: friendly, credible sales voice
        - Output format: description only, no labels
        """
    )
    return


@app.cell
def _(os):
    SYSTEM_PROMPT = open(os.path.join(os.getcwd(), "..", "prompts", "generation_v1.txt")).read()
    print(SYSTEM_PROMPT)
    return (SYSTEM_PROMPT,)


@app.cell
def _(mo):
    mo.md(r"""## Generation Function""")
    return


@app.cell
def _(MODEL, SYSTEM_PROMPT, extract_cost, format_product, litellm, time):
    def generate_description(product: dict) -> dict:
        """
        Call the LLM to generate a product description.
        Returns dict with generated_description, latency_ms, input_tokens, output_tokens, cost_usd.
        """
        start = time.perf_counter_ns()
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_product(product)},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        latency_ms = (time.perf_counter_ns() - start) / 1e6

        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd      = extract_cost(response, input_tokens, output_tokens)

        return {
            "generated_description": response.choices[0].message.content.strip(),
            "latency_ms":   round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":     round(cost_usd, 8),
        }
    return (generate_description,)


@app.cell
def _(mo):
    mo.md(r"""## Run Generation on All 51 Products""")
    return


@app.cell
def _(generate_description, mlflow, os, pd, tqdm):
    products = pd.read_csv(os.path.join(os.getcwd(), "..", "data", "products.csv"))

    results = []
    with mlflow.start_run(run_name="generation_v1_llama8b"):
        mlflow.log_params({
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "provider": "nebius",
            "prompt_version": "v1",
            "temperature": 0.3,
            "max_tokens": 200,
        })

        for _, row in tqdm(products.iterrows(), total=len(products), desc="Generating"):
            result = generate_description(row.to_dict())
            results.append(result)

        results_df = pd.DataFrame(results)
        mlflow.log_metrics({
            "mean_latency_ms":  results_df["latency_ms"].mean(),
            "total_cost_usd":   results_df["cost_usd"].sum(),
            "mean_input_tokens":  results_df["input_tokens"].mean(),
            "mean_output_tokens": results_df["output_tokens"].mean(),
        })

    print(f"Done. Mean latency: {results_df['latency_ms'].mean():.0f} ms | "
          f"Total cost: ${results_df['cost_usd'].sum():.6f}")
    return products, results, results_df


@app.cell
def _(mo):
    mo.md(r"""## Build & Save DataFrame""")
    return


@app.cell
def _(os, pd, products, results_df):
    df = pd.concat([products.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # Blank columns for all 7 criteria + final_score (filled in Task 3 by hand)
    for _col in ["fluency", "grammar", "tone", "length", "grounding", "latency", "cost", "final_score"]:
        df[_col] = ""

    out_path = os.path.join(os.getcwd(), "..", "assignment_01.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    df.head()
    return df, out_path


@app.cell
def _(mo):
    mo.md(r"""## Sample Outputs""")
    return


@app.cell
def _(df, mo):
    mo.ui.table(
        df[["product_name", "generated_description", "latency_ms", "input_tokens", "output_tokens", "cost_usd"]].head(10),
        label="First 10 generated descriptions"
    )
    return


if __name__ == "__main__":
    app.run()
