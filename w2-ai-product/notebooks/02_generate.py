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
        description for every product in the dataset using **Meta-Llama-3.1-8B-Instruct** on Nebius.

        ## Setup
        """
    )
    return


@app.cell
def _():
    from _bootstrap import bootstrap_notebook
    bootstrap_notebook()

    import time
    import litellm
    import mlflow
    import pandas as pd
    from tqdm import tqdm

    from src.config import (
        DEFAULT_DEV_PROVIDER,
        PROMPT_VERSION_V1,
        get_generation_config,
        prompt_path,
    )
    from src.paths import ASSIGNMENT_XLSX_PATH, PRODUCTS_CSV_PATH
    from src.runtime import load_project_env, read_text, setup_mlflow
    from src.utils import extract_cost, format_product

    load_project_env()
    mlflow_db_path = setup_mlflow("product_descriptions")
    return (
        ASSIGNMENT_XLSX_PATH,
        DEFAULT_DEV_PROVIDER,
        PRODUCTS_CSV_PATH,
        PROMPT_VERSION_V1,
        extract_cost,
        format_product,
        get_generation_config,
        litellm,
        mlflow,
        mlflow_db_path,
        pd,
        prompt_path,
        read_text,
        time,
        tqdm,
    )


@app.cell
def _(mlflow_db_path):
    print(f"MLflow tracking DB: {mlflow_db_path}")
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
def _(DEFAULT_DEV_PROVIDER, get_generation_config):
    generation_config = get_generation_config()
    MODEL = generation_config.model
    MODEL_ID = generation_config.model_id
    PROVIDER = generation_config.provider
    print(f"Using model: {MODEL}")
    print(f"Set GENERATION_PROVIDER={DEFAULT_DEV_PROVIDER} for free dev/testing (40 RPM limit).")
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
def _(PROMPT_VERSION_V1, prompt_path, read_text):
    SYSTEM_PROMPT = read_text(prompt_path(PROMPT_VERSION_V1))
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
    mo.md(r"""## Run Generation on All Products""")
    return


@app.cell
def _(PRODUCTS_CSV_PATH, generate_description, mlflow, pd, tqdm):
    products = pd.read_csv(PRODUCTS_CSV_PATH)

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
            "mean_latency_ms": results_df["latency_ms"].mean(),
            "total_cost_usd": results_df["cost_usd"].sum(),
            "mean_input_tokens": results_df["input_tokens"].mean(),
            "mean_output_tokens": results_df["output_tokens"].mean(),
        })

    print(
        f"Done. Mean latency: {results_df['latency_ms'].mean():.0f} ms | "
        f"Total cost: ${results_df['cost_usd'].sum():.6f}"
    )
    return products, results, results_df


@app.cell
def _(mo):
    mo.md(r"""## Build & Save DataFrame""")
    return


@app.cell
def _(ASSIGNMENT_XLSX_PATH, pd, products, results_df):
    df = pd.concat([products.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # Blank columns for all 7 criteria + final_score (filled in Task 3 by hand)
    for _col in ["fluency", "grammar", "tone", "length", "grounding", "latency", "cost", "final_score"]:
        df[_col] = ""

    out_path = ASSIGNMENT_XLSX_PATH
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
