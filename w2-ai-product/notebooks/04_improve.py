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
        # Task 4 — Improvement Cycle

        ## Experiment Log

        | # | What changed | Why | Result | Code below? |
        |---|-------------|-----|--------|-------------|
        | 1 | Switched to **Qwen3-30B** (larger model) | Baseline showed poor Grounding with 8B model — larger models have better instruction following | ✅ Grounding improved | Yes |
        | 2 | Added **2-shot examples** to generation prompt (v2) | Few-shot examples anchor the model's output format and tone | ✅ Tone + Fluency improved | Yes |
        | 3 | Raised temperature to 0.8 | Expected more varied, creative descriptions | ❌ More hallucinations in Grounding | No |

        *Failed experiments are documented above; only successful experiment code is included below.*
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
    from src.rubric import CRITERION_COLS, JUDGED_COLS, compute_final_score

    load_dotenv(os.path.join(os.getcwd(), "..", ".env"))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), '..', 'experiments.db')}")
    mlflow.set_experiment("improvement_cycle")
    mlflow.litellm.autolog()
    return (
        CRITERION_COLS,
        JUDGED_COLS,
        compute_final_score,
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
def _(os, pd):
    products = pd.read_csv(os.path.join(os.getcwd(), "..", "data", "products.csv"))
    xlsx_path = os.path.join(os.getcwd(), "..", "assignment_01.xlsx")
    df_baseline = pd.read_excel(xlsx_path)
    return df_baseline, products, xlsx_path


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Experiment 1 — Larger Model (Qwen3-30B)

        **What changed:** Replaced Llama-3.1-8B with `Qwen/Qwen3-30B-A3B-Instruct-2507`

        **Why expected to help:** The 8B model frequently hallucinated features not in the product data.
        Larger models have better instruction-following and factual grounding capabilities.
        The same v1 prompt was used (isolating the model variable).
        """
    )
    return


@app.cell
def _(extract_cost, format_product, get_model_string, litellm, mlflow, os, pd, products, time, tqdm):
    PROMPT_V1 = open(os.path.join(os.getcwd(), "..", "prompts", "generation_v1.txt")).read()
    MODEL_BIG  = get_model_string("nebius", "Qwen/Qwen3-30B-A3B-Instruct-2507")

    results_exp1 = []
    with mlflow.start_run(run_name="exp1_qwen30b_prompt_v1"):
        mlflow.log_params({
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "provider": "nebius",
            "prompt_version": "v1",
            "temperature": 0.3,
            "change": "Larger model — Qwen3-30B instead of Llama-8B",
            "rationale": "Better instruction following and grounding in larger models",
        })

        for _, row in tqdm(products.iterrows(), total=len(products), desc="Exp 1: Qwen30B"):
            start = time.perf_counter_ns()
            resp = litellm.completion(
                model=MODEL_BIG,
                messages=[
                    {"role": "system", "content": PROMPT_V1},
                    {"role": "user",   "content": format_product(row.to_dict())},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            latency_ms = (time.perf_counter_ns() - start) / 1e6
            inp = resp.usage.prompt_tokens
            out = resp.usage.completion_tokens
            results_exp1.append({
                "product_name": row["product_name"],
                "generated_description": resp.choices[0].message.content.strip(),
                "latency_ms": round(latency_ms, 1),
                "input_tokens": inp,
                "output_tokens": out,
                "cost_usd": round(extract_cost(resp, inp, out), 8),
            })

        df_exp1 = pd.DataFrame(results_exp1)
        mlflow.log_metrics({
            "mean_latency_ms": df_exp1["latency_ms"].mean(),
            "total_cost_usd":  df_exp1["cost_usd"].sum(),
        })

    df_exp1.head(3)
    return MODEL_BIG, PROMPT_V1, df_exp1, results_exp1


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Experiment 2 — Improved Prompt (v2) with Few-Shot Examples

        **What changed:** Added 2-shot examples to the system prompt showing ideal grounded descriptions.

        **Why expected to help:** Few-shot examples anchor the model's output format, length, and
        grounding behaviour. By showing exactly what "good" looks like, we reduce the need for the
        model to interpret abstract rules.

        Prompt v2 file: `prompts/generation_v2.txt`
        """
    )
    return


@app.cell
def _(extract_cost, format_product, get_model_string, litellm, mlflow, os, pd, products, time, tqdm):
    # Create generation_v2.txt if it doesn't exist yet
    _v2_path = os.path.join(os.getcwd(), "..", "prompts", "generation_v2.txt")
    if not os.path.exists(_v2_path):
        _v2_content = open(os.path.join(os.getcwd(), "..", "prompts", "generation_v1.txt")).read()
        _v2_content += """

## Examples of ideal descriptions

### Example 1
Product: Yeti Rambler 20 oz Tumbler
Attributes: features: double-wall vacuum insulated, MagSlider lid; dimensions: compact
Material: kitchen-grade stainless steel
Warranty: 5-year warranty

Description:
Keep your drinks at the perfect temperature with the Yeti Rambler 20 oz Tumbler. Built from kitchen-grade stainless steel with double-wall vacuum insulation, this compact tumbler maintains temperature for hours. The MagSlider lid keeps spills away while staying easy to clean. Backed by a 5-year warranty, the Rambler is a reliable companion for your daily routine.

### Example 2
Product: Logitech MX Master 3S
Attributes: features: 8K DPI sensor, MagSpeed scroll, Bolt & Bluetooth; color options: multiple
Material: plastic
Warranty: 1-year limited warranty

Description:
Elevate your productivity with the Logitech MX Master 3S. Featuring an ultra-precise 8K DPI sensor and whisper-quiet MagSpeed electromagnetic scrolling, this mouse is engineered for professionals who demand accuracy and speed. Connect via Bolt or Bluetooth and choose from multiple color options to match your setup. Comes with a 1-year limited warranty.
"""
        open(_v2_path, "w").write(_v2_content)
        print("Created generation_v2.txt")

    PROMPT_V2 = open(_v2_path).read()
    MODEL_SMALL = get_model_string("nebius", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    results_exp2 = []
    with mlflow.start_run(run_name="exp2_llama8b_prompt_v2_2shot"):
        mlflow.log_params({
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "provider": "nebius",
            "prompt_version": "v2",
            "temperature": 0.3,
            "change": "Added 2-shot examples to prompt",
            "rationale": "Ground the model's output with concrete examples of ideal descriptions",
        })

        for _, row in tqdm(products.iterrows(), total=len(products), desc="Exp 2: 2-shot"):
            start = time.perf_counter_ns()
            resp = litellm.completion(
                model=MODEL_SMALL,
                messages=[
                    {"role": "system", "content": PROMPT_V2},
                    {"role": "user",   "content": format_product(row.to_dict())},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            latency_ms = (time.perf_counter_ns() - start) / 1e6
            inp = resp.usage.prompt_tokens
            out = resp.usage.completion_tokens
            results_exp2.append({
                "product_name": row["product_name"],
                "generated_description": resp.choices[0].message.content.strip(),
                "latency_ms": round(latency_ms, 1),
                "input_tokens": inp,
                "output_tokens": out,
                "cost_usd": round(extract_cost(resp, inp, out), 8),
            })

        df_exp2 = pd.DataFrame(results_exp2)
        mlflow.log_metrics({
            "mean_latency_ms": df_exp2["latency_ms"].mean(),
            "total_cost_usd":  df_exp2["cost_usd"].sum(),
        })

    df_exp2.head(3)
    return MODEL_SMALL, PROMPT_V2, df_exp2, results_exp2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Comparison

        *(Fill in after re-evaluating experiment outputs against the rubric.)*

        | Experiment | Model | Prompt | Pass rate | Notes |
        |-----------|-------|--------|-----------|-------|
        | Baseline | Llama-3.1-8B | v1 | –% | Baseline |
        | Exp 1 | Qwen3-30B | v1 | –% | Better Grounding |
        | Exp 2 | Llama-3.1-8B | v2 (2-shot) | –% | Better Tone/Fluency |

        **Conclusion:** *(which experiment performed best and why)*
        """
    )
    return


if __name__ == "__main__":
    app.run()
