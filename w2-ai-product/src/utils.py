"""
utils.py — shared helpers for product formatting, cost calculation, async batching.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import time
from typing import Any

import litellm
from tqdm.asyncio import tqdm as atqdm


# ── Provider config ───────────────────────────────────────────────────────────
# Change PROVIDER to "nvidia_nim" for dev (free, 40 RPM), "nebius" for final runs.
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "nebius")
GENERATION_MODEL    = os.getenv("GENERATION_MODEL",    "meta-llama/Meta-Llama-3.1-8B-Instruct")
JUDGE_PROVIDER      = os.getenv("JUDGE_PROVIDER",      "nebius")
JUDGE_MODEL         = os.getenv("JUDGE_MODEL",         "google/gemma-2-9b-it")

def get_model_string(provider: str, model: str) -> str:
    return f"{provider}/{model}"


# ── Product formatting ────────────────────────────────────────────────────────
def format_product(product: dict) -> str:
    """Format a product row as structured text for the generation prompt."""
    return (
        f"Product: {product.get('product_name', '')}\n"
        f"Attributes: {product.get('Product_attribute_list', '')}\n"
        f"Material: {product.get('material', '')}\n"
        f"Warranty: {product.get('warranty', '')}"
    )


def format_judge_input(product: dict, description: str) -> str:
    """Format the judge user message: original product data + generated description."""
    product_text = format_product(product)
    return (
        f"## Original Product Data\n{product_text}\n\n"
        f"## Generated Description\n{description}"
    )


# ── Cost extraction ───────────────────────────────────────────────────────────
# Nebius pricing fallback (per token)
_NEBIUS_INPUT_PRICE  = 0.02 / 1_000_000   # $0.02 / 1M input tokens
_NEBIUS_OUTPUT_PRICE = 0.06 / 1_000_000   # $0.06 / 1M output tokens


def extract_cost(response: Any, input_tokens: int, output_tokens: int) -> float:
    """
    Extract cost from LiteLLM response. Falls back to hardcoded Nebius pricing.
    LiteLLM stores cost in response._hidden_params['response_cost'].
    """
    try:
        cost = litellm.completion_cost(completion_response=response)
        if cost and cost > 0:
            return cost
    except Exception:
        pass
    # Fallback: manual calculation
    return (input_tokens * _NEBIUS_INPUT_PRICE) + (output_tokens * _NEBIUS_OUTPUT_PRICE)


# ── Async batch runner ────────────────────────────────────────────────────────
async def run_async_batch(
    coro_fn,
    items: list,
    max_concurrency: int = 5,
    desc: str = "Processing",
) -> list:
    """
    Run coro_fn(item) for each item, up to max_concurrency at a time.
    Returns results in order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(item):
        async with semaphore:
            return await coro_fn(item)

    tasks = [bounded(item) for item in items]
    results = []
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        results.append(await coro)
    return results
