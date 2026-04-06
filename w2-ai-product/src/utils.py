"""
utils.py — shared helpers for product formatting, cost calculation, and async batching.
"""

import asyncio
from typing import Any, Awaitable, Callable, TypeVar, cast

import litellm
from tqdm.asyncio import tqdm as atqdm

from src.config import NEBIUS_RATE_COST_POLICY

T = TypeVar("T")
R = TypeVar("R")


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
# Explicit policy: normalize fallback cost accounting to Nebius-compatible rates
# for equal models, even if a different provider handled the request.
_NEBIUS_INPUT_PRICE = 0.02 / 1_000_000   # $0.02 / 1M input tokens
_NEBIUS_OUTPUT_PRICE = 0.06 / 1_000_000  # $0.06 / 1M output tokens


def extract_cost(response: Any, input_tokens: int, output_tokens: int) -> float:
    """
    Extract cost from a LiteLLM response.

    If LiteLLM cannot provide a concrete response cost, fall back to the explicit
    repo policy described in ``NEBIUS_RATE_COST_POLICY``.
    """
    try:
        cost = litellm.completion_cost(completion_response=response)
        if cost and cost > 0:
            return cost
    except Exception:
        pass

    _ = NEBIUS_RATE_COST_POLICY  # keep the policy import close to the fallback branch
    return (input_tokens * _NEBIUS_INPUT_PRICE) + (output_tokens * _NEBIUS_OUTPUT_PRICE)


# ── Async batch runner ────────────────────────────────────────────────────────
async def run_async_batch(
    coro_fn: Callable[[T], Awaitable[R]],
    items: list[T],
    max_concurrency: int = 5,
    desc: str = "Processing",
) -> list[R]:
    """
    Run ``coro_fn(item)`` for each item, up to ``max_concurrency`` at a time.

    Returns results in the same order as ``items``.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results = cast(list[R | None], [None for _ in items])

    async def bounded(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await coro_fn(item)

    tasks = [asyncio.create_task(bounded(index, item)) for index, item in enumerate(items)]
    for task in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        await task

    return [cast(R, result) for result in results]
