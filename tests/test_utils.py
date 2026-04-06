# pyright: reportMissingImports=false

import asyncio
from types import SimpleNamespace

import src.utils as utils


def test_format_product() -> None:
    product = {
        "product_name": "Widget",
        "Product_attribute_list": "size: medium",
        "material": "steel",
        "warranty": "2 years",
    }

    assert utils.format_product(product) == (
        "Product: Widget\n"
        "Attributes: size: medium\n"
        "Material: steel\n"
        "Warranty: 2 years"
    )


def test_format_judge_input() -> None:
    product = {
        "product_name": "Widget",
        "Product_attribute_list": "size: medium",
        "material": "steel",
        "warranty": "2 years",
    }

    rendered = utils.format_judge_input(product, "A polished description")
    assert "## Original Product Data" in rendered
    assert "## Generated Description" in rendered
    assert "A polished description" in rendered


def test_extract_cost_falls_back_to_nebius_rates(monkeypatch) -> None:
    monkeypatch.setattr(utils.litellm, "completion_cost", lambda completion_response: 0)

    response = SimpleNamespace()
    cost = utils.extract_cost(response, input_tokens=1_000, output_tokens=2_000)

    expected = (1_000 * (0.02 / 1_000_000)) + (2_000 * (0.06 / 1_000_000))
    assert cost == expected


def test_run_async_batch_preserves_input_order() -> None:
    async def worker(item: int) -> int:
        await asyncio.sleep(0.01 * (3 - item))
        return item

    results = asyncio.run(utils.run_async_batch(worker, [1, 2, 3], max_concurrency=3, desc="test"))
    assert results == [1, 2, 3]
