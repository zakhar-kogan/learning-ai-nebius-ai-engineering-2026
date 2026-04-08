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

def test_run_async_batch_limits_concurrency_without_progress_bar() -> None:
    active_workers = 0
    max_active_workers = 0

    async def worker(item: int) -> int:
        nonlocal active_workers, max_active_workers
        active_workers += 1
        max_active_workers = max(max_active_workers, active_workers)
        await asyncio.sleep(0.01)
        active_workers -= 1
        return item * 2

    results = asyncio.run(utils.run_async_batch(worker, [1, 2, 3, 4], max_concurrency=2))

    assert results == [2, 4, 6, 8]
    assert max_active_workers == 2

def test_run_async_batch_sync_works_inside_running_event_loop() -> None:
    async def worker(item: int) -> int:
        await asyncio.sleep(0.01)
        return item + 1

    async def outer() -> list[int]:
        return utils.run_async_batch_sync(worker, [1, 2, 3], max_concurrency=2)

    assert asyncio.run(outer()) == [2, 3, 4]

def test_run_sync_batch_preserves_order_and_limits_concurrency() -> None:
    active_workers = 0
    max_active_workers = 0

    def worker(item: int) -> int:
        import time

        nonlocal active_workers, max_active_workers
        active_workers += 1
        max_active_workers = max(max_active_workers, active_workers)
        time.sleep(0.01 * (5 - item))
        active_workers -= 1
        return item * 10

    results = utils.run_sync_batch(worker, [1, 2, 3, 4], max_concurrency=2)

    assert results == [10, 20, 30, 40]
    assert max_active_workers == 2
