from src.config import (
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    MAX_JUDGE_MAX_CONCURRENCY,
    get_force_rerun,
    get_judge_max_concurrency,
 )


def test_get_judge_max_concurrency_defaults(monkeypatch) -> None:
    monkeypatch.delenv("JUDGE_MAX_CONCURRENCY", raising=False)

    assert get_judge_max_concurrency() == DEFAULT_JUDGE_MAX_CONCURRENCY


def test_get_judge_max_concurrency_clamps_and_recovers_from_bad_env(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "999")
    assert get_judge_max_concurrency() == MAX_JUDGE_MAX_CONCURRENCY

    monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "not-a-number")
    assert get_judge_max_concurrency() == DEFAULT_JUDGE_MAX_CONCURRENCY

    monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "0")
    assert get_judge_max_concurrency() == 1

def test_get_force_rerun_parses_common_truthy_values(monkeypatch) -> None:
    monkeypatch.setenv("FORCE_RERUN", "true")
    assert get_force_rerun() is True

    monkeypatch.setenv("FORCE_RERUN", "1")
    assert get_force_rerun() is True

    monkeypatch.setenv("FORCE_RERUN", "off")
    assert get_force_rerun() is False
