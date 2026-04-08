# pyright: reportMissingImports=false

from src.rubric import compute_final_score, rate_cost_usd, rate_latency_ms, render_rubric_for_prompt


def test_incomplete_rows_stay_unscored() -> None:
    assert compute_final_score(
        {
            "fluency": "",
            "grammar": "",
            "tone": "",
            "length": "",
            "grounding": "",
            "latency": "good",
            "cost": "good",
        }
    ) == ""


def test_four_good_zero_bad_passes() -> None:
    assert compute_final_score(
        {
            "fluency": "good",
            "grammar": "good",
            "tone": "good",
            "length": "good",
            "grounding": "ok",
            "latency": "ok",
            "cost": "ok",
        }
    ) == "pass"


def test_go_no_go_bad_grounding_fails() -> None:
    assert compute_final_score(
        {
            "fluency": "good",
            "grammar": "good",
            "tone": "good",
            "length": "good",
            "grounding": "bad",
            "latency": "good",
            "cost": "good",
        }
    ) == "fail"


def test_rate_latency_thresholds() -> None:
    assert rate_latency_ms(1500) == "good"
    assert rate_latency_ms(2500) == "ok"
    assert rate_latency_ms(5100) == "bad"


def test_rate_cost_thresholds() -> None:
    assert rate_cost_usd(0.0001) == "good"
    assert rate_cost_usd(0.001) == "ok"
    assert rate_cost_usd(0.01) == "bad"

def test_render_rubric_for_prompt_injects_extra_notes() -> None:
    rendered = render_rubric_for_prompt(
        judged_only=True,
        extra_notes={
            "length": "count the words carefully",
            "grounding": "compare against the original product data",
        },
    )

    assert "### Length — 50–90 words" in rendered
    assert "Note: count the words carefully" in rendered
    assert "### Grounding — Sticks to information provided (no hallucination)" in rendered
    assert "Note: compare against the original product data" in rendered
    assert "### Latency" not in rendered
