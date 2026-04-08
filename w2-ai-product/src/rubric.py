"""
rubric.py — single source of truth for evaluation criteria.
Used by: generation prompt rendering, judge prompt rendering, pass/fail scoring.
"""
from enum import Enum
from dataclasses import dataclass


class Rating(str, Enum):
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


@dataclass
class CriterionDefinition:
    name: str
    description: str
    good: str
    ok: str
    bad: str
    judged: bool = True  # False = programmatic only (latency, cost)


# ── Rubric ──────────────────────────────────────────────────────────────────
RUBRIC: list[CriterionDefinition] = [
    CriterionDefinition(
        name="Fluency",
        description="Natural, easy-to-read sentences",
        good="Natural, easy-to-read flow; no awkward phrasing or unnatural constructions",
        ok="Mostly natural but 1–2 awkward phrases or choppy transitions",
        bad="Stilted, robotic, or hard to follow; multiple unnatural constructions",
    ),
    CriterionDefinition(
        name="Grammar",
        description="Correct spelling and punctuation",
        good="Zero spelling or punctuation errors",
        ok="1–2 minor errors that don't impede understanding",
        bad="3+ errors, or any error that changes meaning or is immediately visible",
    ),
    CriterionDefinition(
        name="Tone",
        description="Matches friendly, credible sales voice",
        good="Friendly, credible sales voice; persuasive without being pushy or hyperbolic",
        ok="Mostly appropriate but occasionally too formal, too casual, or generic",
        bad="Inappropriate tone: overly aggressive, robotic, or clearly mismatched to product",
    ),
    CriterionDefinition(
        name="Length",
        description="50–90 words",
        good="50–90 words (inclusive)",
        ok="40–49 words or 91–110 words",
        bad="Fewer than 40 words or more than 110 words",
    ),
    CriterionDefinition(
        name="Grounding",
        description="Sticks to information provided (no hallucination)",
        good="Every claim is traceable to the provided product name, attributes, material, or warranty; no invented features",
        ok="Minor embellishment that is reasonable for the product category but not explicitly in the data (e.g. 'premium feel')",
        bad="Fabricated specs, invented features, or claims that contradict the provided data",
    ),
    CriterionDefinition(
        name="Latency",
        description="Avg. time per API call (ms)",
        good="< 2 000 ms",
        ok="2 000–5 000 ms",
        bad="> 5 000 ms",
        judged=False,
    ),
    CriterionDefinition(
        name="Cost",
        description="Avg. price per API call (USD)",
        good="< $0.000 5",
        ok="$0.000 5–$0.002",
        bad="> $0.002",
        judged=False,
    ),
]

# Only the criteria the judge evaluates (excludes Latency & Cost)
JUDGED_CRITERIA = [c for c in RUBRIC if c.judged]

# Column names used in the DataFrame / Excel
CRITERION_COLS = [c.name.lower() for c in RUBRIC]           # all 7
JUDGED_COLS    = [c.name.lower() for c in JUDGED_CRITERIA]  # 5


# ── Pass / Fail ──────────────────────────────────────────────────────────────
GO_NO_GO = {"grounding", "length"}   # bad in either → auto-fail
PASS_MIN_GOOD = 4                     # at least 4 "good" out of 7
PASS_MAX_BAD  = 0                     # zero "bad" allowed


def rate_latency_ms(latency_ms: float) -> str:
    if latency_ms < 2000:
        return Rating.GOOD.value
    if latency_ms < 5000:
        return Rating.OK.value
    return Rating.BAD.value


def rate_cost_usd(cost_usd: float) -> str:
    if cost_usd < 0.0005:
        return Rating.GOOD.value
    if cost_usd < 0.002:
        return Rating.OK.value
    return Rating.BAD.value


def compute_final_score(ratings: dict[str, str]) -> str:
    """
    Apply cumulative pass bar + go/no-go rules.

    Args:
        ratings: mapping of criterion_name.lower() → "good" | "ok" | "bad" | ""
    Returns:
        "pass" | "fail" | "" (empty if not all criteria rated)
    """
    normalized = {
        criterion: str(ratings.get(criterion, "")).strip().lower()
        for criterion in CRITERION_COLS
    }
    if any(value not in ("good", "ok", "bad") for value in normalized.values()):
        return ""
    
    # Treat partially scored rows as unscored, not failed.
    for criterion in GO_NO_GO:
        if normalized[criterion] == "bad":
            return "fail"
    
    goods = sum(1 for value in normalized.values() if value == "good")
    bads = sum(1 for value in normalized.values() if value == "bad")
    
    if goods >= PASS_MIN_GOOD and bads <= PASS_MAX_BAD:
        return "pass"
    return "fail"


# ── Prompt rendering ─────────────────────────────────────────────────────────
def render_rubric_for_prompt(
    judged_only: bool = True,
    extra_notes: dict[str, str] | None = None,
) -> str:
    """Format rubric as plain text to embed in prompts."""
    criteria = JUDGED_CRITERIA if judged_only else RUBRIC
    note_by_name = {key.lower(): value for key, value in (extra_notes or {}).items()}
    lines = []
    for c in criteria:
        lines.append(f"### {c.name} — {c.description}")
        lines.append(f"- good: {c.good}")
        lines.append(f"- ok:   {c.ok}")
        lines.append(f"- bad:  {c.bad}")
        note = note_by_name.get(c.name.lower())
        if note:
            lines.append(f"Note: {note}")
        lines.append("")
    return "\n".join(lines).strip()
