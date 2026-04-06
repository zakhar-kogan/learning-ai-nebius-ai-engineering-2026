"""
schemas.py — Pydantic output schemas for the judge model.

Key design choice: explanation BEFORE verdict.
When the model generates `explanation` first, it performs chain-of-thought
reasoning before committing to a verdict. If `verdict` came first, the model
would anchor on a rating and then confabulate a justification — anchoring bias.
Forcing the reasoning to precede the conclusion yields more calibrated verdicts.
"""
from pydantic import BaseModel
from src.rubric import Rating


class CriterionJudgment(BaseModel):
    explanation: str  # reasoning BEFORE verdict — chain-of-thought
    verdict: Rating   # "good" | "ok" | "bad"


class JudgeOutput(BaseModel):
    """All-at-once judge output: 5 judged criteria (excludes latency & cost)."""
    fluency: CriterionJudgment
    grammar: CriterionJudgment
    tone: CriterionJudgment
    length: CriterionJudgment
    grounding: CriterionJudgment

    def to_ratings(self) -> dict[str, str]:
        """Return {criterion: verdict_value} dict for compute_final_score()."""
        return {
            "fluency":    self.fluency.verdict.value,
            "grammar":    self.grammar.verdict.value,
            "tone":       self.tone.verdict.value,
            "length":     self.length.verdict.value,
            "grounding":  self.grounding.verdict.value,
        }
