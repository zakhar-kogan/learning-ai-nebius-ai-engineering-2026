from __future__ import annotations

import json
from collections.abc import Callable

import litellm
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from src.rubric import CriterionDefinition, render_rubric_for_prompt
from src.schemas import CriterionJudgment, JudgeOutput


def build_all_criteria_prompt(prompt_template: str) -> str:
    rubric_block = render_rubric_for_prompt(
        judged_only=True,
        extra_notes={
            "length": "count the words in the description carefully before deciding.",
            "grounding": "for grounding, compare the description against the ORIGINAL PRODUCT DATA carefully. Only accept claims that appear in the data.",
        },
    )
    return prompt_template.format(rubric_block=rubric_block)


def create_all_at_once_judge(
    *,
    model: str,
    prompt: str,
    format_judge_input: Callable[[dict, str], str],
) -> Callable[[dict, str], JudgeOutput]:
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, KeyError)),
    )
    def run_judge(product: dict, description: str) -> JudgeOutput:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": format_judge_input(product, description)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "JudgeOutput",
                    "schema": JudgeOutput.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=0.0,
            max_tokens=1024,
        )
        return JudgeOutput.model_validate_json(response.choices[0].message.content)

    return run_judge


def create_single_judges(
    *,
    model: str,
    prompt_template: str,
    judged_criteria: list[CriterionDefinition],
    format_judge_input: Callable[[dict, str], str],
) -> dict[str, Callable[[dict, str], CriterionJudgment]]:
    def _make_single_judge(criterion: CriterionDefinition) -> Callable[[dict, str], CriterionJudgment]:
        extra = ""
        if criterion.name == "Grounding":
            extra = (
                "Note: compare the description against the ORIGINAL PRODUCT DATA carefully. "
                "Only accept claims traceable to the provided fields."
            )
        elif criterion.name == "Length":
            extra = "Note: count the words in the description carefully before deciding."

        prompt = prompt_template.format(
            criterion_name=criterion.name,
            criterion_description=criterion.description,
            good=criterion.good,
            ok=criterion.ok,
            bad=criterion.bad,
            extra_instructions=extra,
        )

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, KeyError)),
        )
        def _judge_single(product: dict, description: str) -> CriterionJudgment:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": format_judge_input(product, description)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CriterionJudgment",
                        "schema": CriterionJudgment.model_json_schema(),
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=512,
            )
            return CriterionJudgment.model_validate_json(response.choices[0].message.content)

        return _judge_single

    return {criterion.name.lower(): _make_single_judge(criterion) for criterion in judged_criteria}
