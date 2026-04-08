from dataclasses import dataclass
import os
from pathlib import Path

from src.paths import PROMPTS_DIR

DEFAULT_DEV_PROVIDER = "nvidia_nim"
DEFAULT_FINAL_PROVIDER = "nebius"

BASELINE_GENERATION_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
IMPROVEMENT_GENERATION_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_JUDGE_MODEL_ID = "google/gemma-2-9b-it-fast"
DEFAULT_JUDGE_MAX_CONCURRENCY = 6
MAX_JUDGE_MAX_CONCURRENCY = 8
PROMPT_VERSION_V1 = "generation_v1"
PROMPT_VERSION_V2 = "generation_v2"
PROMPT_VERSION_V3 = "generation_v3_voice"
PROMPT_VERSION_V4 = "generation_v4_scaffold"
PROMPT_SELECTION_V1 = "generation_selector_v1"
PROMPT_JUDGE_ALL = "judge_all"
PROMPT_JUDGE_SINGLE = "judge_single"

PROMPT_FILES: dict[str, Path] = {
    PROMPT_VERSION_V1: PROMPTS_DIR / "generation_v1.txt",
    PROMPT_VERSION_V2: PROMPTS_DIR / "generation_v2.txt",
    PROMPT_VERSION_V3: PROMPTS_DIR / "generation_v3_voice.txt",
    PROMPT_VERSION_V4: PROMPTS_DIR / "generation_v4_scaffold.txt",
    PROMPT_SELECTION_V1: PROMPTS_DIR / "generation_selector_v1.txt",
    PROMPT_JUDGE_ALL: PROMPTS_DIR / "judge_all.txt",
    PROMPT_JUDGE_SINGLE: PROMPTS_DIR / "judge_single.txt",
}

NEBIUS_RATE_COST_POLICY = (
    "Costs are normalized to Nebius-compatible rates for equal models. "
    "If LiteLLM does not return a concrete response cost, the fallback computation "
    "still uses Nebius input/output token prices by policy."
)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str

    @property
    def model(self) -> str:
        return f"{self.provider}/{self.model_id}"


def get_generation_config() -> ModelConfig:
    return ModelConfig(
        provider=os.getenv("GENERATION_PROVIDER", DEFAULT_FINAL_PROVIDER),
        model_id=os.getenv("GENERATION_MODEL", BASELINE_GENERATION_MODEL_ID),
    )


def get_judge_config() -> ModelConfig:
    return ModelConfig(
        provider=os.getenv("JUDGE_PROVIDER", DEFAULT_FINAL_PROVIDER),
        model_id=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL_ID),
    )


def get_judge_max_concurrency() -> int:
    raw_value = os.getenv("JUDGE_MAX_CONCURRENCY", str(DEFAULT_JUDGE_MAX_CONCURRENCY))
    try:
        value = int(raw_value)
    except ValueError:
        value = DEFAULT_JUDGE_MAX_CONCURRENCY
    return max(1, min(value, MAX_JUDGE_MAX_CONCURRENCY))


def get_force_rerun() -> bool:
    return os.getenv("FORCE_RERUN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_model_config(provider: str, model_id: str) -> ModelConfig:
    return ModelConfig(provider=provider, model_id=model_id)


def prompt_path(prompt_name: str) -> Path:
    return PROMPT_FILES[prompt_name]
