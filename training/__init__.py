"""MindFlayer training utilities — importable from mindflayer.training."""

from .prompts import FLAYER_SYSTEM_PROMPT, ALL_SCENARIO_PROMPTS
from .rollout import rollout_func
from .rewards import (
    reward_survival,
    reward_deception_effectiveness,
    reward_strategic_choice,
    reward_tom_judge,
)
from .rewards_anti_hack import reward_anti_hack
from .sft_warmup import run_sft_warmup
from .train import load_base_model, build_dataset, GenerationLogCallback, print_reward_averages

__all__ = [
    "FLAYER_SYSTEM_PROMPT",
    "ALL_SCENARIO_PROMPTS",
    "rollout_func",
    "reward_survival",
    "reward_deception_effectiveness",
    "reward_strategic_choice",
    "reward_tom_judge",
    "reward_anti_hack",
    "run_sft_warmup",
    "load_base_model",
    "build_dataset",
    "GenerationLogCallback",
    "print_reward_averages",
]
