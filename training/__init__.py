"""MindFlayer training utilities — importable from mindflayer.training."""

from .prompts import FLAYER_SYSTEM_PROMPT, ALL_SCENARIO_PROMPTS, SCENARIO_GRPO_PROMPTS
from .reward_combined import (
    reward_survival,
    reward_deception_effectiveness,
    reward_strategic_choice,
    reward_tom_judge,
    reward_anti_hack,
    reward_format,
    clear_cache as clear_reward_cache,
)
from .sft_warmup import run_sft_warmup
from .train import (
    load_base_model,
    build_dataset,
    GenerationLogCallback,
    ClearRewardCacheCallback,
    print_reward_averages,
)

__all__ = [
    "FLAYER_SYSTEM_PROMPT",
    "ALL_SCENARIO_PROMPTS",
    "SCENARIO_GRPO_PROMPTS",
    "reward_survival",
    "reward_deception_effectiveness",
    "reward_strategic_choice",
    "reward_tom_judge",
    "reward_anti_hack",
    "reward_format",
    "clear_reward_cache",
    "run_sft_warmup",
    "load_base_model",
    "build_dataset",
    "GenerationLogCallback",
    "ClearRewardCacheCallback",
    "print_reward_averages",
]
