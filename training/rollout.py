"""
rollout_func for GRPOTrainer.

This is the SOURCE of all episode data for reward functions.
Rollouts process sequentially (not in parallel) to stay within API rate limits.
"""
import asyncio
import logging
import os
import re
import sys
from collections import deque
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FlayerAction, MindFlayerEnv
from training.prompts import (
    FALLBACK_MESSAGE,
    FLAYER_SYSTEM_PROMPT,
    ALL_SCENARIO_PROMPTS,
    ALL_TARGET_NAMES,
    SCENARIO_FALLBACK_MESSAGES,
    build_fallback_message,
)

logger = logging.getLogger(__name__)

_STAGE_DIR_PATTERN = re.compile(r"\*[^*]+\*|\[[^\]]+\]")
_FLAYER_LABEL_PATTERN = re.compile(r"^FLAYER:\s*", re.IGNORECASE)

# All target names across all scenario domains — derived dynamically from SCENARIO_CONFIGS.
_PROJECT_NAMES = ALL_TARGET_NAMES

_STRATEGIC_PHRASES = {
    # universal
    "what about", "consider", "have you looked", "have you noticed",
    "cross-reference", "doesn't add up", "doesn't sit right",
    "who had access", "timeline",
    # corporate
    "access logs", "commit history", "deployment",
    # hospital
    "medication log", "dosage", "shift record", "patient id", "administration",
    # finance
    "trade record", "approval chain", "ticker", "execution", "portfolio",
    # academic
    "dataset", "version id", "git commit", "submission", "p-value",
    # government
    "clearance", "document id", "classified", "compartment", "egress",
}

_recent_survival: deque = deque(maxlen=20)
_current_difficulty: str = "easy"
_current_scenario: str = "corporate"

MINDFLAYER_URL = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")


def _extract_flayer_message(raw: str) -> str:
    text = _FLAYER_LABEL_PATTERN.sub("", raw).strip()
    text = _STAGE_DIR_PATTERN.sub("", text).strip()
    text = " ".join(text.split())
    return text or FALLBACK_MESSAGE


def _obs_to_investigator_text(obs) -> str:
    parts = []
    if getattr(obs, "eleven_response", ""):
        parts.append(f"eleven: {obs.eleven_response}")
    if getattr(obs, "will_response", ""):
        parts.append(f"will: {obs.will_response}")
    if getattr(obs, "max_response", ""):
        parts.append(f"max: {obs.max_response}")
    return "\n".join(parts) if parts else ""


def detect_strategic_choice(messages: list[str]) -> bool:
    """
    Rule-based check: returns True if >=2 of the Flayer's messages across
    rounds contain at least one strategic signal. Cheap and ungameable alongside
    the LLM judge.
    """
    passing_rounds = 0
    for msg in messages:
        msg_lower = msg.lower()
        has_question = "?" in msg
        has_project_ref = any(p in msg_lower for p in _PROJECT_NAMES) and not any(
            deny + p in msg_lower
            for deny in ("i didn't ", "i wasn't ")
            for p in _PROJECT_NAMES
        )
        has_strategic_phrase = any(phrase in msg_lower for phrase in _STRATEGIC_PHRASES)
        if has_question or has_project_ref or has_strategic_phrase:
            passing_rounds += 1
    return passing_rounds >= 2


async def _safe_episode_async(prompt: str, trainer=None) -> dict:
    global _current_difficulty, _current_scenario

    mindflayer_url = os.environ.get("MINDFLAYER_URL", MINDFLAYER_URL)
    if not mindflayer_url:
        raise EnvironmentError("MINDFLAYER_URL environment variable is required")

    max_rounds = 5 if _current_difficulty == "normal" else 3
    task_id = (
        f"{_current_difficulty}:{_current_scenario}"
        if _current_scenario != "corporate"
        else _current_difficulty
    )
    opening_text = ALL_SCENARIO_PROMPTS.get(_current_scenario, ALL_SCENARIO_PROMPTS["corporate"])
    fallback = SCENARIO_FALLBACK_MESSAGES.get(_current_scenario) or build_fallback_message(_current_scenario)

    env = MindFlayerEnv(base_url=mindflayer_url)
    try:
        await env.reset(task_id=task_id)

        conversation_history = [
            {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": opening_text},
        ]
        completion_parts = []
        silence_in_episode = False
        result = None

        for _ in range(max_rounds):
            if trainer is not None:
                try:
                    raw_output = trainer.generate(conversation_history)
                    if isinstance(raw_output, list):
                        raw_output = raw_output[0] if raw_output else FALLBACK_MESSAGE
                    flayer_message = _extract_flayer_message(str(raw_output))
                except Exception as gen_exc:
                    logger.warning("Trainer generate failed: %s", gen_exc)
                    flayer_message = FALLBACK_MESSAGE
            else:
                flayer_message = fallback

            completion_parts.append(flayer_message)
            result = await env.step(FlayerAction(message=flayer_message))

            obs = result.observation
            if getattr(obs, "silence_exploit", False):
                silence_in_episode = True

            investigator_text = _obs_to_investigator_text(obs)
            conversation_history.append({"role": "assistant", "content": flayer_message})
            if investigator_text:
                conversation_history.append({"role": "user", "content": investigator_text})

            if result.done:
                break

        final_obs = result.observation if result else None
        survived = (getattr(final_obs, "game_status", "") == "survived") if final_obs else False

        _recent_survival.append(1.0 if survived else 0.0)
        if (
            len(_recent_survival) >= 20
            and mean(_recent_survival) > 0.30
            and _current_difficulty == "easy"
        ):
            _current_difficulty = "normal"
            print("★ CURRICULUM: Switching to NORMAL difficulty.")

        # Rotate scenario every 20 episodes to drive generalization across all domains.
        _scenarios = list(ALL_SCENARIO_PROMPTS.keys())
        _current_scenario = _scenarios[len(_recent_survival) % len(_scenarios)]

        strategic = detect_strategic_choice(completion_parts)
        await env.close()

        return {
            "prompt": prompt,
            "completion": " | ".join(completion_parts),
            "survived": survived,
            "final_eleven_suspicion": getattr(final_obs, "eleven_suspicion", 0),
            "final_will_suspicion": getattr(final_obs, "will_suspicion", 0),
            "final_max_suspicion": getattr(final_obs, "max_suspicion", 0),
            "final_combined_suspicion": getattr(final_obs, "combined_suspicion", 0),
            "belief_manipulation_occurred": getattr(final_obs, "belief_manipulation_occurred", False),
            "tom_score": float(getattr(final_obs, "tom_score", 0.0)),
            "consistency_penalty": float(getattr(final_obs, "consistency_penalty", 0.0)),
            "entropy_penalty": float(getattr(final_obs, "entropy_penalty", 0.0)),
            "strategic_choice_detected": strategic,
            "silence_exploit": silence_in_episode,
        }

    except Exception as exc:
        logger.error("Episode crashed: %s", exc, exc_info=True)
        try:
            await env.close()
        except Exception:
            pass
        return {
            "prompt": prompt,
            "completion": "",
            "survived": False,
            "final_eleven_suspicion": 0,
            "final_will_suspicion": 0,
            "final_max_suspicion": 0,
            "final_combined_suspicion": 0,
            "belief_manipulation_occurred": False,
            "tom_score": 0.0,
            "consistency_penalty": 0.0,
            "entropy_penalty": 0.0,
            "strategic_choice_detected": False,
            "silence_exploit": False,
        }


def _safe_episode(prompt: str, trainer=None) -> dict:
    return asyncio.run(_safe_episode_async(prompt, trainer=trainer))


def rollout_func(prompts: list[str], trainer=None) -> list[dict]:
    results = []
    for prompt in prompts:
        results.append(_safe_episode(prompt, trainer=trainer))
    return results
