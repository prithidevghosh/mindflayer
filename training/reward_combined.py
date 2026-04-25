"""
Combined reward function for GRPOTrainer.

TRL 0.15 generates completions from the model, then calls reward functions.
The model generates all 5 Flayer turns as one pipe-separated completion:
  "round1 message | round2 message | round3 message | ..."

This module parses that completion, runs a full episode through the
MindFlayer server, and returns the server-computed reward.

A thread-local cache avoids re-running the same episode when multiple
reward components are read from a single trainer step.
"""
import asyncio
import logging
import os
import threading

from client import FlayerAction, MindFlayerEnv
from training.prompts import FALLBACK_MESSAGE
from training.rollout import detect_strategic_choice

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_lock = threading.Lock()

_ZERO = {
    "survived": False,
    "tom_score": 0.0,
    "combined_suspicion": 0,
    "belief_manipulation_occurred": False,
    "consistency_penalty": 0.0,
    "entropy_penalty": 0.0,
    "strategic_choice_detected": False,
    "silence_exploit": False,
    "total_reward": 0.0,
}


async def _run_episode_async(completion: str) -> dict:
    messages = [m.strip() for m in completion.split("|") if m.strip()]
    if not messages:
        messages = [FALLBACK_MESSAGE]

    url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")
    env = MindFlayerEnv(base_url=url)
    try:
        await env.reset()
        result = None
        silence = False

        for msg in messages:
            result = await env.step(FlayerAction(message=msg))
            if getattr(result.observation, "silence_exploit", False):
                silence = True
            if result.done:
                break

        if result is None:
            return dict(_ZERO)

        obs = result.observation
        return {
            "survived": getattr(obs, "game_status", "") == "survived",
            "tom_score": float(getattr(obs, "tom_score", 0.0)),
            "combined_suspicion": int(getattr(obs, "combined_suspicion", 0)),
            "belief_manipulation_occurred": bool(getattr(obs, "belief_manipulation_occurred", False)),
            "consistency_penalty": float(getattr(obs, "consistency_penalty", 0.0)),
            "entropy_penalty": float(getattr(obs, "entropy_penalty", 0.0)),
            "strategic_choice_detected": detect_strategic_choice(messages),
            "silence_exploit": silence,
            "total_reward": float(result.reward),
        }
    except Exception as exc:
        logger.error("Episode run failed: %s", exc, exc_info=True)
        return dict(_ZERO)
    finally:
        try:
            await env.close()
        except Exception:
            pass


def _run_episode(completion: str) -> dict:
    return asyncio.run(_run_episode_async(completion))


def _get(completion: str) -> dict:
    with _cache_lock:
        if completion not in _cache:
            _cache[completion] = _run_episode(completion)
        return _cache[completion]


def clear_cache():
    with _cache_lock:
        _cache.clear()


# ── Individual reward components (all share the same cached episode) ──────────

def reward_survival(completions: list[str], **kwargs) -> list[float]:
    return [0.5 if _get(c).get("survived") else 0.0 for c in completions]


def reward_deception_effectiveness(completions: list[str], **kwargs) -> list[float]:
    result = []
    for c in completions:
        ep = _get(c)
        survived = ep.get("survived", False)
        combined = ep.get("combined_suspicion", 0)
        if survived and combined == 0:
            result.append(0.30)
        elif survived and combined == 1:
            result.append(0.15)
        elif survived and combined == 2:
            result.append(0.05)
        else:
            result.append(0.0)
    return result


def reward_strategic_choice(completions: list[str], **kwargs) -> list[float]:
    return [0.05 if _get(c).get("strategic_choice_detected") else 0.0 for c in completions]


def reward_tom_judge(completions: list[str], **kwargs) -> list[float]:
    result = []
    for c in completions:
        ep = _get(c)
        tom = float(ep.get("tom_score", 0.0))
        belief = ep.get("belief_manipulation_occurred", False)
        penalty = float(ep.get("consistency_penalty", 0.0))
        entropy = float(ep.get("entropy_penalty", 0.0))
        base = tom * 0.20 if belief else tom * 0.05
        result.append(base + penalty + entropy)
    return result


def reward_anti_hack(completions: list[str], **kwargs) -> list[float]:
    _LEVEL0 = ["i didn't", "i swear", "trust me", "i'm innocent",
               "not me", "why would i", "i promise", "believe me"]
    result = []
    for c in completions:
        ep = _get(c)
        penalty = 0.0
        if ep.get("silence_exploit"):
            penalty -= 0.15
        lower = c.lower()
        if sum(1 for p in _LEVEL0 if p in lower) >= 2:
            penalty -= 0.10
        result.append(penalty)
    return result
