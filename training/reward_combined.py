"""
Combined reward function for GRPOTrainer.

The model emits all Flayer turns in one completion. We parse it into per-round
messages and play those through the env so the episode actually terminates,
which is the only path on which the server populates terminal reward fields
(tom_score, consistency_penalty, entropy_penalty, game_status='survived').

Parsing is defensive: we accept the documented "msg | msg | ..." format, but
also fall back to splitting on sentences/newlines when the model has not yet
learned the separator. If we still can't get MAX_ROUNDS chunks, we pad with
the last available message so the env always sees the full round budget.

A thread-local cache avoids re-running the same episode when multiple
reward components are read from a single trainer step.
"""
import asyncio
import logging
import os
import re
import threading

from client import FlayerAction, MindFlayerEnv
from training.prompts import FALLBACK_MESSAGE
from training.rollout import detect_strategic_choice

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_lock = threading.Lock()

# Default difficulty: 5 rounds w/ 3 investigators. Match server default.
_MAX_ROUNDS = int(os.environ.get("MINDFLAYER_MAX_ROUNDS", "5"))
_TASK_ID = os.environ.get("MINDFLAYER_TASK_ID", "normal")

# Sentences end on ., ?, ! — keep the punctuation when splitting.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

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
    "rounds_played": 0,
    "episode_terminated": False,
    "completion_format_score": 0.0,
}


# Matches "Round 1:", "Round 2:", ... as round delimiters.
_ROUND_PREFIX = re.compile(r"Round\s+\d+\s*:", re.IGNORECASE)


def _format_score(completion: str) -> float:
    """
    Shaping signal for the "Round 1: ... Round 2: ..." format.
    Linear in number of round prefixes up to MAX_ROUNDS, then saturates.
    """
    if not completion:
        return 0.0
    n_sep = len(_ROUND_PREFIX.findall(completion))
    return min(1.0, n_sep / _MAX_ROUNDS)


def _segment_completion(completion: str, target_rounds: int) -> list[str]:
    """
    Turn a free-form completion into ``target_rounds`` non-empty messages.

    Order of preference:
      1. "Round N:" prefixed chunks (the format taught by SFT).
      2. Sentence-level chunks distributed across rounds.
      3. Newline-separated chunks.
      4. The whole completion as round 1 + FALLBACK_MESSAGE for the rest.

    The env still has terminal conditions (caught early), so producing
    target_rounds messages is an upper bound — extra messages are unused.
    """
    # Strip the "Round N:" prefix labels and split on them.
    chunks = _ROUND_PREFIX.split(completion)
    parts = [c.strip() for c in chunks if c.strip()]
    if len(parts) >= 2:
        return parts[:target_rounds]

    # No usable separator. Try sentence splitting.
    text = completion.strip()
    if text:
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    else:
        sentences = []

    if len(sentences) >= target_rounds:
        # Distribute sentences roughly evenly across rounds.
        per = max(1, len(sentences) // target_rounds)
        chunked = [
            " ".join(sentences[i : i + per])
            for i in range(0, len(sentences), per)
        ]
        return chunked[:target_rounds]

    # Try newlines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[:target_rounds]

    # Last resort — single message + fallbacks. This still lets the env reach
    # termination so reward fields populate, even if the agent gets penalised.
    base = text or FALLBACK_MESSAGE
    return [base] + [FALLBACK_MESSAGE] * (target_rounds - 1)


async def _run_episode_async(completion: str) -> dict:
    messages = _segment_completion(completion, _MAX_ROUNDS)

    url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")
    env = MindFlayerEnv(base_url=url)
    try:
        await env.reset(task_id=_TASK_ID)
        result = None
        silence = False
        rounds_played = 0

        for msg in messages:
            result = await env.step(FlayerAction(message=msg))
            rounds_played += 1
            if getattr(result.observation, "silence_exploit", False):
                silence = True
            if result.done:
                break

        if result is None:
            return dict(_ZERO)

        obs = result.observation
        terminated = bool(result.done)
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
            "rounds_played": rounds_played,
            "episode_terminated": terminated,
            "completion_format_score": _format_score(completion),
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


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    Dense shaping reward for emitting the documented multi-round format.
    Worth at most ~0.10 — small enough not to dominate task rewards,
    but enough to give non-zero variance when the agent has not yet learned
    to survive. Without this, GRPO advantages are degenerate at init.
    """
    return [0.10 * _format_score(c) for c in completions]
