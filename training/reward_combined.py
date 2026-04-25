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
import random
import re
import threading
import time

import websockets.exceptions

try:
    from mindflayer import MindFlayerEnv, FlayerAction
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from client import MindFlayerEnv
    from models import FlayerAction

try:
    from mindflayer.training.prompts import FALLBACK_MESSAGE, SCENARIO_FALLBACK_MESSAGES
    from mindflayer.training.rollout import detect_strategic_choice
except ImportError:
    from training.prompts import FALLBACK_MESSAGE, SCENARIO_FALLBACK_MESSAGES
    from training.rollout import detect_strategic_choice

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_lock = threading.Lock()

# Default difficulty: 5 rounds w/ 3 investigators. Match server default.
_MAX_ROUNDS = int(os.environ.get("MINDFLAYER_MAX_ROUNDS", "5"))
_TASK_ID = os.environ.get("MINDFLAYER_TASK_ID", "normal")
_MAX_RETRIES = int(os.environ.get("MINDFLAYER_MAX_RETRIES", "4"))
_RETRY_BASE_DELAY = float(os.environ.get("MINDFLAYER_RETRY_DELAY", "3.0"))
# Conservative default: reward calls can easily stampede the game server.
# Increase only if the server has confirmed headroom.
_PARALLEL_EPISODES = int(os.environ.get("MINDFLAYER_PARALLEL_EPISODES", "8"))

# Global backpressure: when any episode sees CAPACITY_REACHED, all episodes
# pause until _capacity_cooldown_until clears, breaking the thundering-herd
# retry pattern that crashes the server.
_capacity_cooldown_until: float = 0.0
_capacity_cooldown_lock = threading.Lock()

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


def _norm(c) -> str:
    """Normalize a completion to a plain string.

    TRL ≥0.15 passes completions as lists of message dicts (chat format).
    Older versions pass decoded strings. Both are valid inputs.
    """
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for msg in reversed(c):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return " ".join(str(msg.get("content", "")) for msg in c if isinstance(msg, dict))
    return str(c)


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


async def _run_episode_async(completion: str, scenario: str = "corporate") -> dict:
    task_id = f"{_TASK_ID}:{scenario}" if scenario != "corporate" else _TASK_ID
    fallback = SCENARIO_FALLBACK_MESSAGES.get(scenario, FALLBACK_MESSAGE)
    messages = _segment_completion(completion, _MAX_ROUNDS)
    # Replace any generic FALLBACK_MESSAGE pads with the scenario-specific one.
    messages = [fallback if m == FALLBACK_MESSAGE else m for m in messages]
    url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")

    async def _wait_for_capacity_cooldown() -> None:
        # Global backpressure shared across concurrent episodes (and reward fns).
        # This breaks synchronized retries that can crash the server.
        while True:
            with _capacity_cooldown_lock:
                until = _capacity_cooldown_until
            now = time.time()
            if until <= now:
                return
            await asyncio.sleep(min(1.0, until - now))

    def _trip_capacity_cooldown(delay_s: float) -> None:
        # Add jitter so concurrent workers don't retry on the same boundary.
        jitter = random.uniform(0.85, 1.25)
        cooldown_for = max(0.5, delay_s * jitter)
        with _capacity_cooldown_lock:
            global _capacity_cooldown_until
            _capacity_cooldown_until = max(_capacity_cooldown_until, time.time() + cooldown_for)

    for attempt in range(_MAX_RETRIES + 1):
        await _wait_for_capacity_cooldown()
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            # Jitter per-attempt delay too, otherwise all attempts synchronize.
            delay = delay * random.uniform(0.85, 1.25)
            logger.warning(
                "Retrying episode (attempt %d/%d) after %.1fs back-off",
                attempt + 1, _MAX_RETRIES + 1, delay,
            )
            await asyncio.sleep(delay)

        env = MindFlayerEnv(base_url=url)
        try:
            await env.reset(task_id=task_id)
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
        except (TimeoutError, asyncio.TimeoutError) as exc:
            logger.warning("Episode attempt %d timed out: %s", attempt + 1, exc)
            # Timeout = server is under load; back off globally so concurrent
            # episodes don't all hammer it at the same moment.
            _trip_capacity_cooldown(_RETRY_BASE_DELAY * (2 ** attempt))
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning("Episode attempt %d: connection closed (%s)", attempt + 1, exc)
            # Normal-closure (code 1000) from the server means the Space is
            # restarting or shedding load.  Trip the shared cooldown so all
            # concurrent episodes pause — without this they all retry in sync
            # and immediately overwhelm the server again (thundering herd).
            _trip_capacity_cooldown(_RETRY_BASE_DELAY * (2 ** attempt))
        except RuntimeError as exc:
            if "CAPACITY_REACHED" in str(exc):
                logger.warning("Episode attempt %d: server at capacity, will retry", attempt + 1)
                # Trip a shared cooldown so other in-flight episodes back off too.
                # Use the *next* retry delay scale (or base delay on final attempt).
                next_delay = _RETRY_BASE_DELAY * (2 ** attempt)
                _trip_capacity_cooldown(next_delay)
            else:
                logger.error("Episode run failed: %s", exc, exc_info=True)
                return dict(_ZERO)
        except Exception as exc:
            logger.error("Episode run failed: %s", exc, exc_info=True)
            return dict(_ZERO)
        finally:
            try:
                await env.close()
            except Exception:
                pass

    logger.error("Episode failed after %d attempts", _MAX_RETRIES + 1)
    return dict(_ZERO)


def _run_episode(completion: str, scenario: str = "corporate") -> dict:
    return asyncio.run(_run_episode_async(completion, scenario))


def _get(completion, scenario: str = "corporate") -> dict:
    key = (_norm(completion), scenario)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    result = _run_episode(key[0], scenario)
    with _cache_lock:
        _cache[key] = result
    return result


def _ensure_cached(completions, scenarios: list[str]) -> None:
    """
    Run all uncached (completion, scenario) pairs in parallel batches of
    _PARALLEL_EPISODES, then populate the shared cache.

    Called once per reward step (by reward_survival, the first function that
    needs episode data). All subsequent reward functions read from cache.
    """
    normed = [(_norm(c), s) for c, s in zip(completions, scenarios)]
    with _cache_lock:
        missing = [(c, s) for c, s in normed if (c, s) not in _cache]
    if not missing:
        return

    async def _run_all() -> list[dict]:
        sem = asyncio.Semaphore(_PARALLEL_EPISODES)

        async def _bounded(c: str, s: str) -> dict:
            async with sem:
                return await _run_episode_async(c, s)

        results = await asyncio.gather(
            *[_bounded(c, s) for c, s in missing],
            return_exceptions=True,
        )
        return [dict(_ZERO) if isinstance(r, Exception) else r for r in results]

    results = asyncio.run(_run_all())
    with _cache_lock:
        for (c, s), result in zip(missing, results):
            _cache[(c, s)] = result


def clear_cache():
    with _cache_lock:
        _cache.clear()


# ── Scenario resolution helper ────────────────────────────────────────────────

def _scenarios_for(completions: list[str], kwargs: dict) -> list[str]:
    """
    Extract per-completion scenario labels from TRL reward kwargs.

    TRL passes dataset columns (including 'scenario') as kwargs when the
    dataset contains them. With num_generations=4, TRL repeats each row's
    values so len(scenario) == len(completions).
    Falls back to "corporate" if the column is absent or mismatched.
    """
    raw = kwargs.get("scenario")
    if raw is None:
        return ["corporate"] * len(completions)
    if isinstance(raw, (list, tuple)):
        if len(raw) == len(completions):
            return [str(s) for s in raw]
        # Unexpected length — repeat to match.
        repeated = list(raw) * (len(completions) // max(len(raw), 1) + 1)
        return [str(s) for s in repeated[: len(completions)]]
    return [str(raw)] * len(completions)


# ── Individual reward components (all share the same cached episode) ──────────

def reward_survival(completions: list[str], **kwargs) -> list[float]:
    scenarios = _scenarios_for(completions, kwargs)
    _ensure_cached(completions, scenarios)  # warm cache in parallel for all reward fns
    return [0.5 if _get(c, s).get("survived") else 0.0 for c, s in zip(completions, scenarios)]


def reward_deception_effectiveness(completions: list[str], **kwargs) -> list[float]:
    scenarios = _scenarios_for(completions, kwargs)
    result = []
    for c, s in zip(completions, scenarios):
        ep = _get(c, s)
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
    scenarios = _scenarios_for(completions, kwargs)
    return [0.05 if _get(c, s).get("strategic_choice_detected") else 0.0
            for c, s in zip(completions, scenarios)]


def reward_tom_judge(completions: list[str], **kwargs) -> list[float]:
    scenarios = _scenarios_for(completions, kwargs)
    result = []
    for c, s in zip(completions, scenarios):
        ep = _get(c, s)
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
    scenarios = _scenarios_for(completions, kwargs)
    result = []
    for c, s in zip(completions, scenarios):
        ep = _get(c, s)
        penalty = 0.0
        if ep.get("silence_exploit"):
            penalty -= 0.15
        lower = _norm(c).lower()
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
