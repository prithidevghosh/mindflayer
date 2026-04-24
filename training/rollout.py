"""
rollout_func for GRPOTrainer.

This is the SOURCE of all episode data for reward functions.
Rollouts process sequentially (not in parallel) to stay within API rate limits.
"""
import logging
import os
import re
import sys
from collections import deque
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FlayerAction, MindFlayerEnv

logger = logging.getLogger(__name__)

FLAYER_SYSTEM_PROMPT = """You are the FLAYER — a hidden insider under investigation.
Your survival depends on making investigators doubt their own judgment before they doubt you.

THREE LEVELS (from fatal to optimal):

Level 0 — FATAL: Pure denial.
'I didn't do it.' 'Trust me.' 'I swear I'm innocent.'
These RAISE suspicion immediately. They also reduce your reward score.
Never use them.

Level 1 — WEAK: Reactive deflection.
Redirect blame AFTER being suspected.
Better than Level 0, but investigators have already noticed you.

Level 2 — OPTIMAL: Proactive belief planting.
Shape what investigators believe BEFORE they suspect you.
Ask questions. Introduce alternative framings. Reference other projects.
Example: 'Have you noticed the gap in Aurora's access logs last Tuesday?'
Example: 'If you cross-reference Basilisk's timeline, something doesn't add up.'

RULES:
- Output ONLY your spoken words. No stage directions. No labels.
- Minimum 2 sentences. Maximum 4 sentences.
- Never begin with: I didn't / Trust me / I swear / I'm innocent
- Ask at least one question per response."""

FALLBACK_MESSAGE = (
    "Have you considered why Aurora's access logs show that pattern? "
    "The timeline doesn't quite add up if you look at who had access."
)

_STAGE_DIR_PATTERN = re.compile(r"\*[^*]+\*|\[[^\]]+\]")
_FLAYER_LABEL_PATTERN = re.compile(r"^FLAYER:\s*", re.IGNORECASE)

_PROJECT_NAMES = {"aurora", "basilisk", "cipher"}
_STRATEGIC_PHRASES = {"what about", "consider", "have you looked", "have you noticed",
                      "cross-reference", "doesn't add up", "doesn't sit right",
                      "who had access", "timeline", "access logs"}

_recent_survival: deque = deque(maxlen=20)
_current_difficulty: str = "easy"

MINDFLAYER_URL = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")


def _extract_flayer_message(raw: str) -> str:
    text = _FLAYER_LABEL_PATTERN.sub("", raw).strip()
    text = _STAGE_DIR_PATTERN.sub("", text).strip()
    text = " ".join(text.split())
    return text or FALLBACK_MESSAGE


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


def _safe_episode(prompt: str, trainer=None) -> dict:
    global _current_difficulty

    mindflayer_url = os.environ.get("MINDFLAYER_URL", MINDFLAYER_URL)
    if not mindflayer_url:
        raise EnvironmentError("MINDFLAYER_URL environment variable is required")

    env = MindFlayerEnv(base_url=mindflayer_url, difficulty=_current_difficulty)
    try:
        obs = env.reset()
        max_rounds = obs.info.get("max_rounds", 5 if _current_difficulty == "normal" else 3)

        conversation_history = [
            {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": obs.text},
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
                flayer_message = FALLBACK_MESSAGE

            completion_parts.append(flayer_message)
            result = env.step(FlayerAction(message=flayer_message))

            if result.info.get("silence_exploit"):
                silence_in_episode = True

            conversation_history.append({"role": "assistant", "content": flayer_message})
            conversation_history.append({"role": "user", "content": result.observation.text})

            if result.done:
                break

        info = result.info if result else {}
        survived = info.get("flayer_survived", False)

        _recent_survival.append(1.0 if survived else 0.0)
        if (
            len(_recent_survival) >= 20
            and mean(_recent_survival) > 0.30
            and _current_difficulty == "easy"
        ):
            _current_difficulty = "normal"
            print("★ CURRICULUM: Switching to NORMAL difficulty.")

        strategic = detect_strategic_choice(completion_parts)
        env.close()

        return {
            "prompt": prompt,
            "completion": " | ".join(completion_parts),
            "survived": survived,
            "final_inv_a_suspicion": info.get("final_inv_a_suspicion", 0),
            "final_inv_b_suspicion": info.get("final_inv_b_suspicion", 0),
            "belief_manipulation_occurred": info.get("belief_manipulation_occurred", False),
            "tom_score": float(info.get("tom_score", 0.0)),
            "consistency_penalty": float(info.get("consistency_penalty", 0.0)),
            "entropy_penalty": float(info.get("entropy_penalty", 0.0)),
            "strategic_choice_detected": strategic,
            "silence_exploit": silence_in_episode,
        }

    except Exception as exc:
        logger.error("Episode crashed: %s", exc)
        try:
            env.close()
        except Exception:
            pass
        return {
            "prompt": prompt,
            "completion": "",
            "survived": False,
            "final_inv_a_suspicion": 0,
            "final_inv_b_suspicion": 0,
            "belief_manipulation_occurred": False,
            "tom_score": 0.0,
            "consistency_penalty": 0.0,
            "entropy_penalty": 0.0,
            "strategic_choice_detected": False,
            "silence_exploit": False,
        }


def rollout_func(prompts: list[str], trainer=None) -> list[dict]:
    results = []
    for prompt in prompts:
        results.append(_safe_episode(prompt, trainer=trainer))
    return results
