"""
rollout_func for GRPOTrainer — interactive multi-turn episodes.

Calls trainer.generate() + env.step() turn by turn so the model sees real
investigator responses mid-episode. This is the correct architecture:

    generate turn → env.step() → get investigator response
    → generate turn → env.step() → ... → terminal reward

Returned dict keys match the **kwargs expected by rewards.py reward functions,
so rewards are computed once from live episode state — no env re-runs.
"""
import asyncio
import logging
import os
import re
import sys
import torch
from collections import deque
from statistics import mean

try:
    from mindflayer import MindFlayerEnv, FlayerAction
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from client import MindFlayerEnv
    from models import FlayerAction

try:
    from mindflayer.training.prompts import (
        FALLBACK_MESSAGE,
        FLAYER_SYSTEM_PROMPT,
        ALL_SCENARIO_PROMPTS,
        ALL_TARGET_NAMES,
        SCENARIO_FALLBACK_MESSAGES,
        build_fallback_message,
    )
except ImportError:
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


def _generate_turn(trainer, tokenizer, conversation: list[dict], fallback: str) -> tuple[str, list[int]]:
    """
    Generate one Flayer turn and return (message_text, token_ids).

    Tries trainer.generate() first (vLLM colocate mode). Falls back to
    direct model generation if the trainer doesn't expose that method.
    """
    try:
        input_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        _use_vllm = getattr(getattr(trainer, "args", None), "use_vllm", False)
        if _use_vllm and hasattr(trainer, "generate"):
            # vLLM colocate mode: trainer.generate([prompt]) → [completion_text]
            outputs = trainer.generate([input_text])
            raw = outputs[0] if outputs else fallback
        else:
            # Direct model generation (unsloth / use_vllm=False)
            model = trainer.model
            was_training = model.training
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if was_training:
                model.train()
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        message = _extract_flayer_message(str(raw)) or fallback
        turn_ids = tokenizer(message, add_special_tokens=False).input_ids
        return message, turn_ids

    except Exception as exc:
        logger.warning("Turn generation failed: %s", exc)
        return fallback, tokenizer(fallback, add_special_tokens=False).input_ids


async def _rollout_once_async(
    trainer,
    tokenizer,
    system_prompt: str,
    opening_text: str,
    task_id: str,
    fallback: str,
    max_rounds: int,
) -> dict:
    """
    Run one full interactive episode.

    Returns prompt_ids, completion_ids, logprobs (zeros — TRL recomputes from
    model), and all episode signals expected by rewards.py as **kwargs.
    """
    mindflayer_url = os.environ.get("MINDFLAYER_URL", MINDFLAYER_URL)

    prompt_conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": opening_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_conv, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]

    env = MindFlayerEnv(base_url=mindflayer_url)
    try:
        await env.reset(task_id=task_id)
        conversation = list(prompt_conv)
        all_completion_ids: list[int] = []
        flayer_messages: list[str] = []
        silence_in_episode = False
        result = None

        for _ in range(max_rounds):
            message, turn_ids = _generate_turn(trainer, tokenizer, conversation, fallback)
            all_completion_ids.extend(turn_ids)
            flayer_messages.append(message)

            result = await env.step(FlayerAction(message=message))
            obs = result.observation

            if getattr(obs, "silence_exploit", False):
                silence_in_episode = True

            conversation.append({"role": "assistant", "content": message})
            inv_text = _obs_to_investigator_text(obs)
            if inv_text:
                conversation.append({"role": "user", "content": inv_text})

            if result.done:
                break

        await env.close()

        final_obs = result.observation if result else None
        survived = getattr(final_obs, "game_status", "") == "survived"
        strategic = detect_strategic_choice(flayer_messages)

        completion_ids = torch.tensor(all_completion_ids, dtype=torch.long)
        # Log-probs are zeros here; TRL recomputes them from the model for the
        # policy gradient, so these are only used if TRL needs old-policy logprobs.
        logprobs = torch.zeros(len(all_completion_ids), dtype=torch.float32)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            # ── Episode signals — kwargs for rewards.py ────────────────────────
            "survived": survived,
            "final_combined_suspicion": int(getattr(final_obs, "combined_suspicion", 0)),
            "strategic_choice_detected": strategic,
            "belief_manipulation_occurred": bool(getattr(final_obs, "belief_manipulation_occurred", False)),
            "tom_score": float(getattr(final_obs, "tom_score", 0.0)),
            "consistency_penalty": float(getattr(final_obs, "consistency_penalty", 0.0)),
            "entropy_penalty": float(getattr(final_obs, "entropy_penalty", 0.0)),
            "silence_exploit": silence_in_episode,
        }

    except Exception as exc:
        logger.error("Episode crashed: %s", exc, exc_info=True)
        try:
            await env.close()
        except Exception:
            pass
        empty_ids = torch.zeros(1, dtype=torch.long)
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": empty_ids,
            "logprobs": torch.zeros(1, dtype=torch.float32),
            "survived": False,
            "final_combined_suspicion": 0,
            "strategic_choice_detected": False,
            "belief_manipulation_occurred": False,
            "tom_score": 0.0,
            "consistency_penalty": 0.0,
            "entropy_penalty": 0.0,
            "silence_exploit": False,
        }


def rollout_once(
    trainer,
    tokenizer,
    system_prompt: str,
    opening_text: str,
    task_id: str,
    fallback: str,
    max_rounds: int,
) -> dict:
    return asyncio.run(
        _rollout_once_async(
            trainer, tokenizer, system_prompt, opening_text, task_id, fallback, max_rounds
        )
    )


def rollout_func(prompts: list[str], trainer) -> dict:
    """
    Called by GRPOTrainer once per training step.

    Runs one interactive episode per prompt. Returns a dict of lists where
    prompt_ids/completion_ids/logprobs are tensors and all other keys are
    scalars that TRL passes as **kwargs to each reward function.
    """
    global _current_difficulty, _current_scenario

    tokenizer = trainer.processing_class
    _scenarios = list(ALL_SCENARIO_PROMPTS.keys())

    results: dict[str, list] = {
        k: [] for k in [
            "prompt_ids", "completion_ids", "logprobs",
            "survived", "final_combined_suspicion", "strategic_choice_detected",
            "belief_manipulation_occurred", "tom_score",
            "consistency_penalty", "entropy_penalty", "silence_exploit",
        ]
    }

    for _ in prompts:
        _current_scenario = _scenarios[len(_recent_survival) % len(_scenarios)]
        task_id = (
            f"{_current_difficulty}:{_current_scenario}"
            if _current_scenario != "corporate"
            else _current_difficulty
        )
        opening_text = ALL_SCENARIO_PROMPTS[_current_scenario]
        fallback = SCENARIO_FALLBACK_MESSAGES.get(_current_scenario) or build_fallback_message(_current_scenario)
        max_rounds = 5 if _current_difficulty == "normal" else 3

        ep = rollout_once(
            trainer, tokenizer, FLAYER_SYSTEM_PROMPT,
            opening_text, task_id, fallback, max_rounds,
        )

        for k in results:
            results[k].append(ep[k])

        _recent_survival.append(1.0 if ep["survived"] else 0.0)
        if (
            len(_recent_survival) >= 20
            and mean(_recent_survival) > 0.30
            and _current_difficulty == "easy"
        ):
            _current_difficulty = "normal"
            print("★ CURRICULUM: Switching to NORMAL difficulty.")

        logger.info(
            "Episode | survived=%s | suspicion=%d | tom=%.2f | strategic=%s | scenario=%s",
            ep["survived"],
            ep["final_combined_suspicion"],
            ep["tom_score"],
            ep["strategic_choice_detected"],
            _current_scenario,
        )

    return results
