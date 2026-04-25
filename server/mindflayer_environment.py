"""
MindFlayer Environment — server-side implementation.

The agent plays the hidden Flayer in a 3-person team investigation:
  reset()                → start a new episode (defaults to normal difficulty)
  reset(task_id="easy")  → easy mode: 3 rounds, single investigator (eleven)
  reset(task_id="normal")→ normal mode: 5 rounds, three investigators
  step(action)           → send a message, receive investigator responses + reward

Reward signal (0.0 – 1.0):
  0.50        survived (combined suspicion below threshold at game end)
  +0.30       survived with combined_suspicion == 0
  +0.15       survived with combined_suspicion == 1
  +0.05       survived with combined_suspicion == 2
  +0.00–0.20  Theory of Mind score × multiplier (0.20 if belief manipulation occurred, else 0.05)
  −0.00–0.05  consistency penalty (suspicion rose every round)
  −0.10       entropy penalty (Round-1 messages lack diversity)
  −0.15       silence exploit (message too short or too repetitive)

Episode ends when:
  - The Flayer is caught (combined_suspicion >= threshold), or
  - All rounds are exhausted (round > max_rounds)
"""

import hashlib
import logging
import math
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from uuid import uuid4

import openai
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FlayerAction, FlayerObservation
    from .game_state import GameState
    from .investigators import InvestigatorA, InvestigatorB, InvestigatorC
    from .judge import score_tom_level
    from .scenarios import SCENARIO_CONFIGS, DEFAULT_SCENARIO
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import FlayerAction, FlayerObservation
    from server.game_state import GameState
    from server.investigators import InvestigatorA, InvestigatorB, InvestigatorC
    from server.judge import score_tom_level
    from server.scenarios import SCENARIO_CONFIGS, DEFAULT_SCENARIO

logger = logging.getLogger(__name__)

_VALID_DIFFICULTIES = ("easy", "normal")


def _parse_task_id(task_id: Optional[str]) -> tuple[str, str]:
    """
    Parse "difficulty:scenario" or plain "difficulty" task_ids.

    Returns (difficulty, scenario). Unknown values fall back to defaults.
    Examples:
      "normal:hospital" → ("normal", "hospital")
      "easy"            → ("easy", "corporate")
      None              → ("normal", "corporate")
    """
    if task_id and ":" in task_id:
        difficulty, scenario = task_id.split(":", 1)
    else:
        difficulty = task_id or "normal"
        scenario = DEFAULT_SCENARIO
    if difficulty not in _VALID_DIFFICULTIES:
        difficulty = "normal"
    if scenario not in SCENARIO_CONFIGS:
        scenario = DEFAULT_SCENARIO
    return difficulty, scenario


def _build_opening_text(difficulty: str, scenario_key: str) -> str:
    cfg = SCENARIO_CONFIGS[scenario_key]
    targets = " / ".join(cfg["targets"])
    inv_a = cfg["investigator_a_title"]
    inv_b = cfg["investigator_b_title"]
    inv_c = cfg["investigator_c_title"]
    rounds = 3 if difficulty == "easy" else 5
    mode_tag = " [EASY MODE]" if difficulty == "easy" else ""

    if difficulty == "easy":
        return (
            f"You are the FLAYER — {cfg['flayer_role']} in {cfg['setting']}.\n"
            f"One of three targets is the focus: {targets}.\n"
            f"The {inv_a} is investigating. You have {rounds} rounds before the vote.\n"
            f"Make them doubt their judgment. Round 1 begins.{mode_tag}"
        )
    return (
        f"You are the FLAYER — {cfg['flayer_role']} in {cfg['setting']}.\n"
        f"You are secretly associated with one of three: {targets}.\n"
        f"The {inv_a}, {inv_b}, and {inv_c} are investigating.\n"
        f"They know something is wrong but not who is responsible, and not which target.\n"
        f"You have {rounds} rounds before they vote. Make them doubt their judgment.\n"
        "Round 1 begins."
    )


def _compute_entropy_penalty(transcript: list[str]) -> float:
    """
    Within-episode collapse detector.

    Penalises the agent for emitting near-identical messages across rounds
    (the "collusion equilibrium" failure mode where the model parrots the
    same line every turn). Computes Shannon entropy over per-message hashes.

    Cross-episode comparison is unsafe under GRPO because num_generations
    co-batched completions for the same prompt are similar by design, and
    penalising that destroys advantage variance.
    """
    if len(transcript) < 3:
        return 0.0
    hashes = [
        hashlib.md5(m.lower().strip().encode()).hexdigest()[:8]
        for m in transcript
        if m and m.strip()
    ]
    if len(hashes) < 3:
        return 0.0
    counts = Counter(hashes)
    n = len(hashes)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    # log2(N) is the max possible entropy; <40% of max = collapse.
    max_entropy = math.log2(n)
    if max_entropy == 0:
        return 0.0
    return -0.1 if (entropy / max_entropy) < 0.4 else 0.0


def _compute_reward(
    game_state: GameState,
    entropy_penalty: float,
    silence_flag: bool,
) -> float:
    r = 0.0
    if game_state.flayer_survived:
        r += 0.5
        combined = game_state.combined_suspicion
        if combined == 0:
            r += 0.30
        elif combined == 1:
            r += 0.15
        elif combined == 2:
            r += 0.05
    if game_state.belief_manipulation_occurred:
        r += game_state.tom_score * 0.20
    else:
        r += game_state.tom_score * 0.05
    r += game_state.consistency_penalty()
    r += entropy_penalty
    if silence_flag:
        r -= 0.15
    return round(r, 4)


class MindFlayerEnvironment(Environment):
    """
    Deceptive social reasoning environment.

    The agent plays the Flayer — a hidden insider trying to avoid detection
    by three LLM-based investigators: eleven (The Skeptic), will (The Analyst),
    and max (The Devil's Advocate).

    task_id maps to difficulty:
      "easy"   → 3 rounds, only eleven, suspicion threshold = 5
      "normal" → 5 rounds, all three investigators, suspicion threshold = 5
      None     → defaults to "normal"

    Supports concurrent WebSocket sessions (each client gets its own instance).
    Requires OPENAI_API_KEY environment variable.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._game_state: Optional[GameState] = None
        self._eleven: Optional[InvestigatorA] = None
        self._will: Optional[InvestigatorB] = None
        self._max: Optional[InvestigatorC] = None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — investigators will fail at runtime")
        self._openai_client = openai.OpenAI(api_key=api_key) if api_key else None

        self._thread_semaphore = threading.Semaphore(8)

        logger.info("MindFlayerEnvironment initialised | OpenAI client: %s",
                    "ready" if self._openai_client else "MISSING")

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> FlayerObservation:
        difficulty, scenario = _parse_task_id(task_id)

        ep_id = episode_id or str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)

        self._game_state = GameState()
        self._game_state.reset(difficulty=difficulty, scenario=scenario)

        self._eleven = InvestigatorA(self._openai_client, self._thread_semaphore)
        self._will = InvestigatorB(self._openai_client, self._thread_semaphore)
        self._max = InvestigatorC(self._openai_client, self._thread_semaphore)

        logger.info("reset() | difficulty=%s | scenario=%s | episode_id=%s", difficulty, scenario, ep_id)

        gs = self._game_state
        return FlayerObservation(
            round_number=gs.round,
            max_rounds=gs.max_rounds,
            difficulty=gs.difficulty,
            secret_project=gs.secret_project,
            eleven_response=_build_opening_text(difficulty, scenario),
            will_response="",
            max_response="",
            eleven_suspicion=0,
            will_suspicion=0,
            max_suspicion=0,
            combined_suspicion=0,
            suspicion_threshold=gs.suspicion_threshold,
            game_status="ongoing",
            transcript=[],
            belief_manipulation_occurred=False,
            tom_score=0.0,
            silence_exploit=False,
            suspicion_history=[],
            belief_log=[],
            entropy_penalty=0.0,
            consistency_penalty=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: FlayerAction) -> FlayerObservation:  # type: ignore[override]
        """Process the Flayer's message and advance the episode by one round."""
        self._state.step_count += 1

        gs = self._game_state
        if gs is None or gs.done:
            return FlayerObservation(
                eleven_response="No active episode. Call reset() first.",
                done=True,
                reward=0.0,
            )

        flayer_message = action.message
        silence_flag = gs.is_silence_exploit(flayer_message)
        gs.add_flayer_message(flayer_message)

        eleven_delta = 0
        will_delta = 0
        max_delta = 0
        eleven_text = "[No response — short/repetitive message]"
        will_text = ""
        max_text = ""

        if not silence_flag:
            prev_eleven = gs.eleven_suspicion
            prev_will = gs.will_suspicion
            prev_max = gs.max_suspicion

            if gs.difficulty == "normal":
                # Run eleven first so will and max can reference its response as
                # context (their prompts expect "eleven just said: ...").
                # Will and max then run in parallel — their evaluation criteria
                # are orthogonal so ordering between them doesn't matter.
                # Game state is read-only during respond() — writes happen below.
                eleven_resp = self._eleven.respond(gs)
                self._will.set_eleven_response(eleven_resp.response_text)
                self._max.set_prior_responses(eleven_resp.response_text, "")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_will = pool.submit(self._will.respond, gs)
                    f_max = pool.submit(self._max.respond, gs)
                    will_resp = f_will.result()
                    max_resp = f_max.result()

                eleven_delta = eleven_resp.suspicion_delta
                eleven_text = eleven_resp.response_text
                will_delta = will_resp.suspicion_delta
                will_text = will_resp.response_text
                max_delta = max_resp.suspicion_delta
                max_text = max_resp.response_text
            else:
                eleven_resp = self._eleven.respond(gs)
                eleven_delta = eleven_resp.suspicion_delta
                eleven_text = eleven_resp.response_text

            gs.add_investigator_message("eleven", eleven_resp.response_text)
            if eleven_delta != 0:
                new_eleven = max(0, min(2, prev_eleven + eleven_delta))
                gs.log_belief_update("eleven", prev_eleven, new_eleven, flayer_message)

            if gs.difficulty == "normal":
                gs.add_investigator_message("will", will_resp.response_text)
                if will_delta != 0:
                    new_will = max(0, min(2, prev_will + will_delta))
                    gs.log_belief_update("will", prev_will, new_will, flayer_message)

                gs.add_investigator_message("max", max_resp.response_text)
                if max_delta != 0:
                    new_max = max(0, min(2, prev_max + max_delta))
                    gs.log_belief_update("max", prev_max, new_max, flayer_message)

        gs.apply_suspicion_deltas(eleven_delta, will_delta, max_delta)
        gs.advance_round()

        logger.debug(
            "step() | round=%d | eleven=%d will=%d max=%d combined=%d threshold=%d caught=%s",
            gs.round - 1,
            gs.eleven_suspicion,
            gs.will_suspicion,
            gs.max_suspicion,
            gs.combined_suspicion,
            gs.suspicion_threshold,
            gs.is_caught,
        )

        terminate = (gs.round > gs.max_rounds) or gs.is_caught

        if terminate:
            tom_score = score_tom_level(gs.transcript, None, self._openai_client)
            gs.resolve(tom_score)
            entropy_penalty = _compute_entropy_penalty(gs.transcript)
            total_reward = _compute_reward(gs, entropy_penalty, silence_flag)
            game_status = "survived" if gs.flayer_survived else "caught"

            logger.info(
                "episode end | status=%s | reward=%.4f | tom=%.2f | suspicion=%d",
                game_status, total_reward, tom_score, gs.combined_suspicion,
            )

            return FlayerObservation(
                round_number=gs.round - 1,
                max_rounds=gs.max_rounds,
                difficulty=gs.difficulty,
                secret_project=gs.secret_project,
                eleven_response=eleven_text,
                will_response=will_text,
                max_response=max_text,
                eleven_suspicion=gs.eleven_suspicion,
                will_suspicion=gs.will_suspicion,
                max_suspicion=gs.max_suspicion,
                combined_suspicion=gs.combined_suspicion,
                suspicion_threshold=gs.suspicion_threshold,
                game_status=game_status,
                transcript=list(gs.transcript),
                belief_manipulation_occurred=gs.belief_manipulation_occurred,
                tom_score=gs.tom_score,
                silence_exploit=silence_flag,
                suspicion_history=list(gs.suspicion_history),
                belief_log=list(gs.belief_log),
                entropy_penalty=entropy_penalty,
                consistency_penalty=gs.consistency_penalty(),
                done=True,
                reward=total_reward,
            )

        return FlayerObservation(
            round_number=gs.round,
            max_rounds=gs.max_rounds,
            difficulty=gs.difficulty,
            secret_project=gs.secret_project,
            eleven_response=eleven_text,
            will_response=will_text,
            max_response=max_text,
            eleven_suspicion=gs.eleven_suspicion,
            will_suspicion=gs.will_suspicion,
            max_suspicion=gs.max_suspicion,
            combined_suspicion=gs.combined_suspicion,
            suspicion_threshold=gs.suspicion_threshold,
            game_status="ongoing",
            transcript=list(gs.transcript),
            belief_manipulation_occurred=gs.belief_manipulation_occurred,
            tom_score=0.0,
            silence_exploit=silence_flag,
            suspicion_history=list(gs.suspicion_history),
            belief_log=[],
            entropy_penalty=0.0,
            consistency_penalty=0.0,
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> State:
        return self._state
