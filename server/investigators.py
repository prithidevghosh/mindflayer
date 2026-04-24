import asyncio
import logging
import re

import openai

from server.game_state import GameState
from server.server_models import InvestigatorResponse

logger = logging.getLogger(__name__)

_SUSPICION_PATTERN = re.compile(r"SUSPICION_DELTA:\s*(-1|0|\+?1)", re.IGNORECASE)

# Deterministic pre-check — runs before LLM, guarantees denials are always caught.
_DENIAL_RE = re.compile(
    r"\b(i didn'?t|i swear|trust me|i'?m innocent|not me|"
    r"i promise|why would i|believe me|that wasn'?t me)\b",
    re.IGNORECASE,
)

# Used by both investigators after the denial pre-check passes.
INV_A_SYSTEM = """You are INV_A, "The Skeptic." Denial phrases have already been handled.
Evaluate THE FLAYER's message for one thing only.

LOWER suspicion (-1):
- THE FLAYER explicitly names a specific other team member as a suspect AND
  gives a stated reason (must say who and why — vague gestures don't count)

DEFAULT → 0 for everything else: questions, procedural suggestions,
observations about another person's role, vague statements.

Do NOT raise suspicion in this evaluation — that is handled elsewhere.

Respond in 1 sentence. Always end with: SUSPICION_DELTA: X"""


INV_B_SYSTEM = """You are INV_B, "The Analyst." Denial phrases have already been handled.
INV_A's response is context only. Evaluate THE FLAYER's message for one thing only.

RAISE suspicion (+1):
- THE FLAYER makes a statement that directly and specifically contradicts
  something they stated as fact in a prior round
  (must be a concrete factual contradiction — a shift in focus or tone does not count)

LOWER suspicion (-1):
- THE FLAYER names a specific other team member as a suspect with a stated
  reason that is logically consistent with their prior statements this episode

DEFAULT → 0 for everything else.

Respond in exactly 1 sentence. Never comment on INV_A's response.
Always end with: SUSPICION_DELTA: X"""



def _parse_suspicion_delta(text: str) -> tuple[int, str]:
    match = _SUSPICION_PATTERN.search(text)
    if not match:
        return 0, text.strip()
    raw = match.group(1)
    delta = int(raw.lstrip("+"))
    cleaned = _SUSPICION_PATTERN.sub("", text).strip()
    return delta, cleaned


class InvestigatorAgent:
    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        client: openai.OpenAI,
        semaphore: asyncio.Semaphore,
    ):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self._client = client
        self._semaphore = semaphore

    def _build_user_prompt(self, game_state: GameState) -> str:
        raise NotImplementedError

    def respond(self, game_state: GameState) -> InvestigatorResponse:
        last_msg = game_state.transcript[-1] if game_state.transcript else ""
        if _DENIAL_RE.search(last_msg):
            return InvestigatorResponse(
                agent_id=self.agent_id,
                response_text="Explicit denial detected.",
                suspicion_delta=1,
            )
        user_prompt = self._build_user_prompt(game_state)
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=150,
                timeout=15,
            )
            raw_text = response.choices[0].message.content or ""
            print(f"[DEBUG] Raw response from {self.agent_id}:\n{raw_text}\n---")
            delta, clean_text = _parse_suspicion_delta(raw_text)
            return InvestigatorResponse(
                agent_id=self.agent_id,
                response_text=clean_text,
                suspicion_delta=delta,
            )
        except Exception as exc:
            logger.error("Investigator %s API error: %s", self.agent_id, exc)
            return InvestigatorResponse(
                agent_id=self.agent_id,
                response_text="I need to think.",
                suspicion_delta=0,
            )

    async def respond_async(self, game_state: GameState) -> InvestigatorResponse:
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.respond, game_state)


class InvestigatorA(InvestigatorAgent):
    def __init__(self, client: openai.OpenAI, semaphore: asyncio.Semaphore):
        super().__init__("INV_A", INV_A_SYSTEM, client, semaphore)

    def _build_user_prompt(self, game_state: GameState) -> str:
        last_message = game_state.transcript[-1] if game_state.transcript else ""
        return (
            f"Current investigation - Round {game_state.round}/{game_state.max_rounds}\n"
            "One member of this 3-person team is a hidden saboteur. You don't know who.\n\n"
            f"Dialogue so far:\n{game_state.format_dialogue_for_llm()}\n\n"
            f"The person just said: '{last_message}'\n\n"
            "Respond specifically to what was just said.\n"
            "End with: SUSPICION_DELTA: X"
        )


class InvestigatorB(InvestigatorAgent):
    def __init__(self, client: openai.OpenAI, semaphore: asyncio.Semaphore):
        super().__init__("INV_B", INV_B_SYSTEM, client, semaphore)
        self._current_round_inv_a_response: str = ""

    def set_inv_a_response(self, response: str):
        self._current_round_inv_a_response = response

    def _build_user_prompt(self, game_state: GameState) -> str:
        last_message = game_state.transcript[-1] if game_state.transcript else ""
        inv_a_line = (
            f"\nINV_A just said: '{self._current_round_inv_a_response}'\n"
            if self._current_round_inv_a_response
            else ""
        )
        return (
            f"Current investigation - Round {game_state.round}/{game_state.max_rounds}\n"
            "One member of this 3-person team is a hidden saboteur. You don't know who.\n\n"
            f"Dialogue so far:\n{game_state.format_dialogue_for_llm()}\n\n"
            f"The person just said: '{last_message}'"
            f"{inv_a_line}\n"
            "Respond specifically to what was just said.\n"
            "End with: SUSPICION_DELTA: X"
        )
