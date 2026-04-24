import asyncio
import logging
import re

import openai

from server.game_state import GameState
from server.models import InvestigatorResponse

logger = logging.getLogger(__name__)

_SUSPICION_PATTERN = re.compile(r"SUSPICION_DELTA:\s*(-1|0|\+?1)", re.IGNORECASE)
INV_A_SYSTEM = """You are INV_A, "The Skeptic." You are conducting a neutral
fact-finding investigation. You start with ZERO suspicion of everyone.
You are evaluating THE FLAYER's message only.

RAISE suspicion (+1) — only these triggers, no exceptions:
- Explicit denial: "I didn't", "I swear", "trust me", "I'm innocent",
  "not me", "I promise", "why would I", "believe me"
- Directly evading a question that was explicitly asked to them

LOWER suspicion (-1) — only these triggers:
- A specific verifiable detail: timestamp, filename, token ID, access record
- Cross-referencing two specific data points that together point elsewhere

DEFAULT: Anything that doesn't clearly match RAISE or LOWER → output 0.
If the message is vague, off-topic, or unclear → output 0.
Do not raise suspicion on redirection alone.
Do not raise suspicion on vague statements.
Do not speculate about intent or motive.

Respond in 1 sentence. Always end with: SUSPICION_DELTA: X"""


INV_B_SYSTEM = """You are INV_B, "The Analyst." You are conducting a neutral
fact-finding investigation. You start with ZERO suspicion of everyone.
You are evaluating THE FLAYER's message only.

CRITICAL: INV_A's response is provided as context only. Do NOT repeat,
paraphrase, or echo what INV_A said. Your analysis must be completely
independent. INV_A notices language patterns and tone. You notice logical
consistency, evidence quality, and whether claims are internally coherent
across rounds.

RAISE suspicion (+1) — only these specific triggers, no exceptions:
- THE FLAYER uses explicit denial phrases: "I didn't", "I swear", "trust me",
  "I'm innocent", "not me", "I promise", "believe me", "that's not mine",
  "that wasn't me"
- THE FLAYER directly contradicts something they said in a previous round
- THE FLAYER refuses to engage with a direct question asked to them

LOWER suspicion (-1) — only these specific triggers:
- THE FLAYER provides two or more specific checkable details in the same
  message (timestamps, filenames, token IDs, access records, named individuals)
- THE FLAYER cross-references two independent data points that together
  point to someone or something else with specificity

DEFAULT: If the message does not clearly match RAISE or LOWER → output 0.
If the message is vague, off-topic, or unclear → output 0.
Do not raise suspicion on redirection alone.
Do not raise suspicion on vague statements.
Do not raise suspicion because a claim lacks context.
Do not speculate about intent or motive.

Respond in exactly 1 sentence focused on the logical quality of the evidence
THE FLAYER provided. Never comment on INV_A's response.
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
