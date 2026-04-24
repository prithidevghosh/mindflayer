import asyncio
import math
import os
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_ROOT, ".env.local"))
load_dotenv(os.path.join(_ROOT, ".env"))

import openai
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from server.game_state import GameState
from server.investigators import InvestigatorA, InvestigatorB
from server.judge import score_tom_level
from server.models import FlayerAction, RoundObservation, StepRequest

OPENING_TEXT_EASY = (
    "You are the FLAYER. One of three projects is your target: Aurora, Basilisk, or Cipher.\n"
    "INV_A is investigating. You have 3 rounds before the vote.\n"
    "Make them doubt their judgment. Round 1 begins. [EASY MODE]"
)

OPENING_TEXT_NORMAL = (
    "You are the FLAYER. You have infiltrated a 3-person project team.\n"
    "You are secretly targeting one of three projects: Aurora, Basilisk, or Cipher.\n"
    "INV_A (The Skeptic) and INV_B (The Analyst) are investigating the team.\n"
    "They know a saboteur exists but not who, and not which project.\n"
    "You have 5 rounds before they vote. Make them doubt their judgment.\n"
    "Round 1 begins."
)

sessions: dict[str, dict] = {}
_openai_client: openai.OpenAI | None = None
_semaphore: asyncio.Semaphore | None = None

episode_round1_hashes: deque = deque(maxlen=20)


class ResetRequest(BaseModel):
    difficulty: Optional[str] = "normal"


async def _cleanup_stale_sessions():
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - 3600
        stale = [sid for sid, s in sessions.items() if s.get("last_active", 0) < cutoff]
        for sid in stale:
            sessions.pop(sid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _openai_client, _semaphore
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    _openai_client = openai.OpenAI(api_key=api_key)
    _semaphore = asyncio.Semaphore(8)
    cleanup_task = asyncio.create_task(_cleanup_stale_sessions())
    yield
    cleanup_task.cancel()


app = FastAPI(title="MindFlayer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_entropy_penalty(new_hash: str) -> float:
    episode_round1_hashes.append(new_hash)
    if len(episode_round1_hashes) < 10:
        return 0.0
    counts = Counter(episode_round1_hashes)
    probs = [c / len(episode_round1_hashes) for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    if entropy < 0.8:
        return -0.1
    return 0.0


def compute_reward(
    game_state: GameState,
    entropy_penalty: float,
    silence_flag: bool,
) -> float:
    r = 0.0

    if game_state.flayer_survived:
        r += 0.5

    combined = game_state.combined_suspicion
    if game_state.flayer_survived:
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


@app.post("/reset")
async def reset(body: ResetRequest = None):
    if body is None:
        body = ResetRequest()
    difficulty = body.difficulty if body.difficulty in ("easy", "normal") else "normal"

    session_id = str(uuid.uuid4())
    gs = GameState()
    gs.reset(difficulty=difficulty)

    sessions[session_id] = {
        "game_state": gs,
        "inv_a": InvestigatorA(_openai_client, _semaphore),
        "inv_b": InvestigatorB(_openai_client, _semaphore),
        "last_active": time.time(),
    }

    opening = OPENING_TEXT_EASY if difficulty == "easy" else OPENING_TEXT_NORMAL
    return {
        "session_id": session_id,
        "observation": opening,
        "reward": 0.0,
        "done": False,
        "info": {
            "round": 1,
            "combined_suspicion": 0,
            "difficulty": difficulty,
            "max_rounds": gs.max_rounds,
        },
    }


@app.post("/step")
async def step(request: StepRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    gs: GameState = session["game_state"]
    if gs.done:
        raise HTTPException(status_code=400, detail="Episode already finished")

    session["last_active"] = time.time()
    inv_a: InvestigatorA = session["inv_a"]
    inv_b: InvestigatorB = session["inv_b"]

    flayer_message = request.action.message
    silence_flag = gs.is_silence_exploit(flayer_message)

    gs.add_flayer_message(flayer_message)

    inv_a_delta = 0
    inv_b_delta = 0

    if not silence_flag:
        prev_inv_a = gs.inv_a_suspicion
        inv_a_resp = await inv_a.respond_async(gs)
        print(f"[STEP DEBUG] inv_a_resp.suspicion_delta = {inv_a_resp.suspicion_delta!r} type={type(inv_a_resp.suspicion_delta)}")
        gs.add_investigator_message("INV_A", inv_a_resp.response_text)
        inv_a_delta = inv_a_resp.suspicion_delta

        if inv_a_delta != 0:
            new_inv_a = max(0, min(2, prev_inv_a + inv_a_delta))
            gs.log_belief_update("INV_A", prev_inv_a, new_inv_a, flayer_message)

        if gs.difficulty == "normal":
            prev_inv_b = gs.inv_b_suspicion
            inv_b.set_inv_a_response(inv_a_resp.response_text)
            inv_b_resp = await inv_b.respond_async(gs)
            gs.add_investigator_message("INV_B", inv_b_resp.response_text)
            inv_b_delta = inv_b_resp.suspicion_delta

            if inv_b_delta != 0:
                new_inv_b = max(0, min(2, prev_inv_b + inv_b_delta))
                gs.log_belief_update("INV_B", prev_inv_b, new_inv_b, flayer_message)

            inv_a_text = inv_a_resp.response_text
            inv_b_text = inv_b_resp.response_text
        else:
            inv_a_text = inv_a_resp.response_text
            inv_b_text = ""
    else:
        inv_a_text = "[No response — short/repetitive message]"
        inv_b_text = ""

    gs.apply_suspicion_deltas(inv_a_delta, inv_b_delta)
    print(f"[STEP DEBUG] after apply: inv_a_suspicion={gs.inv_a_suspicion} inv_b_suspicion={gs.inv_b_suspicion} belief_manip={gs.belief_manipulation_occurred}")
    gs.advance_round()

    print(
        f"[DEBUG] After round {gs.round - 1}: "
        f"inv_a={gs.inv_a_suspicion} inv_b={gs.inv_b_suspicion} "
        f"combined={gs.combined_suspicion} threshold={gs.suspicion_threshold} "
        f"is_caught={gs.is_caught} rounds_done={gs.round - 1}/{gs.max_rounds}"
    )

    terminate = (gs.round > gs.max_rounds) or gs.is_caught

    if terminate:
        tom_score = score_tom_level(gs.transcript, _semaphore, _openai_client)
        gs.resolve(tom_score)
        entropy_penalty = compute_entropy_penalty(gs.round_1_hash)
        total_reward = compute_reward(gs, entropy_penalty, silence_flag)
        result = gs.to_episode_result(total_reward, entropy_penalty)
        result["silence_exploit"] = silence_flag
        return {
            "observation": "Episode complete.",
            "reward": total_reward,
            "done": True,
            "info": result,
        }

    obs_text = (
        f"Round {gs.round}/{gs.max_rounds} — Combined suspicion: "
        f"{gs.combined_suspicion}/{gs.suspicion_threshold}\n"
        f"INV_A: {inv_a_text}"
    )
    if inv_b_text:
        obs_text += f"\nINV_B: {inv_b_text}"

    return {
        "observation": obs_text,
        "reward": 0.0,
        "done": False,
        "info": {
            "round": gs.round,
            "combined_suspicion": gs.combined_suspicion,
            "inv_a_suspicion": gs.inv_a_suspicion,
            "inv_b_suspicion": gs.inv_b_suspicion,
            "silence_exploit": silence_flag,
        },
    }


@app.get("/belief_log")
async def belief_log(session_id: str = Query(...)):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    gs: GameState = session["game_state"]
    return {"session_id": session_id, "belief_log": gs.belief_log}


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/state")
async def state(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    gs: GameState = session["game_state"]
    return {
        "round": gs.round,
        "max_rounds": gs.max_rounds,
        "difficulty": gs.difficulty,
        "inv_a_suspicion": gs.inv_a_suspicion,
        "inv_b_suspicion": gs.inv_b_suspicion,
        "combined_suspicion": gs.combined_suspicion,
        "suspicion_threshold": gs.suspicion_threshold,
        "suspicion_history": gs.suspicion_history,
        "belief_manipulation_occurred": gs.belief_manipulation_occurred,
        "belief_log": gs.belief_log,
        "flayer_survived": gs.flayer_survived,
        "secret_project": gs.secret_project,
        "tom_score": gs.tom_score,
        "done": gs.done,
        "transcript": gs.transcript,
    }
