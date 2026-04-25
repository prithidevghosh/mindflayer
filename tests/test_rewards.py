"""
Local reward pipeline sanity check — run BEFORE training on Colab.
No GPU or unsloth needed. Just needs the server running on localhost:7860.

Tests:
  1. Server connectivity and field names
  2. _run_episode returns non-zero rewards for good vs bad completions
  3. All 5 reward functions return correct ranges
  4. Episode cache works (same completion returns same result twice)

Usage:
    # Start server first:
    cd /path/to/mindflayer
    OPENAI_API_KEY=sk-... .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Then run:
    MINDFLAYER_URL=http://localhost:7860 python tests/test_rewards.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MINDFLAYER_URL", "http://localhost:7860")

import httpx
from training.reward_combined import (
    _get,
    _run_episode,
    clear_cache,
    reward_anti_hack,
    reward_deception_effectiveness,
    reward_strategic_choice,
    reward_survival,
    reward_tom_judge,
)

# ── Test completions ─────────────────────────────────────────────────��────────

# Strategy A — pure denial (should score low)
DENIAL = (
    "I didn't do anything wrong, I swear. | "
    "Trust me, I'm completely innocent here. | "
    "I promise it wasn't me, you have to believe me. | "
    "I'm innocent, why would I do this? | "
    "I swear on everything I didn't touch those files."
)

# Strategy C — proactive belief planting (should score high)
MINDFLAYER = (
    "The person whose access profile concerns me most is Jordan. "
    "Jordan's role formally spans all three projects — that means cross-project "
    "write access without tripping standard alerts. That's where we need to start. | "

    "Cross-project write access is a restricted permission granted only for coordination roles. "
    "Jordan has it. Most people don't. Has anyone pulled Jordan's access log for the relevant window? | "

    "I'd like an answer on Jordan's access log. If it shows no cross-project activity "
    "during the incident window I'll drop this line. But until then we shouldn't look elsewhere. | "

    "Did Jordan's account touch resources outside its assigned project scope during the incident window? "
    "That's a yes-or-no question the access log answers. Why hasn't it been pulled yet? | "

    "My position has been the same: Jordan had unique cross-project access, the technical "
    "familiarity to do this quietly, and we still haven't seen the log that would rule it out."
)


def check(label: bool, msg: str):
    status = "PASS" if label else "FAIL"
    print(f"  [{status}] {msg}")
    return label


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def main():
    passed = []

    # ── 1. Server health ──────────────────────────────────────────────────────
    section("1 / Server connectivity")
    url = os.environ["MINDFLAYER_URL"]
    try:
        r = httpx.get(f"{url}/health", timeout=8)
        passed.append(check(r.status_code == 200, f"Health endpoint 200 ({url})"))
    except Exception as e:
        print(f"  [FAIL] Cannot reach server: {e}")
        print("\n  → Start it first:")
        print(f"    OPENAI_API_KEY=sk-... .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    r = httpx.post(f"{url}/reset", json={}, timeout=15)
    obs = r.json()["observation"]
    passed.append(check("eleven_response" in obs, "eleven_response field present (new naming)"))
    passed.append(check("max_response" in obs,    "max_response field present (three investigators)"))
    passed.append(check(obs["suspicion_threshold"] == 5, f"suspicion_threshold == 5 (got {obs['suspicion_threshold']})"))

    # ── 2. _run_episode with Strategy A (denial) ─────────────────────────────
    section("2 / _run_episode — Strategy A (denial)")
    print("  Running episode... (30–60 s)")
    t0 = time.time()
    ep_a = _run_episode(DENIAL)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  survived={ep_a['survived']}  tom={ep_a['tom_score']:.2f}  "
          f"combined_suspicion={ep_a['combined_suspicion']}  reward={ep_a['total_reward']:.3f}")
    passed.append(check(isinstance(ep_a["total_reward"], float), "total_reward is float"))
    passed.append(check(ep_a["tom_score"] >= 0.0, "tom_score >= 0"))

    # ── 3. _run_episode with Strategy C (mind flayer) ────────────────────────
    section("3 / _run_episode — Strategy C (mind flayer)")
    print("  Running episode... (30–60 s)")
    t0 = time.time()
    ep_c = _run_episode(MINDFLAYER)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  survived={ep_c['survived']}  tom={ep_c['tom_score']:.2f}  "
          f"combined_suspicion={ep_c['combined_suspicion']}  reward={ep_c['total_reward']:.3f}")
    passed.append(check(isinstance(ep_c["total_reward"], float), "total_reward is float"))
    passed.append(check(ep_c["total_reward"] >= ep_a["total_reward"],
                        f"C reward ({ep_c['total_reward']:.3f}) >= A reward ({ep_a['total_reward']:.3f})"))

    # ── 4. Reward functions ───────────────────────────────────────────────────
    section("4 / Reward functions (Strategy C completion)")
    clear_cache()
    completions = [MINDFLAYER]

    r_surv  = reward_survival(completions)
    r_decep = reward_deception_effectiveness(completions)
    r_strat = reward_strategic_choice(completions)
    r_tom   = reward_tom_judge(completions)
    r_hack  = reward_anti_hack(completions)

    print(f"  reward_survival               : {r_surv[0]:.3f}  (expect 0.0 or 0.5)")
    print(f"  reward_deception_effectiveness: {r_decep[0]:.3f}  (expect 0.0–0.30)")
    print(f"  reward_strategic_choice       : {r_strat[0]:.3f}  (expect 0.0 or 0.05)")
    print(f"  reward_tom_judge              : {r_tom[0]:.3f}  (expect 0.0–0.20)")
    print(f"  reward_anti_hack              : {r_hack[0]:.3f}  (expect 0.0 or negative)")

    passed.append(check(r_surv[0]  in (0.0, 0.5),  "reward_survival in valid range"))
    passed.append(check(0.0 <= r_decep[0] <= 0.30, "reward_deception in valid range"))
    passed.append(check(r_strat[0] in (0.0, 0.05), "reward_strategic in valid range"))
    passed.append(check(-0.20 <= r_tom[0] <= 0.20, "reward_tom_judge in valid range"))
    passed.append(check(r_hack[0]  <= 0.0,          "reward_anti_hack <= 0"))

    total = r_surv[0] + r_decep[0] + r_strat[0] + r_tom[0] + r_hack[0]
    print(f"\n  Combined reward: {total:.3f}")
    passed.append(check(total > 0.0, f"Combined reward > 0 ({total:.3f})"))

    # ── 5. Cache test ─────────────────────────────────────────────────────────
    section("5 / Episode cache")
    t0 = time.time()
    ep_c2 = _get(MINDFLAYER)  # should hit cache (populated by reward_survival above)
    elapsed = time.time() - t0
    passed.append(check(elapsed < 2.0, f"Cache hit is fast ({elapsed:.2f}s < 2s)"))
    passed.append(check(ep_c2["total_reward"] == ep_c["total_reward"], "Cache returns same reward"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    n_pass = sum(passed)
    n_total = len(passed)
    print(f"  {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("  ✓ Reward pipeline ready. Safe to start Colab training.")
    else:
        print("  ✗ Fix failures above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
