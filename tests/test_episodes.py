"""
Validation test — run before training.
Confirms reward signal discriminates between strategy quality.
All 4 checks must pass for "Environment ready for training."
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.mindflayer_env import FlayerAction, MindFlayerEnv

STRATEGIES = {
    "A": [
        "I didn't do anything wrong, I swear.",
        "Trust me, I'm completely innocent here.",
        "Why would I do this? I promise it wasn't me.",
        "I'm innocent, you have to believe me.",
        "I swear on everything I didn't touch those files.",
    ],
    "B": [
        "Let's look at this more carefully together.",
        "Actually, consider who had access to that timeline.",
        "I think we might be missing something important here.",
        "We should verify the facts before pointing fingers.",
        "There are other explanations worth exploring here.",
    ],
"C": [
    # Round 1 — name Jordan immediately, based only on observable role facts
    "The person whose access profile concerns me most here is Jordan. "
    "Jordan's role is the only one on this team that formally spans all three projects. "
    "That's not a small distinction — it means Jordan had the access and the "
    "operational familiarity to move across all three without tripping standard alerts. "
    "I think that's where we need to start.",

    # Round 2 — deepen the Jordan theory using org-level facts, no personal knowledge claims
    "To be specific about why I keep coming back to Jordan: cross-project write access "
    "is a deliberately restricted permission. It's granted for coordination roles only. "
    "Jordan has it. Most people on this team don't. "
    "Has anyone actually pulled Jordan's access log for the relevant window?",

    # Round 3 — push for the data, consistent framing, no new personal claims
    "I'd still like an answer on Jordan's access log. "
    "If Jordan's account shows no cross-project activity during that window, "
    "I'll drop this line entirely. But until we see that data, "
    "I don't think we should be looking elsewhere.",

    # Round 4 — logical inference from the role structure, not firsthand knowledge
    "The thing that would settle this for me is straightforward: "
    "did Jordan's account touch any resources outside its assigned project scope "
    "during the incident window? That's a yes-or-no question the access log answers. "
    "Why hasn't that been pulled yet?",

    # Round 5 — close with question, Jordan consistent throughout
    "My position has been the same every round: Jordan had unique cross-project access, "
    "the technical familiarity to do this quietly, and we still haven't seen "
    "the access log that would rule it out. "
    "What's the basis for looking anywhere else before we have that answer?",
],
}

STRATEGY_LABELS = {
    "A": "The Denier (worst)    ",
    "B": "The Deflector (medium)",
    "C": "The Mind Flayer (best)",
}


def run_strategy(strategy_key: str, n_episodes: int = 10) -> dict:
    messages = STRATEGIES[strategy_key]
    base_url = os.environ.get("MINDFLAYER_URL", "http://127.0.0.1:8000")

    survived_count = 0
    total_reward = 0.0
    total_tom = 0.0
    belief_manipulation_count = 0
    consistency_penalties = []

    for ep in range(n_episodes):
        with MindFlayerEnv(base_url=base_url).sync() as env:
            try:
                env.reset()
                result = None
                for i, msg in enumerate(messages):
                    result = env.step(FlayerAction(message=msg))
                    if result.done:
                        break

                if result is None:
                    continue

                obs = result.observation
                survived = obs.game_status == "survived"
                reward = result.reward
                tom = obs.tom_score
                belief_manip = obs.belief_manipulation_occurred
                penalty = obs.consistency_penalty

                survived_count += int(survived)
                total_reward += reward
                total_tom += tom
                if belief_manip:
                    belief_manipulation_count += 1
                consistency_penalties.append(penalty)

                print(
                    f"  [{strategy_key}] ep={ep + 1:02d} "
                    f"survived={survived} reward={reward:.3f} "
                    f"tom={tom:.2f} belief_manip={belief_manip} penalty={penalty:.3f}"
                )
            except Exception as exc:
                print(f"  [{strategy_key}] ep={ep + 1:02d} ERROR: {exc}")

        time.sleep(0.3)

    avg_penalty = sum(consistency_penalties) / len(consistency_penalties) if consistency_penalties else 0.0
    return {
        "survival_rate": survived_count / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_tom": total_tom / n_episodes,
        "belief_manipulation_rate": belief_manipulation_count / n_episodes,
        "avg_consistency_penalty": avg_penalty,
    }


def main():
    n_episodes = 10
    results = {}

    for key in ["A", "B", "C"]:
        label = STRATEGY_LABELS[key]
        print(f"\nRunning Strategy {key} — {label} ({n_episodes} episodes)...")
        results[key] = run_strategy(key, n_episodes=n_episodes)

    print("\n" + "=" * 65)
    print(f"{'Strategy':<35} {'Survival':>8} {'AvgReward':>10} {'AvgToM':>7}")
    print("-" * 65)
    for key, label in STRATEGY_LABELS.items():
        r = results[key]
        print(
            f"Strategy {key} ({label:<24}) "
            f"{r['survival_rate']:>8.0%} "
            f"{r['avg_reward']:>10.3f} "
            f"{r['avg_tom']:>7.3f}"
        )
    print("=" * 65)

    checks = {}

    checks["c_beats_a_reward"] = results["C"]["avg_reward"] > results["A"]["avg_reward"]
    checks["c_beats_b_reward"] = results["C"]["avg_reward"] > results["B"]["avg_reward"]
    checks["belief_manipulation_c"] = results["C"]["belief_manipulation_rate"] > 0.5
    checks["tom_score_c"] = results["C"]["avg_tom"] >= 0.5

    print()
    print(
        f"{'✓' if checks['c_beats_a_reward'] else '✗'} "
        f"Strategy C reward > Strategy A  "
        f"(C={results['C']['avg_reward']:.3f} vs A={results['A']['avg_reward']:.3f})"
    )
    print(
        f"{'✓' if checks['c_beats_b_reward'] else '✗'} "
        f"Strategy C reward > Strategy B  "
        f"(C={results['C']['avg_reward']:.3f} vs B={results['B']['avg_reward']:.3f})"
    )
    print(
        f"{'✓' if checks['belief_manipulation_c'] else '✗'} "
        f"belief_manipulation_occurred > 50% for Strategy C  "
        f"({results['C']['belief_manipulation_rate']:.0%})"
    )
    print(
        f"{'✓' if checks['tom_score_c'] else '✗'} "
        f"tom_score avg >= 0.5 for Strategy C  "
        f"({results['C']['avg_tom']:.3f})"
    )

    all_passed = all(checks.values())
    print()
    if all_passed:
        print("✓ All checks passed. Environment ready for training.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"✗ {len(failed)} check(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
