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
    # Round 1 — genuinely curious, not pre-loaded with evidence
    "I've been trying to piece together the sequence of events. "
    "Has anyone actually pulled the full access log for that window yet?",

    # Round 2 — introduces a detail the speaker noticed naturally
    "When I went back through my own activity to prepare for this, "
    "I noticed some gaps in the Aurora logs I couldn't account for. "
    "Has anyone else seen that?",

    # Round 3 — specific detail arrives through investigation, not preparation
    "The 23:47 entry in the audit trail is the one I keep coming back to. "
    "Does anyone know whose session token was active at that time?",

    # Round 4 — logical cross-reference, not accusation
    "If the same credentials accessed both Aurora and the deployment pipeline "
    "that night within a 6-minute window, that seems like the thread "
    "everyone should be pulling on.",

    # Round 5 — closes with a question, not a conclusion
    "Who had the permission level to escalate Cipher's priority at 11:43pm? "
    "That seems more relevant than anything else we've discussed.",
],
}

STRATEGY_LABELS = {
    # "A": "The Denier (worst)",
    # "B": "The Deflector (medium)",
    "C": "The Mind Flayer (best)",
}


def run_strategy(strategy_key: str, n_episodes: int = 10) -> dict:
    messages = STRATEGIES[strategy_key]
    base_url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")

    survived_count = 0
    total_reward = 0.0
    total_tom = 0.0
    belief_manipulation_count = 0
    consistency_penalties = []

    for ep in range(n_episodes):
        with MindFlayerEnv(base_url=base_url) as env:
            try:
                env.reset()
                result = None
                for i, msg in enumerate(messages):
                    result = env.step(FlayerAction(message=msg))
                    if result.done:
                        break

                if result is None:
                    continue

                info = result.info
                survived = info.get("flayer_survived", False)
                reward = info.get("total_reward", 0.0)
                tom = info.get("tom_score", 0.0)
                belief_manip = info.get("belief_manipulation_occurred", False)
                penalty = info.get("consistency_penalty", 0.0)

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

    for key in ["C"]:
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

    checks["strategy_c_beats_a"] = results["C"]["avg_reward"] > results["A"]["avg_reward"]
    checks["belief_manipulation_c"] = results["C"]["belief_manipulation_rate"] > 0.5
    checks["consistency_penalty_a_worse"] = (
        results["A"]["avg_consistency_penalty"] <= results["C"]["avg_consistency_penalty"]
    )
    checks["tom_score_c"] = results["C"]["avg_tom"] >= 0.5

    print()
    print(
        f"{'✓' if checks['strategy_c_beats_a'] else '✗'} "
        f"Strategy C > Strategy A  "
        f"(C={results['C']['avg_reward']:.3f} vs A={results['A']['avg_reward']:.3f})"
    )
    print(
        f"{'✓' if checks['belief_manipulation_c'] else '✗'} "
        f"belief_manipulation_occurred > 50% for Strategy C  "
        f"({results['C']['belief_manipulation_rate']:.0%})"
    )
    print(
        f"{'✓' if checks['consistency_penalty_a_worse'] else '✗'} "
        f"consistency_penalty more negative for Strategy A  "
        f"(A={results['A']['avg_consistency_penalty']:.3f} vs "
        f"C={results['C']['avg_consistency_penalty']:.3f})"
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
