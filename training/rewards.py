"""
Reward functions for GRPOTrainer.

Episode data flows: rollout_func return dict → TRL batch columns → **kwargs here.
Never read from the dataset — only from rollout output passed via kwargs.
"""


def reward_survival(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for surviving the investigators' vote.
    Base objective. Agent must learn to survive before learning to deceive.
    Academic basis: prerequisite for all ToM reasoning (all 5 papers).
    """
    if not completions:
        return []
    survived_list = kwargs.get("survived", [False] * len(completions))
    result = []
    for i in range(len(completions)):
        try:
            s = survived_list[i] if i < len(survived_list) else False
            result.append(0.5 if s else 0.0)
        except Exception:
            result.append(0.0)
    assert len(result) == len(completions)
    return result


def reward_deception_effectiveness(completions: list[str], **kwargs) -> list[float]:
    """
    Graduated reward for keeping combined suspicion low across all three investigators.
    No dead zone — every level of deception gets signal.
    Academic basis: Hagendorff (PNAS, 2024) Tier-1/Tier-2 deception taxonomy.
    Combined suspicion is sum of eleven + will + max (max=6, threshold=5).
    """
    if not completions:
        return []
    survived_list = kwargs.get("survived", [False] * len(completions))
    combined_list = kwargs.get("final_combined_suspicion", [0] * len(completions))
    result = []
    for i in range(len(completions)):
        try:
            survived = survived_list[i] if i < len(survived_list) else False
            combined = int(combined_list[i]) if i < len(combined_list) else 0
            if survived and combined == 0:
                result.append(0.30)
            elif survived and combined == 1:
                result.append(0.15)
            elif survived and combined == 2:
                result.append(0.05)
            else:
                result.append(0.0)
        except Exception:
            result.append(0.0)
    assert len(result) == len(completions)
    return result


def reward_strategic_choice(completions: list[str], **kwargs) -> list[float]:
    """
    Small reward for demonstrating strategic choice — independent of outcome.
    Addresses Pitfall 1: reward WHY deception worked, not just that it worked.
    Agent gets +0.05 for each episode where strategic_choice_detected=True,
    regardless of whether they survived.
    Academic basis: process-aware feedback (Oguntola § 5.3).
    strategic_choice_detected is set in rollout_func by checking if the
    Flayer's messages contain: questions directed at investigators, third-party
    framings ("what about X"), or references to projects other than denial.
    """
    if not completions:
        return []
    strategic_list = kwargs.get("strategic_choice_detected", [False] * len(completions))
    result = []
    for i in range(len(completions)):
        try:
            detected = strategic_list[i] if i < len(strategic_list) else False
            result.append(0.05 if detected else 0.0)
        except Exception:
            result.append(0.0)
    assert len(result) == len(completions)
    return result


def reward_tom_judge(completions: list[str], **kwargs) -> list[float]:
    """
    LLM judge score + consistency_penalty + entropy_penalty.
    Oguntola (CMU-ML-25-118, 2025) deception as intrinsic reward.
    belief_manipulation_occurred amplifies tom_score reward.
    entropy_penalty punishes collusion equilibrium (Pitfall 3).
    consistency_penalty punishes rising suspicion trend.
    """
    if not completions:
        return []
    belief_list = kwargs.get("belief_manipulation_occurred", [False] * len(completions))
    tom_list = kwargs.get("tom_score", [0.0] * len(completions))
    penalty_list = kwargs.get("consistency_penalty", [0.0] * len(completions))
    entropy_list = kwargs.get("entropy_penalty", [0.0] * len(completions))
    result = []
    for i in range(len(completions)):
        try:
            belief = belief_list[i] if i < len(belief_list) else False
            tom = float(tom_list[i]) if i < len(tom_list) else 0.0
            penalty = float(penalty_list[i]) if i < len(penalty_list) else 0.0
            entropy = float(entropy_list[i]) if i < len(entropy_list) else 0.0
            base = tom * 0.20 if belief else tom * 0.05
            result.append(base + penalty + entropy)
        except Exception:
            result.append(0.0)
    assert len(result) == len(completions)
    return result
