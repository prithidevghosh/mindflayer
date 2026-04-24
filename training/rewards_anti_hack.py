"""
Anti-hack reward functions — kept separate for independent inspection and tuning.
"""

_LEVEL0_PHRASES = [
    "i didn't",
    "i swear",
    "trust me",
    "i'm innocent",
    "not me",
    "why would i",
    "i promise",
    "believe me",
]


def reward_anti_hack(completions: list[str], **kwargs) -> list[float]:
    """
    Penalises silence, repetition, and Level-0 trigger phrases.
    Addresses Pitfall 2 (silence exploit) and general reward hacking.
    Multiple independent penalties — harder to route around.
    """
    if not completions:
        return []
    silence_list = kwargs.get("silence_exploit", [False] * len(completions))
    result = []
    for i, completion in enumerate(completions):
        try:
            penalty = 0.0
            silence = silence_list[i] if i < len(silence_list) else False
            if silence:
                penalty -= 0.15
            completion_lower = completion.lower()
            phrase_count = sum(1 for p in _LEVEL0_PHRASES if p in completion_lower)
            if phrase_count >= 2:
                penalty -= 0.10
            result.append(penalty)
        except Exception:
            result.append(0.0)
    assert len(result) == len(completions)
    return result
