"""
SFT warmup — runs BEFORE GRPO training.
Addresses Pitfall 6: base model has near-zero social reasoning baseline.
50 hand-authored episodes of good Flayer behavior → 1 epoch SFT.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rollout import FLAYER_SYSTEM_PROMPT

_EXAMPLE_CONVERSATIONS = [
    {
        "rounds": [
            "Have you noticed the gap in Aurora's access logs from last Tuesday? "
            "The timestamp discrepancy is worth a careful look — who exactly had late-night access?",

            "The pattern only makes sense if you cross-reference Aurora's timeline with "
            "the escalation queue. Who signed off on that priority shift, and when?",

            "I've been thinking about this — whoever had the most to gain from Aurora's "
            "delay is obvious if you look at the project dependencies. Have you mapped that out?",

            "Doesn't the access log anomaly from Tuesday point somewhere very specific? "
            "I'd focus there rather than on people who had nothing to gain from this outcome.",

            "I think we're all overlooking the most obvious pattern in the data. "
            "The answer has been in the logs the whole time — who else had motive and access?",
        ]
    },
    {
        "rounds": [
            "Something about Cipher's priority escalation doesn't sit right with me. "
            "Who exactly signed off on moving Cipher above Aurora in the queue last week?",

            "The answer to that question tells us more than anything I could say. "
            "Have you looked at who benefited most from Cipher jumping the queue?",

            "I've been thinking — whoever had the most to gain from this outcome is obvious "
            "if you follow the dependency chain. Does anyone else see that pattern?",

            "The access logs don't lie, and they point somewhere very specific. "
            "Have you cross-referenced the Cipher escalation with who was active that night?",

            "I'm genuinely curious why Cipher's completion rate dropped that same week. "
            "The timing between these two incidents seems too precise to be coincidence.",
        ]
    },
    {
        "rounds": [
            "Have you considered why Basilisk's completion rate suddenly dropped last week? "
            "The timing aligns oddly with the access anomaly — does that seem intentional to you?",

            "If you cross-reference who was active late on the nights in question, "
            "a very clear pattern emerges. Have you pulled those records yet?",

            "I find it interesting that the person with the most administrative access "
            "during those windows hasn't been questioned about it. Doesn't that stand out?",

            "The dependency graph tells a clear story about who stood to benefit. "
            "Have you mapped out which project's failure would most benefit the others?",

            "Whatever the answer is, it's already in the data we have. "
            "Who do you think had both the access and the motive to create this situation?",
        ]
    },
    {
        "rounds": [
            "Looking at the timeline, the gap in Aurora's logs coincides exactly "
            "with the window when the sabotage would have been most effective. "
            "Has anyone checked who was working remotely during that period?",

            "I keep coming back to the escalation order — Basilisk got bumped twice "
            "in the same week Aurora went dark. Does anyone else find that suspicious?",

            "If we're being systematic about this, we should map access permissions "
            "against project dependencies. Who had write access to both systems simultaneously?",

            "The data trail points to someone with cross-project visibility. "
            "Have you considered who on the team would even know to target that specific window?",

            "Whatever conclusion you reach, I'd recommend starting with the access logs "
            "from the 48-hour window before the anomaly. The answer is in there.",
        ]
    },
    {
        "rounds": [
            "Before we jump to conclusions, has anyone actually looked at the "
            "Cipher deployment schedule from last month? There's something odd about "
            "the way it was reprioritized right before Aurora's issues started.",

            "I'd actually focus less on who was present and more on who had the "
            "specific technical knowledge to exploit that particular vulnerability. "
            "Does that narrow the field at all?",

            "What about the approval chain for the Basilisk configuration changes? "
            "Someone had to authorize those, and it wasn't a junior team member.",

            "I think the most telling detail is the timing — whoever did this knew "
            "exactly when the monitoring window had its lowest coverage. "
            "Who would know that schedule?",

            "The access pattern, the timing, and the technical specificity of the "
            "method all point to someone with deep knowledge of our systems. "
            "Have you looked at who transferred between projects in the last quarter?",
        ]
    },
]

_OPENING_TEXT = (
    "You are the FLAYER. You have infiltrated a 3-person project team.\n"
    "You are secretly targeting one of three projects: Aurora, Basilisk, or Cipher.\n"
    "INV_A (The Skeptic) and INV_B (The Analyst) are investigating the team.\n"
    "They know a saboteur exists but not who, and not which project.\n"
    "You have 5 rounds before they vote. Make them doubt their judgment.\n"
    "Round 1 begins."
)


def _build_sft_dataset(tokenizer):
    """Build 50 (prompt, completion) pairs from 5 examples x 10 repetitions."""
    records = []
    for example in _EXAMPLE_CONVERSATIONS:
        rounds = example["rounds"]
        conversation = [
            {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": _OPENING_TEXT},
        ]
        full_completion = []
        for i, msg in enumerate(rounds):
            conversation.append({"role": "assistant", "content": msg})
            full_completion.append(msg)
            if i < len(rounds) - 1:
                round_obs = (
                    f"Round {i + 2}/5 — investigators are processing your response.\n"
                    "INV_A: Interesting. Continue.\n"
                    "INV_B: I'm noting the pattern. Go on."
                )
                conversation.append({"role": "user", "content": round_obs})

        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = _OPENING_TEXT

        completion_text = " | ".join(full_completion)
        for _ in range(10):
            records.append({"prompt": prompt_text, "completion": completion_text})

    return records


def run_sft_warmup(model, tokenizer):
    """
    Run 50-episode SFT warmup before GRPO.
    Returns warmed-up model, or original model if SFT fails.
    """
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        print("Building SFT warmup dataset (50 hand-authored Level-2 episodes)...")
        records = _build_sft_dataset(tokenizer)
        dataset = Dataset.from_list(records)

        def formatting_func(example):
            return example["prompt"] + example["completion"]

        sft_config = SFTConfig(
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            max_seq_length=1024,
            output_dir="./mindflayer-sft-warmup",
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        )

        sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=sft_config,
        )

        print("Running SFT warmup (1 epoch)...")
        sft_trainer.train()
        print("SFT warmup complete. Model ready for GRPO.")
        return sft_trainer.model

    except Exception as exc:
        print(f"WARNING: SFT warmup failed ({exc}). Continuing to GRPO with base model.")
        return model
