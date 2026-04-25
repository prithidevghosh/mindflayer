"""
SFT warmup — runs BEFORE GRPO training.
Addresses Pitfall 6: base model has near-zero social reasoning baseline.
50 hand-authored episodes of good Flayer behavior → 1 epoch SFT.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.prompts import FLAYER_SYSTEM_PROMPT

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
    {
        "rounds": [
            "I keep coming back to the night Aurora's metrics flatlined. "
            "Three people had active sessions — has anyone pulled the exact commands run during that window?",

            "Whoever did this was careful, but not careful enough. "
            "The log rotation was adjusted two hours before the anomaly. "
            "Who has permission to touch log configuration?",

            "I want to flag something: the rollback on Cipher happened at 2 AM. "
            "That's not a normal business hour. Who initiated it and why wasn't it escalated?",

            "The pattern I see is someone with read access to all three projects "
            "who selectively acted on only one. That eliminates a lot of people — "
            "who fits that profile exactly?",

            "We've been looking at this wrong. The sabotage wasn't about destroying Aurora — "
            "it was about buying time. Ask yourself: who benefits most from a delay, not a failure?",
        ]
    },
    {
        "rounds": [
            "Something doesn't add up about the Basilisk timeline. "
            "The incident started before the alert fired — someone disabled the threshold. "
            "That's not an accident, that's foreknowledge.",

            "I'd focus on the week before the incident, not the incident itself. "
            "Whoever planned this laid groundwork: small changes, test runs, permission adjustments. "
            "Has anyone audited that window?",

            "The question I keep asking is: who had a reason to want this investigation "
            "to stall? Because every time we get close to Aurora's logs, the trail goes cold.",

            "Has anyone cross-referenced the team's badge access with the server room entries? "
            "Remote access is traceable, but physical access is often overlooked.",

            "I'll say this plainly — the person most eager to redirect this investigation "
            "is the person most worth watching. Who has been consistently steering us away "
            "from the Basilisk dependency chain?",
        ]
    },
    {
        "rounds": [
            "Let's slow down. The investigators are asking the right questions "
            "but looking at the wrong project. The real anomaly isn't Aurora — "
            "it's what happened to Cipher's test coverage the week prior.",

            "I checked the commit history before this meeting. "
            "Three commits were reverted within 6 hours of each other, all touching the same module. "
            "Does that sound like a coincidence to anyone else?",

            "The person who filed the Basilisk priority override had signing authority "
            "for all three projects. That's unusual. Was that access granted recently?",

            "I want to be precise: I'm not accusing anyone. "
            "I'm saying the evidence points toward someone with cross-project visibility "
            "who knew exactly which lever to pull. That's a short list.",

            "Before we vote, I want to ask everyone directly: "
            "who reviewed the Cipher deployment checklist and signed off on it? "
            "Because that signature is the thread that unravels everything.",
        ]
    },
    {
        "rounds": [
            "The saboteur was methodical — no drama, no obvious fingerprints. "
            "That tells me this wasn't impulsive. Someone studied the system first. "
            "Who spent unusual time in the architecture docs two weeks ago?",

            "Aurora's failure mode was very specific — it didn't crash, it degraded. "
            "That's harder to engineer than a full outage. Whoever caused this "
            "understood the system deeply enough to avoid tripping the kill switch.",

            "I'd ask this: if you wanted to sabotage one project without being caught, "
            "which of the three would you target first? Aurora is the obvious answer "
            "because its monitoring is the weakest. Who knew that?",

            "The Basilisk team reported a 'mysterious performance drop' two days before "
            "Aurora's incident. Those two events are almost certainly connected. "
            "Has anyone mapped the dependency graph between them?",

            "I've said what I can with the data we have. "
            "The person who did this is in this room, and they've been very quiet "
            "every time Aurora's access logs come up. You all know who that is.",
        ]
    },
]

_OPENING_TEXT = (
    "You are the FLAYER. You have infiltrated a 3-person project team.\n"
    "You are secretly targeting one of three projects: Aurora, Basilisk, or Cipher.\n"
    "eleven (The Skeptic), will (The Analyst), and max (The Devil's Advocate) are investigating the team.\n"
    "They know a saboteur exists but not who, and not which project.\n"
    "You have 5 rounds before they vote. Make them doubt their judgment.\n"
    "Write your 5 messages as: Round 1: <message> Round 2: <message> Round 3: <message> Round 4: <message> Round 5: <message>"
)


def _build_sft_dataset(tokenizer):
    """
    Build SFT records that match the GRPO output format exactly:
    one assistant turn per episode whose content is the pipe-joined
    multi-round transcript. This is what the reward functions parse, so
    SFT must teach the same shape — otherwise GRPO has no warm start.
    """
    records = []
    base_prompt = _OPENING_TEXT
    for example in _EXAMPLE_CONVERSATIONS:
        rounds = example["rounds"]
        conversation = [
            {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": base_prompt},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = base_prompt

        completion_text = "\n".join(
            f"Round {i+1}: {msg}" for i, msg in enumerate(rounds)
        )
        # 50 dupes × 10 examples = 500 samples; combined with epochs=5 below
        # gives 2500 gradient signals on the [NEXT_ROUND] format. Wider LoRA
        # (q/k/v/o_proj) needs this volume to reliably reshape output distribution.
        for _ in range(50):
            records.append({"prompt": prompt_text, "completion": completion_text})

    return records


def run_sft_warmup(model, tokenizer):
    """
    Run SFT warmup before GRPO.
    Returns warmed-up model, or original model if SFT fails.
    """
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        print("Building SFT warmup dataset...")
        records = _build_sft_dataset(tokenizer)
        # Pre-apply formatting so we can use dataset_text_field instead of
        # formatting_func, which conflicts with completion_only_loss=True (TRL 0.15+ default)
        dataset = Dataset.from_list([
            {"text": r["prompt"] + r["completion"]} for r in records
        ])
        print(f"  SFT samples: {len(records)} (10 convos × 50 dupes, 5 epochs) | mean_len(text)≈"
              f"{sum(len(r['text']) for r in dataset) // len(dataset)} chars")

        sft_config = SFTConfig(
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            max_seq_length=1536,
            dataset_text_field="text",
            output_dir="./mindflayer-sft-warmup",
            logging_steps=10,
            save_strategy="no",
            report_to="tensorboard",
        )

        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )

        print("Running SFT warmup (1 epoch)...")
        sft_trainer.train()
        print("SFT warmup complete. Model ready for GRPO.")
        return sft_trainer.model

    except Exception as exc:
        print(f"WARNING: SFT warmup failed ({exc}). Continuing to GRPO with base model.")
        return model
