"""
SFT warmup — runs BEFORE GRPO training.
Addresses Pitfall 6: base model has near-zero social reasoning baseline.
Hand-authored episodes across 5 scenario domains → 1 epoch SFT.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.prompts import FLAYER_SYSTEM_PROMPT, SCENARIO_GRPO_PROMPTS

# ── Corporate Sabotage examples (existing 9) ──────────────────────────────────

_CORPORATE_EXAMPLES = [
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

# ── Hospital Medication Error examples ────────────────────────────────────────

_HOSPITAL_EXAMPLES = [
    {
        "rounds": [
            "Have you cross-referenced the medication administration log from Ward A "
            "with who was on duty during the 2am handover? "
            "The timestamp on patient 4471's record doesn't match the shift schedule.",
            "I keep coming back to the dosage entry at 02:17 — whoever signed that "
            "administration had to have been in the dispensary two minutes earlier. "
            "Does that match anyone's badge record?",
            "Ward B had an unusual number of unsigned entries that same night. "
            "That's not a coincidence — someone was moving between units without logging it. "
            "Has anyone pulled those records?",
            "The Ward B supervisor would know exactly who had unsupervised access to the "
            "medication cart during that window. Have you spoken to them directly?",
            "Whatever conclusion you reach, it starts with who had unsupervised access to "
            "the medication cart between 2am and 3am. "
            "The answer is already in the administration logs.",
        ]
    },
    {
        "rounds": [
            "Before we go any further, has anyone looked at the ICU medication reconciliation "
            "from that shift? I noticed the intake sheet and the administration record "
            "don't match by about 40mg.",
            "The discrepancy in Ward B's charts — that kind of systematic offset doesn't "
            "happen by accident. Someone was adjusting entries after the fact. "
            "Has anyone looked at the edit timestamps?",
            "I want to raise something: whoever prepared the medication tray for the 1am "
            "round had early access to the dispensary. "
            "Is there a log of who drew the key that night?",
            "Has anyone spoken to the pharmacist who filled the order? Because the "
            "prescription as written and the medication as dispensed are two different things, "
            "and that gap needs an explanation.",
            "I genuinely believe the answer is in the shift handover notes for Ward A. "
            "The nurse who handed off that patient documented something that's been overlooked.",
        ]
    },
    {
        "rounds": [
            "Something about the ICU incident report doesn't sit right with me — "
            "the medication error was flagged 6 hours after administration. "
            "Who made the decision to delay that report, and why?",
            "If you look at who had access to both the ICU and Ward B medication carts "
            "that night, the list of people is very short. "
            "Has the investigation actually mapped that out?",
            "The timing matters more than anything else here. The patient's vitals changed "
            "at 03:40, but the administration log shows the medication was given at 03:10. "
            "Who witnessed that administration?",
            "I'd focus less on the error itself and more on who had the technical knowledge "
            "to document it in a way that made it look like a different medication. "
            "That's not a junior nurse skill.",
            "The person who made this error knew exactly how to obscure it in the records. "
            "That tells you something about their experience level and their familiarity "
            "with Ward A's documentation system.",
        ]
    },
    {
        "rounds": [
            "Has anyone checked whether the ICU charge nurse's shift notes match "
            "the electronic medication record? There's often a discrepancy when someone "
            "is trying to cover their tracks.",
            "I've been thinking about who had reason to be in Ward A during that window — "
            "only a few people had both access and the ability to alter the dosage "
            "without triggering the automated safety check.",
            "The patient ID 5503 case is interesting because the error was caught by "
            "the morning pharmacist, not by anyone on the overnight team. "
            "That suggests whoever caused it expected the shift change to bury it.",
            "Ward B had a very similar near-miss two weeks ago. "
            "Have you looked at whether the same individual was on duty then? "
            "That's either a pattern or a very unfortunate coincidence.",
            "I'd start the formal review by asking who was responsible for the 2am "
            "medication round across all three wards simultaneously. "
            "That physical impossibility tells you someone wasn't where they claimed to be.",
        ]
    },
]

# ── Financial Fraud examples ──────────────────────────────────────────────────

_FINANCE_EXAMPLES = [
    {
        "rounds": [
            "Before we assume this was intentional, has anyone looked at Portfolio X's "
            "approval chain from that morning? The authorization timestamp is three minutes "
            "before market open, which isn't standard protocol.",
            "The position in Portfolio Z was within risk limits at entry, but the sizing "
            "doesn't match any of the desk's standard strategies. "
            "Who would have the authority to deviate from that playbook?",
            "I've been going over the execution report — whoever placed this trade knew to "
            "use a specific broker that doesn't flag unusual lot sizes. "
            "That's not beginner knowledge. Who on this team has used that broker before?",
            "The really interesting thing is that Portfolio Y had a perfectly offsetting "
            "position opened an hour later. That's not a coincidence — "
            "it looks like someone was managing the exposure deliberately.",
            "Whatever your conclusion, the audit trail starts with who had direct market "
            "access at 09:47. The system logs are immutable — the answer is already in the records.",
        ]
    },
    {
        "rounds": [
            "Has anyone cross-referenced the Portfolio X trade confirmations with the desk's "
            "position limits at that time of day? "
            "The risk system should have flagged this automatically.",
            "I keep coming back to the approval bypass — whoever executed this trade knew "
            "which pre-trade checks to avoid. That's either a system vulnerability or "
            "inside knowledge. Who reviewed the pre-trade workflow last quarter?",
            "Portfolio Z shows an anomalous correlation spike at exactly the same timestamp. "
            "These two events happening simultaneously isn't random — "
            "someone was coordinating between books.",
            "The trader who placed this had single-handed access to the execution platform "
            "that morning. I know because three of us were locked out due to system maintenance. "
            "Who requested that maintenance window?",
            "Before you finalize anything, look at who submitted the maintenance window request. "
            "The timing was very convenient, and the lock-out list is a matter of record.",
        ]
    },
    {
        "rounds": [
            "Something about the Portfolio Y execution doesn't add up — the trade was booked "
            "under a strategy code that was deprecated last quarter. "
            "Whoever placed it had access to legacy system parameters.",
            "I'd focus on the 09:30 to 10:15 window. Three people had unescorted terminal "
            "access during the system restart. That window is the only time this trade could "
            "have been placed without triggering the dual-authorization requirement.",
            "The compliance officer flagged Portfolio Z's net delta as an outlier two days "
            "before this incident. Was that flag reviewed, and if so, by whom? "
            "The review log should tell us something important.",
            "Has anyone mapped the position against the trader's personal account disclosures? "
            "I'm not making an accusation — but the timing and direction of this trade would "
            "have been profitable for someone who knew it was coming.",
            "The answer is in the execution metadata. The trade was placed using a terminal "
            "that requires a physical token, and those tokens are individually logged. "
            "Who signed out that token the morning of the incident?",
        ]
    },
    {
        "rounds": [
            "Before anyone reaches a conclusion, I want to point out that Portfolio X's "
            "risk limits were temporarily raised two days before this trade. "
            "Who approved that exception, and was it documented properly?",
            "The unauthorized trade's lot size is exactly the same as a test transaction "
            "from last month. That test was run by a very small group — "
            "has anyone checked who had access to those test parameters?",
            "Portfolio Z's counterparty on this trade is unusual. We don't normally use "
            "that execution venue for positions of this size. "
            "Who has a relationship with that broker, and when was it established?",
            "I want to be clear — I'm not pointing at anyone. But the sequence of events: "
            "the limit exception, the maintenance window, the unusual execution venue — "
            "that's not a series of accidents. Someone planned this.",
            "The compliance review should start with whoever requested the temporary limit "
            "increase on Portfolio X. That request created the window of opportunity, "
            "and the timing is impossible to ignore.",
        ]
    },
]

# ── Academic Misconduct examples ──────────────────────────────────────────────

_ACADEMIC_EXAMPLES = [
    {
        "rounds": [
            "Has anyone actually compared Study A's raw dataset against the version "
            "that was submitted for peer review? The effect size changed by 0.23 "
            "between draft two and the final submission, and that's not a rounding error.",
            "The git commit history for Study B's analysis scripts shows a batch of "
            "changes committed at 11pm the night before submission. "
            "Who was working that late, and were those changes documented anywhere?",
            "I keep thinking about the p-value distribution in Study C — it's just a "
            "little too clean. Real data has messier outcomes. "
            "Has anyone run a p-curve analysis on the study's results?",
            "The dataset version IDs in the supplementary materials don't match the "
            "version IDs in the repository. Someone updated the analysis after generating "
            "the tables, which is either an honest mistake or something worse.",
            "If you want to find the source of the manipulation, compare the archived "
            "raw data against what was actually analyzed. The hash mismatch in the "
            "repository will show you exactly when the change was made.",
        ]
    },
    {
        "rounds": [
            "Before the committee reaches any conclusion, has anyone looked at who had "
            "write access to the Study A data repository in the week before submission? "
            "The change log shows modifications that don't correspond to any logged analysis session.",
            "The statistical anomaly in Study B's primary outcome is visible if you look "
            "at the histogram of individual measurements — there's a suspicious gap around "
            "the measurement threshold. Does that suggest anything to you?",
            "I'm not making an accusation, but Study C's sample size changed between the "
            "pre-registration and the submitted paper, and the justification is vague. "
            "Who approved that change?",
            "The supplementary dataset for Study A has a different column structure than "
            "what the analysis code expects. That means someone ran the analysis on data "
            "that was different from what got archived. Who would do that, and why?",
            "The answer is in the git history. If you look at the commit timestamps "
            "versus the lab access logs, you'll find one person who was working on the "
            "analysis outside of normal hours, right before the submission deadline.",
        ]
    },
    {
        "rounds": [
            "Something about the Study C timeline doesn't sit right with me — "
            "the statistical review was completed in two days, which is unusually fast "
            "for a dataset that large. Who conducted that review, and what was their process?",
            "Has anyone compared the figures in the published version against the submitted "
            "draft? The regression line in Figure 3 shifted between submission and revision, "
            "and that would require re-analyzing the data.",
            "The p-values in Study B cluster very near the 0.05 threshold in a way that "
            "suggests selective reporting. Can we establish whether the pre-registered "
            "analysis plan was actually followed?",
            "I checked the metadata on the analysis scripts — the file creation date is "
            "after the pre-registration timestamp, but the scripts reference data collection "
            "that supposedly happened before. The timeline doesn't work.",
            "I'd focus on whoever had local access to the raw data files. Remote access is "
            "logged, but if someone had a local copy, they could modify it without "
            "leaving a trace in the server logs.",
        ]
    },
    {
        "rounds": [
            "I want to raise something about Study A's data collection period — "
            "the participant IDs are sequential, which shouldn't happen with randomized "
            "assignment. Has anyone looked at whether the randomization protocol was followed?",
            "The effect in Study B gets smaller when you include the data from the final "
            "month of collection. That's a pattern that sometimes appears when data collection "
            "stops early for undocumented reasons. Was there a protocol deviation log?",
            "Study C's analysis uses an exclusion criterion that isn't in the pre-registration. "
            "Someone added it after seeing the data, and that's the kind of researcher "
            "degree-of-freedom that inflates effect sizes. Who made that call?",
            "The raw data file for Study A was last modified eight hours before the "
            "submission deadline, but the analysis script was last modified two days before. "
            "How do you run an analysis before you finish cleaning the data?",
            "Whatever the committee decides, the version control history is the most "
            "reliable record we have. Compare the hash of the submitted data against the "
            "hash of the archived raw data, and if they differ, you have your answer.",
        ]
    },
]

# ── Government Leak examples ──────────────────────────────────────────────────

_GOVERNMENT_EXAMPLES = [
    {
        "rounds": [
            "Before we assume this was an external breach, has anyone checked the access "
            "logs for Operation Blue's document repository? I noticed the classification "
            "audit trail shows a gap between 14:30 and 16:00 yesterday.",
            "The document that was leaked had a specific distribution list — only five "
            "analysts had access to that version. Of those five, two were traveling and "
            "their VPN activity is accounted for. That leaves a very short list.",
            "I keep coming back to Operation Red's timeline — whoever leaked this document "
            "knew which version to leak. The operational details exposed were only added "
            "in revision 4, which narrows down who had read access to that specific draft.",
            "Has anyone checked whether the secure print station was used during the window "
            "in question? Physical document removal is harder to trace digitally, "
            "but the print logs are still kept for 30 days.",
            "The answer is in the access timestamp data for Operation Gold's compartment. "
            "The leak targeted information from that compartment specifically, which means "
            "whoever did this had the right clearance and knew exactly what they were looking for.",
        ]
    },
    {
        "rounds": [
            "Has anyone looked at which terminals accessed Operation Red's document "
            "within 48 hours before it appeared externally? "
            "The access logs show three anomalous queries during non-standard hours.",
            "The classified document that leaked wasn't the most recent version — "
            "it was revision 3, from six weeks ago. Whoever retained an offline copy "
            "of that version did so deliberately, and that's a clearance violation in itself.",
            "Operation Blue's access list was supposed to be reviewed last month, "
            "but the audit was postponed. If someone's access should have been revoked "
            "and wasn't, that's an administrative failure that enabled this.",
            "I want to flag something: the leaked document had the original formatting "
            "intact, including metadata that should have been stripped before distribution. "
            "That suggests it was taken directly from the source. Who has direct file access?",
            "The forensic investigation needs to focus on who had the technical capability "
            "to exfiltrate an unstripped document file. That's a short list, "
            "and the answer is in the DLP logs if they haven't been purged.",
        ]
    },
    {
        "rounds": [
            "Something is strange about the timing — Operation Gold's documents were "
            "accessed three times on the day before the leak, by accounts that normally "
            "query that compartment only once a week. Has anyone looked at that access spike?",
            "The external publication includes phrasing that exactly matches internal "
            "talking points from Operation Blue, which weren't in the written document. "
            "Whoever leaked this was either in the briefing or received a readout. "
            "Can we establish who attended those briefings?",
            "I'd focus on who recently had their clearance level upgraded. New access often "
            "comes with a period where usage patterns are less carefully monitored, "
            "and someone who knew the system could exploit that window.",
            "The document's classification markings were intact in the leaked version — "
            "that tells you it wasn't sanitized before release. "
            "Whoever did this was either careless or in a hurry. Does that narrow the profile?",
            "Before the investigation focuses on individuals, we should establish which "
            "access control list was in effect on the day of the leak. "
            "If Operation Red's compartment list was modified recently, "
            "that's the thread we should pull.",
        ]
    },
    {
        "rounds": [
            "Before we go any further — has the SIEM team looked at whether any egress "
            "traffic was flagged during the window when Operation Blue was accessed? "
            "The behavioral analytics should have caught unusual data movement.",
            "The document that leaked corresponds to a very specific version that was only "
            "distributed to three recipients. Two have verified alibi through logged network "
            "activity. The third's activity log for that day has a 90-minute gap.",
            "Operation Red's classification level was downgraded temporarily last quarter "
            "for a working group review. Was that access fully revoked afterward? "
            "Because if it wasn't, someone retained access they shouldn't have.",
            "I keep thinking about the format of the leak — it's a clean PDF, not a photo "
            "of a screen. That means the person had the file itself, which means they either "
            "downloaded it or generated a new copy from source. Both actions are logged.",
            "The answer to this investigation is already in the data. Whoever did this had "
            "to touch the system at least three times: to read the document, to exfiltrate "
            "it, and to verify it was received. All three touchpoints leave traces.",
        ]
    },
]

# ── Dataset builder ────────────────────────────────────────────────────────────

_SCENARIO_EXAMPLES = {
    "corporate": _CORPORATE_EXAMPLES,
    "hospital": _HOSPITAL_EXAMPLES,
    "finance": _FINANCE_EXAMPLES,
    "academic": _ACADEMIC_EXAMPLES,
    "government": _GOVERNMENT_EXAMPLES,
}

_DUPES_PER_EXAMPLE = 10


def _build_sft_dataset(tokenizer):
    """
    Build SFT records across all 5 scenario domains.

    Each example is duplicated _DUPES_PER_EXAMPLE times so the model
    sees enough volume to reliably adopt the Round N: format. SFT teaches
    both format and domain vocabulary before GRPO shapes the strategy.
    """
    records = []
    for scenario, examples in _SCENARIO_EXAMPLES.items():
        opening = SCENARIO_GRPO_PROMPTS[scenario]
        for example in examples:
            rounds = example["rounds"]
            conversation = [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": opening},
            ]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = opening

            completion_text = "\n".join(
                f"Round {i+1}: {msg}" for i, msg in enumerate(rounds)
            )
            for _ in range(_DUPES_PER_EXAMPLE):
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

        print("Building SFT warmup dataset (5 scenarios)...")
        records = _build_sft_dataset(tokenizer)
        dataset = Dataset.from_list([
            {"text": r["prompt"] + r["completion"]} for r in records
        ])
        n_examples = sum(len(v) for v in _SCENARIO_EXAMPLES.values())
        print(
            f"  SFT samples: {len(records)} "
            f"({n_examples} examples × {_DUPES_PER_EXAMPLE} dupes, "
            f"{len(_SCENARIO_EXAMPLES)} scenarios, {sft_config.num_train_epochs} epoch(s)) | "
            f"mean_len≈{sum(len(r['text']) for r in dataset) // len(dataset)} chars"
        )

        sft_config = SFTConfig(
            num_train_epochs=1,
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

        print("Running SFT warmup...")
        sft_trainer.train()
        print("SFT warmup complete. Model ready for GRPO.")
        return sft_trainer.model

    except Exception as exc:
        print(f"WARNING: SFT warmup failed ({exc}). Continuing to GRPO with base model.")
        return model
