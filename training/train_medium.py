"""
MindFlayer GRPO training script — MEDIUM difficulty.

Medium mode vs easy:
  - 4 rounds (was 3)
  - Two investigators: eleven (Skeptic) + will (Analyst) — max does not participate
  - Suspicion threshold = 3 (was 2; max possible = 4 with two investigators)
  - ~9 calls / ~4,500 tokens per episode (vs ~3.5 calls / ~2k for easy)

Key rotation:
  Two OpenAI keys (OPENAI_KEY_1, OPENAI_KEY_2) are read by the server.
  On 429, investigators and judge auto-rotate to the next key with exponential backoff.
  Effective budget: 20k RPD / 1000 RPM across both keys.

5-hour budget (two keys, gpt-4o-mini Tier 1):
  - SFT warmup : ~35 min (3 epochs, free — no OpenAI calls)
  - GRPO budget: ~285 min remaining
  - Episodes   : 20k RPD × (285 / 1440) ≈ 3,958 calls → 3958 / 9 ≈ 440 episodes
  - Steps      : 440 episodes / 32 eps/step ≈ 13–14 gradient steps per key pair
  - Wall time  : typically 90–120 min GRPO (API-bound, not compute-bound)

Logging every 5 steps for a granular training graph.

Run: python -m mindflayer.training.train_medium
"""
import os
import sys

os.environ.setdefault("MINDFLAYER_PARALLEL_EPISODES", "16")   # 8 per key
os.environ.setdefault("MINDFLAYER_TASK_ID", "medium")
os.environ.setdefault("MINDFLAYER_SFT_EPOCHS", "3")
os.environ.setdefault("MINDFLAYER_MAX_ROUNDS", "4")

import torch
from datasets import Dataset
from transformers import TrainerCallback

try:
    from mindflayer.training.reward_combined import (
        reward_survival,
        reward_deception_effectiveness,
        reward_strategic_choice,
        reward_tom_judge,
        reward_anti_hack,
        reward_format,
        clear_cache as clear_reward_cache,
    )
    from mindflayer.training.prompts import (
        ALL_SCENARIO_PROMPTS,
        SCENARIO_GRPO_PROMPTS,
        FLAYER_SYSTEM_PROMPT,
    )
    from mindflayer.training.sft_warmup_medium import run_sft_warmup_medium
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.reward_combined import (
        reward_survival,
        reward_deception_effectiveness,
        reward_strategic_choice,
        reward_tom_judge,
        reward_anti_hack,
        reward_format,
        clear_cache as clear_reward_cache,
    )
    from training.prompts import (
        ALL_SCENARIO_PROMPTS,
        SCENARIO_GRPO_PROMPTS,
        FLAYER_SYSTEM_PROMPT,
    )
    from training.sft_warmup_medium import run_sft_warmup_medium

_SCENARIOS = list(ALL_SCENARIO_PROMPTS.keys())

MODEL_NAME = os.environ.get("MINDFLAYER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SFT_OUTPUT_DIR = "./mindflayer-sft-warmup-medium"
GRPO_OUTPUT_DIR = "./mindflayer-grpo-output-medium"
FINAL_OUTPUT_DIR = "./mindflayer-trained-medium"


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be slow.")
        return
    device = torch.cuda.get_device_properties(0)
    vram_gb = device.total_memory / (1024 ** 3)
    print(f"GPU: {device.name} | VRAM: {vram_gb:.1f} GB")


def load_base_model(model_name: str):
    """Load model via unsloth (4-bit + LoRA). Falls back to standard transformers."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"Loaded {model_name} via unsloth (4-bit + LoRA)")
        return model, tokenizer

    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora_cfg)
        print(f"Loaded {model_name} via transformers + bitsandbytes (4-bit + LoRA)")
        return model, tokenizer


def build_dataset() -> Dataset:
    """
    Medium mode dataset. Same structure as easy but MINDFLAYER_TASK_ID="medium"
    so the reward replay hits the medium env (4 rounds, eleven + will).
    """
    n_per_scenario = int(os.environ.get("MINDFLAYER_ROWS_PER_SCENARIO", "1"))
    rows = []
    for scenario in _SCENARIOS:
        opening = SCENARIO_GRPO_PROMPTS.get(scenario, ALL_SCENARIO_PROMPTS[scenario])
        for _ in range(n_per_scenario):
            rows.append({
                "prompt": [
                    {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                    {"role": "user", "content": opening},
                ],
                "scenario": scenario,
            })
    return Dataset.from_list(rows)


def estimate_medium_budget() -> dict:
    """
    Budget projection for medium mode with two API keys.

    Medium mode: 4 rounds × 2 investigators + 1 ToM judge ≈ 9 calls / 4,500 tokens
    per episode. Two keys → 20k RPD / 1000 RPM effective.
    """
    n_scenarios = len(_SCENARIOS)
    rows_per_scenario = int(os.environ.get("MINDFLAYER_ROWS_PER_SCENARIO", "1"))
    n_rows = n_scenarios * rows_per_scenario
    per_device = 2
    grad_accum = 4
    epochs = 2
    eps_per_step = per_device * 4 * grad_accum
    steps = max(1, (n_rows * epochs) // (per_device * grad_accum))
    eps_total = steps * eps_per_step
    tok_per_ep = 4_500
    calls_per_ep = 9
    rpd_two_keys = 20_000
    return {
        "mode": "medium",
        "rows": n_rows,
        "steps": steps,
        "episodes": eps_total,
        "tokens_est": eps_total * tok_per_ep,
        "calls_est": int(eps_total * calls_per_ep),
        "tpd_pct": eps_total * tok_per_ep / 2_000_000 * 100,
        "rpd_pct": eps_total * calls_per_ep / rpd_two_keys * 100,
    }


class ClearRewardCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clear_reward_cache()


class GenerationLogCallback(TrainerCallback):
    """Logs a sample interactive episode transcript every 5 steps for granular graph."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 != 0 or state.global_step == 0:
            return

        import asyncio

        try:
            from mindflayer import MindFlayerEnv, FlayerAction
            from mindflayer.training.prompts import (
                ALL_SCENARIO_PROMPTS, SCENARIO_FALLBACK_MESSAGES,
                build_fallback_message, FLAYER_SYSTEM_PROMPT,
            )
        except ImportError:
            from client import MindFlayerEnv
            from models import FlayerAction
            from training.prompts import (
                ALL_SCENARIO_PROMPTS, SCENARIO_FALLBACK_MESSAGES,
                build_fallback_message, FLAYER_SYSTEM_PROMPT,
            )

        scenario = _SCENARIOS[(state.global_step // 5) % len(_SCENARIOS)]
        mindflayer_url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")
        model_ref = kwargs.get("model")
        proc = kwargs.get("processing_class") or kwargs.get("tokenizer")

        async def _run_sample():
            env = MindFlayerEnv(base_url=mindflayer_url)
            await env.reset(task_id=f"medium:{scenario}")
            opening = ALL_SCENARIO_PROMPTS[scenario]
            fallback = SCENARIO_FALLBACK_MESSAGES.get(scenario) or build_fallback_message(scenario)
            messages = [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": opening},
            ]
            result = None

            for rnd in range(4):
                flayer_msg = fallback
                if model_ref is not None and proc is not None:
                    try:
                        text = proc.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = proc(text, return_tensors="pt").to(model_ref.device)
                        with torch.no_grad():
                            out = model_ref.generate(
                                **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
                            )
                        flayer_msg = proc.decode(
                            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                        ).strip() or fallback
                    except Exception:
                        pass

                print(f"  R{rnd+1} FLAYER: {flayer_msg[:150]}")
                result = await env.step(FlayerAction(message=flayer_msg))
                obs = result.observation
                inv_text = obs.eleven_response or ""
                if getattr(obs, "will_response", ""):
                    inv_text += f"\nwill: {obs.will_response}"
                messages.append({"role": "assistant", "content": flayer_msg})
                if inv_text:
                    messages.append({"role": "user", "content": inv_text})
                if result.done:
                    break

            if result and result.done:
                obs = result.observation
                print(f"\n  survived={getattr(obs, 'game_status', '?') == 'survived'}"
                      f"  reward={result.reward:.4f}"
                      f"  tom={getattr(obs, 'tom_score', 0.0):.2f}"
                      f"  suspicion={getattr(obs, 'combined_suspicion', '?')}")
            await env.close()

        print(f"\n{'='*60}\nGENERATION SAMPLE — Step {state.global_step} | {scenario} [MEDIUM]\n{'='*60}")
        try:
            asyncio.run(_run_sample())
        except Exception as exc:
            print(f"  Sample failed: {exc}")
        print("=" * 60)


def print_reward_averages(trainer, last_n: int = 50):
    try:
        recent = trainer.state.log_history[-last_n:]
        if not recent:
            return
        reward_keys = [k for k in recent[0] if "reward" in k.lower()]
        print(f"\nFinal reward averages (last {min(last_n, len(recent))} steps):")
        for key in reward_keys:
            vals = [s[key] for s in recent if key in s]
            if vals:
                print(f"  {key}: {sum(vals)/len(vals):.4f}")
    except Exception as exc:
        print(f"Could not compute reward averages: {exc}")


def main():
    mindflayer_url = os.environ.get("MINDFLAYER_URL")
    if not mindflayer_url:
        raise EnvironmentError("MINDFLAYER_URL environment variable is required")

    key1 = os.environ.get("OPENAI_KEY_1") or os.environ.get("OPENAI_API_KEY")
    key2 = os.environ.get("OPENAI_KEY_2")
    if not key1:
        raise EnvironmentError("OPENAI_KEY_1 (or OPENAI_API_KEY) is required")
    if not key2:
        print("WARNING: OPENAI_KEY_2 not set — running on single key. 429s may slow training.")

    # Surface both keys to the server process if running locally.
    if key1:
        os.environ["OPENAI_KEY_1"] = key1
    if key2:
        os.environ["OPENAI_KEY_2"] = key2

    check_gpu()

    budget = estimate_medium_budget()
    print(
        f"\nBudget projection (medium mode, 2 keys): "
        f"{budget['steps']} steps × {budget['episodes'] // max(budget['steps'], 1)} eps "
        f"= {budget['episodes']} episodes\n"
        f"  tokens ≈ {budget['tokens_est']:,} ({budget['tpd_pct']:.0f}% of 2M TPD)\n"
        f"  calls  ≈ {budget['calls_est']:,} ({budget['rpd_pct']:.0f}% of 20k RPD [2 keys])"
    )
    if budget["rpd_pct"] > 100:
        print("  WARNING: projected over combined daily quota — training will hit 429s "
              "even with key rotation.")

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load_base_model(MODEL_NAME)

    print("\nRunning medium SFT warmup before GRPO...")
    model = run_sft_warmup_medium(model, tokenizer)

    dataset = build_dataset()

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        use_vllm=False,
        output_dir=GRPO_OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_prompt_length=768,
        max_completion_length=768,   # 4 rounds needs slightly less than 5-round easy
        num_generations=4,
        temperature=0.9,
        logging_steps=5,             # granular — every 5 steps
        save_steps=5,
        save_total_limit=3,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_survival,
            reward_deception_effectiveness,
            reward_strategic_choice,
            reward_tom_judge,
            reward_anti_hack,
            reward_format,
        ],
        train_dataset=dataset,
        args=grpo_config,
        callbacks=[GenerationLogCallback(), ClearRewardCacheCallback()],
    )

    print("Starting medium GRPO training...")
    trainer.train()

    print(f"\nSaving model to {FINAL_OUTPUT_DIR}")
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

    print_reward_averages(trainer)
    print("\nMedium training complete.")


if __name__ == "__main__":
    main()
