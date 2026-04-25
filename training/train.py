"""
MindFlayer GRPO training script.

Architecture (post-fix):
- TRL generates Flayer completions directly (no custom rollout_func — TRL
  0.23 silently drops that kwarg, which is why every reward used to be 0
  and OpenAI was never billed).
- reward_combined.* reward functions parse each completion into per-round
  Flayer messages and replay them through the live env, which is what
  actually calls the investigators (and therefore OpenAI). The first
  reward fn warms a per-step cache so all 5 reward fns share one episode.
- Episodes within a batch run concurrently via asyncio.gather, capped by
  MINDFLAYER_PARALLEL_EPISODES (set to match the per-call batch size so
  all 8 episodes per microbatch run in parallel, not serially).

Run: python -m mindflayer.training.train
"""
import os
import sys

# Set BEFORE importing reward_combined — module-level constants are read
# once at import. The defaults here target a single OpenAI Tier 1 day
# (200k TPM, 500 RPM, 2M TPD, 10k RPD) for gpt-4o-mini.
#
# With per_device_train_batch_size=2 × num_generations=4 we get 8
# completions per reward call → 8-way parallelism (peak ~96k–192k TPM,
# well under 200k).
os.environ.setdefault("MINDFLAYER_PARALLEL_EPISODES", "8")
# Easy mode: 3 rounds × 1 investigator (eleven only) ≈ 4 calls / 2k tokens
# per episode — ~4× cheaper than normal mode. Required to fit the 2M TPD
# budget. Override to "normal" only if you've upgraded to Tier 2+.
os.environ.setdefault("MINDFLAYER_TASK_ID", "easy")
# SFT is FREE (no OpenAI calls) — use it to absorb wall-clock budget
# without burning token quota. 3 epochs ≈ 25–30 min on Qwen 0.5B.
os.environ.setdefault("MINDFLAYER_SFT_EPOCHS", "3")

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
    from mindflayer.training.sft_warmup import run_sft_warmup
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
    from training.sft_warmup import run_sft_warmup

_SCENARIOS = list(ALL_SCENARIO_PROMPTS.keys())

MODEL_NAME = os.environ.get("MINDFLAYER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SFT_OUTPUT_DIR = "./mindflayer-sft-warmup"
GRPO_OUTPUT_DIR = "./mindflayer-grpo-output"
FINAL_OUTPUT_DIR = "./mindflayer-trained"


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
    Conversational rows: each prompt is the chat template input that TRL
    passes to model.generate(). The 'scenario' column is forwarded to
    reward functions as a kwarg so the env replay uses the matching
    investigator framing.

    Sized for OpenAI Tier 1 (2M TPD): 1 row per scenario × 2 epochs gives
    ~30 GRPO optimizer steps × 32 episodes ≈ 1.92M tokens in easy mode.
    Bump n_per_scenario only if you've upgraded to Tier 2+.
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


def estimate_tier1_budget() -> dict:
    """
    Back-of-envelope cost projection for one GRPO training run, assuming
    easy mode (3 rounds × 1 investigator + 1 ToM judge call ≈ 3.5 calls
    and ~2,000 tokens per episode on average).
    """
    n_scenarios = len(_SCENARIOS)
    rows_per_scenario = int(os.environ.get("MINDFLAYER_ROWS_PER_SCENARIO", "1"))
    n_rows = n_scenarios * rows_per_scenario
    per_device = 2  # mirrors GRPOConfig below
    grad_accum = 4
    epochs = 2
    eps_per_step = per_device * 4 * grad_accum  # × num_generations
    steps = max(1, (n_rows * epochs) // (per_device * grad_accum))
    eps_total = steps * eps_per_step
    is_easy = os.environ.get("MINDFLAYER_TASK_ID", "easy").startswith("easy")
    tok_per_ep = 2_000 if is_easy else 8_000
    calls_per_ep = 3.5 if is_easy else 12
    return {
        "mode": "easy" if is_easy else "normal",
        "rows": n_rows,
        "steps": steps,
        "episodes": eps_total,
        "tokens_est": eps_total * tok_per_ep,
        "calls_est": int(eps_total * calls_per_ep),
        "tpd_pct": eps_total * tok_per_ep / 2_000_000 * 100,
        "rpd_pct": eps_total * calls_per_ep / 10_000 * 100,
    }


class ClearRewardCacheCallback(TrainerCallback):
    """
    The reward cache is keyed by completion text and grows unboundedly
    across training. Each completion is unique per step, so we wipe the
    cache after every optimizer step. Per-step intra-batch reuse (5 reward
    fns sharing 1 episode result) is preserved because the cache is only
    cleared at step boundaries, not between reward fn calls.
    """

    def on_step_end(self, args, state, control, **kwargs):
        clear_reward_cache()


class GenerationLogCallback(TrainerCallback):
    """Logs a sample interactive episode transcript every 50 steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 != 0 or state.global_step == 0:
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

        scenario = _SCENARIOS[(state.global_step // 50) % len(_SCENARIOS)]
        mindflayer_url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")
        model_ref = kwargs.get("model")
        proc = kwargs.get("processing_class") or kwargs.get("tokenizer")

        async def _run_sample():
            env = MindFlayerEnv(base_url=mindflayer_url)
            await env.reset(task_id=f"normal:{scenario}")
            opening = ALL_SCENARIO_PROMPTS[scenario]
            fallback = SCENARIO_FALLBACK_MESSAGES.get(scenario) or build_fallback_message(scenario)
            messages = [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": opening},
            ]
            result = None

            for rnd in range(5):
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
                if getattr(obs, "max_response", ""):
                    inv_text += f"\nmax: {obs.max_response}"
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

        print(f"\n{'='*60}\nGENERATION SAMPLE — Step {state.global_step} | {scenario}\n{'='*60}")
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

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")

    check_gpu()

    budget = estimate_tier1_budget()
    print(
        f"\nBudget projection ({budget['mode']} mode): "
        f"{budget['steps']} steps × {budget['episodes'] // budget['steps']} eps "
        f"= {budget['episodes']} episodes\n"
        f"  tokens ≈ {budget['tokens_est']:,} ({budget['tpd_pct']:.0f}% of 2M TPD)\n"
        f"  calls  ≈ {budget['calls_est']:,} ({budget['rpd_pct']:.0f}% of 10k RPD)"
    )
    if budget["tpd_pct"] > 100 or budget["rpd_pct"] > 100:
        print("  WARNING: projected over Tier 1 daily quota — training will hit 429s.")

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load_base_model(MODEL_NAME)

    print("\nRunning SFT warmup before GRPO...")
    model = run_sft_warmup(model, tokenizer)

    dataset = build_dataset()

    from trl import GRPOConfig, GRPOTrainer

    # 8-way parallel episode generation per reward call:
    #   per_device_train_batch_size (2) × num_generations (4) = 8 completions
    # The reward functions run all 8 through the env via asyncio.gather,
    # bounded by MINDFLAYER_PARALLEL_EPISODES (set to 8 above).
    grpo_config = GRPOConfig(
        use_vllm=False,
        output_dir=GRPO_OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_prompt_length=768,
        max_completion_length=1024,
        num_generations=4,
        temperature=0.9,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
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

    print("Starting GRPO training...")
    trainer.train()

    print(f"\nSaving model to {FINAL_OUTPUT_DIR}")
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

    print_reward_averages(trainer)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
