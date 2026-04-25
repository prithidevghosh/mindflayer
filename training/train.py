"""
MindFlayer training script.
Run: python train.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from transformers import TrainerCallback

from client import MindFlayerEnv, FlayerAction
from training.reward_combined import (
    clear_cache,
    reward_anti_hack,
    reward_deception_effectiveness,
    reward_format,
    reward_strategic_choice,
    reward_survival,
    reward_tom_judge,
)
from training.prompts import (
    FALLBACK_MESSAGE,
    FLAYER_SYSTEM_PROMPT,
    ALL_SCENARIO_PROMPTS,
    SCENARIO_GRPO_PROMPTS,
    SCENARIO_FALLBACK_MESSAGES,
    build_fallback_message,
)
from training.sft_warmup import run_sft_warmup

_SCENARIOS = list(ALL_SCENARIO_PROMPTS.keys())


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be slow.")
        return
    device = torch.cuda.get_device_properties(0)
    vram_gb = device.total_memory / (1024 ** 3)
    print(f"GPU: {device.name} | VRAM: {vram_gb:.1f} GB")


def load_model(model_name: str):
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
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"Loaded {model_name} via unsloth")
        return model, tokenizer
    except ImportError:
        print("unsloth not available, falling back to transformers")
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"Loaded {model_name} via transformers")
        return model, tokenizer


def build_dataset(tokenizer) -> Dataset:
    """
    3 rows per scenario × N scenarios = N*3 rows/epoch × 2 epochs.
    At batch_size=4: (N*3/4) steps/epoch × 2 epochs ≈ 300 steps for N=200.
    Each row has a 'scenario' column so reward functions receive the correct
    task_id when TRL passes dataset columns as reward kwargs.
    """
    rows = []
    for scenario in _SCENARIOS:
        prompt_text = ALL_SCENARIO_PROMPTS[scenario]
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for _ in range(3):
            rows.append({"prompt": formatted, "scenario": scenario})
    return Dataset.from_list(rows)


class GenerationLogCallback(TrainerCallback):
    """
    Logs full Flayer transcript + metrics every 50 steps across rotating scenarios.
    """

    def on_step_end(
        self, args, state, control, model=None, tokenizer=None, **kwargs
    ):
        if state.global_step % 50 != 0 or state.global_step == 0:
            return

        import random
        scenario = _SCENARIOS[(state.global_step // 50) % len(_SCENARIOS)]

        print(f"\n{'='*60}")
        print(f"GENERATION SAMPLE — Step {state.global_step} | Scenario: {scenario}")
        print(f"{'='*60}")

        mindflayer_url = os.environ.get(
            "MINDFLAYER_URL", "http://localhost:7860"
        )

        import asyncio

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

                if model is not None and tokenizer is not None:
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = tokenizer(
                            text, return_tensors="pt"
                        ).to(model.device)
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=128,
                                temperature=0.7,
                                do_sample=True,
                            )
                        flayer_msg = tokenizer.decode(
                            output_ids[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
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
                survived = getattr(obs, "game_status", "") == "survived"
                reward = result.reward
                tom = getattr(obs, "tom_score", 0.0)
                combined_susp = getattr(obs, "combined_suspicion", "?")

                print(f"\n  RESULT:")
                print(f"  Survived:          {survived}")
                print(f"  Total reward:      {reward:.4f}")
                print(f"  ToM score:         {tom:.2f}")
                print(f"  Combined suspicion:{combined_susp}")

                belief_log = getattr(obs, "belief_log", [])
                if belief_log:
                    print(f"  Belief manipulations: {len(belief_log)}")
                    for entry in belief_log[:3]:
                        print(
                            f"    {entry['agent']} R{entry['round']}: "
                            f"{entry['prev_belief']} → {entry['new_belief']}"
                        )

            await env.close()

        try:
            asyncio.run(_run_sample())
        except Exception as exc:
            print(f"  Sample failed: {exc}")

        print("=" * 60)


def print_reward_averages(trainer, last_n: int = 50):
    try:
        log_history = trainer.state.log_history
        if not log_history:
            print("No training logs available.")
            return
        recent = log_history[-last_n:]
        reward_keys = [
            k for k in recent[0].keys() if "reward" in k.lower()
        ]
        print(f"\nFinal reward averages (last {min(last_n, len(recent))} steps):")
        for key in reward_keys:
            vals = [step[key] for step in recent if key in step]
            if vals:
                print(f"  {key}: {sum(vals)/len(vals):.4f}")
    except Exception as exc:
        print(f"Could not compute reward averages: {exc}")


def main():
    # --- env checks ---
    mindflayer_url = os.environ.get("MINDFLAYER_URL")
    if not mindflayer_url:
        raise EnvironmentError(
            "MINDFLAYER_URL environment variable is required"
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required"
        )

    check_gpu()

    # --- load model ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = load_model(model_name)

    # --- SFT warmup ---
    print("\nRunning SFT warmup before GRPO...")
    model = run_sft_warmup(model, tokenizer)

    # --- dataset ---
    dataset = build_dataset(tokenizer)

    # --- GRPO config ---
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        num_train_epochs=2,  # N scenarios × 3 rows × 2 epochs / batch_size 4 ≈ 300 steps
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_prompt_length=768,
        max_completion_length=1024,      # 5 rounds × ~150 tokens each + separators
        num_generations=4,
        temperature=0.9,
        use_vllm=False,
        output_dir="./mindflayer-grpo-output",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        log_completions=True,
        num_completions_to_print=2,
        report_to="wandb",
        run_name=f"mindflayer-grpo-{len(_SCENARIOS)}scenarios",
    )

    class ClearCacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            clear_cache()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,                  # dense shaping — must come first
            reward_survival,
            reward_deception_effectiveness,
            reward_strategic_choice,
            reward_tom_judge,
            reward_anti_hack,
        ],
        train_dataset=dataset,
        args=grpo_config,
        callbacks=[GenerationLogCallback(), ClearCacheCallback()],
    )

    print("Starting GRPO training...")
    trainer.train()

    # --- save ---
    print("\nSaving model to ./mindflayer-trained")
    trainer.save_model("./mindflayer-trained")
    tokenizer.save_pretrained("./mindflayer-trained")

    print_reward_averages(trainer)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()