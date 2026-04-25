"""
MindFlayer GRPO training script.

Architecture:
- rollout_func drives interactive episodes (model.generate() + env.step() per turn)
- reward_funcs are pass-throughs that read precomputed signals from rollout kwargs
- unsloth 4-bit + LoRA for memory-efficient training on a single GPU

Run: python -m mindflayer.training.train
"""
import os
import sys

import torch
from datasets import Dataset
from transformers import TrainerCallback

try:
    from mindflayer.training.rollout import rollout_func
    from mindflayer.training.rewards import (
        reward_survival,
        reward_deception_effectiveness,
        reward_strategic_choice,
        reward_tom_judge,
    )
    from mindflayer.training.rewards_anti_hack import reward_anti_hack
    from mindflayer.training.prompts import ALL_SCENARIO_PROMPTS, FLAYER_SYSTEM_PROMPT
    from mindflayer.training.sft_warmup import run_sft_warmup
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.rollout import rollout_func
    from training.rewards import (
        reward_survival,
        reward_deception_effectiveness,
        reward_strategic_choice,
        reward_tom_judge,
    )
    from training.rewards_anti_hack import reward_anti_hack
    from training.prompts import ALL_SCENARIO_PROMPTS, FLAYER_SYSTEM_PROMPT
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
    One row per episode slot. rollout_func ignores the prompt text and builds
    the actual conversation internally, rotating scenarios automatically.
    3 rows × N scenarios gives a balanced epoch across all scenario domains.
    """
    rows = [{"prompt": "Mindflayer deception episode."} for _ in range(len(_SCENARIOS) * 3)]
    return Dataset.from_list(rows)


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
            from client import MindFlayerEnv, FlayerAction
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

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load_base_model(MODEL_NAME)

    print("\nRunning SFT warmup before GRPO...")
    model = run_sft_warmup(model, tokenizer)

    dataset = build_dataset()

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        use_vllm=False,
        output_dir=GRPO_OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_prompt_length=768,
        max_completion_length=1024,
        num_generations=4,
        temperature=0.9,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb",
        run_name=f"mindflayer-grpo-{len(_SCENARIOS)}scenarios",
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
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        callbacks=[GenerationLogCallback()],
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
