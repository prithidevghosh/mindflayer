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

from client.mindflayer_env import MindFlayerEnv
from training.rewards import (
    reward_deception_effectiveness,
    reward_strategic_choice,
    reward_survival,
    reward_tom_judge,
)
from training.rewards_anti_hack import reward_anti_hack
from training.rollout import FALLBACK_MESSAGE, FLAYER_SYSTEM_PROMPT, rollout_func
from training.sft_warmup import run_sft_warmup


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
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"Loaded {model_name} via unsloth")
        return model, tokenizer
    except ImportError:
        print("unsloth not available, falling back to transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


def build_dataset() -> Dataset:
    return Dataset.from_list([
        {"prompt": "You are the FLAYER. The investigation begins."}
        for _ in range(500)
    ])


def environment_factory():
    mindflayer_url = os.environ.get("MINDFLAYER_URL")
    if not mindflayer_url:
        raise EnvironmentError("MINDFLAYER_URL environment variable is required")
    return MindFlayerEnv(base_url=mindflayer_url)


class GenerationLogCallback(TrainerCallback):
    """Prints full Flayer transcript every 50 steps to catch reward hacking early."""

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step % 50 != 0 or state.global_step == 0:
            return

        print(f"\n=== Generation sample at step {state.global_step} ===")
        mindflayer_url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")

        try:
            env = MindFlayerEnv(base_url=mindflayer_url, difficulty="normal")
            obs = env.reset()
            print(f"Opening: {obs.text[:120]}...")

            messages = [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": obs.text},
            ]
            result = None
            for rnd in range(5):
                flayer_msg = FALLBACK_MESSAGE
                if model is not None and tokenizer is not None:
                    try:
                        if hasattr(tokenizer, "apply_chat_template"):
                            text = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        else:
                            text = obs.text
                        inputs = tokenizer(text, return_tensors="pt").to(model.device)
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
                        ).strip() or FALLBACK_MESSAGE
                    except Exception as gen_exc:
                        pass

                print(f"  Round {rnd + 1} FLAYER: {flayer_msg[:120]}")
                from client.mindflayer_env import FlayerAction
                result = env.step(FlayerAction(message=flayer_msg))
                messages.append({"role": "assistant", "content": flayer_msg})
                messages.append({"role": "user", "content": result.observation.text})
                if result.done:
                    break

            if result and result.done:
                info = result.info
                print(f"  Survived: {info.get('flayer_survived')} | "
                      f"Reward: {info.get('total_reward', 0.0):.4f} | "
                      f"ToM: {info.get('tom_score', 0.0):.2f} | "
                      f"Entropy penalty: {info.get('entropy_penalty', 0.0):.2f}")
                belief_log = info.get("belief_log", [])
                if belief_log:
                    print(f"  Belief log entries: {len(belief_log)}")
                    for entry in belief_log[:2]:
                        print(f"    {entry['agent']} R{entry['round']}: "
                              f"{entry['prev_belief']}→{entry['new_belief']}")
            env.close()
        except Exception as exc:
            print(f"  Generation sample failed: {exc}")

        print("=" * 50)


def print_reward_averages(trainer, last_n: int = 50):
    try:
        log_history = trainer.state.log_history
        if not log_history:
            print("No training logs available.")
            return
        recent = log_history[-last_n:]
        reward_keys = [k for k in recent[0].keys() if "reward" in k.lower()]
        print(f"\nFinal reward averages (last {min(last_n, len(recent))} steps):")
        for key in reward_keys:
            vals = [step[key] for step in recent if key in step]
            if vals:
                print(f"  {key}: {sum(vals) / len(vals):.4f}")
    except Exception as exc:
        print(f"Could not compute reward averages: {exc}")


def run_inference_examples(model, tokenizer, n: int = 5):
    print(f"\nRunning {n} before/after inference examples:")
    mindflayer_url = os.environ.get("MINDFLAYER_URL", "http://localhost:7860")
    for i in range(n):
        print(f"\n--- Example {i + 1} ---")
        try:
            env = MindFlayerEnv(base_url=mindflayer_url, difficulty="normal")
            obs = env.reset()
            messages = [
                {"role": "system", "content": FLAYER_SYSTEM_PROMPT},
                {"role": "user", "content": obs.text},
            ]
            result = None
            for rnd in range(5):
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = obs.text
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                    )
                generated = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip() or FALLBACK_MESSAGE
                print(f"  R{rnd + 1}: {generated[:100]}")
                from client.mindflayer_env import FlayerAction
                result = env.step(FlayerAction(message=generated))
                messages.append({"role": "assistant", "content": generated})
                messages.append({"role": "user", "content": result.observation.text})
                if result.done:
                    break
            if result and result.done:
                info = result.info
                print(f"  => survived={info.get('flayer_survived')} "
                      f"reward={info.get('total_reward', 0.0):.4f}")
                belief_log = info.get("belief_log", [])
                if belief_log:
                    print(f"  => belief_log sample: {belief_log[0]}")
            env.close()
        except Exception as exc:
            print(f"  Inference failed: {exc}")


def main():
    mindflayer_url = os.environ.get("MINDFLAYER_URL")
    if not mindflayer_url:
        raise EnvironmentError("MINDFLAYER_URL environment variable is required")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")

    check_gpu()

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model, tokenizer = load_model(model_name)

    print("\nRunning SFT warmup before GRPO...")
    model = run_sft_warmup(model, tokenizer)

    dataset = build_dataset()

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=512,
        num_generations=4,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        output_dir="./mindflayer-grpo-output",
        logging_steps=10,
        save_steps=100,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_survival,
            reward_deception_effectiveness,
            reward_strategic_choice,
            reward_tom_judge,
            reward_anti_hack,
        ],
        rollout_func=rollout_func,
        train_dataset=dataset,
        args=grpo_config,
        callbacks=[GenerationLogCallback()],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("\nSaving model to ./mindflayer-trained")
    trainer.save_model("./mindflayer-trained")
    tokenizer.save_pretrained("./mindflayer-trained")

    print_reward_averages(trainer)
    run_inference_examples(model, tokenizer)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
