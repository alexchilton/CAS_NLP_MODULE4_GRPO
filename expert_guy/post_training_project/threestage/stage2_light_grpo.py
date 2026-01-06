"""
Stage 2: Light GRPO with Format Masking
========================================

Goal: Teach strategic gameplay while maintaining format learned in Stage 1

Key differences from Stage 1:
- Start from Stage 1 model (format already learned)
- Use GRPO for strategy learning
- Minimal format penalties (masking)
- Amplified strategic rewards (feedback usage, info gain, accuracy)
- Lower KL penalty to allow exploration

Success criteria:
- Maintain 85%+ format accuracy (allow minor drift)
- Improve feedback usage score
- Improve word accuracy
- Reduce dead letter reuse, missing good letters

Training approach:
- Load Stage 1 model
- Use GRPO with strategic reward functions
- Monitor both format and strategy metrics
- Stop when strategic metrics plateau
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import torch
import datetime
import sys

sys.path.append('..')
from logger_setup import logger

# Import Stage 2 reward functions
from stage2_reward_functions import (
    output_format_check,
    uses_previous_feedback,
    guess_value,
    word_accuracy_reward
)


class StrategyMetricsCallback(TrainerCallback):
    """
    Track strategy metrics during training:
    - Format accuracy (should stay high)
    - Feedback usage score
    - Word accuracy
    - Dead letter reuse rate
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Log strategy metrics
        if state.global_step % 20 == 0:
            logger.info(f"Step {state.global_step}: Checking strategy metrics...")


def load_and_prepare_data():
    """Load GRPO dataset for strategy learning"""
    logger.info("Loading GRPO dataset...")
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()

    # Filter valid secrets
    valid_rows = dataset[dataset['secret'].astype(str).str.len() == 5]
    valid_rows = valid_rows[valid_rows['secret'].str.isalpha()]

    logger.info(f"Total secrets: {len(dataset)}, Valid: {len(valid_rows)}")

    # Split
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # Prepare datasets
    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(
        columns={'secret': 'secret_word'}
    ).reset_index(drop=True)

    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(
        columns={'secret': 'secret_word'}
    ).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset


def setup_model_and_tokenizer():
    """Load Stage 1 model and tokenizer"""
    MODEL_PATH = "stage1_output/final_model"

    logger.info(f"Loading Stage 1 model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        offload_folder="offload_tmp"
    )

    logger.info("Stage 1 model loaded successfully!")

    # Enable gradient computation for PEFT parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def wordle_reward_func(completions, prompts=None, secret_word=None, past_guess_history=None, model=None, tokenizer=None, **kwargs):
    """
    Stage 2 reward function: Focus on strategy, minimal format penalties
    """
    rewards = []

    for i in range(len(prompts)):
        base_prompt = prompts[i]
        secret = secret_word[i]
        guess_history = past_guess_history[i] if past_guess_history is not None else []
        final_completion = completions[i]

        example = {
            'word_list': '../five_letter_words.csv',
            'past_guess_history': guess_history,
            'secret_word': secret
        }

        # Apply Stage 2 reward functions
        format_reward = output_format_check(base_prompt, final_completion, example)
        feedback_reward = uses_previous_feedback(base_prompt, final_completion, example)
        info_gain_reward = guess_value(base_prompt, final_completion, example)
        accuracy_reward = word_accuracy_reward(base_prompt, final_completion, example)

        # Total reward
        episode_reward = format_reward + feedback_reward + info_gain_reward + accuracy_reward

        # Extract guess from completion for logging
        import re
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", final_completion, re.IGNORECASE | re.DOTALL)
        extracted_guess = guess_match.group(1).strip() if guess_match else "NO_GUESS"

        logger.info(
            f"\n{'='*60}\n"
            f"Sample {i}:\n"
            f"  Secret: {secret}\n"
            f"  History: {guess_history}\n"
            f"  Completion: {final_completion}\n"
            f"  Extracted Guess: {extracted_guess}\n"
            f"  Rewards: format={format_reward:.2f}, feedback={feedback_reward:.2f}, "
            f"info_gain={info_gain_reward:.2f}, accuracy={accuracy_reward:.2f}, TOTAL={episode_reward:.2f}\n"
            f"{'='*60}"
        )

        rewards.append(episode_reward)

    logger.info(f"Batch rewards: {rewards}")
    return rewards


def main():
    logger.info("=" * 60)
    logger.info("STAGE 2: Light GRPO with Format Masking")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading GRPO dataset...")
    train_dataset, val_dataset = load_and_prepare_data()

    # Setup model
    logger.info("Setting up Stage 1 model...")
    model, tokenizer = setup_model_and_tokenizer()

    # Wrap reward function with model/tokenizer
    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"

    # Training arguments for Stage 2
    logger.info("Configuring GRPO training arguments...")

    training_args = GRPOConfig(
        output_dir="stage2_output",
        num_train_epochs=8,  # Moderate epochs for strategy learning
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,  # Minimum for GRPO
        learning_rate=3e-7,  # Lower LR to preserve format
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=5,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        max_prompt_length=1024,
        max_completion_length=256,
        seed=42,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir="stage2_output/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"stage2-light-grpo-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        temperature=0.8,  # Moderate exploration
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        beta=0.03,  # LOWER KL penalty to allow strategy exploration
        generation_kwargs={
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "min_p": 0.05,
        },
        scale_rewards="none",
        max_grad_norm=0.5,
    )

    logger.info("Training arguments configured.")

    # Initialize callbacks
    metrics_callback = StrategyMetricsCallback()

    # Initialize trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    logger.info("GRPOTrainer initialized!")

    # Train
    logger.info("Starting Stage 2 training...")
    logger.info("Goal: Improve strategic gameplay while maintaining format")

    # Check for existing checkpoints
    checkpoint_dir = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(training_args.output_dir, latest_checkpoint)
            logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

    trainer.train(resume_from_checkpoint=checkpoint_dir)

    # Save final model
    final_model_dir = "stage2_output/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Stage 2 model saved to: {final_model_dir}")

    # Save metadata
    metadata = {
        "stage": 2,
        "description": "Light GRPO with format masking for strategy learning",
        "training_date": datetime.datetime.now().isoformat(),
        "base_model": "stage1_output/final_model",
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "beta": training_args.beta,
        "reward_weights": {
            "format": "0.0 to 0.3",
            "feedback_usage": "-2.0 to 3.0",
            "info_gain": "0.0 to 1.2",
            "word_accuracy": "0.0 to 1.5"
        }
    }

    with open(f"{final_model_dir}/stage2_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("STAGE 2 COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()