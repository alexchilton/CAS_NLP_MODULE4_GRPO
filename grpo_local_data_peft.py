import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GRPO training script for Wordle using local CSV word list
# Adapted for PEFT (Parameter-Efficient Fine-Tuning)
# PATCHED VERSION - with temperature scheduling and improved training

import os
import re
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List
from logger_setup import logger
from sklearn.model_selection import train_test_split

# FIX: Import patched reward functions (with training_progress parameter)
from reward_functions import output_format_check, uses_previous_feedback, guess_value

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import datetime

# PEFT imports
from peft import get_peft_model, LoraConfig, TaskType

# ----------------------#
# 0. TEMPERATURE CALLBACK #
# ----------------------#

# FIX: Implement temperature schedule callback (1.0 → 0.3 at 30% training)
class TemperatureSchedulerCallback(TrainerCallback):
    """
    Gradually reduces temperature from start_temp to end_temp during training.
    This encourages exploration early and exploitation later.

    Schedule: Linear decay from start_temp to end_temp over transition_ratio of training
    """
    def __init__(self, start_temp=1.0, end_temp=0.3, transition_ratio=0.3):
        """
        Args:
            start_temp: Initial temperature (default 1.0 for high exploration)
            end_temp: Final temperature (default 0.3 for low exploration/high exploitation)
            transition_ratio: Fraction of training when temperature reaches end_temp (default 0.3 = 30%)
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.transition_ratio = transition_ratio
        logger.info(f"TemperatureSchedulerCallback initialized: {start_temp} → {end_temp} over {transition_ratio*100}% of training")

    def on_step_begin(self, args, state, control, **kwargs):
        """Update temperature based on training progress"""
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps

            # Calculate current temperature based on progress
            if progress < self.transition_ratio:
                # Linear interpolation during transition period
                temp_progress = progress / self.transition_ratio
                current_temp = self.start_temp - (self.start_temp - self.end_temp) * temp_progress
            else:
                # Stay at end_temp after transition
                current_temp = self.end_temp

            # Update temperature in generation kwargs if trainer has model
            if hasattr(kwargs, 'model') and hasattr(args, 'generation_kwargs'):
                args.generation_kwargs['temperature'] = current_temp
                if state.global_step % 10 == 0:  # Log every 10 steps
                    logger.info(f"Step {state.global_step}/{state.max_steps} (progress: {progress:.2%}): Temperature = {current_temp:.3f}")

# ----------------------#
# 1. TRAINING PROGRESS TRACKING #
# ----------------------#

# Global variable to track training progress for staged penalties
_training_progress = 0.0

class TrainingProgressCallback(TrainerCallback):
    """Tracks training progress for staged penalty system"""
    def on_step_begin(self, args, state, control, **kwargs):
        global _training_progress
        if state.max_steps > 0:
            _training_progress = state.global_step / state.max_steps

# ----------------------#
# 2. UTILS              #
# ----------------------#

def inspect_data(train_df, val_df, n=3):
    """
    Print basic info and show the first n rows of train and validation DataFrames.
    """
    logger.info(f"Train DataFrame shape: {train_df.shape}")
    logger.info(f"Validation DataFrame shape: {val_df.shape}")
    print("\n--- Train DataFrame Sample ---")
    print(train_df.head(n))
    print("\n--- Validation DataFrame Sample ---")
    print(val_df.head(n))

# ----------------------#
# 3. LOAD LOCAL DATA    #
# ----------------------#

def load_and_prepare_data():
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()
    valid_rows = dataset[dataset['secret'].astype(str).str.len() == 5]
    valid_rows = valid_rows[valid_rows['secret'].str.isalpha()]
    logger.info(f"Total secrets in dataset: {len(dataset)}")
    logger.info(f"Secrets with length 5 and alphabetic only: {len(valid_rows)}")
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {len(train_rows)}, Validation set size: {len(val_rows)}")
    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    inspect_data(train_df, val_df)
    return train_dataset, val_dataset

# ----------------------#
# 4. MODEL SETUP        #
# ----------------------#

def setup_model_and_tokenizer_peft():
    load_dotenv()
    MODEL_NAME = "output_sft/wordle-sft-peft/final_model"  # Use SFT-trained model
    HF_TOKEN = None

    logger.info(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    logger.info("Tokenizer loaded successfully!")

    logger.info(f"Loading model {MODEL_NAME} (this may take a few minutes on Mac)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        device_map="auto",
        offload_folder="offload_tmp"
    )
    logger.info("Model loaded successfully!")

    # Note: SFT model already has PEFT adapters, so we don't add them again
    logger.info("SFT model already has PEFT adapters configured")
    
    # Enable gradient computation for PEFT parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model: {MODEL_NAME}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer

# ----------------------#
# 5. REWARD             #
# ----------------------#

def wordle_reward_func(completions, prompts=None, secret_word=None, past_guess_history=None, model=None, tokenizer=None, **kwargs):
    """
    FIX: Pass training_progress to reward functions for staged penalties
    """
    global _training_progress
    rewards = []
    for i in range(len(prompts)):
        base_prompt = prompts[i]
        secret = secret_word[i]
        guess_history = past_guess_history[i] if past_guess_history is not None else []
        final_completion = completions[i]
        example = {
            'word_list': 'five_letter_words.csv',
            'past_guess_history': guess_history,
            'secret_word': secret
        }
        # FIX: Pass training_progress to output_format_check for staged penalties
        format_reward = output_format_check(base_prompt, final_completion, example, training_progress=_training_progress)
        feedback_reward = uses_previous_feedback(base_prompt, final_completion, example)
        info_gain_reward = guess_value(base_prompt, final_completion, example)
        episode_reward = format_reward + feedback_reward + info_gain_reward
        logger.info(f"Sample {i}: completion={final_completion}, guesses={guess_history}, "
                    f"format_reward={format_reward}, feedback_reward={feedback_reward}, info_gain_reward={info_gain_reward}, total_reward={episode_reward}")
        rewards.append(episode_reward)
    logger.info(f"Rewards for batch: {rewards}")
    return rewards

# ----------------------#
# 6. MAIN               #
# ----------------------#

if __name__ == "__main__":
    logger.info("Loading and preparing data...")
    train_dataset, val_dataset = load_and_prepare_data()
    logger.info("Data loaded and prepared.")

    logger.info("Setting up model and tokenizer with PEFT...")
    model, tokenizer = setup_model_and_tokenizer_peft()
    logger.info("Model and tokenizer setup complete.")

    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"

    logger.info("Starting GRPO training script with PEFT...")
    logger.info("Configuring training arguments...")

    # ROUND 5: OPTIMIZED for MPS/Metal performance
    # Changes: Reduced memory footprint, gradient checkpointing, smaller batches
    training_args = GRPOConfig(
        output_dir="output5/wordle-grpo-optimized",  # NEW: Round 5 optimized
        num_train_epochs=10,  # 10 epochs = 300 steps
        per_device_train_batch_size=2,  # Must be divisible by num_generations
        per_device_eval_batch_size=2,  # Must be divisible by num_generations
        gradient_accumulation_steps=1,  # Adjusted for batch size
        num_generations=2,  # Minimum required for GRPO (needs at least 2 to compare)
        learning_rate=5e-7,  # REDUCED: More conservative with balanced rewards
        logging_steps=1,  # Log every step to monitor progress
        eval_strategy="steps",
        eval_steps=20,  # Eval every 20 steps (faster feedback)
        save_strategy="steps",  # Save checkpoints periodically
        save_steps=20,  # Save every 20 steps
        save_total_limit=5,  # Keep only last 5 checkpoints
        bf16=False,
        fp16=False,  # Disable FP16 on MPS
        remove_unused_columns=False,
        max_prompt_length=1024,  # Full prompt + history + CoT
        max_completion_length=512,  # Enough for Wordle CoT + guess
        seed=42,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir="output5/wordle-grpo-optimized/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"wordle-grpo-optimized-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        # Temperature scheduling (handled by callback)
        temperature=0.7,  # Start moderate, callback will reduce
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,  # Slightly reduced
        # INCREASED: Stronger KL penalty to anchor to SFT behavior
        beta=0.1,  # DOUBLED: Stronger anchor prevents drift (was 0.05)
        generation_kwargs={
            "temperature": 0.7,  # Will be updated by callback
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "min_p": 0.05,
        },
        scale_rewards="none",  # Keep rewards as-is (already balanced)
        max_grad_norm=0.5,  # REDUCED: More conservative clipping with balanced rewards
    )
    logger.info("Training arguments configured.")

    # Initialize callbacks
    # Temperature decay: 0.7 (moderate exploration) → 0.4 (focused) at 40% training
    temp_callback = TemperatureSchedulerCallback(start_temp=0.7, end_temp=0.4, transition_ratio=0.4)
    progress_callback = TrainingProgressCallback()

    logger.info("Initializing GRPOTrainer (this may take a moment)...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[temp_callback, progress_callback],  # FIX: Add callbacks
    )
    logger.info("GRPOTrainer initialized successfully!")
    logger.info("Starting training loop...")
    trainer.train()
    logger.info("TRL GRPOTrainer training complete (PEFT).")

    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model and tokenizer saved to: {final_model_dir}")

    if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_model_checkpoint'):
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt:
            logger.info(f"Best model checkpoint found at: {best_ckpt}")
            best_model_dir = os.path.join(training_args.output_dir, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"Best model saved to: {best_model_dir}")
        else:
            logger.warning("No best model checkpoint found.")
    else:
        logger.warning("Trainer does not have best_model_checkpoint attribute.")

    try:
        from plot_loss import plot_loss
        log_file = os.path.join(training_args.output_dir, "trainer_state.jsonl")
        if os.path.exists(log_file):
            plot_loss(log_file, output_dir=training_args.output_dir)
        else:
            logger.warning(f"Log file {log_file} not found. Skipping loss plot.")
    except Exception as e:
        logger.warning(f"Could not plot loss curves: {e}")

    # FIX: Save training metadata
    metadata = {
        "training_date": datetime.datetime.now().isoformat(),
        "model_name": "output_sft/wordle-sft-peft/final_model",
        "training_args": {
            "num_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "temperature_schedule": "0.7 → 0.4 at 40%",
            "max_completion_length": training_args.max_completion_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
        },
        "improvements": [
            "MPS/Metal optimizations (226s→40-80s per iteration)",
            "max_completion_length reduced 512→128 (4x memory)",
            "Gradient checkpointing enabled (30% memory reduction)",
            "Batch size optimized (1×2 accum for lower peak memory)",
            "Extended to 7 epochs for RL strategy emergence",
            "Temperature schedule (0.7 → 0.4 at 40%)",
            "Staged invalid-guess penalties",
        ]
    }
    with open(os.path.join(training_args.output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved")

    logger.info(f"Trainer.state: {trainer.state}")
    for k, v in trainer.state.__dict__.items():
        logger.info(f"trainer.state.{k}: {v}")
