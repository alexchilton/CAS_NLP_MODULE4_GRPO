import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_MODE"] = "offline"
import sys

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# GRPO training script for Wordle using local CSV word list
# Adapted for PEFT (Parameter-Efficient Fine-Tuning)
# PATCHED VERSION - with temperature scheduling and improved training

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import re
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List
from logger_setup import logger
import argparse
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType

# # FIX: Import patched reward functions (with training_progress parameter)
# from reward_functions import output_format_check, uses_previous_feedback, guess_value

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import datetime

from transformers import BitsAndBytesConfig

from reward_functions import wordle_reward_func
from reward_functions import extract_guess_from_completion


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
    def __init__(self, start_temp=1.2, end_temp=0.8, transition_ratio=0.3):
        """
        Args:
            start_temp: Initial temperature (default 1.0 for high exploration)
            end_temp: Final temperature (default 0.3 for low exploration/high exploitation)
            transition_ratio: Fraction of training when temperature reaches end_temp (default 0.3 = 30%)
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.transition_ratio = transition_ratio
        logger.info(f"TemperatureSchedulerCallback initialized: {start_temp} -> {end_temp} over {transition_ratio*100}% of training")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps

            if progress < self.transition_ratio:
                temp_progress = progress / self.transition_ratio
                current_temp = self.start_temp - (self.start_temp - self.end_temp) * temp_progress
            else:
                current_temp = self.end_temp

            # ✅ Just update args.generation_kwargs directly
            if getattr(args, "generation_kwargs", None) is not None:
                args.generation_kwargs["temperature"] = float(current_temp)

            if state.global_step % 10 == 0:
                logger.info(
                    f"Step {state.global_step}/{state.max_steps} (progress: {progress:.2%}): "
                    f"Temperature = {current_temp:.3f}"
                )

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

def evaluate_avg_reward(model, tokenizer, dataset, n_samples=10):
    model.eval()
    rewards = []

    for i in range(min(n_samples, len(dataset))):
        example = dataset[i]
        prompt = example["prompt"]
        secret = example["secret_word"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        history = [example["past_guess_history"]]
        batch_rewards = wordle_reward_func(
            [completion],
            prompts=[prompt],
            secret_word=[secret],
            past_guess_history=history,
            word_list_path="../five_letter_words.csv",
        )
        reward = batch_rewards[0]
        rewards.append(reward)

        guess = extract_guess_from_completion(completion)

        print(f"Example {i}:")
        print(f"  secret = {secret}")
        print(f"  guess  = {guess}")
        print(f"  reward = {reward:.3f}")
        print(f"  raw    = {completion.strip()[:120]}...\n")

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Average reward over {len(rewards)} examples: {avg:.3f}")
    model.train()
    return avg


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

    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(
        columns={'secret': 'secret_word'}
    ).reset_index(drop=True)

    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(
        columns={'secret': 'secret_word'}
    ).reset_index(drop=True)

    # 🔹 Add strong instruction to always output a single 5-letter word
    PROMPT_SUFFIX = (
        "\n\nInstructions: Think step-by-step. "
        "1. List letters that are definitely NOT in the word. "
        "2. List letters you MUST include. "
        "3. Pick a valid 5-letter word that follows these rules. "
        "Format: <think>your reasoning</think><solution>WORD</solution>"
    )
    train_df["prompt"] = "Solve Wordle. Use the feedback history to find the secret.\n" + \
                         train_df["prompt"] + PROMPT_SUFFIX
    val_df["prompt"] = val_df["prompt"] + PROMPT_SUFFIX

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    inspect_data(train_df, val_df)
    return train_dataset, val_dataset

# ----------------------#
# 4. MODEL SETUP        #
# ----------------------#


def setup_model_and_tokenizer_peft():
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,    # Native 16-bit
        device_map="auto",
        attn_implementation="sdpa" # 4090 loves this
    )
    
    # Standard LoRA Setup
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer

# ----------------------#
# 5. REWARD             #
# ----------------------#

# in his own file

# ----------------------#
# 6. MAIN               #
# ----------------------#

def run_train(train_dataset, val_dataset):
    logger.info("Setting up model and tokenizer with PEFT...")
    model, tokenizer = setup_model_and_tokenizer_peft()
    logger.info("Model and tokenizer setup complete.")
    logger.info(f"Dataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}")

    training_args = GRPOConfig(
        output_dir="output_4090/wordle-grpo",

        # --- Time Control ---
        max_steps=150,
        logging_steps=1,


        # --- 4090 VRAM Sweet Spot ---
        per_device_train_batch_size=2,    # Small train batch
        num_generations=16,               # High generation count for better RL
        generation_batch_size=16,         # Match num_generations for efficiency
        gradient_accumulation_steps=8,    # Keeps total batch size stable
        
        # --- Performance ---
        bf16=True,                        # Use BF16 (Native on 4090)
        gradient_checkpointing=True,      # Keep this ON to save VRAM for longer <think> blocks
        
        # --- Learning Params ---
        learning_rate=1e-5,               # Standard for LoRA
        beta=0.04,                        # KL penalty to keep reasoning coherent
        
        # --- Reasoning Depth ---
        max_prompt_length=256,
        max_completion_length=512,        # 512 tokens allows for very deep thinking tags
        
        # --- Cluster Safety ---
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",                 # Use "none" unless you've verified internet
    )
    logger.info("Training arguments configured.")

    # ----- CALLBACKS -----
    temp_callback = TemperatureSchedulerCallback(start_temp=0.7, end_temp=0.4, transition_ratio=0.4)
    progress_callback = TrainingProgressCallback()

    # ----- WRAP REWARD -----
    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(
            *args, 
            word_list_path="../five_letter_words.csv",
            **kwargs
            )
    reward_func_with_model.__name__ = "wordle_reward_func"

    # ----- TRAINER -----
    logger.info("Initializing GRPOTrainer (this may take a moment)...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[temp_callback, progress_callback],
    )
    logger.info("GRPOTrainer initialized successfully!")

    # ----- TRAIN LOOP -----
    logger.info("Starting training loop...")
    torch.cuda.empty_cache()
    trainer.train()
    logger.info("TRL GRPOTrainer training complete (PEFT).")

    # ----- SAVE FINAL MODEL -----
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model and tokenizer saved to: {final_model_dir}")

    # ----- SAVE BEST MODEL IF AVAILABLE -----
    if hasattr(trainer, "state") and hasattr(trainer.state, "best_model_checkpoint"):
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt:
            logger.info(f"Best model checkpoint found at: {best_ckpt}")
            best_model_dir = os.path.join(training_args.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            # you can re-load from best_ckpt here if you want, but simplest:
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"Best model saved to: {best_model_dir}")
        else:
            logger.warning("No best model checkpoint found.")
    else:
        logger.warning("Trainer does not have best_model_checkpoint attribute.")

    # ----- PLOT LOSS IF AVAILABLE -----
    try:
        from plot_loss import plot_loss
        log_file = os.path.join(training_args.output_dir, "trainer_state.jsonl")
        if os.path.exists(log_file):
            plot_loss(log_file, output_dir=training_args.output_dir)
        else:
            logger.warning(f"Log file {log_file} not found. Skipping loss plot.")
    except Exception as e:
        logger.warning(f"Could not plot loss curves: {e}")

    # ----- SAVE METADATA -----
    metadata = {
        "training_date": datetime.datetime.now().isoformat(),
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_args": {
            "num_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "temperature_schedule": "0.7 → 0.4 at 40%",
            "max_completion_length": training_args.max_completion_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
        },
        "notes": [
            "Dense reward based on secret/guess overlap",
            "Penalty for invalid guesses",
            "Penalty for repeated guesses",
            "Extraction from <solution>WORD</solution> where available",
        ],
    }
    meta_path = os.path.join(training_args.output_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved to: {meta_path}")

    logger.info(f"Trainer.state: {trainer.state}")

def run_eval(checkpoint_dir, n_samples=10):
    logger.info(f"Loading model for evaluation from: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map="auto",
    )

    # Load data (train + val, but we only use val)
    train_dataset, val_dataset = load_and_prepare_data()

    logger.info(f"Evaluating on {n_samples} samples from validation set...")

    avg = evaluate_avg_reward(
        model,
        tokenizer,
        val_dataset,
        n_samples=n_samples
    )

    logger.info(f"Average reward on validation: {avg:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Whether to train the model or evaluate a saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output5/wordle-grpo-rtx4070/final_model",
        help="Path to model checkpoint for eval mode.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate in eval mode.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        logger.info("Loading and preparing data for TRAIN mode...")
        train_dataset, val_dataset = load_and_prepare_data()
        # optional mini-set:
        # train_dataset = train_dataset.select(range(20))
        # val_dataset   = val_dataset.select(range(5))
        logger.info("Data loaded and prepared.")
        run_train(train_dataset, val_dataset)
    else:
        logger.info("Running in EVAL mode...")
        run_eval(args.checkpoint, n_samples=args.n_samples)
