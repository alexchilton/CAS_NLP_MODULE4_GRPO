import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GRPO training script for Gemma-3-4b starting from SFT checkpoint
# Loads the SFT-trained LoRA adapters and continues with GRPO

import os
import re
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List
from logger_setup import logger

from sklearn.model_selection import train_test_split

from reward_functions import output_format_check, uses_previous_feedback, guess_value, word_accuracy_reward

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import datetime

# PEFT imports
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# ----------------------#
# 0. TEMPERATURE CALLBACK #
# ----------------------#

class TemperatureSchedulerCallback(TrainerCallback):
    """Gradually reduces temperature during training"""
    def __init__(self, start_temp=1.0, end_temp=0.3, transition_ratio=0.3):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.transition_ratio = transition_ratio
        logger.info(f"TemperatureScheduler: {start_temp} → {end_temp} over {transition_ratio*100}% of training")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps
            if progress < self.transition_ratio:
                temp_progress = progress / self.transition_ratio
                current_temp = self.start_temp - (self.start_temp - self.end_temp) * temp_progress
            else:
                current_temp = self.end_temp
            
            if hasattr(kwargs, 'model') and hasattr(args, 'generation_kwargs'):
                args.generation_kwargs['temperature'] = current_temp
                if state.global_step % 10 == 0:
                    logger.info(f"Step {state.global_step}/{state.max_steps}: Temperature = {current_temp:.3f}")

# Global training progress tracker
_training_progress = 0.0

class TrainingProgressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        global _training_progress
        if state.max_steps > 0:
            _training_progress = state.global_step / state.max_steps

# ----------------------#
# 1. UTILS              #
# ----------------------#

def inspect_data(train_df, val_df, n=3):
    logger.info(f"Train DataFrame shape: {train_df.shape}")
    logger.info(f"Validation DataFrame shape: {val_df.shape}")
    print("\n--- Train DataFrame Sample ---")
    print(train_df.head(n))
    print("\n--- Validation DataFrame Sample ---")
    print(val_df.head(n))

# ----------------------#
# 2. DATA LOADING       #
# ----------------------#

def load_and_prepare_data():
    """Load Wordle word list and create GRPO training prompts"""
    logger.info("Loading five_letter_words.csv...")
    words_df = pd.read_csv("five_letter_words.csv")
    logger.info(f"Loaded {len(words_df)} five-letter words")
    logger.info(f"CSV columns: {words_df.columns.tolist()}")
    
    # REDUCE: Use only a subset for faster training (e.g., 500 words instead of 10k+)
    SUBSET_SIZE = 500
    words_df = words_df.sample(n=min(SUBSET_SIZE, len(words_df)), random_state=42)
    logger.info(f"Using subset of {len(words_df)} words for training")
    
    # Build dataset for GRPO
    dataset_rows = []
    for idx, row in words_df.iterrows():
        word = row['Word'].upper()  # Column is 'Word' not 'word'
        prompt = f"Play Wordle. The secret word is a 5-letter word. Make your first guess.\n<think>"
        dataset_rows.append({
            'prompt': prompt,
            'secret': word,
            'past_guess_history': '[]'
        })
    
    logger.info(f"Created {len(dataset_rows)} training examples")
    
    # Split into train/val
    valid_rows = pd.DataFrame(dataset_rows)
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_rows)}, Validation: {len(val_rows)}")
    
    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    inspect_data(train_df, val_df)
    return train_dataset, val_dataset

# ----------------------#
# 3. MODEL SETUP        #
# ----------------------#

def setup_model_and_tokenizer_gemma_sft():
    """Load Gemma-3-4b base model + SFT LoRA adapters"""
    load_dotenv()
    
    BASE_MODEL_NAME = "google/gemma-3-4b-it"
    ADAPTER_PATH = "output_sft/wordle-sft-peft/final_model"
    HF_TOKEN = None

    logger.info(f"Loading base model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        token=HF_TOKEN, 
        device_map="auto",
        torch_dtype=torch.float32  # Use FP32 for numerical stability
    )
    logger.info("Base model loaded!")

    logger.info(f"Loading SFT LoRA adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    logger.info("SFT adapters loaded!")
    
    # Enable gradient checkpointing for the adapters
    model.enable_input_require_grads()
    
    # Ensure LoRA parameters are trainable
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
    logger.info(f"Trainable: {trainable_params:,} || Total: {total_params:,} || %: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer

# ----------------------#
# 4. REWARD FUNCTION    #
# ----------------------#

def wordle_reward_func(prompts, completions, secret_word, past_guess_history, model, tokenizer, **kwargs):
    """Reward function for Wordle GRPO training"""
    rewards = []
    
    # Get training progress for staged penalties
    global _training_progress
    
    for prompt, completion, secret, past_hist in zip(prompts, completions, secret_word, past_guess_history):
        total_reward = 0.0
        
        # Build example dict for reward functions
        example = {
            'secret_word': secret,
            'past_guess_history': past_hist
        }
        
        # 1. Format check (takes training_progress)
        format_score = output_format_check(prompt, completion, example, training_progress=_training_progress)
        total_reward += format_score
        
        # 2. Feedback usage (no training_progress param)
        feedback_score = uses_previous_feedback(prompt, completion, example)
        total_reward += feedback_score
        
        # 3. Guess value (no training_progress param)
        guess_score = guess_value(prompt, completion, example)
        total_reward += guess_score
        
        # 4. Word accuracy (dense signal)
        accuracy_score = word_accuracy_reward(prompt, completion, example)
        total_reward += accuracy_score
        
        rewards.append(total_reward)
    
    return rewards

# ----------------------#
# 5. MAIN               #
# ----------------------#

if __name__ == "__main__":
    logger.info("Starting GRPO training for Gemma-3-4b (from SFT checkpoint)...")
    
    logger.info("Loading data...")
    train_dataset, val_dataset = load_and_prepare_data()
    
    logger.info("Setting up Gemma + SFT adapters...")
    model, tokenizer = setup_model_and_tokenizer_gemma_sft()
    
    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"
    
    logger.info("Configuring GRPO training...")
    
    training_args = GRPOConfig(
        output_dir="output_grpo_gemma3/wordle-grpo-gemma",
        num_train_epochs=3,  # REDUCED: 3 epochs instead of 5
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        learning_rate=5e-7,
        logging_steps=5,  # Log every 5 steps (less spam)
        eval_strategy="steps",
        eval_steps=50,  # Eval less frequently
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,  # Keep fewer checkpoints
        bf16=False,
        fp16=False,  # Disable FP16 - use FP32 for MPS stability
        remove_unused_columns=False,
        max_prompt_length=1024,
        max_completion_length=512,
        seed=42,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir="output_grpo_gemma3/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"wordle-grpo-gemma-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        beta=0.1,  # KL penalty
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
    
    logger.info("Training config ready!")
    
    # Callbacks
    temp_callback = TemperatureSchedulerCallback(start_temp=0.8, end_temp=0.4, transition_ratio=0.4)
    progress_callback = TrainingProgressCallback()
    
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Changed from 'tokenizer' to 'processing_class'
        callbacks=[temp_callback, progress_callback],
    )
    
    logger.info("Starting GRPO training...")
    trainer.train()
    
    logger.info("Training complete!")
    
    # Save final model
    final_dir = os.path.join(training_args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to: {final_dir}")
    
    logger.info("GRPO training complete! Model ready for testing.")
