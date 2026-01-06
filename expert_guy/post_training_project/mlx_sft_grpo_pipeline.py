
#!/usr/bin/env python3
"""
MLX-GRPO Wordle Training Pipeline
Based on: https://github.com/Doriandarko/MLX-GRPO

STEP 1: Convert model (run this first, separately)
STEP 2: 2-Stage Training:
  - SFT: Supervised fine-tuning with LoRA
  - GRPO: Group Relative Policy Optimization with strict rewards

Uses 4-bit quantization for Apple Silicon optimization with Gemma-3-4b
"""

import argparse
import json
import math
import time
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import ast

from logger_setup import logger

# Import strict reward functions
from reward_functions_strict import (
    output_format_check,
    uses_previous_feedback,
    guess_value
)

# ========================
# CONFIGURATION
# ========================

class Config:
    # Model - Gemma-3-4b
    hf_model_name = "google/gemma-3-4b-it"  # Original HuggingFace model
    mlx_model_path = "mlx_models/gemma-3-4b-it-4bit"  # 4-bit quantized for memory efficiency
    
    # Paths
    output_dir = "mlx_output"
    word_list = "five_letter_words.csv"
    
    # SFT
    sft_epochs = 3
    sft_batch_size = 1
    sft_lr = 2e-5
    sft_iters = 1000
    
    # GRPO
    grpo_epochs = 10
    grpo_batch_size = 2
    grpo_lr = 1e-3
    grpo_iters = 3000
    num_generations = 2
    temperature_start = 0.3
    temperature_end = 0.1
    beta = 0.1  # KL penalty
    gamma = 0.99  # Discount factor
    
    # LoRA
    lora_rank = 64
    lora_alpha = 16
    lora_dropout = 0.1
    lora_layers = 16
    
    # Generation
    max_tokens = 256
    top_p = 0.9


# ========================
# STEP 1: MODEL CONVERSION
# ========================

def convert_model_to_mlx():
    """
    Convert HuggingFace Gemma-3-4b to MLX format with 4-bit quantization
    
    This should be run ONCE before training.
    Uses mlx-lm's convert command.
    """
    output_path = Path(Config.mlx_model_path)
    
    # Check if already converted
    if output_path.exists() and (output_path / "config.json").exists():
        logger.info(f"✓ MLX model already exists at {output_path}")
        logger.info("  Skipping conversion. Delete directory to reconvert.")
        return str(output_path)
    
    logger.info("=" * 80)
    logger.info("CONVERTING GEMMA-3-4B TO MLX 4-BIT FORMAT")
    logger.info("=" * 80)
    logger.info(f"Source: {Config.hf_model_name}")
    logger.info(f"Target: {output_path}")
    logger.info("\nThis will take several minutes and download ~8GB...")
    logger.info("The model will be quantized to 4-bit (~2GB final size)")
    logger.info("=" * 80)

    # Note: mlx_lm.convert will create the directory itself
    # Don't create it beforehand or it will error

    # Run mlx-lm convert command
    cmd = [
        "python", "-m", "mlx_lm", "convert",
        "--hf-path", Config.hf_model_name,
        "--mlx-path", str(output_path),
        "--quantize",
        "--q-bits", "4",
        "--q-group-size", "64"
    ]
    
    logger.info(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("\n" + "=" * 80)
            logger.info("✓ CONVERSION SUCCESSFUL!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {output_path}")
            logger.info(f"Size: ~2GB (4-bit quantized)")
            logger.info("\nYou can now run training with --train flag")
            logger.info("=" * 80)
            return str(output_path)
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        logger.error("=" * 80)
        logger.error("✗ CONVERSION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error code: {e.returncode}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check internet connection (needs to download model)")
        logger.error("2. Ensure mlx-lm is installed: pip install mlx-lm")
        logger.error("3. Check disk space (~8GB needed during conversion)")
        logger.error("=" * 80)
        raise


# ========================
# DATA PREPARATION
# ========================

def prepare_sft_data():
    """Load and prepare SFT dataset"""
    logger.info("Loading SFT dataset...")
    dataset = load_dataset("predibase/wordle-sft", split="train")
    df = dataset.to_pandas()
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    def format_example(row):
        return {"text": row['prompt'] + row['completion']}
    
    train_data = [format_example(row) for _, row in train_df.iterrows()]
    val_data = [format_example(row) for _, row in val_df.iterrows()]
    
    logger.info(f"SFT data: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data


def prepare_grpo_data():
    """Load and prepare GRPO dataset with stop instructions"""
    logger.info("Loading GRPO dataset...")
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()
    
    # Filter valid words
    valid = dataset[dataset['secret'].astype(str).str.len() == 5]
    valid = valid[valid['secret'].str.isalpha()]
    
    train_df, val_df = train_test_split(valid, test_size=0.2, random_state=42)
    
    # Add stop instruction
    def add_stop_instruction(prompt):
        if '<|im_end|>' in prompt:
            parts = prompt.split('<|im_end|>', 1)
            instruction = "\n\n**IMPORTANT**: Provide ONLY your <think></think> reasoning and <guess>WORD</guess>. Do NOT generate any text after the </guess> tag. Stop immediately after </guess>."
            return parts[0] + instruction + '<|im_end|>' + parts[1]
        return prompt
    
    train_df = train_df.copy()
    train_df['prompt'] = train_df['prompt'].apply(add_stop_instruction)
    
    # Parse history
    def parse_history(h):
        if isinstance(h, str):
            try:
                return ast.literal_eval(h)
            except:
                return []
        return h if h else []
    
    train_df['past_guess_history'] = train_df['past_guess_history'].apply(parse_history)
    
    logger.info(f"GRPO data: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


# ========================
# REWARD COMPUTATION
# ========================

def compute_reward(prompt: str, completion: str, secret: str, past_history: list) -> Tuple[float, float, float, float]:
    """Compute total reward using strict reward functions"""
    example = {
        'word_list': Config.word_list,
        'past_guess_history': past_history,
        'secret_word': secret
    }
    
    format_r = output_format_check(prompt, completion, example)
    feedback_r = uses_previous_feedback(prompt, completion, example)
    info_r = guess_value(prompt, completion, example)
    
    total = format_r + feedback_r + info_r
    return total, format_r, feedback_r, info_r


# ========================
# STAGE 1: SFT TRAINING
# ========================

def train_sft(model, tokenizer, train_data, val_data, adapter_path: str):
    """
    SFT training with LoRA using MLX (works with quantized models)
    Uses proper QLoRA approach with trainable adapters
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: SFT TRAINING")
    logger.info("=" * 80)

    # Apply LoRA layers
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Freeze base model weights
    model.freeze()

    # Add LoRA adapters (these will be trainable even with quantized base)
    lora_config = {
        "rank": Config.lora_rank,
        "scale": Config.lora_alpha,
        "dropout": Config.lora_dropout
    }
    linear_to_lora_layers(model, Config.lora_layers, lora_config)

    # DON'T call model.train() - keep base model frozen!
    # Only LoRA adapters should be trainable

    # Verify trainable parameters
    trainable_params = [(n, p) for n, p in model.parameters().items() if p.requires_grad]
    logger.info(f"Trainable parameters: {len(trainable_params)}")

    # Optimizer only for LoRA parameters
    optimizer = optim.Adam(learning_rate=Config.sft_lr)

    # Training loop
    for epoch in range(Config.sft_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{Config.sft_epochs}")
        total_loss = 0

        for i in range(0, len(train_data), Config.sft_batch_size):
            batch = train_data[i:i + Config.sft_batch_size]

            batch_loss = 0
            for item in batch:
                # Tokenize
                tokens = tokenizer.encode(item['text'])
                if len(tokens) < 2:
                    continue

                input_ids = mx.array(tokens[:-1])[None, :]  # Add batch dimension
                targets = mx.array(tokens[1:])

                # Define loss function
                def loss_fn(mdl):
                    logits = mdl(input_ids)[0]
                    return nn.losses.cross_entropy(logits, targets, reduction='mean')

                # Compute loss and gradients
                loss_value, grads = mx.value_and_grad(loss_fn)(model)

                # Update only LoRA parameters
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                batch_loss += loss_value.item()

            total_loss += batch_loss

            if (i // Config.sft_batch_size) % 10 == 0:
                logger.info(f"  Step {i}/{len(train_data)}, Loss: {batch_loss/max(len(batch), 1):.4f}")

        avg_loss = total_loss / max(len(train_data), 1)
        logger.info(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # Save adapter weights only
    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    # Save only LoRA adapters
    adapter_weights = {k: v for k, v in model.parameters().items() if 'lora' in k.lower()}
    mx.save_safetensors(str(Path(adapter_path) / "adapters.safetensors"), adapter_weights)

    logger.info(f"SFT complete! Adapters saved to {adapter_path}")
    return model


# ========================
# STAGE 2: GRPO TRAINING
# ========================

def train_grpo(model, tokenizer, train_data, val_data, adapter_path: str):
    """
    GRPO training using MLX
    Based on https://github.com/Doriandarko/MLX-GRPO
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: GRPO TRAINING")
    logger.info("=" * 80)
    
    # Optimizer with higher LR for RL
    optimizer = optim.SGD(learning_rate=Config.grpo_lr, momentum=0.9)
    
    step = 0
    total_steps = Config.grpo_iters
    
    for epoch in range(Config.grpo_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{Config.grpo_epochs}")
        
        for idx, row in train_data.iterrows():
            if step >= total_steps:
                break
            
            prompt = row['prompt']
            secret = row['secret']
            past_history = row['past_guess_history']
            
            # Temperature schedule
            progress = step / total_steps
            temperature = Config.temperature_start - (Config.temperature_start - Config.temperature_end) * progress
            
            # Generate multiple completions
            completions = []
            rewards = []
            
            for _ in range(Config.num_generations):
                # Generate
                response = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=Config.max_tokens,
                    temp=temperature,
                    top_p=Config.top_p
                )
                
                # Compute reward
                total_r, format_r, feedback_r, info_r = compute_reward(
                    prompt, response, secret, past_history
                )
                
                completions.append(response)
                rewards.append(total_r)
                
                # Log for debugging
                if step % 10 == 0 and _ == 0:
                    logger.info(f"\n  Secret: {secret}")
                    logger.info(f"  Generation: {response[:100]}...")
                    logger.info(f"  Rewards: format={format_r:.2f}, feedback={feedback_r:.2f}, info={info_r:.2f}, total={total_r:.2f}")
            
            # GRPO: Group relative policy optimization
            if len(rewards) >= 2:
                # Compute advantages relative to group mean
                rewards_array = np.array(rewards)
                mean_reward = rewards_array.mean()
                advantages = rewards_array - mean_reward
                
                # Policy gradient update
                for i, (completion, advantage) in enumerate(zip(completions, advantages)):
                    if abs(advantage) > 0.01:  # Only update if significant
                        # Tokenize completion
                        tokens = tokenizer.encode(completion)
                        input_ids = mx.array(tokens[:-1])[None, :]  # Add batch dimension
                        targets = mx.array(tokens[1:])

                        # Define policy loss function
                        def policy_loss_fn(mdl):
                            logits = mdl(input_ids)[0]  # Remove batch dimension
                            log_probs = nn.log_softmax(logits, axis=-1)
                            selected_log_probs = mx.take_along_axis(
                                log_probs,
                                targets[..., None],
                                axis=-1
                            ).squeeze(-1)
                            return -advantage * selected_log_probs.mean()

                        # Compute loss and gradients
                        loss_value, grads = mx.value_and_grad(policy_loss_fn)(model)
                        optimizer.update(model, grads)
                        mx.eval(model.parameters(), optimizer.state)
                
                if step % 10 == 0:
                    logger.info(f"Step {step}: Mean reward={mean_reward:.2f}, Advantages=[{advantages.min():.2f}, {advantages.max():.2f}]")
            
            step += 1
            
            # Save checkpoint
            if step % 500 == 0:
                checkpoint_path = Path(adapter_path) / f"checkpoint-{step}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                model.save_weights(str(checkpoint_path / "adapters.npz"))
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if step >= total_steps:
            break
    
    # Save final
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    model.save_weights(str(Path(adapter_path) / "adapters.npz"))
    
    logger.info(f"GRPO complete! Adapters saved to {adapter_path}")
    return model


# ========================
# MAIN PIPELINE
# ========================

def run_training():
    """Run the 2-stage training pipeline"""
    logger.info("=" * 80)
    logger.info("MLX-GRPO WORDLE TRAINING PIPELINE")
    logger.info(f"Model: {Config.hf_model_name}")
    logger.info("=" * 80)
    
    # Check if model is converted
    if not Path(Config.mlx_model_path).exists():
        logger.error("=" * 80)
        logger.error("ERROR: MLX model not found!")
        logger.error("=" * 80)
        logger.error(f"Expected path: {Config.mlx_model_path}")
        logger.error("\nYou need to convert the model first:")
        logger.error("  python mlx_sft_grpo_pipeline.py --convert")
        logger.error("=" * 80)
        return
    
    # Create output directory
    Path(Config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load 4-bit quantized model
    logger.info(f"Loading {Config.mlx_model_path}...")
    model, tokenizer = load(Config.mlx_model_path)
    logger.info("Model loaded successfully!")
    
    # Stage 1: SFT
    train_sft_data, val_sft_data = prepare_sft_data()
    sft_adapter_path = Path(Config.output_dir) / "sft_adapters"
    model = train_sft(model, tokenizer, train_sft_data, val_sft_data, str(sft_adapter_path))
    
    # Stage 2: GRPO
    train_grpo_data, val_grpo_data = prepare_grpo_data()
    grpo_adapter_path = Path(Config.output_dir) / "grpo_adapters"
    model = train_grpo(model, tokenizer, train_grpo_data, val_grpo_data, str(grpo_adapter_path))
    
    # Save metadata
    metadata = {
        "model": Config.hf_model_name,
        "mlx_model": Config.mlx_model_path,
        "quantization": "4-bit",
        "sft_epochs": Config.sft_epochs,
        "grpo_epochs": Config.grpo_epochs,
        "temperature_schedule": f"{Config.temperature_start} → {Config.temperature_end}",
        "sft_path": str(sft_adapter_path),
        "grpo_path": str(grpo_adapter_path),
    }
    
    with open(Path(Config.output_dir) / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MLX-GRPO Wordle Training")
    parser.add_argument("--convert", action="store_true", help="Convert model to MLX 4-bit format")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    
    args = parser.parse_args()
    
    if args.convert:
        convert_model_to_mlx()
    elif args.train:
        run_training()
    else:
        print("MLX-GRPO Wordle Training Pipeline")
        print("=" * 80)
        print("\nUsage:")
        print("  Step 1 - Convert model (run once):")
        print("    python mlx_sft_grpo_pipeline.py --convert")
        print("\n  Step 2 - Run training:")
        print("    python mlx_sft_grpo_pipeline.py --train")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
