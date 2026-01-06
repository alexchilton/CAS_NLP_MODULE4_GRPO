#!/usr/bin/env python3
"""
MLX-GRPO Wordle Training Pipeline (Simplified)
Uses MLX-LM's built-in commands which properly handle 4-bit quantized models

STEP 1: Convert model with 4-bit quantization
STEP 2: Train LoRA adapters (SFT) using mlx_lm.lora
STEP 3: Load model + adapters and do GRPO training manually
"""

import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer

from logger_setup import logger
from reward_functions import output_format_check, uses_previous_feedback, guess_value


class Config:
    # Model
    hf_model_name = "google/gemma-3-4b-it"
    mlx_model_path = "mlx_models/gemma-3-4b-it-4bit"

    # Paths
    output_dir = "mlx_output"

    # LoRA
    lora_rank = 64
    lora_alpha = 16
    lora_layers = 16

    # SFT
    sft_iters = 600  # ~10 epochs for 65 train examples (600 / 65 ≈ 9.2 epochs)
    sft_lr = 1e-5
    batch_size = 1
    steps_per_eval = 50  # Evaluate every 50 steps
    save_every = 100  # Save checkpoint every 100 steps
    max_seq_length = 4096  # Optimal: ~74% of data (61 examples), ~37GB peak memory (est.)

    # GRPO
    grpo_num_samples = 4  # Number of responses to generate per prompt
    grpo_temperature = 0.3  # Balanced temperature
    max_tokens = 512  # Standard length

    # Wandb
    use_wandb = True
    wandb_project = "wordle-mlx-grpo"
    wandb_run_name = None  # Auto-generated if None


def run_command(cmd, description):
    """Run a command and stream output"""
    logger.info(f"\n{'='*80}")
    logger.info(description)
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}\n")

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

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    logger.info(f"\n✓ {description} complete!\n")


def convert_model():
    """Convert HF model to MLX 4-bit"""
    output_path = Path(Config.mlx_model_path)

    if output_path.exists() and (output_path / "config.json").exists():
        logger.info(f"✓ Model already converted: {output_path}")
        return

    cmd = [
        "python", "-m", "mlx_lm", "convert",
        "--hf-path", Config.hf_model_name,
        "--mlx-path", str(output_path),
        "--quantize",
        "--q-bits", "4",
        "--q-group-size", "64"
    ]

    run_command(cmd, "Converting to MLX 4-bit")


def prepare_sft_data():
    """Prepare SFT data in JSONL format for mlx_lm.lora
    
    Strategy: Always keep full completions. If prompt+completion exceeds max_seq_length,
    skip the example (affects ~8.5% of data at 8192 length).
    """
    # MLX-LM expects a directory with train.jsonl and valid.jsonl
    data_dir = Path(Config.output_dir) / "sft_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        logger.info(f"✓ SFT data already exists in {data_dir}")
        return str(data_dir)

    logger.info("Preparing SFT dataset...")
    dataset = load_dataset("predibase/wordle-sft", split="train")

    # Split into train and validation (80/20)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset.select(range(split_idx))
    valid_data = dataset.select(range(split_idx, len(dataset)))

    max_length = Config.max_seq_length
    
    # Load tokenizer to get EXACT token counts - no estimation!
    logger.info("Loading tokenizer for accurate filtering...")
    tokenizer = AutoTokenizer.from_pretrained(Config.hf_model_name)

    def write_data_preserve_completions(data, filepath):
        """Write data, GUARANTEEING no truncation by using actual tokenization"""
        written = 0
        skipped = 0
        with open(filepath, 'w') as f:
            for item in data:
                # Always concatenate prompt + completion for proper SFT
                text = item['prompt'] + item['completion']
                
                # Actually tokenize to get EXACT count (no estimates!)
                actual_tokens = len(tokenizer.encode(text))
                
                if actual_tokens <= max_length:
                    data_item = {"text": text}
                    f.write(json.dumps(data_item) + '\n')
                    written += 1
                else:
                    skipped += 1
        return written, skipped

    # Write training data
    train_written, train_skipped = write_data_preserve_completions(train_data, train_path)

    # Write validation data
    valid_written, valid_skipped = write_data_preserve_completions(valid_data, valid_path)

    logger.info(f"✓ Train: {train_written} saved, {train_skipped} skipped (too long)")
    logger.info(f"✓ Valid: {valid_written} saved, {valid_skipped} skipped (too long)")
    logger.info(f"✓ Using max_seq_length={max_length}")
    logger.info(f"✓ GUARANTEED: No truncation - all completions are intact!")

    return str(data_dir)


def train_sft():
    """Train LoRA adapters using mlx_lm.lora (handles quantized models correctly)"""
    data_dir = prepare_sft_data()
    adapter_path = Path(Config.output_dir) / "sft_adapters"

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", Config.mlx_model_path,
        "--train",
        "--data", data_dir,  # Directory containing train.jsonl
        "--adapter-path", str(adapter_path),
        "--iters", str(Config.sft_iters),
        "--steps-per-eval", str(Config.steps_per_eval),
        "--save-every", str(Config.save_every),
        "--learning-rate", str(Config.sft_lr),
        "--num-layers", str(Config.lora_layers),
        "--batch-size", str(Config.batch_size),
        "--max-seq-length", str(Config.max_seq_length),
    ]

    run_command(cmd, "SFT Training with LoRA")

    return str(adapter_path)


# ==============================================================================
# GRPO TRAINING
# ==============================================================================

def pad_sequences(token_lists: List[List[int]], pad_value: int) -> mx.array:
    """Pad token sequences to same length"""
    if not token_lists:
        return mx.array([], dtype=mx.int32)
    max_len = max(len(tokens) for tokens in token_lists)
    padded = [tokens + [pad_value] * (max_len - len(tokens)) for tokens in token_lists]
    return mx.array(padded)


def get_log_probs(model: nn.Module, prompt_ids: mx.array, response_ids: mx.array, pad_token_id: int) -> mx.array:
    """Get log probabilities of generated tokens"""
    full_sequence = mx.concatenate([prompt_ids, response_ids], axis=1)
    logits = model(full_sequence)

    prompt_len = prompt_ids.shape[1]
    gen_len = response_ids.shape[1]

    # Get logits for generated tokens only
    output_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]

    # Log probabilities
    log_probs = nn.log_softmax(output_logits, axis=-1)

    # Select log probs of actual tokens
    chosen_log_probs = mx.take_along_axis(
        log_probs, response_ids[..., None], axis=-1
    ).squeeze(-1)

    # Mask padding
    mask = (response_ids != pad_token_id).astype(chosen_log_probs.dtype)
    masked_log_probs = chosen_log_probs * mask

    return mx.sum(masked_log_probs, axis=-1)


def compute_reward(prompt: str, completion: str, secret: str, past_history: list) -> float:
    """Compute total reward using reward functions"""
    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': past_history,
        'secret_word': secret
    }

    format_r = output_format_check(prompt, completion, example)
    feedback_r = uses_previous_feedback(prompt, completion, example)
    info_r = guess_value(prompt, completion, example)

    return format_r + feedback_r + info_r


def train_grpo(sft_adapter_path: str, num_iterations: int = 760):
    """
    GRPO training using 4-bit quantized reference model
    Based on: https://github.com/Doriandarko/MLX-GRPO
    
    Default: 760 iterations = ~10 epochs on 76 GRPO examples
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: GRPO TRAINING")
    logger.info("=" * 80)

    # Initialize wandb
    if Config.use_wandb:
        run_name = Config.wandb_run_name or f"grpo-{Config.hf_model_name.split('/')[-1]}"
        wandb.init(
            project=Config.wandb_project,
            name=run_name,
            config={
                "model": Config.hf_model_name,
                "lora_rank": Config.lora_rank,
                "lora_alpha": Config.lora_alpha,
                "lora_layers": Config.lora_layers,
                "grpo_num_samples": Config.grpo_num_samples,
                "grpo_temperature": Config.grpo_temperature,
                "max_tokens": Config.max_tokens,
                "iterations": num_iterations,
            }
        )
        logger.info(f"✓ Wandb initialized: {wandb.run.url}")

    # Load reference model (4-bit, frozen)
    logger.info("Loading reference model (4-bit)...")
    ref_model, tokenizer = load(Config.mlx_model_path)
    ref_model.freeze()

    # Load policy model from base (no SFT adapters)
    logger.info(f"Loading policy model from base model...")
    policy_model, _ = load(Config.mlx_model_path)
    
    # Apply LoRA adapters using mlx-lm's built-in method
    from mlx_lm.tuner.utils import linear_to_lora_layers
    logger.info("Adding LoRA adapters to model...")
    
    linear_to_lora_layers(
        policy_model.model,
        Config.lora_layers,
        Config.lora_rank,
        Config.lora_alpha,
    )

    # Get trainable parameters (only LoRA adapters)
    trainable_params = {k: v for k, v in tree_flatten(policy_model.parameters()) if "lora" in k.lower()}
    logger.info(f"Found {len(trainable_params)} trainable LoRA parameters")

    if not trainable_params:
        raise ValueError("No LoRA parameters found!")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-6)

    # Load GRPO dataset
    logger.info("Loading GRPO dataset...")
    dataset = load_dataset("predibase/wordle-grpo", split="train")

    # Training loop
    logger.info(f"\nStarting GRPO training for {num_iterations} iterations...")

    # Create sampler for generation
    sampler = make_sampler(temp=Config.grpo_temperature)

    for step in tqdm(range(num_iterations), desc="GRPO"):
        # Sample from dataset
        sample = dataset[step % len(dataset)]
        base_prompt = sample['prompt']
        secret = sample['secret']
        
        # Add format instruction to prompt (model learned this in SFT)
        prompt = base_prompt + "\n<|im_start|>assistant\n<think>"

        # Generate multiple responses
        responses = []
        rewards = []

        for i in range(Config.grpo_num_samples):
            full_output = generate(
                policy_model,
                tokenizer,
                prompt,
                max_tokens=Config.max_tokens,
                sampler=sampler
            )
            
            # Extract only the completion (remove prompt)
            response = full_output[len(prompt):] if full_output.startswith(prompt) else full_output

            reward = compute_reward(base_prompt, response, secret, [])
            responses.append(response)
            rewards.append(reward)
            
            # Log EVERY step for debugging
            if i == 0:
                logger.info(f"\n--- Sample at step {step} ---")
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Full output: {full_output}")
                logger.info(f"Response (stripped): {response}")
                logger.info(f"Reward: {reward}")
                logger.info(f"Secret: {secret}")
                print(f"\n[Step {step}] Response:\n{response}")
                print(f"[Step {step}] Reward: {reward}")

        # Find winner and losers
        if len(responses) < 2:
            continue

        winner_idx = np.argmax(rewards)
        winner_response = responses[winner_idx]

        # Prepare batches
        winner_tokens = tokenizer.encode(winner_response)
        prompt_tokens = tokenizer.encode(prompt)

        winner_toks_list = []
        loser_toks_list = []
        prompt_toks_list = []

        for i, response in enumerate(responses):
            if i != winner_idx:
                winner_toks_list.append(winner_tokens)
                loser_toks_list.append(tokenizer.encode(response))
                prompt_toks_list.append(prompt_tokens)

        if not loser_toks_list:
            continue

        # Pad sequences
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        winner_toks = pad_sequences(winner_toks_list, pad_id)
        loser_toks = pad_sequences(loser_toks_list, pad_id)
        prompt_toks = pad_sequences(prompt_toks_list, pad_id)

        # Compute GRPO loss
        def grpo_loss_fn(params):
            # Update policy model with current parameters
            policy_model.update(tree_unflatten(list(params.items())))

            # Log probs from policy
            log_probs_winner = get_log_probs(policy_model, prompt_toks, winner_toks, pad_id)
            log_probs_loser = get_log_probs(policy_model, prompt_toks, loser_toks, pad_id)

            # Log probs from reference (frozen)
            log_probs_ref_winner = get_log_probs(ref_model, prompt_toks, winner_toks, pad_id)
            log_probs_ref_loser = get_log_probs(ref_model, prompt_toks, loser_toks, pad_id)

            # GRPO loss
            pi_logratios = log_probs_winner - log_probs_loser
            ref_logratios = log_probs_ref_winner - log_probs_ref_loser

            logits = pi_logratios - ref_logratios
            loss = -mx.mean(nn.log_sigmoid(0.1 * logits))  # beta = 0.1

            return loss

        # Compute gradients
        loss, grads = mx.value_and_grad(grpo_loss_fn)(trainable_params)

        # Update
        trainable_params = optimizer.apply_gradients(grads, trainable_params)
        policy_model.update(tree_unflatten(list(trainable_params.items())))
        mx.eval(policy_model.parameters(), optimizer.state)

        # Logging
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)

        if Config.use_wandb:
            wandb.log({
                "loss": loss.item(),
                "avg_reward": avg_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "step": step
            })

        if step % 10 == 0:
            logger.info(f"Step {step}: loss={loss.item():.4f}, avg_reward={avg_reward:.2f}, max={max_reward:.2f}, min={min_reward:.2f}")

    # Save final adapters
    grpo_adapter_path = Path(Config.output_dir) / "grpo_adapters"
    grpo_adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_weights = {k: v for k, v in trainable_params.items()}
    mx.save_safetensors(str(grpo_adapter_path / "adapters.safetensors"), adapter_weights)

    logger.info(f"\n✓ GRPO complete! Adapters saved to {grpo_adapter_path}")

    if Config.use_wandb:
        wandb.finish()

    return str(grpo_adapter_path)


def main():
    parser = argparse.ArgumentParser(description="MLX-GRPO Training (Simplified)")
    parser.add_argument("--convert", action="store_true", help="Convert model to MLX 4-bit")
    parser.add_argument("--sft", action="store_true", help="Run SFT training")
    parser.add_argument("--grpo", action="store_true", help="Run GRPO training")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.convert or args.all:
        convert_model()

    if args.sft or args.all:
        adapter_path = train_sft()
        logger.info(f"\n{'='*80}")
        logger.info("SFT TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Adapters saved to: {adapter_path}")
        logger.info(f"{'='*80}\n")

    if args.grpo or args.all:
        # Use checkpoint 100 (closest to best val loss - checkpoints saved every 100 steps)
        checkpoint_file = Path(Config.output_dir) / "sft_adapters" / "0000100_adapters.safetensors"
        
        if checkpoint_file.exists():
            # Create temp dir and copy checkpoint + config
            import shutil
            best_path = Path(Config.output_dir) / "sft_adapters_best"
            best_path.mkdir(exist_ok=True)
            shutil.copy(checkpoint_file, best_path / "adapters.safetensors")
            # Copy adapter config too
            config_file = Path(Config.output_dir) / "sft_adapters" / "adapter_config.json"
            if config_file.exists():
                shutil.copy(config_file, best_path / "adapter_config.json")
            sft_path = best_path
            logger.info("Using checkpoint 100 (closest to best validation loss)")
        else:
            sft_path = Path(Config.output_dir) / "sft_adapters"
            logger.warning("Checkpoint 100 not found, using latest adapters")
        
        if not sft_path.exists():
            logger.error("SFT adapters not found! Run --sft first.")
            return

        grpo_path = train_grpo(str(sft_path))
        logger.info(f"\n{'='*80}")
        logger.info("GRPO TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Final adapters: {grpo_path}")
        logger.info(f"{'='*80}\n")

    if not (args.convert or args.sft or args.grpo or args.all):
        print("MLX-GRPO Wordle Training (Simplified)")
        print("="*80)
        print("\nUsage:")
        print("  python mlx_sft_grpo_simple.py --convert    # Convert model")
        print("  python mlx_sft_grpo_simple.py --sft        # Train SFT")
        print("  python mlx_sft_grpo_simple.py --grpo       # Train GRPO")
        print("  python mlx_sft_grpo_simple.py --all        # Do everything")
        print("="*80)


if __name__ == "__main__":
    main()
