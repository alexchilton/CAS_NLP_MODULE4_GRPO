#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for Wordle model using a standard PyTorch loop.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from utils.device import get_device, print_device_info
from utils.logging import setup_logger
from model.setup import load_model_and_tokenizer, save_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Wordle")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to SFT configuration YAML file"
    )
    return parser.parse_args()

def run_simple_test(model, tokenizer, device, prompt):
    """Runs a simple generation test with the given model."""
    print("\n--- Running Simple Test ---")
    print(f"Prompt: {prompt}")
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nGenerated Text:")
    print(generated_text)
    print("--- Test Complete ---\n")

def main():
    """Main SFT training function."""
    args = parse_args()

    # 1. Load configuration
    print("Step 1/8: Loading configuration...")
    config = load_config(args.config)

    # 2. Setup logging
    print("\nStep 2/8: Setting up logging...")
    logger = setup_logger("wordle_sft", config.output.log_dir)
    logger.info("SFT Training Started")

    # 3. Setup device
    print("\nStep 3/8: Detecting device...")
    device = get_device()
    print_device_info()
    logger.info(f"Using device: {device}")

    # 4. Load base model and tokenizer
    print("\nStep 4/8: Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config, device)
    print(f"Model loaded on device: {model.device}")

    # 5. Pre-SFT Test
    print("\nStep 5/8: Running pre-SFT test...")
    test_prompt = """<|im_start|>system
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM -> Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE -> Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK -> Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to first add your step-by-step thought process within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.
<|im_end|>
<|im_start|>user
Make your first 5-letter word guess.<|im_end|>
<|im_start|>assistant
"""
    run_simple_test(model, tokenizer, device, test_prompt)

    # 6. Load and tokenize dataset
    print("\nStep 6/8: Loading and tokenizing dataset...")
    dataset = load_dataset(config.data.dataset_name, split=config.data.train_split)
    
    def tokenize_function(examples):
        text = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        return tokenizer(text, truncation=True, padding="max_length", max_length=config.training.max_seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    dataloader = DataLoader(tokenized_dataset, batch_size=config.training.batch_size, shuffle=True)
    logger.info(f"Dataset loaded and tokenized: {len(dataloader.dataset)} samples")

    # 7. Initialize optimizer and scheduler
    print("\nStep 7/8: Initializing optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=float(config.training.learning_rate))
    num_training_steps = len(dataloader) * config.training.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # 8. Training loop
    print("\nStep 8/8: Starting SFT training...")
    model.train()
    losses = []
    for epoch in range(config.training.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.training.epochs} ---")
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()

            if (i + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            if i % config.training.logging_steps == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
                logger.info(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    # Save final model
    print("\n--- Saving Final Model ---")
    save_model(model, tokenizer, config.output.model_dir)
    logger.info(f"Final model saved to {config.output.model_dir}")

    # Plot and save loss graph
    print("\n--- Generating Loss Graph ---")
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(config.output.loss_graph_path)
    print(f"Loss graph saved to {config.output.loss_graph_path}")

    # Post-SFT Test
    print("\n--- Running Post-SFT Test ---")
    run_simple_test(model, tokenizer, device, test_prompt)
    
    print("\nSFT training and testing complete.")

if __name__ == "__main__":
    main()