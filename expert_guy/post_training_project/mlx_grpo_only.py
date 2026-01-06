#!/usr/bin/env python3
"""
MLX-GRPO Training for Wordle (NO SFT)
Uses the MLX-GRPO library directly on base model
"""

import mlx.core as mx
from mlx_grpo import GRPOConfig, GRPOTrainer
from mlx_lm import load
from datasets import load_dataset
from transformers import AutoTokenizer
from reward_functions import output_format_check, uses_previous_feedback, guess_value


class Config:
    # Model
    hf_model_name = "google/gemma-3-4b-it"
    mlx_model_path = "mlx_models/gemma-3-4b-it-4bit"
    
    # Output
    output_dir = "mlx_output/grpo_only"
    
    # LoRA
    lora_rank = 64
    lora_alpha = 16
    lora_layers = 16
    
    # GRPO
    num_iterations = 760
    learning_rate = 1e-6
    num_samples = 4
    temperature = 0.3
    max_tokens = 512
    beta = 0.1  # KL penalty


def compute_reward(prompt: str, completion: str, secret: str) -> float:
    """Compute total reward"""
    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [],
        'secret_word': secret
    }
    
    format_r = output_format_check(prompt, completion, example)
    feedback_r = uses_previous_feedback(prompt, completion, example)
    info_r = guess_value(prompt, completion, example)
    
    return format_r + feedback_r + info_r


def main():
    print("="*80)
    print("MLX-GRPO TRAINING (NO SFT)")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("predibase/wordle-grpo", split="train")
    
    # Format dataset for GRPO
    def format_example(example):
        # Add assistant marker to prompt
        prompt = example['prompt'] + "\n<|im_start|>assistant\n"
        return {
            'query': prompt,
            'secret': example['secret']
        }
    
    formatted_dataset = dataset.map(format_example)
    
    # Load model and tokenizer
    print(f"\nLoading model from {Config.mlx_model_path}...")
    model, tokenizer = load(Config.mlx_model_path)
    
    # GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir=Config.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        learning_rate=Config.learning_rate,
        lora_rank=Config.lora_rank,
        lora_alpha=Config.lora_alpha,
        num_lora_layers=Config.lora_layers,
        num_generations=Config.num_samples,
        temperature=Config.temperature,
        max_new_tokens=Config.max_tokens,
        beta=Config.beta,
        logging_steps=10,
        save_steps=100,
    )
    
    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=formatted_dataset,
        reward_function=lambda prompts, completions, metadata: [
            compute_reward(p, c, m['secret']) 
            for p, c, m in zip(prompts, completions, metadata)
        ]
    )
    
    # Train
    print(f"\nStarting GRPO training for {Config.num_iterations} steps...")
    trainer.train()
    
    print(f"\n✓ Training complete! Model saved to {Config.output_dir}")


if __name__ == "__main__":
    main()
