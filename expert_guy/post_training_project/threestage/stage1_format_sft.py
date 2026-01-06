"""
Stage 1: Pure Format SFT Training
==================================

Goal: Teach the model the correct output format to 90%+ accuracy
- Input: Wordle game state
- Output: <think>reasoning</think><guess>WORD</guess>

This stage focuses ONLY on format, not strategy.
We train until the model consistently produces valid formatted outputs.

Success criteria:
- 90%+ of outputs have correct <think></think><guess></guess> structure
- 90%+ of guesses are valid 5-letter words from the word list
- Low perplexity on validation set

Training approach:
- Use synthetic data (1000 examples)
- Simple language modeling objective
- Higher learning rate for faster format learning
- Stop when validation format accuracy > 90%
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import datetime
import sys
import wandb

# Add parent directory to path for logger
sys.path.append('..')
from logger_setup import logger


class FormatAccuracyCallback(TrainerCallback):
    """
    Callback to evaluate format accuracy during training.
    Stops training early when format accuracy > 90%.
    """

    def __init__(self, tokenizer, eval_dataset, word_list_path="../five_letter_words.csv", patience=5):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.word_list = pd.read_csv(word_list_path)["Word"].str.upper().values
        self.best_format_accuracy = 0.0
        self.best_composite_score = 0.0  # Track best combined metric
        self.patience = patience  # Number of evals without improvement before stopping
        self.evals_without_improvement = 0

    def evaluate_format_accuracy(self, model):
        """Evaluate format accuracy on a sample of validation data"""
        model.eval()
        n_samples = min(10, len(self.eval_dataset))  # Reduced from 50 to 10 for faster eval
        correct_format = 0
        valid_words = 0

        logger.info(f"Evaluating format accuracy on {n_samples} samples...")

        for i in range(n_samples):
            if i % 5 == 0:
                logger.info(f"  Processing sample {i+1}/{n_samples}...")
            prompt = self.eval_dataset.iloc[i]["prompt"]

            # Generate completion
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Check format - must have BOTH opening and closing tags
            has_format = bool(re.search(r"<think>.*?</think>.*?<guess>.*?</guess>", completion, re.DOTALL))
            if has_format:
                correct_format += 1

            # Check valid word
            guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
            if guess_match:
                guess_text = guess_match.group(1).strip()
                guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
                letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
                if len(letters_only) == 5 and letters_only in self.word_list:
                    valid_words += 1

        format_acc = correct_format / n_samples
        word_acc = valid_words / n_samples

        model.train()
        return format_acc, word_acc

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Evaluate format accuracy after each evaluation"""
        model = kwargs.get('model')
        if model is None:
            return

        format_acc, word_acc = self.evaluate_format_accuracy(model)

        logger.info(f"Format Accuracy: {format_acc:.2%} | Valid Word Accuracy: {word_acc:.2%}")

        # Calculate composite score: both format AND word accuracy must be good
        # Use harmonic mean so both metrics must be high (penalizes imbalance)
        if format_acc > 0 and word_acc > 0:
            composite_score = 2 * (format_acc * word_acc) / (format_acc + word_acc)
        else:
            composite_score = 0.0

        # Log to WandB
        if wandb.run is not None:
            wandb.log({
                "format_accuracy": format_acc,
                "word_accuracy": word_acc,
                "composite_score": composite_score,
                "eval/format_accuracy": format_acc,
                "eval/word_accuracy": word_acc,
                "eval/composite_score": composite_score,
            }, step=state.global_step)

        # Track best format accuracy separately for logging
        if format_acc > self.best_format_accuracy:
            self.best_format_accuracy = format_acc

        # Save best model based on COMPOSITE SCORE (format + word, no strategy)
        if composite_score > self.best_composite_score:
            self.best_composite_score = composite_score
            self.evals_without_improvement = 0  # Reset patience counter
            logger.info(f"🌟 New best: composite={composite_score:.2%} (format={format_acc:.2%}, word={word_acc:.2%})")

            # IMMEDIATELY save best model (don't wait for next checkpoint)
            best_model_dir = "stage1_output/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)
            logger.info(f"💾 Saved to {best_model_dir} (step {state.global_step})")

            # Save metadata with this best model
            import json
            metadata = {
                "step": state.global_step,
                "format_accuracy": format_acc,
                "word_accuracy": word_acc,
                "composite_score": composite_score,
                "eval_loss": metrics.get("eval_loss") if metrics else None,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with open(f"{best_model_dir}/best_model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            # No improvement
            self.evals_without_improvement += 1
            logger.info(f"No improvement for {self.evals_without_improvement}/{self.patience} evaluations")

        # Early stopping: either reached target OR plateaued
        if format_acc >= 0.98 and word_acc >= 0.98:
            logger.info(f"🎉 Reached 98% format+word accuracy! Stopping training.")
            control.should_training_stop = True
        elif self.evals_without_improvement >= self.patience:
            logger.info(f"⚠️ No improvement for {self.patience} evaluations. Stopping (best composite: {self.best_composite_score:.2%})")
            control.should_training_stop = True


def load_synthetic_data():
    """Load the synthetic SFT data"""
    data_path = Path("sft_synthetic_data.jsonl")

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Please run generate_sft_data.py first."
        )

    logger.info(f"Loading synthetic data from {data_path}...")

    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    df = pd.DataFrame(examples)
    logger.info(f"Loaded {len(df)} examples")

    # Split 80/20 and limit validation to 100 samples for faster evaluation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = val_df.head(100)  # Limit validation to 100 samples
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    return train_df, val_df


def setup_model_and_tokenizer():
    """Setup base model with PEFT for efficient training"""
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    logger.info(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer loaded successfully!")

    logger.info(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32
    )
    logger.info("Model loaded successfully!")

    # Setup tokenizer
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # PEFT config - higher rank for better format learning
    logger.info("Configuring PEFT (LoRA)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Higher rank for format learning
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )

    model = get_peft_model(model, peft_config)
    logger.info("Model wrapped with PEFT")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def preprocess_function(examples, tokenizer):
    """Tokenize the data for supervised training"""
    # Combine prompt + completion
    texts = [prompt + completion for prompt, completion in zip(examples["prompt"], examples["completion"])]

    model_inputs = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
    )

    # For language modeling, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def main():
    logger.info("=" * 60)
    logger.info("STAGE 1: Pure Format SFT Training")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading synthetic SFT data...")
    train_df, val_df = load_synthetic_data()

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df[["prompt", "completion"]])
    val_dataset = Dataset.from_pandas(val_df[["prompt", "completion"]])

    # Setup model
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Training arguments
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir="stage1_output",
        num_train_epochs=15,  # Higher epochs for format mastery
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,  # Higher LR for faster format learning
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,  # Evaluate every 25 steps (catches oscillation peaks)
        save_strategy="steps",
        save_steps=100,  # Save checkpoints every 100 steps (saves disk space)
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        logging_dir="stage1_output/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"stage1-format-sft-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        warmup_steps=20,
        max_grad_norm=1.0,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create format accuracy callback
    format_callback = FormatAccuracyCallback(tokenizer, val_df)

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    
    # Force eval_steps to be respected even when resuming from checkpoint
    # by re-setting it after Trainer initialization
    original_eval_steps = training_args.eval_steps
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[format_callback],
    )
    
    # Ensure eval_steps stays at our desired value
    trainer.args.eval_steps = original_eval_steps
    logger.info(f"Confirmed eval_steps set to: {trainer.args.eval_steps}")

    # Train (with automatic checkpoint resumption)
    logger.info("Starting Stage 1 training...")
    logger.info("Goal: Format accuracy > 98%")

    # Check for existing checkpoints
    checkpoint_dir = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Delete ALL training_args.bin files so our new eval_steps takes effect
            for checkpoint in checkpoints:
                args_file = os.path.join(training_args.output_dir, checkpoint, "training_args.bin")
                if os.path.exists(args_file):
                    os.remove(args_file)
                    logger.info(f"Removed {args_file}")
            
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(training_args.output_dir, latest_checkpoint)
            logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
            logger.info(f"Using NEW eval_steps={trainer.args.eval_steps}")

    trainer.train(resume_from_checkpoint=checkpoint_dir)

    # Save final model
    final_model_dir = "stage1_output/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Stage 1 model saved to: {final_model_dir}")

    # Save metadata
    metadata = {
        "stage": 1,
        "description": "Pure format SFT training",
        "training_date": datetime.datetime.now().isoformat(),
        "training_examples": len(train_df),
        "validation_examples": len(val_df),
        "best_format_accuracy": format_callback.best_format_accuracy,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
    }

    with open(f"{final_model_dir}/stage1_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE!")
    logger.info(f"Best format accuracy: {format_callback.best_format_accuracy:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()