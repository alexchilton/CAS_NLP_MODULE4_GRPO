import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# SFT (Supervised Fine-Tuning) training script for Wordle
# This teaches the model basic format and game rules before GRPO refinement

import os
import pandas as pd
from dataclasses import dataclass
from logger_setup import logger
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import torch
import datetime

# PEFT imports
from peft import get_peft_model, LoraConfig, TaskType

# ----------------------#
# 1. LOAD SFT DATA      #
# ----------------------#

def load_and_prepare_sft_data():
    """
    Load Wordle SFT dataset from Hugging Face (Predibase dataset).
    This dataset contains examples of good Wordle gameplay.
    """
    logger.info("Loading SFT dataset from Hugging Face...")

    # Load the Predibase SFT dataset
    dataset = load_dataset("predibase/wordle-sft", split="train")
    logger.info(f"Loaded predibase/wordle-sft dataset with {len(dataset)} examples")

    # Convert to pandas for inspection and split
    dataset_df = dataset.to_pandas()
    logger.info(f"Dataset columns: {dataset_df.columns.tolist()}")
    logger.info(f"Sample row:\n{dataset_df.head(1)}")

    # Split into train/validation (80/20)
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    logger.info(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset

# ----------------------#
# 2. MODEL SETUP        #
# ----------------------#

def setup_model_and_tokenizer_peft():
    """Setup model with PEFT (LoRA) for efficient fine-tuning"""
    load_dotenv()
    MODEL_NAME = "google/gemma-3-4b-it"
    HF_TOKEN = None

    logger.info(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    logger.info("Tokenizer loaded successfully!")

    logger.info(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        token=HF_TOKEN, 
        device_map="auto",
        torch_dtype=torch.float16  # Use FP16 to reduce memory
    )
    logger.info("Model loaded successfully!")

    # Setup tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model: {MODEL_NAME}")

    # PEFT config (LoRA) - matching Predibase SFT settings
    logger.info("Configuring PEFT (LoRA) for SFT...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Predibase uses rank 64 for SFT
        lora_alpha=16,  # Standard alpha = r/4
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    logger.info("Wrapping model with PEFT...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("Model wrapped with PEFT (LoRA)")

    # Ensure model is in training mode
    model.train()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer

# ----------------------#
# 3. DATA PROCESSING    #
# ----------------------#

def preprocess_function(examples, tokenizer):
    """
    Tokenize the text data for supervised training.
    For SFT, we train the model to predict completions given prompts.
    """
    # Tokenize the prompts
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=1024,
        truncation=True,
        padding="max_length",
    )

    # For language modeling, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

# ----------------------#
# 4. MAIN               #
# ----------------------#

if __name__ == "__main__":
    logger.info("Starting SFT training script...")

    # Load data
    logger.info("Loading and preparing SFT data...")
    train_dataset, val_dataset = load_and_prepare_sft_data()
    logger.info("Data loaded and prepared.")

    # Setup model
    logger.info("Setting up model and tokenizer with PEFT...")
    model, tokenizer = setup_model_and_tokenizer_peft()
    logger.info("Model and tokenizer setup complete.")

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
    logger.info("Datasets preprocessed.")

    # Training arguments for SFT (adjusted for Mac memory constraints)
    logger.info("Configuring SFT training arguments...")
    training_args = TrainingArguments(
        output_dir="output_sft/wordle-sft-peft",
        num_train_epochs=10,  # Predibase uses 10 epochs for SFT
        per_device_train_batch_size=1,  # REDUCED: 4→1 for Mac memory
        per_device_eval_batch_size=1,  # REDUCED: 4→1 for Mac memory
        gradient_accumulation_steps=8,  # INCREASED: Maintain effective batch size
        optim="sgd",  # Use SGD instead of AdamW to save memory
        learning_rate=1e-3,  # Higher LR for SGD (typically 10-50x AdamW)
        optim_args="momentum=0.9",  # Standard momentum for SGD
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,  # Reduced to save disk space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=True,  # Enable FP16 for memory savings
        logging_dir="output_sft/wordle-sft-peft/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"wordle-sft-peft-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=False,
        warmup_steps=10,
        max_grad_norm=1.0,
        gradient_checkpointing=False,  # Disable - causes issues with PEFT
    )
    logger.info("Training arguments configured.")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    logger.info("Trainer initialized successfully!")

    # Train
    logger.info("Starting SFT training loop...")
    trainer.train()
    logger.info("SFT training complete!")

    # Save final model
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final SFT model saved to: {final_model_dir}")

    # Save best model if available
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        logger.info(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")

    logger.info("SFT training complete! Ready for GRPO refinement.")
    logger.info(f"To use this model for GRPO, load from: {final_model_dir}")
