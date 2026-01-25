import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from trl import SFTTrainer
from logger_setup import logger
import json

# --- 0. CLUSTER & MEMORY SETUP ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_MODE"] = "offline"

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def run_sft_train():
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    
    # --- 1. QLoRA CONFIG ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 2. LOAD MODEL & TOKENIZER ---
    logger.info(f"Loading {MODEL_NAME} with sdpa...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Essential for SFT consistency
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        attn_implementation="sdpa", # Optimized for 4090/A100
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- 3. PEFT PREP ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # --- 4. DATA LOADING (Predibase SFT) ---
    # This dataset typically has 'prompt' and 'completion' columns
    
    # --- 4. DATA LOADING (Wordle SFT) ---
    logger.info("Loading willcb/V3-wordle...")

    def format_instruction(sample):
        prompt = sample["prompt"]
        completion = sample["completion"]

        # If stored as JSON strings, parse them
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(completion, str):
            completion = json.loads(completion)

        messages = prompt + completion  # full chat

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = load_dataset("willcb/V3-wordle", split="train")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    logger.info(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # --- 5. TRAINING ARGUMENTS ---
    sft_config = SFTConfig(
        output_dir="./qwen2.5-3b-wordle-sft",
        dataset_text_field="text",      # ✅ Moved here
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # --- 6. INITIALIZE TRAINER ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config, # Pass the sft_config here
    )

    # --- 7. START ---
    logger.info("Starting SFT training loop...")
    trainer.train()
    
    # Save the adapter
    trainer.save_model("./qwen2.5-3b-wordle-sft/final_adapter")
    logger.info("Training complete and adapter saved.")

if __name__ == "__main__":
    run_sft_train()