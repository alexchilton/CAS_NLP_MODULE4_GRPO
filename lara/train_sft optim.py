# import os
# import torch
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig
# from trl import SFTTrainer, SFTConfig

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ["WANDB_MODE"] = "offline"

# def run_sft_train():
#     # TODO: change this to your actual Qwen3 base if you mean Qwen3
#     MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         dtype=torch.bfloat16,
#         device_map="auto",
#         attn_implementation="sdpa",
#     )

#     lora_config = LoraConfig(
#         r=32,
#         lora_alpha=64,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj",
#         ],
#     )

#     dataset = load_dataset(
#         "json",
#         data_files={"train": "wordle_teacher_trajs.jsonl"},
#     )["train"].shuffle(seed=42)

#     dataset = dataset.train_test_split(test_size=0.05, seed=42)

#     sft_config = SFTConfig(
#         output_dir="./qwen-wordle-sft-teacher",
#         dataset_text_field="text",

#         packing=True,

#         learning_rate=5e-5,
#         lr_scheduler_type="cosine",
#         warmup_ratio=0.03,
#         num_train_epochs=0.5,

#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=32,

#         logging_steps=10,

#         save_strategy="steps",
#         save_steps=500,
#         save_total_limit=3,

#         eval_strategy="steps",
#         eval_steps=500,

#         bf16=True,
#         tf32=True,

#         gradient_checkpointing=True,
#         gradient_checkpointing_kwargs={"use_reentrant": False},

#         optim="paged_adamw_8bit",
#         report_to="none",
#         remove_unused_columns=False,
#     )

#     trainer = SFTTrainer(
#         model=model,
#         args=sft_config,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#         peft_config=lora_config,
#         processing_class=tokenizer,
#     )

#     trainer.train()


#     # Saves LoRA adapter (and tokenizer config in output_dir)
#     trainer.save_model("./qwen-wordle-sft-teacher/final_adapter")

# if __name__ == "__main__":
#     run_sft_train()


import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_MODE"] = "offline"

def formatting_func(ex):
    # Build a chat where assistant content is ONLY the completion
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["prompt"].split("\n\n",1)[1]},  # or just use ex["prompt"] if you prefer
    ]
    # simpler: don't split; just put prompt into user:
    messages = [{"role":"system","content":SYSTEM},{"role":"user","content":ex["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + ex["completion"]

def run_sft_train():
    # TODO: change this to your actual Qwen3 base if you mean Qwen3
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = load_dataset("json", data_files={"train": "wordle_sft_pc.jsonl"})["train"].shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    sft_config = SFTConfig(
        output_dir="./qwen-wordle-sft-teacher-masked",
        completion_only_loss=True,
        packing=False,                 # important (avoid packing warnings)
        num_train_epochs=0.3,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        tf32=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()


    # Saves LoRA adapter (and tokenizer config in output_dir)
    trainer.save_model("./qwen-wordle-sft-teacher-masked/final_adapter")

if __name__ == "__main__":
    run_sft_train()