import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "./qwen-wordle-sft-teacher/final_adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

SYSTEM = (
    "You are an expert Wordle solver.\n"
    "Follow the feedback constraints exactly.\n"
    "Output exactly one line: <guess>ABCDE</guess>\n"
    "No other text."
)

USER = """Previous guesses (word -> feedback):
CRANE: C(x) R(-) A(x) N(x) E(x)

Return your next guess.
Output exactly one line: <guess>ABCDE</guess>
No other text."""

messages = [{"role":"system","content":SYSTEM}, {"role":"user","content":USER}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(out[0], skip_special_tokens=True)
# extract last tag
m = re.findall(r"<guess>[A-Z]{5}</guess>", text)
print(text[-400:])
print("PARSED:", m[-1] if m else None)
