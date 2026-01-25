import json
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

INP = "wordle_teacher_trajs.jsonl"
OUT = "wordle_sft_pc.jsonl"

with open(INP) as fin, open(OUT, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        messages = [
            {"role":"system", "content": ex["system"].strip()},
            {"role":"user", "content": ex["user"].strip()},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = ex["label"].strip() + tok.eos_token  # or "\n"
        fout.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


