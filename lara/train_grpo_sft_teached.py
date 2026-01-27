import os
import re
import math
import random
from collections import Counter

import torch
import pandas as pd
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

# ----------------------------
# CONFIG
# ----------------------------
WORD_LIST_PATH = "five_letter_words.csv"
WORD_COL = "Word"

DATA_JSONL = "wordle_teacher_trajs.jsonl"   #  teacher states
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# SFT_ADAPTER_PATH = "./qwen-wordle-sft-teacher/final_adapter"  # start GRPO from your SFT LoRA
SFT_ADAPTER_PATH = "./qwen-wordle-grpo/final_adapter"  # start GRPO from SFT LoRA

OUT_DIR = "./qwen-wordle-grpo"

SEED = 42
random.seed(SEED)

os.environ["WANDB_MODE"] = "offline"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ----------------------------
# WORDLE UTILS
# ----------------------------
def get_wordle_feedback(guess, target):
    guess = (guess[:5] + "     ")[:5].upper()
    target = target.upper()

    result = [""] * 5
    target_counts = Counter(target)

    # greens
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = f"{guess[i]}(✓)"
            target_counts[guess[i]] -= 1

    # yellows / grays
    for i in range(5):
        if result[i]:
            continue
        ch = guess[i]
        if target_counts[ch] > 0:
            result[i] = f"{ch}(-)"
            target_counts[ch] -= 1
        else:
            result[i] = f"{ch}(x)"

    return " ".join(result)

def normalize_word(w: str) -> str:
    w = re.sub(r"[^A-Z]", "", str(w).upper())
    return w[:5]

def filter_candidates(candidates, guess, feedback):
    guess = normalize_word(guess)
    if len(guess) != 5:
        return candidates
    fb_norm = " ".join(feedback.strip().split())
    out = []
    for w in candidates:
        w = normalize_word(w)
        if len(w) != 5:
            continue
        sim = get_wordle_feedback(guess, w)
        sim = " ".join(sim.strip().split())
        if sim == fb_norm:
            out.append(w)
    return out

def candidates_from_past(words_list, past_guesses):
    """
    Recompute candidate list implied by past guesses/feedback.
    past_guesses comes from your jsonl: list like [["AUDIO","A(x) ..."], ...]
    """
    cands = list(words_list)
    for g, fb in past_guesses:
        cands = filter_candidates(cands, g, fb)
        if not cands:
            break
    return cands

# ----------------------------
# LOAD WORD LIST (global)
# ----------------------------
df_words = pd.read_csv(WORD_LIST_PATH)
col = WORD_COL if WORD_COL in df_words.columns else df_words.columns[0]
WORDS = df_words[col].astype(str).str.strip().str.upper().tolist()
WORDS = [w for w in WORDS if re.fullmatch(r"[A-Z]{5}", w)]
WORDS_SET = set(WORDS)

# ----------------------------
# MODEL LOADING (policy initialized from SFT adapter)
# ----------------------------
def load_policy():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # SFT LoRA as starting point
    policy = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
    policy.train()
    return policy, tok

# ----------------------------
# DATASET: map jsonl rows -> conversational prompts + keep target/past
# ----------------------------
def build_dataset():
    ds = load_dataset("json", data_files={"train": DATA_JSONL})["train"].shuffle(seed=SEED)

    def to_prompt(ex):
        # GRPOTrainer supports conversational prompts (list of {role, content})
        ex["prompt"] = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        # keep only what we need
        return ex

    ds = ds.map(to_prompt, remove_columns=[])
    # keep columns we use
    keep = {"prompt", "target", "past_guesses", "n_candidates"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds

# ----------------------------
# REWARD FUNCTION (PURE: no projection, OOV gets punished)
# Signature per TRL: reward_func(completions, **kwargs) or reward_func(prompts, completions, extra_cols..., **kwargs)
# ----------------------------
GUESS_RE = re.compile(r"<guess>[A-Z]{5}</guess>")

def wordle_reward(prompts, completions, target, past_guesses, n_candidates, **kwargs):
    """
    completions: list of list of messages, e.g. [[{"role":"assistant","content":"..."}], ...]
    Return list[float] rewards aligned with batch.
    """
    rewards = []

    for comp, tgt, past, n_c in zip(completions, target, past_guesses, n_candidates):
        content = comp[0]["content"] if comp and len(comp) > 0 else ""
        m = GUESS_RE.findall(content)
        if not m:
            # not parseable -> strongly negative
            rewards.append(-2.0)
            continue

        guess = m[-1][7:12]  # inside <guess>.....</guess>
        guess = normalize_word(guess)

        # must be 5 letters
        if len(guess) != 5:
            rewards.append(-2.0)
            continue

        # in-vocab (pure, no projection)
        if guess not in WORDS_SET:
            rewards.append(-1.5)
            continue

        # base reward for obeying format + vocab
        r = 0.2 + 0.4  # parse + vocab

        # repeat penalty
        past_words = {normalize_word(g) for g, _ in past}
        if guess in past_words:
            r -= 0.3

        # constraint-consistent:
        # "consistent" means it remains a possible solution under past feedback constraints
        tmp_cands = candidates_from_past(WORDS, past)
        if guess in set(tmp_cands):
            r += 0.6
        else:
            # still allowed (exploration), but contradictory gets mild penalty
            r -= 0.2

        # progress/info gain wrt TRUE target feedback:
        # compute candidate reduction after applying this guess feedback to current candidates
        fb = get_wordle_feedback(guess, tgt)
        before = len(tmp_cands) if tmp_cands else max(int(n_c), 1)
        after_cands = filter_candidates(tmp_cands, guess, fb) if tmp_cands else []
        after = len(after_cands) if after_cands else 1

        # reward shrinking candidate set (log reduction), capped
        if before > 0 and after > 0:
            gain = math.log(before / after)
            r += max(0.0, min(0.8, 0.15 * gain))

        # solve bonus
        if guess == normalize_word(tgt):
            r += 2.0

        rewards.append(float(r))

    return rewards

# ----------------------------
# TRAIN
# ----------------------------
def main():
    policy, tok = load_policy()
    ds = build_dataset()

    cfg = GRPOConfig(
        output_dir=OUT_DIR,

        # speed / stability knobs
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        # GRPO generation settings
        max_completion_length=16,    # enough for "<guess>ABCDE</guess>"
        num_generations=8,           # group size G (try 4–16)
        temperature=0.8,
        top_p=0.95,

        # training length
        max_steps=1000,              # adjust
        logging_steps=10,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,

        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to="none",

        # optional: keep policy close to ref a bit
        beta=0.02,                   # set 0.0 if you want no KL at all
    )

    trainer = GRPOTrainer(
        model=policy,
        processing_class=tok,
        args=cfg,
        train_dataset=ds,
        reward_funcs=wordle_reward,
    )

    trainer.train()
    trainer.save_model(f"{OUT_DIR}/final_adapter")

if __name__ == "__main__":
    main()
