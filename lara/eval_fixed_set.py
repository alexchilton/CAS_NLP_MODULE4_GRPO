#!/usr/bin/env python3
"""
Evaluate multiple Wordle policies (PEFT adapters or full models) on the SAME fixed set of targets.

Metrics per model over N games:
- solve_rate (within 6)
- avg_turns_solved (only among solved)
- avg_turns_with_fail_as_7 (failures counted as 7 turns)
- format_fail_rate (no parsable <guess>ABCDE</guess>)
- oov_rate (guess not in your 5-letter word list)
- repeat_rate (re-guessing a previous guess in the same game)
- avg_log_candidate_reduction (mean log(before/after) per *valid step*)
- avg_frac_candidate_reduction (mean 1 - after/before per *valid step*)

Also saves:
- targets file (targets_500.json) so every run is comparable
- a few full game traces per model for slides (JSONL)

Run:
  python eval_fixed_games.py
"""

import os
import re
import json
import math
import random
from collections import Counter
from typing import List, Dict, Tuple, Any

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ----------------------------
# CONFIG (EDIT ME)
# ----------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
WORD_LIST_PATH = "five_letter_words.csv"
WORD_COL = "Word"
TAGGED_GUESS_RE = re.compile(r"<guess>\s*([A-Za-z]{5})\s*</guess>", re.IGNORECASE)
CLOSE_TAG_ONLY_RE = re.compile(r"([A-Za-z]{5})\s*</guess>", re.IGNORECASE)
ALPHA_RUN_RE = re.compile(r"[A-Za-z]{5,}")  # sequences of letters length >= 5

# Correct model paths (final only)
MODELS = {
    "only grpo": "./output_4090/wordle-grpo/final_model",
    "only sft": "./qwen2.5-3b-wordle-sft/final_adapter",
    "sft_teacher": "./qwen-wordle-sft-teacher/final_adapter",
    "sft_teacher_masked": "./qwen-wordle-sft-teacher-masked/final_adapter",
    "grpo on sft_masked_teach": "./qwen-wordle-grpo/final_adapter",
}

# Evaluation setup
SEED = 123
N_GAMES = 500
MAX_TURNS = 6

# Deterministic eval
TEMPERATURE = 0.0  # keep 0.0 for fair comparisons
TOP_P = 1.0
MAX_NEW_TOKENS = 24  # enough for "<guess>ABCDE</guess>"

# Outputs
OUT_CSV = "eval_500games_results.csv"
TARGETS_JSON = "targets_500.json"
TRACES_DIR = "traces"
N_TRACES_PER_MODEL = 5  # save 3–5; you can change to 10 etc.

# Prompt template (must match your training expectation)
# SYSTEM_MSG = (
#     "You are playing Wordle. Respond with exactly one guess in the format "
#     "<guess>ABCDE</guess> where ABCDE is a valid 5-letter English word in uppercase. "
#     "Do not output anything else."
# )

SYSTEM_MSG = (
  "Wordle. Output ONLY one 5-letter uppercase word. No punctuation, no explanation."
)


# ----------------------------
# WORDLE UTILS
# ----------------------------
GUESS_RE = re.compile(r"<guess>\s*([a-zA-Z]{5})\s*</guess>", re.IGNORECASE)

def normalize_word(w: str) -> str:
    return re.sub(r"[^A-Z]", "", str(w).upper())[:5]

def get_wordle_feedback(guess: str, target: str) -> str:
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

def filter_candidates(candidates: List[str], guess: str, feedback: str) -> List[str]:
    guess = normalize_word(guess)
    if len(guess) != 5:
        return candidates

    fb_norm = " ".join(feedback.strip().split())
    out = []
    for w in candidates:
        w = normalize_word(w)
        if len(w) != 5:
            continue
        sim = " ".join(get_wordle_feedback(guess, w).strip().split())
        if sim == fb_norm:
            out.append(w)
    return out

def candidates_from_past(words_list: List[str], past_guesses: List[Tuple[str, str]]) -> List[str]:
    cands = list(words_list)
    for g, fb in past_guesses:
        cands = filter_candidates(cands, g, fb)
        if not cands:
            break
    return cands


# ----------------------------
# DATA: LOAD WORD LIST + FIXED TARGETS
# ----------------------------
def load_words() -> Tuple[List[str], set]:
    df_words = pd.read_csv(WORD_LIST_PATH)
    col = WORD_COL if WORD_COL in df_words.columns else df_words.columns[0]
    words = df_words[col].astype(str).str.strip().str.upper().tolist()
    words = [w for w in words if re.fullmatch(r"[A-Z]{5}", w)]
    return words, set(words)

def load_or_create_targets(words: List[str]) -> List[str]:
    if os.path.exists(TARGETS_JSON):
        with open(TARGETS_JSON, "r") as f:
            targets = json.load(f)
        # basic validation
        targets = [normalize_word(t) for t in targets if isinstance(t, str)]
        targets = [t for t in targets if len(t) == 5]
        if len(targets) >= 1:
            return targets[:min(len(targets), N_GAMES)]

    random.seed(SEED)
    targets = random.sample(words, k=min(N_GAMES, len(words)))
    with open(TARGETS_JSON, "w") as f:
        json.dump(targets, f, indent=2)
    return targets


# ----------------------------
# PROMPTING
# ----------------------------
def build_user_msg(past: List[Tuple[str, str]]) -> str:
    if not past:
        return "Game start. Provide your first guess."
    lines = ["Past guesses and feedback:"]
    for g, fb in past:
        lines.append(f"- {g} -> {fb}")
    lines.append("Provide your next guess.")
    return "\n".join(lines)

def build_prompt(tok: AutoTokenizer, past: List[Tuple[str, str]]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(past)},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ----------------------------
# MODEL LOADING (PEFT adapter vs full model)
# ----------------------------
def load_model_and_tokenizer(path: str):
    # Tokenizer: always from base model (keeps chat template consistent)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # If adapter_config.json exists, treat as PEFT adapter
    is_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    if is_adapter:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, path)
    else:
        # full model folder
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    model.eval()
    return model, tok


def extract_guess(cont: str, words_set: set) -> Tuple[str, bool, bool]:
    """
    Returns (guess, extracted_ok, tagged_ok)
    - tagged_ok: used <guess>...</guess>
    - extracted_ok: we got some 5-letter guess from content (tagged or inferred)
    """
    m = TAGGED_GUESS_RE.search(cont)
    if m:
        return normalize_word(m.group(1)), True, True

    m = CLOSE_TAG_ONLY_RE.search(cont)
    if m:
        return normalize_word(m.group(1)), True, False

    # Fallback: find a run of letters, then try to choose an in-vocab 5-letter window
    m = ALPHA_RUN_RE.search(cont)
    if not m:
        return "", False, False

    run = m.group(0).upper()

    # Prefer an in-vocab 5-letter window
    for i in range(0, len(run) - 5 + 1):
        cand = run[i:i+5]
        if cand in words_set:
            return cand, True, False

    # If none are in vocab, fall back to first 5 letters
    return run[:5], True, False



@torch.no_grad()
def generate_guess(model, tok, past, words_set) -> Tuple[str, bool, bool, str]:

    """
    Returns (guess, extracted_ok, tagged_ok, raw_continuation_text).
    extracted_ok: we could extract some 5-letter guess (tagged or plain).
    tagged_ok: specifically matched <guess>.....</guess>.
    """
    prompt = build_prompt(tok, past)
    inputs = tok([prompt], return_tensors="pt", padding=True).to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0),
        temperature=(TEMPERATURE if TEMPERATURE > 0 else None),
        top_p=TOP_P,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    full = tok.decode(gen[0], skip_special_tokens=False)
    cont = full[len(prompt):] if full.startswith(prompt) else full

    guess, ok, tagged_ok = extract_guess(cont, words_set)
    if not ok:
        print("\n[PARSE FAIL] continuation was:\n", repr(cont[:300]), "\n")
        return "", False, False, cont

    return guess, True, tagged_ok, cont


# ----------------------------
# EVALUATION
# ----------------------------
def play_one_game(
    model,
    tok,
    target: str,
    words: List[str],
    words_set: set,
) -> Dict[str, Any]:
    """Play up to MAX_TURNS. Return detailed trace + counters."""
    past: List[Tuple[str, str]] = []
    used = set()

    game_steps = []
    solved = False
    format_fails = 0
    oov = 0
    repeats = 0

    # Candidate tracking (optional but great)
    # Use constraints implied by past feedback (which includes true feedback each step)
    valid_step_log_gains = []
    valid_step_frac_reductions = []

    for turn in range(1, MAX_TURNS + 1):
        guess, ok, tagged_ok, raw = generate_guess(model, tok, past, words_set)

        step = {
            "turn": turn,
            "prompt_past": list(past),
            "raw_model_output": raw,
            "parsed_ok": ok,
            "guess": guess if ok else None,
        }

        if not ok:
            format_fails += 1
            # Don't update past with nonsense constraints; just keep state moving:
            # We still need feedback to continue; give a dummy feedback that won't filter candidates meaningfully.
            # (Alternative: break; but then you're changing game dynamics vs other models.)
            dummy_fb = "?(x) ?(x) ?(x) ?(x) ?(x)"
            past.append(("?????", dummy_fb))
            step["feedback"] = dummy_fb
            step["status"] = "format_fail"
            game_steps.append(step)
            continue

        # Repeat?
        if guess in used:
            repeats += 1
        used.add(guess)

        # OOV?
        if guess not in words_set:
            oov += 1
            fb = get_wordle_feedback(guess, target)  # still compute env feedback
            past.append((guess, fb))
            step["feedback"] = fb
            step["status"] = "oov"
            game_steps.append(step)
            continue

        # Candidate reduction metrics (only for valid in-vocab guesses)
        tmp_cands = candidates_from_past(words, past)  # candidates before applying this move
        before = max(len(tmp_cands), 1)
        fb = get_wordle_feedback(guess, target)
        after_cands = filter_candidates(tmp_cands, guess, fb) if tmp_cands else []
        after = max(len(after_cands), 1)

        log_gain = math.log(before / after) if before > 0 and after > 0 else 0.0
        frac_red = 1.0 - (after / before) if before > 0 else 0.0
        valid_step_log_gains.append(log_gain)
        valid_step_frac_reductions.append(frac_red)

        past.append((guess, fb))
        step["feedback"] = fb
        step["status"] = "ok"
        step["candidates_before"] = before
        step["candidates_after"] = after
        step["log_candidate_gain"] = log_gain
        step["frac_candidate_reduction"] = frac_red

        if guess == target:
            solved = True
            step["status"] = "solved"
            game_steps.append(step)
            break

        game_steps.append(step)

    turns_used = len(game_steps)
    return {
        "target": target,
        "solved": solved,
        "turns_used": turns_used,
        "format_fails": format_fails,
        "oov": oov,
        "repeats": repeats,
        "valid_step_log_gains": valid_step_log_gains,
        "valid_step_frac_reductions": valid_step_frac_reductions,
        "steps": game_steps,
    }


def eval_model_on_targets(model_name: str, model_path: str, model, tok, targets: List[str], words: List[str], words_set: set):
    solved_count = 0
    turns_solved_list = []
    turns_with_fail_as_7_list = []

    total_format_fails = 0
    total_oov = 0
    total_repeats = 0

    # Candidate reduction aggregates over valid in-vocab steps only
    all_log_gains = []
    all_frac_reductions = []

    # Traces to save (choose a fixed subset of targets for comparability)
    trace_targets = targets[:min(N_TRACES_PER_MODEL, len(targets))]
    saved_traces = []

    for i, tgt in enumerate(targets):
        game = play_one_game(model, tok, tgt, words, words_set)

        if game["solved"]:
            solved_count += 1
            turns_solved_list.append(game["turns_used"])
            turns_with_fail_as_7_list.append(game["turns_used"])
        else:
            turns_with_fail_as_7_list.append(7)

        total_format_fails += game["format_fails"]
        total_oov += game["oov"]
        total_repeats += game["repeats"]

        all_log_gains.extend(game["valid_step_log_gains"])
        all_frac_reductions.extend(game["valid_step_frac_reductions"])

        if tgt in trace_targets:
            saved_traces.append(game)

        if (i + 1) % 50 == 0:
            print(f"  {model_name}: finished {i+1}/{len(targets)} games...")

    n_games = len(targets)
    solve_rate = solved_count / n_games if n_games else 0.0

    avg_turns_solved = float("nan")
    if turns_solved_list:
        avg_turns_solved = sum(turns_solved_list) / len(turns_solved_list)

    avg_turns_fail_as_7 = sum(turns_with_fail_as_7_list) / len(turns_with_fail_as_7_list) if turns_with_fail_as_7_list else float("nan")

    # Rates: normalize by total turns attempted (<= N_GAMES*6)
    total_turns_attempted = sum(min(g, MAX_TURNS) for g in turns_with_fail_as_7_list)  # approx; failures counted 7 but turns used in game is still <=6
    # Better: count actual turns used per game:
    # We'll derive it from average list length: already in turns_with_fail_as_7_list mixes 7; so use a safer denominator:
    max_total_turns = n_games * MAX_TURNS
    denom_turns = max_total_turns if max_total_turns else 1

    format_fail_rate = total_format_fails / denom_turns
    oov_rate = total_oov / denom_turns
    repeat_rate = total_repeats / denom_turns

    avg_log_gain = (sum(all_log_gains) / len(all_log_gains)) if all_log_gains else float("nan")
    avg_frac_red = (sum(all_frac_reductions) / len(all_frac_reductions)) if all_frac_reductions else float("nan")

    # Save traces
    os.makedirs(TRACES_DIR, exist_ok=True)
    trace_path = os.path.join(TRACES_DIR, f"{model_name.replace(' ', '_')}.jsonl")
    with open(trace_path, "w") as f:
        for g in saved_traces:
            f.write(json.dumps(g) + "\n")

    metrics = {
        "model_name": model_name,
        "path": model_path,
        "n_games": n_games,
        "solve_rate": solve_rate,
        "avg_turns_solved": avg_turns_solved,
        "avg_turns_with_fail_as_7": avg_turns_fail_as_7,
        "format_fail_rate_per_turn": format_fail_rate,
        "oov_rate_per_turn": oov_rate,
        "repeat_rate_per_turn": repeat_rate,
        "avg_log_candidate_reduction": avg_log_gain,
        "avg_frac_candidate_reduction": avg_frac_red,
        "traces_saved_to": trace_path,
    }
    return metrics


def main():
    # Determinism as much as possible
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load words and fixed targets
    words, words_set = load_words()
    targets = load_or_create_targets(words)
    if len(targets) < N_GAMES:
        print(f"[warn] Only {len(targets)} targets available (requested {N_GAMES}).")

    # Quick path sanity checks
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[warn] Missing path for {name}: {path}")
        else:
            if not os.path.exists(os.path.join(path, "adapter_config.json")):
                # Might still be a full model directory; just warn
                print(f"[info] {name}: no adapter_config.json at {path} (will attempt full-model load).")

    rows = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: path not found: {path}")
            continue

        print(f"\n=== Loading {name} from {path} ===")
        model, tok = load_model_and_tokenizer(path)

        print(f"=== Evaluating {name} on {len(targets)} fixed targets ===")
        metrics = eval_model_on_targets(name, path, model, tok, targets, words, words_set)
        rows.append(metrics)

        # Free VRAM between models
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved results to: {OUT_CSV}")
    print(f"Saved targets to: {TARGETS_JSON}")
    print(f"Saved traces to: {TRACES_DIR}/<model>.jsonl\n")

    # Print a nice ranking
    show_cols = [
        "model_name",
        "solve_rate",
        "avg_turns_solved",
        "avg_turns_with_fail_as_7",
        "format_fail_rate_per_turn",
        "oov_rate_per_turn",
        "repeat_rate_per_turn",
        "avg_log_candidate_reduction",
        "avg_frac_candidate_reduction",
    ]
    print(df[show_cols].sort_values("solve_rate", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
