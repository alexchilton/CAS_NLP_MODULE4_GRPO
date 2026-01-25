import re, random
import pandas as pd
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------- CONFIG ----------
WORD_LIST_PATH = "five_letter_words.csv"
COL_NAME = "Word"           # change if your CSV uses a different column
N_GAMES = 500
SEED = 42

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./qwen-wordle-sft-teacher-masked/final_adapter"

MAX_NEW_TOKENS = 16
DETERMINISTIC = True        # True = do_sample=False
TEMPERATURE = 0.7           # only used if DETERMINISTIC=False
TOP_P = 0.9

# ---------- Wordle feedback (same logic you used) ----------
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

def build_user_content(past_guesses):
    if past_guesses:
        lines = ["Previous guesses (word -> feedback):"]
        for guess, feedback in past_guesses:
            lines.append(f"{guess}: {feedback}")
        history = "\n".join(lines)
    else:
        history = "No previous guesses yet."

    return (
        f"{history}\n\n"
        "Return your next guess.\n"
        "Output exactly one line: <guess>ABCDE</guess>\n"
        "No other text."
    )

SYSTEM = (
    "You are an expert Wordle solver.\n"
    "Follow the feedback constraints exactly.\n"
    "Output exactly one line: <guess>ABCDE</guess>\n"
    "No other text."
)

# ---------- Model loading ----------
def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tok

def model_next_guess(model, tok, past_guesses):
    user = build_user_content(past_guesses)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if DETERMINISTIC:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P))

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)
    matches = re.findall(r"<guess>[A-Z]{5}</guess>", text)
    if not matches:
        return None, text
    guess = matches[-1][7:12]
    return guess, text

# ---------- Main evaluation ----------
def main():
    random.seed(SEED)

    df = pd.read_csv(WORD_LIST_PATH)
    col = COL_NAME if COL_NAME in df.columns else df.columns[0]
    words = df[col].astype(str).str.strip().str.upper()
    words = [w for w in words if re.fullmatch(r"[A-Z]{5}", w)]
    words_set = set(words)

    model, tok = load_model()

    parse_ok = 0
    invocab_ok = 0
    constraint_ok = 0
    solved = 0
    total_turns = 0
    repeats = 0

    for _ in range(N_GAMES):
        target = random.choice(words)
        candidates = list(words_set)
        past = []
        used = set()
        won = False

        for turn in range(1, 7):
            guess, raw = model_next_guess(model, tok, past)

            # parseable?
            if guess is None:
                break
            parse_ok += 1

            # in vocab?
            if guess in words_set:
                invocab_ok += 1
            else:
                break

            # repeated?
            if guess in used:
                repeats += 1
                # still continue; repetition is a quality metric

            # constraint-consistent? (i.e. guess should be compatible with past feedback)
            # We treat "consistent" as: guess is in current candidate set OR at least doesn't contradict
            # The safest check is: it must be compatible with *all previous feedback*.
            # Easiest robust way: ensure guess itself remains possible given past feedback:
            # simulate filtering candidates with all past and see if guess survives.
            tmp_cands = list(words_set)
            for g, fb in past:
                tmp_cands = filter_candidates(tmp_cands, g, fb)
            if guess in tmp_cands:
                constraint_ok += 1
            else:
                # still legal exploration, but it might contradict constraints; count as not consistent
                pass

            used.add(guess)

            fb = get_wordle_feedback(guess, target)
            past.append((guess, fb))

            # update candidate set (true constraints)
            candidates = filter_candidates(candidates, guess, fb)

            if guess == target:
                won = True
                solved += 1
                total_turns += turn
                break

        if not won:
            # count 6 turns for avg only if you want "avg turns when solved" vs "overall"
            pass

    # Metrics
    # parse_ok is counted per produced guess, so denominator is total attempts (N_GAMES*<=6) isn't tracked.
    # We'll report rates per guess among those that produced a guess.
    guesses_made = parse_ok if parse_ok > 0 else 1

    print("---- Rollout eval ----")
    print(f"Games: {N_GAMES}")
    print(f"Solved: {solved}/{N_GAMES} = {solved/N_GAMES:.3f}")
    if solved > 0:
        print(f"Avg turns (solved games): {total_turns/solved:.2f}")
    else:
        print("Avg turns (solved games): N/A")

    print(f"Parseable guess rate (per produced guess): {parse_ok/guesses_made:.3f} (always 1.0 by definition here)")
    print(f"In-vocab rate (per produced guess): {invocab_ok/guesses_made:.3f}")
    print(f"Constraint-consistent rate (per produced guess): {constraint_ok/guesses_made:.3f}")
    print(f"Repeat guess count: {repeats}")

if __name__ == "__main__":
    main()
