import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random
import re
import os
from prompt_templates import create_wordle_prompt
from collections import Counter
from peft import PeftModel
import sys; sys.stdout.flush()

# 1. PATH CONFIGURATION
# RELATIVE_PATH = "~/nlp_m4_grpo/output_4090/wordle-grpo/final_model"
# MODEL_PATH = os.path.expanduser(RELATIVE_PATH)
WORD_LIST_PATH = "five_letter_words.csv"
NUM_GAMES = 10

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "./qwen2.5-3b-wordle-sft/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# Load the words from your CSV
# Assumes the first column contains the 5-letter words
words_df = pd.read_csv("five_letter_words.csv")
# words = [str(w).upper() for w in words_df.iloc[:,0].tolist() if len(str(w)) == 5]

col = "Word" if "Word" in words_df.columns else words_df.columns[0]
words = (
    words_df[col]
    .astype(str)
    .str.strip()
    .str.upper()
)

words = [w for w in words if re.fullmatch(r"[A-Z]{5}", w)]
words_set = set(words)

print("Wordlist size:", len(words_set))
print("LOGIC in wordlist?", "LOGIC" in words_set)
print("THINK in wordlist?", "THINK" in words_set)
print("AAHED in wordlist?", "AAHED" in words_set)

# Pick a secret target word for this session
target_word = random.choice(words)
print(f"🎯 The target word for this game is: {target_word}")

# --- 1. DEFINE FUNCTIONS AT THE TOP ---
def get_wordle_feedback(guess, target):
    guess = (guess[:5] + "     ")[:5].upper()
    target = target.upper()

    result = [""] * 5
    target_counts = Counter(target)

    # Greens
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = f"{guess[i]}(✓)"
            target_counts[guess[i]] -= 1

    # Yellows / Grays
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

def extract_guess(text: str) -> str:
    # Prefer tagged format
    m = re.search(r"<guess>\s*([A-Za-z]{5})\s*</guess>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: first standalone 5-letter token anywhere
    m = re.search(r"\b([A-Za-z]{5})\b", text)
    if m:
        return m.group(1).upper()

    return ""

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

def normalize_word(w: str) -> str:
    w = re.sub(r"[^A-Z]", "", str(w).upper())
    return w[:5]

def filter_candidates(candidates, guess, feedback, get_wordle_feedback):
    """
    Keep only those candidate words that would produce exactly the same feedback
    if 'guess' were applied to them as the target.

    This is robust (including duplicate letters) because it delegates to your
    get_wordle_feedback().
    """
    guess = normalize_word(guess)
    if not guess or len(guess) != 5:
        return candidates

    # Normalize feedback spacing so minor formatting differences don't break matches
    fb_norm = " ".join(feedback.strip().split())

    filtered = []
    for w in candidates:
        w = normalize_word(w)
        if len(w) != 5:
            continue
        sim_fb = get_wordle_feedback(guess, w)
        sim_fb = " ".join(sim_fb.strip().split())
        if sim_fb == fb_norm:
            filtered.append(w)
    return filtered

def pick_best_guess(candidates, allowed_words=None, used=None):
    """
    Heuristic:
    - Prefer guesses that cover high-frequency letters in the remaining candidate set.
    - Prefer placing letters in positions where they're frequent.
    - Penalize duplicate letters (information waste).

    candidates: current remaining possible solutions
    allowed_words: list/set of valid guesses to consider (defaults to candidates)
    used: set of words already guessed
    """
    if used is None:
        used = set()

    # If allowed_words is not provided, only guess from remaining candidates.
    # This is "hard mode" and guarantees guess is possible solution.
    if allowed_words is None:
        pool = candidates
    else:
        pool = allowed_words

    pool = [w for w in pool if w not in used]
    if not pool:
        return None

    # If candidates is tiny, just pick from candidates directly.
    if len(candidates) <= 2:
        for w in candidates:
            if w not in used:
                return w
        return pool[0]

    # Letter frequency across remaining candidates (global)
    global_counts = Counter("".join(candidates))

    # Position-specific letter frequency
    pos_counts = [Counter() for _ in range(5)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1

    def score_word(word):
        # position score: reward letters that are common in that position
        s = 0.0
        for i, ch in enumerate(word):
            s += pos_counts[i][ch] * 1.0

        # coverage score: reward unique letters that are globally common
        uniq = set(word)
        s += sum(global_counts[ch] for ch in uniq) * 0.5

        # duplicate penalty: discourage repeated letters
        dup_pen = (5 - len(uniq)) * 2.0
        s -= dup_pen

        return s

    best = max(pool, key=score_word)
    return best


# --- 2. SETUP FOR MULTIPLE GAMES ---
results = {'WIN': 0, 'LOSE': 0}
random.seed(42) 
test_words = random.sample(words, NUM_GAMES)

# --- 3. THE OUTER LOOP (One iteration per word) ---
for game_idx, target_word in enumerate(test_words):
    past_guesses = []  # RESET HISTORY for every new word
    success = False

    candidates = list(words_set) 
    used = set() 
    
    
    print(f"\n{'='*20} GAME {game_idx + 1} / {NUM_GAMES} {'='*20}")
    print(f"🎯 Secret Target: {target_word}")

    # --- 4. THE INNER LOOP (The 6-turn Wordle logic) ---
    for turn in range(1, 7):
        # Create prompt with current game's history
        user_content = build_user_content(past_guesses)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Wordle solver.\n"
                    "Follow the feedback constraints exactly.\n"
                    "Output exactly one line: <guess>ABCDE</guess>\n"
                    "No other text."
                    ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # IMPORTANT
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7, 
                top_p = 0.9, 
                eos_token_id=tokenizer.eos_token_id,
            )

        # 5. Decode ONLY the newly generated tokens
        resp = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        candidate = extract_guess(resp).strip().upper()
        candidate = re.sub(r"[^A-Z]", "", candidate)[:5]

        used = {g for g, _ in past_guesses}

        bad_reasons = []
        if not candidate:
            bad_reasons.append("empty_parse")
        if candidate and candidate not in words_set:
            bad_reasons.append("not_in_wordlist")
        if candidate and candidate in used:
            bad_reasons.append("repeated")
        if candidate and candidate not in candidates:
            bad_reasons.append("violates_constraints")

        # If we are close to the end, NEVER probe. Just solve.
        force_solve = (len(candidates) <= 3)

        if bad_reasons or force_solve:
            if bad_reasons:
                print(f"⚠️ Bad guess '{candidate}' ({','.join(bad_reasons)}). used={sorted(used)}", flush=True)

            if force_solve:
                # hard mode: must pick from remaining solutions
                candidate = pick_best_guess(candidates, allowed_words=None, used=used)
            else:
                # probe mode: can pick any allowed word
                candidate = pick_best_guess(candidates, allowed_words=words, used=used)

            if not candidate:
                candidate = random.choice([w for w in words if w not in used])

        guess = candidate

        feedback = get_wordle_feedback(guess, target_word)
        print(f"--- Turn {turn} ---")
        print(f"Guess: {guess} | Feedback: {feedback}")
        past_guesses.append((guess, feedback))
        used.add(guess)

        candidates = filter_candidates(candidates, guess, feedback, get_wordle_feedback)
        print(f"Remaining candidates: {len(candidates)}", flush=True)

        # print(f"\n--- Turn {turn} ---", flush=True)
        # print(f"Guess: {guess} | Feedback: {feedback}", flush=True)

        if guess.strip() == target_word:
            print(f"✅ WIN! Found '{target_word}' in {turn} turns.")
            results['WIN'] += 1
            success = True
            break
    
    if not success:
        print(f"❌ LOSE! The word was {target_word}")
        results['LOSE'] += 1

# --- 5. SUMMARY ---
print("\n" + "#"*30)
print(f"FINAL SCORE: {results['WIN']} Wins, {results['LOSE']} Losses")
print("#"*30)