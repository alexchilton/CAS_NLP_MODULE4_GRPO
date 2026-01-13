import random, re, json
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

WORD_LIST_PATH = "five_letter_words.csv"
OUT_JSONL = "wordle_teacher_trajs.jsonl"
N_GAMES = 50000
SEED = 42

random.seed(SEED)


# Load the words from your CSV
words_df = pd.read_csv(WORD_LIST_PATH)
col = "Word" if "Word" in words_df.columns else words_df.columns[0]
words = words_df[col].astype(str).str.strip().str.upper()
words = [w for w in words if re.fullmatch(r"[A-Z]{5}", w)]
words_set = set(words)

BASE = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

STARTERS = ["CRANE","SLATE","TRACE","SALET","ROATE","RAISE","ARISE","AUDIO","STARE"]
STARTERS = [w for w in STARTERS if w in words_set]

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
    Heuristic (teacher):
    - scores words by letter + position frequency
    - samples from top-k to avoid deterministic trajectories
    """
    if used is None:
        used = set()

    # choose pool
    pool = candidates if allowed_words is None else allowed_words
    pool = [w for w in pool if w not in used]
    if not pool:
        return None

    # trivial case
    if len(candidates) <= 2:
        return pool[0]

    # --- statistics from remaining candidates ---
    global_counts = Counter("".join(candidates))
    pos_counts = [Counter() for _ in range(5)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1

    def score_word(word):
        s = 0.0

        # positional frequency
        for i, ch in enumerate(word):
            s += pos_counts[i][ch]

        # coverage (unique letters)
        uniq = set(word)
        s += sum(global_counts[ch] for ch in uniq) * 0.5

        # duplicate penalty
        s -= (5 - len(uniq)) * 2.0
        return s

    # --- top-k stochastic selection ---
    scored = [(w, score_word(w)) for w in pool]
    scored.sort(key=lambda x: x[1], reverse=True)

    k = min(5, len(scored))
    top = scored[:k]

    weights = [max(s, 1e-6) for _, s in top]
    return random.choices(
        [w for w, _ in top],
        weights=weights,
        k=1
    )[0]

def to_text(messages, label):
    full = messages + [{"role": "assistant", "content": label}]
    return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)


SYSTEM = (
    "You are an expert Wordle solver.\n"
    "Follow the feedback constraints exactly.\n"
    "Output exactly one line: <guess>ABCDE</guess>\n"
    "No other text."
)

train_targets = random.choices(words, k=N_GAMES)

with open(OUT_JSONL, "w") as f:
    n_rows = 0
    for game_idx, target_word in enumerate(train_targets):
        if (game_idx + 1) % 1000 == 0:
            print(f"Generated {n_rows} rows from {game_idx+1}/{N_GAMES} games...")

        past_guesses = []
        used = set()
        candidates = list(words_set)

        for turn in range(1, 7):
            user_content = build_user_content(past_guesses)

            if turn == 1 and STARTERS:
                guess = random.choice(STARTERS)
            else:
                force_solve = (len(candidates) <= 3)
                guess = pick_best_guess(candidates, allowed_words=None if force_solve else words, used=used)

            if guess is None:
                break

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ]
            label = f"<guess>{guess}</guess>"

            record = {
                "text": to_text(messages, label),
                "system": SYSTEM,
                "user": user_content,
                "label": label,
                "turn": turn,
                "target": target_word,
                "n_candidates": len(candidates),
                "past_guesses": past_guesses,
                "guess": guess,
            }

            f.write(json.dumps(record) + "\n")
            n_rows += 1

            feedback = get_wordle_feedback(guess, target_word)
            past_guesses.append((guess, feedback))
            used.add(guess)
            candidates = filter_candidates(candidates, guess, feedback, get_wordle_feedback)

            if guess == target_word:
                break

print(f"Saved {n_rows} rows to {OUT_JSONL}")
