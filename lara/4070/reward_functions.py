import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import re
import json
import ast
import pandas as pd
from logger_setup import logger

SOLUTION_PATTERN = re.compile(
    r"<solution>\s*([A-Za-z]{5})\s*</solution>",
    re.IGNORECASE,
)


def extract_guess_from_completion(completion: str) -> str:
    # 1. Standard full tag check
    tags_pattern = r"<solution>\s*([A-Za-z]{5})\s*</solution>"
    matches = re.findall(tags_pattern, completion, re.IGNORECASE)
    if matches:
        return matches[-1].upper().strip()
    
    # 2. PARTIAL TAG CHECK (Crucial for clipped completions)
    # If the model wrote <solution>WORD but got cut off before </solution>
    partial_pattern = r"<solution>\s*([A-Za-z]{5})"
    partials = re.findall(partial_pattern, completion, re.IGNORECASE)
    if partials:
        return partials[-1].upper().strip()

    # 3. Fallback: Last 5-letter word
    fallback_pattern = r"\b([A-Za-z]{5})\b"
    fallbacks = re.findall(fallback_pattern, completion)
    if fallbacks:
        # Avoid common words that aren't guesses (like 'think')
        filtered = [f for f in fallbacks if f.lower() not in ['think', 'words', 'guess']]
        if filtered:
            return filtered[-1].upper().strip()
            
    return ""


def parse_history(past_guess_history_item):
    """
    Turn one example's past_guess_history into:
        guesses: [ 'CRANE', 'FROST', ... ]
        feedbacks: [ 'C(x) R(-) ...', ... ]
    """
    guesses = []
    feedbacks = []

    h = past_guess_history_item

    if isinstance(h, str):
        try:
            h = ast.literal_eval(h)
        except Exception:
            h = []

    if isinstance(h, list):
        for item in h:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                g, fb = item[0], item[1]
                if isinstance(g, str) and isinstance(fb, str):
                    guesses.append(g.strip().upper())
                    feedbacks.append(fb.strip())

    return guesses, feedbacks


def letters_constraints_from_history(guesses, feedbacks):
    """
    Build simple constraints from history:

    - greens[pos] = letter that MUST be at that position
    - yellows: letters that MUST appear (somewhere, but not at specific positions)
    - dead: letters that SHOULD NOT appear

    This is a simplification but enough for a reward.
    """
    greens = {}          # pos -> letter
    yellows = set()      # letters known to be in the word
    dead = set()         # letters known NOT to be in the word

    for guess, fb in zip(guesses, feedbacks):
        guess = guess.upper()
        parts = fb.split()

        for i, part in enumerate(parts):
            if i >= len(guess):
                continue
            letter = guess[i]

            if "(✓)" in part or "(√)" in part:
                greens[i] = letter
                yellows.add(letter)  # it's also in the word
            elif "(-)" in part:
                yellows.add(letter)
            elif "(x)" in part or "(X)" in part:
                # only mark dead if we don't already know it's green/yellow
                if letter not in yellows and letter not in greens.values():
                    dead.add(letter)

    return greens, yellows, dead

def format_reward(completion):
    # Reward for using the correct tags
    if "<think>" in completion and "</think>" in completion:
        if "<solution>" in completion and "</solution>" in completion:
            return 0.5  # Bonus for following structure
    return 0.0


# def wordle_reward_func(
#     completions,
#     prompts=None,
#     secret_word=None,
#     past_guess_history=None,
#     word_list_path="five_letter_words.csv",
#     **kwargs,
# ):
#     # Load dictionary (Optimization: consider moving this outside the function if possible)
#     word_list = pd.read_csv(word_list_path)
#     valid_words = set(word_list["Word"].str.upper().tolist())
    
#     # Common strong starters for the START_BONUS
#     strong_starters = {"STARE", "CRANE", "AUDIO", "TRACE", "SLATE", "ROATE"}

#     INVALID_PENALTY = -2.0
#     REPEAT_BASE_PENALTY = -5.0  # Increased to be more aggressive
#     EXACT_BONUS = 10.0
#     CONSTRAINTS_BONUS = 1.0
#     CONSTRAINTS_PENALTY = -1.5
#     REWARD_MIN, REWARD_MAX = -5.0, 15.0

#     rewards = []

#     for i in range(len(completions)):
#         text = completions[i] or ""
#         secret = secret_word[i].strip().upper()
        
#         # 1) Parse history for this specific instance
#         guesses_hist, feedbacks_hist = [], []
#         if past_guess_history is not None:
#             guesses_hist, feedbacks_hist = parse_history(past_guess_history[i])

#         # 2) Extract and validate guess
#         guess = extract_guess_from_completion(text)
#         if not guess or len(guess) != 5 or not guess.isalpha() or guess not in valid_words:
#             rewards.append(float(INVALID_PENALTY))
#             continue

#         # 3) Initialize Reward
#         current_reward = 0.0

#         # 4) Meaningful Start Bonus (Turn 1 only)
#         if len(guesses_hist) == 0:
#             if guess in strong_starters:
#                 current_reward += 0.5
        
#         # 5) Escalating Repeat Penalty
#         # If the word was guessed before, punish based on how many times it appears
#         repeat_count = guesses_hist.count(guess)
#         if repeat_count > 0:
#             current_reward += REPEAT_BASE_PENALTY * (repeat_count + 1)

#         # 6) Progressive Match Rewards (Green/Yellow logic)
#         secret_list = list(secret)
#         guess_list = list(guess)
#         pos_matches = 0
#         yellow_matches = 0

#         # First pass: Count Greens
#         for j in range(5):
#             if guess_list[j] == secret_list[j]:
#                 pos_matches += 1
#                 secret_list[j] = None
#                 guess_list[j] = None
        
#         # Second pass: Count Yellows
#         for j in range(5):
#             if guess_list[j] is not None and guess_list[j] in secret_list:
#                 yellow_matches += 1
#                 secret_list[secret_list.index(guess_list[j])] = None

#         current_reward += (pos_matches * 1.0) + (yellow_matches * 0.4)

#         # 7) Constraint Validation (The most important part for RL)
#         if guesses_hist and feedbacks_hist:
#             greens, yellows, dead = letters_constraints_from_history(guesses_hist, feedbacks_hist)
            
#             violation = False
#             # Check Greens: Must have the letter in the known position
#             for pos, letter in greens.items():
#                 if guess[pos] != letter: violation = True
            
#             # Check Dead: Must not use letters known to be absent
#             for letter in guess:
#                 if letter in dead: violation = True
            
#             # Check Yellows: Must include the letter somewhere
#             for letter in yellows:
#                 if letter not in guess: violation = True

#             current_reward += CONSTRAINTS_PENALTY if violation else CONSTRAINTS_BONUS

#         # 8) Win Bonus
#         if guess == secret:
#             current_reward += EXACT_BONUS

#         # Clamp and append
#         rewards.append(float(max(REWARD_MIN, min(REWARD_MAX, current_reward))))

#     return rewards

def wordle_reward_func(completions, prompts, secret_word, past_guess_history=None, **kwargs):
    rewards = []
    
    # Constants for tuning
    INVALID_PENALTY = -2.0
    REPEAT_PENALTY = -3.0
    CONSTRAINT_VIOLATION_PENALTY = -1.5
    EXACT_BONUS = 20.0

    for i, (completion, secret) in enumerate(zip(completions, secret_word)):
        score = 0.0
        secret = secret.upper()
        
        # 1. Use your robust extraction
        guess = extract_guess_from_completion(completion)
        
        # 2. Format Bonus (using your helper)
        score += format_reward(completion)

        if not guess or len(guess) != 5:
            rewards.append(INVALID_PENALTY)
            continue

        # 3. Use your helpers to parse history
        guesses_hist, feedbacks_hist = [], []
        if past_guess_history is not None:
            guesses_hist, feedbacks_hist = parse_history(past_guess_history[i])

        # 4. Anti-Repetition Check
        if guess in guesses_hist:
            score += REPEAT_PENALTY

        # 5. Logic/Constraint Check (The "Brain" part)
        if guesses_hist and feedbacks_hist:
            greens, yellows, dead = letters_constraints_from_history(guesses_hist, feedbacks_hist)
            
            violation = False
            # Check Greens: Must have the letter in the known position
            for pos, letter in greens.items():
                if guess[pos] != letter: 
                    violation = True
            
            # Check Dead: Must not use letters known to be absent
            for letter in guess:
                if letter in dead: 
                    violation = True
            
            # Check Yellows: Must include the letter somewhere
            for letter in yellows:
                if letter not in guess: 
                    violation = True

            if violation:
                score += CONSTRAINT_VIOLATION_PENALTY

        # 6. Win/Match Scoring
        if guess == secret:
            score += EXACT_BONUS
        else:
            # Add simple character-match scoring (Greens)
            for g_char, s_char in zip(guess, secret):
                if g_char == s_char:
                    score += 1.0
                elif g_char in secret:
                    score += 0.3

        rewards.append(float(score))
        
    return rewards