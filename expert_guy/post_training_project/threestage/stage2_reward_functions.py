"""
Stage 2 Reward Functions: Light GRPO with Format Masking
=========================================================

Key principle: Assume format is mostly learned, focus on STRATEGY

Format masking approach:
- Only penalize SEVERE format violations (no guess tag at all)
- Ignore minor format issues (spacing, capitalization)
- Reduce format rewards to minimal (already learned)
- Amplify strategic rewards (feedback usage, info gain)

Reward balance:
- Format: 0.0 to +0.2 (mostly neutral, already learned)
- Strategy: -2.0 to +3.0 (dominate the signal)
- Total: -2.0 to +3.2

This allows the model to explore strategic improvements without
being punished for minor format variations.
"""

import math
import re
import ast
import pandas as pd


# ===== STAGE 2 FORMAT MASKING CONSTANTS =====

# FORMAT REWARDS (minimal - already learned in Stage 1)
HAS_GUESS_TAG_REWARD = 0.2        # Small reward for having <guess> tag
NO_GUESS_TAG_PENALTY = -0.5       # Only penalize if completely missing

# VALIDITY (lenient - focus on strategy)
VALID_WORD_BONUS = 0.1            # Small bonus (not the focus)
INVALID_WORD_PENALTY = -0.3       # Gentle nudge (not harsh)

# FEEDBACK USAGE REWARDS (amplified for strategy learning)
CORRECT_POSITION_REWARD = 0.6     # DOUBLED from 0.4
NEW_POSITION_REWARD = 0.5         # INCREASED from 0.3
REPEATED_WRONG_POSITION = -0.4    # INCREASED penalty
EXPLORATION_BONUS = 0.1           # DOUBLED from 0.05

# FEEDBACK USAGE PENALTIES (amplified to teach strategy)
DEAD_LETTER_PENALTY = -1.0        # DOUBLED from -0.5
MISSING_GOOD_LETTER_PENALTY = -0.8  # DOUBLED from -0.4
MAX_FEEDBACK_PENALTY = -2.0       # Allow stronger penalties for bad strategy

# DENSE REWARD (amplified)
WORD_ACCURACY_WEIGHT = 1.5        # INCREASED from 1.0

# INFO GAIN (amplified)
INFO_GAIN_WEIGHT = 1.2            # INCREASED from 1.0


def output_format_check(prompt: str, completion: str, example: dict, training_progress: float = 0.0) -> float:
    """
    STAGE 2: Minimal format checking - assume format is learned

    Only check:
    - Has <guess> tag? (yes = +0.2, no = -0.5)
    - Is it a 5-letter word from list? (yes = +0.1, no = -0.3)

    Returns: -0.5 to +0.3 (much smaller range than strategic rewards)
    """
    reward = 0.0

    try:
        # Add synthetic <think> as it's in the prompt
        completion = "<think>" + completion

        # Check for <guess> tag (very lenient)
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)

        if not guess_match:
            # No guess tag at all - penalize
            return NO_GUESS_TAG_PENALTY  # -0.5

        # Has guess tag - small reward
        reward += HAS_GUESS_TAG_REWARD  # +0.2

        # Extract guess
        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        # Check if it's a valid word (lenient)
        if len(guess) != 5:
            return reward + INVALID_WORD_PENALTY  # +0.2 - 0.3 = -0.1

        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess in word_list_upper:
            reward += VALID_WORD_BONUS  # +0.1
        else:
            reward += INVALID_WORD_PENALTY  # -0.3

        return reward

    except Exception as e:
        return NO_GUESS_TAG_PENALTY  # -0.5


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    STAGE 2: Amplified strategic rewards for using feedback correctly

    This is the PRIMARY teaching signal in Stage 2.
    Rewards/penalties are 2x stronger than Stage 1.

    Returns: -2.0 to +3.0 (dominates format rewards)
    """
    reward = 0.0

    try:
        # Extract guess
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        if len(guess) != 5:
            return 0.0

        # Parse past guess history
        past_history = example.get("past_guess_history", [])
        if isinstance(past_history, str):
            try:
                past_history = ast.literal_eval(past_history)
            except:
                past_history = []

        if not past_history:
            # First guess - reward exploration
            reward = len(set(guess)) * EXPLORATION_BONUS
            return reward

        # Track constraints from feedback
        confirmed_positions = {}
        valid_letter_to_positions = {}
        dead_letters = set()

        for past_guess, feedback_str in past_history:
            past_guess = past_guess.upper()
            feedback_parts = feedback_str.split()

            for i, part in enumerate(feedback_parts):
                if i >= len(past_guess):
                    continue

                letter = past_guess[i]

                if '(✓)' in part or '(√)' in part:
                    confirmed_positions[i] = letter
                elif '(-)' in part:
                    if letter not in valid_letter_to_positions:
                        valid_letter_to_positions[letter] = []
                    valid_letter_to_positions[letter].append(i)
                elif '(x)' in part or '(X)' in part:
                    dead_letters.add(letter)

        guess_letters = list(guess)

        # 1. Check confirmed positions (AMPLIFIED REWARDS)
        for pos, letter in confirmed_positions.items():
            if pos < len(guess_letters) and guess_letters[pos] == letter:
                reward += CORRECT_POSITION_REWARD  # +0.6
            else:
                reward += REPEATED_WRONG_POSITION  # -0.4

        # 2. Check valid letters at new positions (AMPLIFIED)
        for letter, tried_positions in valid_letter_to_positions.items():
            if letter in guess_letters:
                current_pos = guess_letters.index(letter)
                if current_pos not in tried_positions:
                    reward += NEW_POSITION_REWARD  # +0.5
                else:
                    reward += REPEATED_WRONG_POSITION  # -0.4
            else:
                reward += MISSING_GOOD_LETTER_PENALTY  # -0.8

        # 3. Penalize dead letters (AMPLIFIED)
        for letter in guess_letters:
            if letter in dead_letters:
                reward += DEAD_LETTER_PENALTY  # -1.0

        # 4. Reward exploration (AMPLIFIED)
        new_letters = set(guess_letters) - set(confirmed_positions.values()) - set(valid_letter_to_positions.keys()) - dead_letters
        reward += len(new_letters) * EXPLORATION_BONUS  # +0.1 each

        # Clamp total penalty
        reward = max(reward, MAX_FEEDBACK_PENALTY)

        return reward

    except Exception as e:
        return 0.0


def filter_words_by_history(word_list_upper, past_history):
    """Helper: Filter word list by feedback constraints"""
    if not past_history:
        return word_list_upper

    confirmed_positions = {}
    valid_letters = {}
    dead_letters = set()

    for past_guess, feedback_str in past_history:
        past_guess = past_guess.upper()
        feedback_parts = feedback_str.split()

        for i, part in enumerate(feedback_parts):
            if i >= len(past_guess):
                continue
            letter = past_guess[i]

            if '(✓)' in part or '(√)' in part:
                confirmed_positions[i] = letter
            elif '(-)' in part:
                if letter not in valid_letters:
                    valid_letters[letter] = []
                valid_letters[letter].append(i)
            elif '(x)' in part or '(X)' in part:
                dead_letters.add(letter)

    candidates = []
    for word in word_list_upper:
        word = word.upper() if isinstance(word, str) else word
        if len(word) != 5:
            continue

        valid = True
        for pos, letter in confirmed_positions.items():
            if word[pos] != letter:
                valid = False
                break
        if not valid:
            continue

        for letter, wrong_positions in valid_letters.items():
            if letter not in word:
                valid = False
                break
            for wrong_pos in wrong_positions:
                if word[wrong_pos] == letter:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        for letter in dead_letters:
            if letter in word:
                valid = False
                break
        if not valid:
            continue

        candidates.append(word)

    return candidates


def word_accuracy_reward(prompt: str, completion: str, example: dict) -> float:
    """
    STAGE 2: Amplified dense reward signal

    Weight increased to 1.5x to provide stronger gradient toward correct answer.

    Returns: 0.0 to 1.5
    """
    try:
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        if len(guess) != 5:
            return 0.0

        hidden = example.get("secret_word", "").upper()
        if len(hidden) != 5:
            return 0.0

        # Count exact matches
        exact = sum(g == h for g, h in zip(guess, hidden))

        # Count letters that exist
        exist = sum(min(guess.count(c), hidden.count(c)) for c in set(guess))

        # Dense reward formula
        raw_score = (exact + 0.2 * max(0, exist - exact)) / 6.0
        accuracy = min(1.0, raw_score * 1.2)

        # Apply weight
        return accuracy * WORD_ACCURACY_WEIGHT

    except Exception as e:
        return 0.0


def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    STAGE 2: Amplified information gain reward

    Weight increased to 1.2x to encourage exploration of high-value guesses.

    Returns: 0.0 to 1.2
    """
    try:
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        if len(guess) != 5:
            return 0.0

        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess not in word_list_upper:
            return 0.0

        # Filter candidates by past feedback
        past_history = example.get("past_guess_history", [])
        if isinstance(past_history, str):
            try:
                past_history = ast.literal_eval(past_history)
            except:
                past_history = []

        candidates = filter_words_by_history(word_list_upper, past_history)
        allowed_letters = set(''.join(candidates)) if candidates else set()

        # Information gain heuristic
        unique_letters = len(set(guess))
        common_letters = set('ETAOINSHRDLU')
        common_count = sum(1 for letter in guess if letter in common_letters and letter in allowed_letters)
        unique_allowed = len(set(guess) & allowed_letters)

        # Normalize
        uniqueness_score = unique_allowed / 5.0
        commonality_score = common_count / 5.0

        # Combine
        info_gain = (uniqueness_score * 0.6 + commonality_score * 0.4)

        # Apply weight
        return info_gain * INFO_GAIN_WEIGHT

    except Exception as e:
        return 0.0