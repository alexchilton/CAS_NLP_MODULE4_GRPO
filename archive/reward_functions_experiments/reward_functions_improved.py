"""
IMPROVED Reward Functions for Wordle GRPO Training

Key improvements:
1. More lenient regex - rewards partial success
2. Balanced reward magnitudes (1.0 scale instead of 10x)
3. Consistent learning signals
4. Progressive guidance (don't punish early, teach incrementally)

Philosophy: Clear, consistent, balanced rewards that guide learning
"""

import math
import re
import ast
import pandas as pd

# ===== BALANCED REWARD CONSTANTS =====
# All rewards on a 0-1 scale for consistency

# Format rewards (encourage correct structure)
VALID_FORMAT_REWARD = 1.0           # Has proper <think></think><guess></guess>
PARTIAL_FORMAT_REWARD = 0.3         # Has <guess> tag but improper format
NO_FORMAT_REWARD = 0.0              # No <guess> tag at all

# Validity rewards/penalties
VALID_WORD_BONUS = 0.5              # Bonus for valid 5-letter word
INVALID_LENGTH_PENALTY = -0.3       # Wrong length (mild penalty)
INVALID_WORD_PENALTY = -0.5         # Not in word list (moderate penalty)

# Feedback usage rewards (encourage learning from feedback)
CORRECT_POSITION_REWARD = 0.4       # Per letter in confirmed ✓ position
NEW_POSITION_REWARD = 0.3           # Per (-) letter tried in NEW position
REPEATED_WRONG_POSITION = -0.2      # Per (-) letter tried in SAME wrong position
DEAD_LETTER_PENALTY = -0.4          # Per dead (x) letter reused
MISSING_GOOD_LETTER_PENALTY = -0.3  # Per (-) letter NOT used in guess
EXPLORATION_BONUS = 0.05            # Per new letter tried

# Maximum possible rewards for scaling
# Perfect guess with 5 correct positions: 5 * 0.4 = 2.0
# Format + valid word: 1.0 + 0.5 = 1.5
# Info gain: 0.0 - 1.0
# Total max realistic: ~4.5


def output_format_check(prompt: str, completion: str, example: dict, training_progress: float = 0.0) -> float:
    """
    IMPROVED: More lenient format checking with partial credit

    Rewards:
    - Perfect format (<think></think><guess>WORD</guess>): +1.5
    - Has <guess> tag with valid word: +0.8
    - Has <guess> tag but invalid: +0.3 to -0.5
    - No <guess> tag: 0.0
    """

    reward = 0.0

    try:
        # Add synthetic <think> as it's already in the prompt
        completion = "<think>" + completion

        # IMPROVED: More lenient regex - just look for <guess> tag
        # This accepts various formats and gives partial credit
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)

        if not guess_match:
            # No <guess> tag found at all
            return 0.0

        # Extract the guess (strip extra text like "guessed-word:")
        guess_text = guess_match.group(1).strip()
        # Remove common prefixes
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

        # Check if format is complete (has </think> before <guess>)
        has_proper_format = bool(re.search(r"</think>.*?<guess>", completion, re.DOTALL))

        if has_proper_format:
            reward += VALID_FORMAT_REWARD  # +1.0 for proper structure
        else:
            reward += PARTIAL_FORMAT_REWARD  # +0.3 for trying

        # Validate word length
        if len(guess) != 5:
            # Scale penalty by training progress (lenient early, harsh late)
            penalty = INVALID_LENGTH_PENALTY * (0.5 + 0.5 * training_progress)
            return reward + penalty

        # Check if it's a valid word
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess in word_list_upper:
            reward += VALID_WORD_BONUS  # +0.5 bonus for valid word
        else:
            # Scale penalty by training progress
            penalty = INVALID_WORD_PENALTY * (0.5 + 0.5 * training_progress)
            reward += penalty

    except Exception as e:
        # On error, return 0 (neutral)
        return 0.0

    return reward


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    IMPROVED: Balanced feedback rewards with consistent scaling

    Encourages:
    - Keeping ✓ positions (+0.4 per letter)
    - Trying (-) letters in new positions (+0.3 per letter)
    - Using ALL (-) letters somewhere

    Penalizes:
    - Reusing dead (x) letters (-0.4 per letter)
    - Repeating wrong positions for (-) letters (-0.2 per letter)
    - NOT using (-) letters (-0.3 per missing letter)
    """

    reward = 0.0

    try:
        # Add synthetic <think>
        completion = "<think>" + completion

        # Extract guess (lenient regex)
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

        if len(guess) != 5:
            return 0.0

        # Parse past guess history
        past_guess_history = ast.literal_eval(example["past_guess_history"])

        if len(past_guess_history) == 0:
            # No past guesses - small exploration bonus
            return 0.1

        # Track letter states from feedback
        correct_positions = {}      # pos -> letter (✓)
        wrong_positions = {}         # letter -> set of wrong positions (-)
        dead_letters = set()         # letters not in word (x)

        for past_guess, past_feedback in past_guess_history:
            feedback_parts = past_feedback.split(" ")
            if len(feedback_parts) != 5:
                continue

            for i, fb in enumerate(feedback_parts):
                letter = fb[0].upper()
                if '✓' in fb:
                    correct_positions[i] = letter
                elif '-' in fb:
                    if letter not in wrong_positions:
                        wrong_positions[letter] = set()
                    wrong_positions[letter].add(i)
                elif 'x' in fb:
                    dead_letters.add(letter)

        # Evaluate each letter in the guess
        guess_letters = set(guess)

        for idx, letter in enumerate(guess):
            letter = letter.upper()

            # Check correct positions (highest priority)
            if idx in correct_positions:
                if letter == correct_positions[idx]:
                    reward += CORRECT_POSITION_REWARD  # +0.4
                else:
                    # Wrong letter in confirmed position (bad!)
                    reward -= 0.5

            # Check if letter is marked as wrong position (-)
            elif letter in wrong_positions:
                if idx not in wrong_positions[letter]:
                    # Good! Trying letter in NEW position
                    reward += NEW_POSITION_REWARD  # +0.3
                else:
                    # Bad! Repeating same wrong position
                    reward += REPEATED_WRONG_POSITION  # -0.2

            # Check if letter is dead (x)
            elif letter in dead_letters:
                reward += DEAD_LETTER_PENALTY  # -0.4

            else:
                # New letter being explored
                reward += EXPLORATION_BONUS  # +0.05

        # Penalize for NOT using letters marked as (-)
        # These letters are CONFIRMED in the word but wrong position
        for letter in wrong_positions.keys():
            if letter not in guess_letters:
                reward += MISSING_GOOD_LETTER_PENALTY  # -0.3

    except Exception:
        return 0.0

    return reward


def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    IMPROVED: Same information gain logic, but with better error handling
    """

    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret.upper())
        guess = guess.upper()

        # Check for correct positions
        for i, (g_char, s_char) in enumerate(zip(guess, secret_list)):
            if g_char == s_char:
                feedback.append(f"{g_char}(✓) ")
                secret_list[i] = None
            else:
                feedback.append(None)

        # Check for misplaced letters
        for i, g_char in enumerate(guess):
            if feedback[i] is None:
                if g_char in secret_list:
                    feedback[i] = f"{g_char}(-) "
                    secret_list[secret_list.index(g_char)] = None
                else:
                    feedback[i] = f"{g_char}(x) "

        if raw_feedback:
            return feedback
        return "".join(feedback).strip()

    def filter_candidates(all_candidate_words, past_guesses):
        filtered = []
        for word in all_candidate_words:
            word = word.upper()
            valid = True
            for past_guess, past_feedback in past_guesses:
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)

        if total_candidates == 0:
            return 0.0, 0.0

        current_entropy = math.log2(total_candidates)

        feedback_groups = {}
        for word in candidates:
            feedback = validate_guess(word, guess, raw_feedback=True)
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x')
                                    for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)

        expected_entropy = 0
        max_info_gain = 0

        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
            info_gain = current_entropy - group_entropy
            max_info_gain = max(max_info_gain, info_gain)

        expected_gain = current_entropy - expected_entropy
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0

        return normalized_expected_gain, normalized_max_gain

    reward = 0.0

    try:
        completion = "<think>" + completion

        # IMPROVED: More lenient extraction
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

        if len(guess) != 5:
            return 0.0

        # Load word list
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess not in word_list_upper:
            return 0.0

        # Extract past guesses
        past_guess_history = ast.literal_eval(example["past_guess_history"])

        # Compute information gain
        normalized_expected_gain, _ = compute_normalized_information_gain(
            word_list["Word"].values,
            past_guess_history,
            guess
        )

        # Scale to 0-1 range (it's already normalized)
        reward = normalized_expected_gain

    except Exception:
        return 0.0

    return reward
