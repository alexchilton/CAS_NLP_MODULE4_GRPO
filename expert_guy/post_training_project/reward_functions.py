"""
ROUND 5 REWARD FUNCTIONS - Moderate Balanced System

Changes from Round 4:
- Format rewards reduced 20-60% (prevent domination)
- Penalties increased 75-400% (clear strategic signals)
- Variance controlled at σ=1.19 (below 1.5 threshold)

Expected impact:
- Invalid words: +0.72 → -1.1 (FIXED)
- Dead letter reuse: +0.38 → -0.4 (FIXED)
- Format domination: 1.5 → 0.8 (REDUCED)
- 72% of bad behaviors now penalized (was 6%)

Date: 2025-12-18
Status: Ready to apply (current training still running)
"""

import math
import re
import ast
import pandas as pd

# ===== ROUND 5 MODERATE BALANCED CONSTANTS =====
# Changed from Round 4 - see ROUND5_IMPROVEMENTS.md for details

# FORMAT REWARDS (reduced by 20-60%)
VALID_FORMAT_REWARD = 0.4           # was 1.0 → 60% reduction
PARTIAL_FORMAT_REWARD = 0.2         # was 0.3 → 33% reduction  
NO_FORMAT_REWARD = 0.0              # unchanged

# VALIDITY REWARDS/PENALTIES
VALID_WORD_BONUS = 0.4              # was 0.5 → 20% reduction
INVALID_LENGTH_PENALTY = -1.5       # was -0.3 → 5x stronger (400% increase)
INVALID_WORD_PENALTY = -1.5         # was -0.5 → 3x stronger (200% increase)

# FEEDBACK USAGE REWARDS (unchanged - these work well)
CORRECT_POSITION_REWARD = 0.4       # unchanged
NEW_POSITION_REWARD = 0.3           # unchanged
REPEATED_WRONG_POSITION = -0.2      # unchanged
EXPLORATION_BONUS = 0.05            # unchanged

# FEEDBACK USAGE PENALTIES (Round 6.5 - softened to prevent over-penalization)
DEAD_LETTER_PENALTY = -0.5          # was -0.7 → rolled back (was too harsh)
MISSING_GOOD_LETTER_PENALTY = -0.4  # was -0.6 → rolled back (was too harsh)
MAX_FEEDBACK_PENALTY = -1.2         # NEW: clamp total feedback penalty per guess

# ROUND 6 ADDITIONS (Dense Reward + Info-Gain Masking)
WORD_ACCURACY_WEIGHT = 1.0          # Weight for dense reward signal

# Maximum possible rewards for scaling
# Perfect guess with 5 correct positions: 5 * 0.4 = 2.0
# Format + valid word: 0.4 + 0.4 = 0.8 (was 1.5)
# Info gain: 0.0 - 1.0
# Word accuracy (dense): 0.0 - 1.0
# Total max realistic: ~4.8 (was 3.8 in Round 5)


def output_format_check(prompt: str, completion: str, example: dict, training_progress: float = 0.0) -> float:
    """
    IMPROVED: More lenient format checking with partial credit
    
    ROUND 5 CHANGES:
    - Reduced VALID_FORMAT_REWARD: 1.0 → 0.4
    - Reduced VALID_WORD_BONUS: 0.5 → 0.4
    - Increased INVALID_LENGTH_PENALTY: -0.3 → -1.5
    - Increased INVALID_WORD_PENALTY: -0.5 → -1.5

    Rewards:
    - Perfect format (<think></think><guess>WORD</guess>): +0.8 (was +1.5)
    - Has <guess> tag with valid word: +0.6 to +0.8
    - Has <guess> tag but invalid: +0.2 to -1.3
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
        # ROUND 6.5 FIX: Better prefix removal - handle colons and spaces
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)

        # ROUND 6.5 FIX: Extract letters only (allow dashes/spaces like "S-L-A-N-T")
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        # Check if format is complete (has </think> before <guess>)
        has_proper_format = bool(re.search(r"</think>.*?<guess>", completion, re.DOTALL))

        if has_proper_format:
            reward += VALID_FORMAT_REWARD  # +0.4 for proper structure (was +1.0)
        else:
            reward += PARTIAL_FORMAT_REWARD  # +0.2 for trying (was +0.3)

        # Validate word length (after stripping non-letters)
        if len(guess) != 5:
            # Scale penalty by training progress (lenient early, harsh late)
            penalty = INVALID_LENGTH_PENALTY * (0.5 + 0.5 * training_progress)
            return reward + penalty

        # Check if it's a valid word
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess in word_list_upper:
            reward += VALID_WORD_BONUS  # +0.4 bonus for valid word (was +0.5)
        else:
            # Scale penalty by training progress
            penalty = INVALID_WORD_PENALTY * (0.5 + 0.5 * training_progress)
            reward += penalty

        return reward

    except Exception as e:
        # If any parsing fails, return 0
        return 0.0


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    UNCHANGED from Round 4 - these mechanics work well
    
    Rewards model for correctly using Wordle feedback symbols:
    - ✓ (correct position): Keep letter at that position
    - (-) (wrong position): Use letter at different position
    - (x) (not in word): Never use this letter again
    
    ROUND 5 CHANGES:
    - Increased DEAD_LETTER_PENALTY: -0.4 → -0.7
    - Increased MISSING_GOOD_LETTER_PENALTY: -0.3 → -0.6
    """
    
    reward = 0.0

    try:
        # Extract the guess from completion
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        # ROUND 6.5 FIX: Better prefix removal + extract letters only
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
            # First guess - give small exploration bonus
            reward = len(set(guess)) * EXPLORATION_BONUS
            return reward

        # Track which letters are confirmed correct (✓)
        confirmed_positions = {}  # {position: letter}
        # Track which letters are in word but wrong position (-)
        valid_letter_to_positions = {}  # {letter: [tried_positions]}
        # Track which letters are dead (x)
        dead_letters = set()

        # Parse all previous feedback
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

        # Now check the current guess
        guess_letters = list(guess)

        # 1. Check if keeping confirmed ✓ positions
        for pos, letter in confirmed_positions.items():
            if pos < len(guess_letters) and guess_letters[pos] == letter:
                reward += CORRECT_POSITION_REWARD  # +0.4 (unchanged)
            else:
                # Not keeping a confirmed position is bad
                reward += REPEATED_WRONG_POSITION  # -0.2 (unchanged)

        # 2. Check if using (-) letters at NEW positions
        for letter, tried_positions in valid_letter_to_positions.items():
            if letter in guess_letters:
                current_pos = guess_letters.index(letter)
                if current_pos not in tried_positions:
                    # Good! Trying this letter at a new position
                    reward += NEW_POSITION_REWARD  # +0.3 (unchanged)
                else:
                    # Bad! Trying same wrong position again
                    reward += REPEATED_WRONG_POSITION  # -0.2 (unchanged)
            else:
                # Bad! Not using a letter we know is in the word
                reward += MISSING_GOOD_LETTER_PENALTY  # -0.6 (was -0.3)

        # 3. Penalize using dead letters
        for letter in guess_letters:
            if letter in dead_letters:
                reward += DEAD_LETTER_PENALTY  # -0.7 (was -0.4)

        # 4. Reward exploring new letters
        new_letters = set(guess_letters) - set(confirmed_positions.values()) - set(valid_letter_to_positions.keys()) - dead_letters
        reward += len(new_letters) * EXPLORATION_BONUS  # +0.05 each (unchanged)

        # ROUND 6.5: Clamp total feedback penalty to prevent over-penalization
        # Even if guess violates 3+ constraints, cap at -1.2
        reward = max(reward, MAX_FEEDBACK_PENALTY)

        return reward

    except Exception as e:
        return 0.0


def filter_words_by_history(word_list_upper, past_history):
    """
    ROUND 6 HELPER: Filter word list to only candidates that satisfy all feedback constraints.

    Returns list of words that could still be the answer based on past feedback.
    Used by word_accuracy_reward and guess_value (info-gain masking).
    """
    if not past_history:
        return word_list_upper

    # Track constraints from feedback
    confirmed_positions = {}  # {position: letter}
    valid_letters = {}  # {letter: [wrong_positions]}
    dead_letters = set()

    # Parse all previous feedback
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

    # Filter candidates
    candidates = []
    for word in word_list_upper:
        word = word.upper() if isinstance(word, str) else word
        if len(word) != 5:
            continue

        # Check confirmed positions
        valid = True
        for pos, letter in confirmed_positions.items():
            if word[pos] != letter:
                valid = False
                break
        if not valid:
            continue

        # Check valid letters are present but not at wrong positions
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

        # Check no dead letters
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
    ROUND 6 NEW: Dense reward based on how close the guess is to the hidden word.

    This gives the model gradient toward the correct answer even when constraints
    are satisfied. Without this, "FROSN" (one letter away) gets the same 0.0 as
    random garbage, making learning extremely sparse.

    ROUND 6.5 ADDITION: ABSTAIN action for unsatisfiable constraints
    - If no word can satisfy feedback constraints, ABSTAIN gets +0.6
    - Other guesses get capped accuracy reward (max 0.1)

    Formula:
    - exact: number of letters in correct positions (0-5)
    - exist: number of letters that exist in word but wrong position
    - reward: (exact + 0.2 * exist) / 6.0

    Returns float between 0.0 and 1.0
    Maximum: 1.0 (perfect match: FROST == FROST gives 5/6 ≈ 0.83, scaled to 1.0)
    """
    try:
        # Extract the guess
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        # ROUND 6.5 FIX: Better prefix removal + extract letters only
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        # ROUND 6.5: Check for ABSTAIN action
        if guess == "ABSTAIN":
            # Check if constraints are unsatisfiable
            past_history = example.get("past_guess_history", [])
            if isinstance(past_history, str):
                try:
                    past_history = ast.literal_eval(past_history)
                except:
                    past_history = []

            # Filter candidates by constraints
            word_list = pd.read_csv(str(example["word_list"]))
            word_list_upper = word_list["Word"].str.upper().values
            candidates = filter_words_by_history(word_list_upper, past_history)

            # If no valid candidates, ABSTAIN is the right move
            if len(candidates) == 0:
                return 0.6  # Better than average accurate guess
            else:
                return 0.0  # Abstaining when solution exists is bad

        if len(guess) != 5:
            return 0.0

        # Get the hidden word
        hidden = example.get("secret_word", "").upper()
        if len(hidden) != 5:
            return 0.0

        # ROUND 6.5: Check if constraints are unsatisfiable (cap accuracy if so)
        past_history = example.get("past_guess_history", [])
        if isinstance(past_history, str):
            try:
                past_history = ast.literal_eval(past_history)
            except:
                past_history = []

        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values
        candidates = filter_words_by_history(word_list_upper, past_history)
        must_abstain = (len(candidates) == 0)

        # Count exact matches (correct position)
        exact = sum(g == h for g, h in zip(guess, hidden))

        # Count letters that exist in word (anywhere)
        exist = sum(min(guess.count(c), hidden.count(c)) for c in set(guess))

        # Dense reward formula: prioritize exact matches, give partial credit for exists
        # Max possible: 5 exact = 5/6 ≈ 0.83
        # Scale to 0-1 range
        raw_score = (exact + 0.2 * max(0, exist - exact)) / 6.0

        # Scale to proper 0-1 range (max raw_score is 5/6 ≈ 0.83)
        accuracy = min(1.0, raw_score * 1.2)

        # ROUND 6.5: Cap accuracy if constraints are unsatisfiable
        # This discourages hallucinating when no valid word exists
        if must_abstain:
            accuracy = min(accuracy, 0.1)

        return accuracy

    except Exception as e:
        return 0.0


def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    ROUND 6 UPDATED: Information gain calculation with candidate masking

    Calculates information gain (entropy reduction) from the guess.
    Rewards guesses that maximize learning about the secret word.

    ROUND 6 CHANGE: Mask info-gain with live candidate set
    - Only reward letters that can still appear (based on feedback)
    - Prevents rewarding dead letters (which would cancel penalty)

    Returns float between 0.0 and 1.0
    """

    try:
        # Extract the guess
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        # ROUND 6.5 FIX: Better prefix removal + extract letters only
        guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)
        letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())
        guess = letters_only

        if len(guess) != 5:
            return 0.0

        # Get word list
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess not in word_list_upper:
            return 0.0

        # ROUND 6: Filter candidates by past feedback
        past_history = example.get("past_guess_history", [])
        if isinstance(past_history, str):
            try:
                past_history = ast.literal_eval(past_history)
            except:
                past_history = []

        candidates = filter_words_by_history(word_list_upper, past_history)

        # Get allowed alphabet from still-possible candidates
        allowed_letters = set(''.join(candidates)) if candidates else set()

        # Simple information gain heuristic:
        # More unique letters = more information
        unique_letters = len(set(guess))

        # ROUND 6 FIX: Only count common letters that are still allowed
        common_letters = set('ETAOINSHRDLU')
        # Intersect with allowed alphabet so dead letters don't get bonus
        common_count = sum(1 for letter in guess if letter in common_letters and letter in allowed_letters)

        # Also adjust uniqueness to only count allowed letters
        unique_allowed = len(set(guess) & allowed_letters)

        # Normalize to 0-1 range
        uniqueness_score = unique_allowed / 5.0  # 5 unique allowed letters = 1.0
        commonality_score = common_count / 5.0    # 5 common allowed letters = 1.0

        # Combine (favor unique common letters)
        info_gain = (uniqueness_score * 0.6 + commonality_score * 0.4)

        return info_gain

    except Exception as e:
        return 0.0
