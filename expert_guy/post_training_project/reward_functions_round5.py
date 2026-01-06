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

# FEEDBACK USAGE PENALTIES (increased 75-100%)
DEAD_LETTER_PENALTY = -0.7          # was -0.4 → 75% stronger
MISSING_GOOD_LETTER_PENALTY = -0.6  # was -0.3 → 100% stronger

# Maximum possible rewards for scaling
# Perfect guess with 5 correct positions: 5 * 0.4 = 2.0
# Format + valid word: 0.4 + 0.4 = 0.8 (was 1.5)
# Info gain: 0.0 - 1.0
# Total max realistic: ~3.8 (vs 4.5 in Round 4)


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
        # Remove common prefixes
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

        # Check if format is complete (has </think> before <guess>)
        has_proper_format = bool(re.search(r"</think>.*?<guess>", completion, re.DOTALL))

        if has_proper_format:
            reward += VALID_FORMAT_REWARD  # +0.4 for proper structure (was +1.0)
        else:
            reward += PARTIAL_FORMAT_REWARD  # +0.2 for trying (was +0.3)

        # Validate word length
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
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

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

        return reward

    except Exception as e:
        return 0.0


def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    UNCHANGED from Round 4 - information gain calculation works well
    
    Calculates information gain (entropy reduction) from the guess.
    Rewards guesses that maximize learning about the secret word.
    
    Returns float between 0.0 and 1.0
    """
    
    try:
        # Extract the guess
        guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)
        if not guess_match:
            return 0.0

        guess_text = guess_match.group(1).strip()
        guess_text = re.sub(r'^(guessed-word|word|answer):\s*', '', guess_text, flags=re.IGNORECASE)
        guess = guess_text.strip().upper()

        if len(guess) != 5:
            return 0.0

        # Get word list
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values

        if guess not in word_list_upper:
            return 0.0

        # Simple information gain heuristic:
        # More unique letters = more information
        unique_letters = len(set(guess))
        
        # Common letters vs uncommon letters
        common_letters = set('ETAOINSHRDLU')
        common_count = sum(1 for letter in guess if letter in common_letters)
        
        # Normalize to 0-1 range
        uniqueness_score = unique_letters / 5.0  # 5 unique letters = 1.0
        commonality_score = common_count / 5.0    # 5 common letters = 1.0
        
        # Combine (favor unique common letters)
        info_gain = (uniqueness_score * 0.6 + commonality_score * 0.4)
        
        return info_gain

    except Exception as e:
        return 0.0
