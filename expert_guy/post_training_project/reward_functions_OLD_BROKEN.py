# This file is adapted from the course:
# https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo
#
# Modifications and additional logic may have been applied for this project.
#
# PATCHED VERSION - Fixes for hallucination, dead-letter reuse, and invalid guesses

import math
import re
import ast
import pandas as pd

from logger_setup import logger

# ===== REWARD/PENALTY CONSTANTS =====
# ROUND 3: Reward-focused approach - incentivize correct behavior
# Philosophy: Reward good choices heavily, penalize bad choices moderately
CORRECT_POSITION_REWARD = 2.0    # DOUBLED: Strong reward for keeping ✓ positions
NEW_POSITION_REWARD = 1.0        # DOUBLED: Strong reward for trying (-) letters at NEW positions
REPEATED_POSITION_PENALTY = -0.5 # Moderate penalty for repeating wrong position
WRONG_LETTER_PENALTY = -0.5      # REDUCED: Light penalty for dead letters (was -1.0)
EXPLORATION_REWARD = 0.1         # DOUBLED: Encourage exploring new letters
MISSING_VALID_LETTER_PENALTY = -0.5  # REDUCED: Light penalty for missing (-) letters (was -1.0)
# Goal: Make correct actions (+2.0, +1.0) much more attractive than incorrect (-0.5)

# FIX: New staged penalty system for invalid guesses
INVALID_LENGTH_PENALTY_EARLY = -0.1  # Lenient early in training
INVALID_LENGTH_PENALTY_LATE = -2.0   # Harsh late in training
INVALID_WORD_PENALTY_EARLY = -0.5    # Lenient early in training
INVALID_WORD_PENALTY_LATE = -3.0     # Harsh late in training


def output_format_check(prompt: str, completion: str, example: dict, training_progress: float = 0.0) -> float:
    """
    Checks if the completion output is in the correct format and if the guess is a valid word.
    Returns a reward score based on format and validity.

    FIX: Added staged penalty system based on training_progress (0.0 = start, 1.0 = end)
    FIX: Stricter validation to prevent 4-letter guesses and non-words

    Args:
        prompt: The input prompt
        completion: Model's completion
        example: Dict containing word_list, past_guess_history, etc.
        training_progress: Float 0.0-1.0 indicating how far along training is
    """

    reward = 0
    try:
        logger.info('Running output_format_check')

        # Add synthetic <think> as it's already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Adjust the regex to capture only the guessed word within <guess> tags
        regex = r"<think>.*?<\/think>\s*<guess>\s*(\w+)\s*<\/guess>"

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is None:
            logger.warning(f'output_format_check: Regex did not match. Completion: {completion}')
            return 0.0

        guess = match.group(1).strip().upper()  # FIX: Convert to uppercase for consistency
        logger.info(f"output_format_check: Guess = {guess}")

        # FIX: Staged penalty for wrong length (lenient early, harsh late)
        if len(guess) != 5:
            penalty = INVALID_LENGTH_PENALTY_EARLY + (INVALID_LENGTH_PENALTY_LATE - INVALID_LENGTH_PENALTY_EARLY) * training_progress
            logger.info(f'output_format_check: Guess length not 5: {guess}. Penalty: {penalty}. Completion: {completion}')
            return penalty  # Return negative penalty instead of 0.1

        # Check if the guess is a valid word compared to a predefined list of words
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values  # FIX: Case-insensitive comparison

        if guess not in word_list_upper:
            # FIX: Staged penalty for invalid words (lenient early, harsh late)
            penalty = INVALID_WORD_PENALTY_EARLY + (INVALID_WORD_PENALTY_LATE - INVALID_WORD_PENALTY_EARLY) * training_progress
            logger.info(f'output_format_check: Guess not in word list: {guess}. Penalty: {penalty}. Completion: {completion}')
            return penalty  # Return negative penalty instead of 0.5

        # If everything is correct, return a reward of 1.0
        reward = 1.0
        logger.info(f'output_format_check: Success, guess={guess}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in output_format_check: {e}")
        return 0.0

    return reward


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    Rewards guesses that make good use of previous feedback.
    Returns a cumulative reward based on letter positions and exploration.

    FIX: Strengthened dead-letter penalty with cumulative punishment
    FIX: Added explicit tracking of dead letters for better enforcement
    """

    reward = 0
    try:
        logger.info('Running uses_previous_feedback')

        # Add synthetic <think> as it's already part of the prompt and prefilled
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\w]+)\s*<\/guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.warning(f'uses_previous_feedback: Regex did not match or wrong group count. Completion: {completion}')
            return 0.0

        guess = match.groups()[0].strip().upper()  # FIX: Convert to uppercase
        if len(guess) != 5:
            logger.info(f'uses_previous_feedback: Guess length not 5: {guess}. Completion: {completion}')
            return 0.0

        past_guess_history = ast.literal_eval(example["past_guess_history"])
        if len(past_guess_history) == 0:
            logger.info('uses_previous_feedback: No past guesses')
            return 0.1

        correct_letter_to_position = {}
        valid_letter_to_position = {}
        wrong_letter_to_position = {}

        # FIX: Track dead letters explicitly (letters marked with 'x')
        dead_letters = set()

        for past_guess, past_feedback in past_guess_history:
            past_feedback_parts = past_feedback.split(" ")
            if len(past_feedback_parts) != 5:
                logger.warning(f"Invalid feedback format for guess: {past_guess}")
                continue
            for i, fb in enumerate(past_feedback_parts):
                letter = fb[0].upper()  # FIX: Case-insensitive handling
                if '✓' in fb:
                    if letter not in correct_letter_to_position:
                        correct_letter_to_position[letter] = set()
                    correct_letter_to_position[letter].add(i)
                elif '-' in fb:
                    if letter not in valid_letter_to_position:
                        valid_letter_to_position[letter] = set()
                    valid_letter_to_position[letter].add(i)
                elif 'x' in fb:
                    # FIX: Track dead letters (not in word at all)
                    dead_letters.add(letter)
                    if letter not in wrong_letter_to_position:
                        wrong_letter_to_position[letter] = set()
                    wrong_letter_to_position[letter].add(i)

        # Track dead letter reuse (but don't escalate penalty)
        dead_letter_count = 0
        guess_letters = set(guess.upper())

        for idx, letter in enumerate(guess):
            letter = letter.upper()
            if letter in correct_letter_to_position and idx in correct_letter_to_position[letter]:
                reward += CORRECT_POSITION_REWARD
            elif letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]:
                reward += NEW_POSITION_REWARD
            elif letter in valid_letter_to_position and idx in valid_letter_to_position[letter]:
                reward += REPEATED_POSITION_PENALTY
            elif letter in dead_letters:
                # ROUND 3: Flat penalty per dead letter (no escalation)
                dead_letter_count += 1
                reward += WRONG_LETTER_PENALTY  # Flat -1.0 per dead letter
                logger.warning(f'uses_previous_feedback: Dead letter reused: {letter} (count: {dead_letter_count})')
            else:
                reward += EXPLORATION_REWARD

        # NEW: Penalize for NOT using valid_letter_to_position letters (the (-) ones)
        # This teaches: "If you see O(-), you MUST use O somewhere!"
        for letter in valid_letter_to_position.keys():
            if letter not in guess_letters:
                reward += MISSING_VALID_LETTER_PENALTY
                logger.warning(f'uses_previous_feedback: Missing valid letter: {letter} (marked as wrong position but not used in guess)')

        logger.info(f'uses_previous_feedback: guess={guess}, dead_letters_used={dead_letter_count}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in uses_previous_feedback: {e}")
        return 0.0

    return reward



# Reward function that computes normalized information gain of the guess, i.e.,
# does the new guess reduce the uncertainty of the secret word the most
def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    Measures how much the guess reduces uncertainty about the secret word (information gain).
    Returns the normalized expected information gain as the reward.

    NO MAJOR CHANGES - This function is already well-designed
    """


    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret.upper())  # FIX: Case-insensitive
        guess = guess.upper()  # FIX: Case-insensitive

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
            word = word.upper()  # FIX: Case-insensitive
            valid = True
            for past_guess, past_feedback in past_guesses:
                # Compute what the feedback would be if 'word' were the secret.
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        # First, filter the candidate words based on past guesses.
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)

        # If no candidates remain, return zeros.
        if total_candidates == 0:
            return 0.0, 0.0

        # Current uncertainty (entropy) before the guess.
        current_entropy = math.log2(total_candidates)

        # Partition candidates by the feedback pattern that would be produced by the current guess.
        feedback_groups = {}
        for word in candidates:
            # Get the raw feedback list (e.g., ['B(✓) ', 'R(✓) ', 'A(x) ', ...])
            feedback = validate_guess(word, guess, raw_feedback=True)
            # Create a simple representation for the feedback pattern.
            # '1' for correct position, '0' for wrong position, 'x' for letter not in word.
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x')
                                    for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)

        expected_entropy = 0
        max_info_gain = 0
        # For each feedback group, compute its contribution to the expected entropy and the info gain.
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            # Entropy if this feedback is received.
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
            # Information gain for this feedback outcome.
            info_gain = current_entropy - group_entropy
            max_info_gain = max(max_info_gain, info_gain)

        # The expected gain is the reduction in entropy on average.
        expected_gain = current_entropy - expected_entropy

        # Normalize by the maximum possible gain, which is current_entropy (if you reduced to one candidate).
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0

        return normalized_expected_gain, normalized_max_gain

    reward = 0
    try:
        logger.info('Running guess_value')
        # Add synthetic <think> as it's already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*(\w+)\s*<\/guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.warning(f'guess_value: Regex did not match or wrong group count. Completion: {completion}')
            return 0.0

        guess = match.groups()[0].strip().upper()  # FIX: Case-insensitive
        if len(guess) != 5:
            logger.info(f'guess_value: Guess length not 5: {guess}. Completion: {completion}')
            return 0.0

        # Load the word list
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values  # FIX: Case-insensitive

        if guess not in word_list_upper:
            logger.info(f'guess_value: Guess not in word list: {guess}. Completion: {completion}')
            return 0.0

        # Extract past guesses and feedback
        past_guess_history = ast.literal_eval(example["past_guess_history"])

        # Compute normalized information gain
        normalized_expected_gain, _ = compute_normalized_information_gain(
            word_list["Word"].values,
            past_guess_history,
            guess
        )

        # Compute reward based on normalized information gain
        reward = normalized_expected_gain
        logger.info(f'guess_value: guess={guess}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in guess_value: {e}")
        return 0.0

    return reward
