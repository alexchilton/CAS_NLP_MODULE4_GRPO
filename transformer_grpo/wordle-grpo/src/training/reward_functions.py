"""
Reward functions for Wordle GRPO training.

This module contains reward functions adapted from the course's reward_functions.py.
Each function evaluates the quality of a model's guess based on different criteria:
- Output format and validity
- Use of previous feedback
- Information gain

These functions are used during GRPO training to provide learning signals.
"""

import ast
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from data.wordle_game import validate_guess, filter_candidates

logger = logging.getLogger(__name__)


def output_format_check(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    word_list_path: Optional[Union[str, Path]] = None
) -> float:
    """
    Check if the model's output follows the expected format and is a valid guess.

    This reward function evaluates:
    1. Format correctness: <think>...</think> followed by <guess>...</guess>
    2. Guess length: Must be exactly 5 characters
    3. Word validity: Must be in the predefined word list

    Reward scale:
    - 1.0: Perfect format and valid word
    - 0.5: Valid format and length, but word not in list
    - 0.1: Valid format but wrong length
    - 0.0: Invalid format

    Args:
        prompt: The input prompt given to the model.
        completion: The model's completion (response).
        example: Dictionary containing example data, must include 'word_list' key if word_list_path is None.
        word_list_path: Optional path to word list CSV. If None, uses example['word_list'].

    Returns:
        Reward score between 0.0 and 1.0.
    """
    reward = 0.0

    try:
        # Add synthetic <think> tag as it's already part of the prompt and prefilled
        completion = "<think>" + completion

        # Check if the format matches expected pattern:
        # <think> content </think> followed by <guess> content </guess>
        regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
            r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        )

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            logger.debug("Output format check failed: Invalid format, trying graduated fallback...")

            # GRADUATED REWARD SHAPING: Provide incremental rewards for partial progress
            partial_reward = 0.0

            # Check for closing </think> tag (+0.1)
            if '</think>' in completion:
                partial_reward += 0.1
                logger.debug("Found </think> tag: +0.1")

            # Check for <guess> opening tag (+0.1)
            if '<guess>' in completion:
                partial_reward += 0.1
                logger.debug("Found <guess> tag: +0.1")

                # Check for closing </guess> tag (+0.1)
                if '</guess>' in completion:
                    partial_reward += 0.1
                    logger.debug("Found </guess> tag: +0.1")

                    # Extract what's between <guess> and </guess>
                    guess_match = re.search(r'<guess>\s*(.*?)\s*</guess>', completion, re.DOTALL)
                    if guess_match:
                        guess_content = guess_match.group(1).strip()

                        # Check if it's exactly 5 characters (+0.2)
                        if len(guess_content) == 5:
                            partial_reward += 0.2
                            logger.debug(f"Guess is 5 letters: +0.2")

                            # Check if it's a valid word (+0.2)
                            word_list = _load_word_list(example, word_list_path)
                            if word_list is not None and guess_content.upper() in word_list:
                                partial_reward += 0.2
                                logger.debug(f"Valid word '{guess_content}': +0.2")

            logger.debug(f"Partial format reward: {partial_reward:.3f}")
            return partial_reward

        guess = match.groups()[1].strip()

        # If the word is not 5 characters, return partial credit
        if len(guess) != 5:
            logger.debug(f"Output format check: Guess '{guess}' has wrong length ({len(guess)})")
            return 0.1

        # Load word list
        word_list = _load_word_list(example, word_list_path)
        if word_list is None:
            logger.warning("Word list not available, returning partial reward")
            return 0.5

        # Check if the guess is a valid word
        if guess.upper() not in word_list:
            logger.debug(f"Output format check: Guess '{guess}' not in word list")
            return 0.5

        reward = 1.0
        logger.debug(f"Output format check passed: '{guess}' is valid")

    except Exception as e:
        logger.error(f"Error in output_format_check: {e}", exc_info=True)
        # REWARD SHAPING: Even on error, try to find any valid word
        try:
            word_list = _load_word_list(example, word_list_path)
            if word_list is not None:
                five_letter_words = re.findall(r'\b[A-Z]{5}\b', completion.upper())
                for word in five_letter_words:
                    if word in word_list:
                        logger.debug(f"Output format check: Found valid word '{word}' on error fallback")
                        return 0.2  # Small credit even with errors
        except:
            pass
        return 0.0

    return reward


def uses_previous_feedback(
    prompt: str,
    completion: str,
    example: Dict[str, Any]
) -> float:
    """
    Evaluate how well the guess uses information from previous feedback.

    This reward function checks if the model's guess respects the constraints
    learned from previous guesses:
    - Reuses letters confirmed in correct positions (+0.2 per letter)
    - Uses letters known to be in word but in new positions (+0.1 per letter)
    - Avoids letters known to be absent (+0.05 per new letter)
    - Penalizes reusing letters in same wrong position (-0.2 per letter)
    - Penalizes using letters known to be absent (-0.5 per letter)

    Args:
        prompt: The input prompt given to the model.
        completion: The model's completion (response).
        example: Dictionary containing 'past_guess_history' key with list of (guess, feedback) tuples.

    Returns:
        Reward score (can be negative if guess violates many constraints).
    """
    reward = 0.0

    try:
        # Add synthetic <think> tag
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.debug("Uses previous feedback: Failed to extract guess")
            return 0.0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            logger.debug(f"Uses previous feedback: Invalid guess length ({len(guess)})")
            return 0.0

        # Parse past guess history
        past_guess_history = _parse_guess_history(example.get("past_guess_history", []))

        if len(past_guess_history) == 0:
            logger.debug("Uses previous feedback: No past guesses, returning base reward")
            return 0.1

        # Extract constraints from history
        correct_letter_to_position = {}
        valid_letter_to_position = {}
        wrong_letter_to_position = {}

        for _, past_feedback in past_guess_history:
            past_feedback_parts = past_feedback.split(" ")
            for i, fb in enumerate(past_feedback_parts):
                if not fb or '(' not in fb:
                    continue

                letter = fb[0]

                if '✓' in fb:
                    if letter not in correct_letter_to_position:
                        correct_letter_to_position[letter] = set()
                    correct_letter_to_position[letter].add(i)
                elif '-' in fb:
                    if letter not in valid_letter_to_position:
                        valid_letter_to_position[letter] = set()
                    valid_letter_to_position[letter].add(i)
                else:  # 'x' in fb
                    if letter not in wrong_letter_to_position:
                        wrong_letter_to_position[letter] = set()
                    wrong_letter_to_position[letter].add(i)

        # Evaluate the guess against constraints
        for idx, letter in enumerate(guess):
            # Positive reward if guess reuses letter in confirmed correct position
            if (letter in correct_letter_to_position and idx in correct_letter_to_position[letter]):
                reward += 0.2
            # Reward if letter known to be in word is used in a new position
            elif (letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]):
                reward += 0.1
            # Penalize reuse of known-in-word letter in same position (not exploring)
            elif (letter in valid_letter_to_position and idx in valid_letter_to_position[letter]):
                reward -= 0.2
            # Penalize use of known-absent letter
            elif letter in wrong_letter_to_position:
                reward -= 0.5
            else:
                # Reward unknown letters with partial credit for exploration
                reward += 0.05

        logger.debug(f"Uses previous feedback reward: {reward:.3f} for guess '{guess}'")

    except Exception as e:
        logger.error(f"Error in uses_previous_feedback: {e}", exc_info=True)
        return 0.0

    return reward


def guess_value(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    word_list_path: Optional[Union[str, Path]] = None
) -> float:
    """
    Compute normalized information gain of the guess.

    This reward function evaluates how much uncertainty the guess reduces
    about the secret word. It uses information theory (entropy) to measure
    how effectively the guess narrows down the possible candidates.

    The reward is the normalized expected information gain:
    - 1.0: Maximum possible information gain (perfectly reduces candidates)
    - 0.0-1.0: Proportional to how much the guess reduces uncertainty
    - 0.0: No information gain or invalid guess

    This encourages the model to make strategic guesses that eliminate
    many possibilities, similar to optimal Wordle strategy.

    Args:
        prompt: The input prompt given to the model.
        completion: The model's completion (response).
        example: Dictionary containing 'past_guess_history' and optionally 'word_list'.
        word_list_path: Optional path to word list CSV.

    Returns:
        Normalized information gain score between 0.0 and 1.0.
    """
    reward = 0.0

    try:
        # Add synthetic <think> tag
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.debug("Guess value: Failed to extract guess")
            return 0.0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            logger.debug(f"Guess value: Invalid guess length ({len(guess)})")
            return 0.0

        # Load the word list
        word_list = _load_word_list(example, word_list_path)
        if word_list is None:
            logger.warning("Guess value: Word list not available")
            return 0.0

        if guess.upper() not in word_list:
            logger.debug(f"Guess value: '{guess}' not in word list")
            return 0.0

        # Extract past guesses and feedback
        past_guess_history = _parse_guess_history(example.get("past_guess_history", []))

        # Compute normalized information gain
        normalized_expected_gain, _ = _compute_normalized_information_gain(
            word_list,
            past_guess_history,
            guess.upper()
        )

        reward = normalized_expected_gain
        logger.debug(f"Guess value reward: {reward:.3f} for guess '{guess}'")

    except Exception as e:
        logger.error(f"Error in guess_value: {e}", exc_info=True)
        return 0.0

    return reward


# Helper functions


def _load_word_list(
    example: Dict[str, Any],
    word_list_path: Optional[Union[str, Path]] = None
) -> Optional[List[str]]:
    """
    Load word list from file or example data.

    Args:
        example: Example dictionary that may contain 'word_list' key.
        word_list_path: Optional path to word list CSV file.

    Returns:
        List of words (uppercase) or None if loading fails.
    """
    try:
        # Try to use provided path first
        if word_list_path is not None:
            path = Path(word_list_path)
            if path.exists():
                df = pd.read_csv(path)
                # Try common column names
                for col in ["Word", "word", "words", "WORD"]:
                    if col in df.columns:
                        return [str(w).upper().strip() for w in df[col].values]
                # If no named column, use first column
                return [str(w).upper().strip() for w in df.iloc[:, 0].values]

        # Try to use word_list from example
        if "word_list" in example:
            word_list_ref = example["word_list"]

            # If it's already a list
            if isinstance(word_list_ref, list):
                return [str(w).upper().strip() for w in word_list_ref]

            # If it's a path string
            path = Path(str(word_list_ref))
            if path.exists():
                df = pd.read_csv(path)
                for col in ["Word", "word", "words", "WORD"]:
                    if col in df.columns:
                        return [str(w).upper().strip() for w in df[col].values]
                return [str(w).upper().strip() for w in df.iloc[:, 0].values]

        logger.warning("Could not load word list from any source")
        return None

    except Exception as e:
        logger.error(f"Error loading word list: {e}")
        return None


def _parse_guess_history(
    past_guess_history: Union[str, List[Tuple[str, str]]]
) -> List[Tuple[str, str]]:
    """
    Parse past guess history into list of tuples.

    Args:
        past_guess_history: Either a string representation or already parsed list.

    Returns:
        List of (guess, feedback) tuples.
    """
    if isinstance(past_guess_history, str):
        try:
            return ast.literal_eval(past_guess_history)
        except Exception as e:
            logger.error(f"Error parsing guess history: {e}")
            return []
    elif isinstance(past_guess_history, list):
        return past_guess_history
    else:
        return []


def _compute_normalized_information_gain(
    all_candidate_words: List[str],
    past_guesses: List[Tuple[str, str]],
    guess: str
) -> Tuple[float, float]:
    """
    Compute normalized information gain for a guess.

    This function calculates how much the guess reduces uncertainty about
    the secret word, using entropy-based information theory.

    Args:
        all_candidate_words: List of all possible words.
        past_guesses: List of (guess, feedback) tuples from previous attempts.
        guess: The current guess to evaluate.

    Returns:
        Tuple of (normalized_expected_gain, normalized_max_gain).
    """
    # Filter candidates based on past guesses
    candidates = filter_candidates(all_candidate_words, past_guesses)
    total_candidates = len(candidates)

    # If no candidates remain, return zeros
    if total_candidates == 0:
        return 0.0, 0.0

    # Current uncertainty (entropy) before the guess
    current_entropy = math.log2(total_candidates)

    # Partition candidates by the feedback pattern that would be produced by the current guess
    feedback_groups = {}
    for word in candidates:
        # Get the raw feedback list
        feedback = validate_guess(word, guess, raw_feedback=True)
        # Create a simple representation for the feedback pattern
        # '1' for correct position, '0' for wrong position, 'x' for letter not in word
        feedback_pattern = "".join(
            '1' if "✓" in fb else ('0' if "-" in fb else 'x')
            for fb in feedback
        )
        feedback_groups.setdefault(feedback_pattern, []).append(word)

    expected_entropy = 0.0
    max_info_gain = 0.0

    # For each feedback group, compute its contribution to expected entropy and info gain
    for group in feedback_groups.values():
        group_size = len(group)
        p = group_size / total_candidates
        # Entropy if this feedback is received
        group_entropy = math.log2(group_size) if group_size > 0 else 0.0
        expected_entropy += p * group_entropy
        # Information gain for this feedback outcome
        info_gain = current_entropy - group_entropy
        max_info_gain = max(max_info_gain, info_gain)

    # The expected gain is the reduction in entropy on average
    expected_gain = current_entropy - expected_entropy

    # Normalize by the maximum possible gain
    normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0.0
    normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0.0

    return normalized_expected_gain, normalized_max_gain


class CombinedReward:
    """
    Combine multiple reward functions with configurable weights.

    This class allows you to create a weighted combination of the three
    reward functions, enabling you to balance different objectives during training.

    Example:
        >>> reward_fn = CombinedReward(
        ...     format_weight=1.0,
        ...     feedback_weight=0.5,
        ...     value_weight=0.3
        ... )
        >>> score = reward_fn(prompt, completion, example)
    """

    def __init__(
        self,
        format_weight: float = 1.0,
        feedback_weight: float = 0.5,
        value_weight: float = 0.3,
        word_list_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the combined reward function.

        Args:
            format_weight: Weight for output_format_check (default: 1.0).
            feedback_weight: Weight for uses_previous_feedback (default: 0.5).
            value_weight: Weight for guess_value (default: 0.3).
            word_list_path: Optional path to word list CSV file.
        """
        self.format_weight = format_weight
        self.feedback_weight = feedback_weight
        self.value_weight = value_weight
        self.word_list_path = word_list_path

        logger.info(
            f"CombinedReward initialized with weights: "
            f"format={format_weight}, feedback={feedback_weight}, value={value_weight}"
        )

    def __call__(
        self,
        prompt: str,
        completion: str,
        example: Dict[str, Any]
    ) -> float:
        """
        Compute the combined reward score.

        Args:
            prompt: The input prompt.
            completion: The model's completion.
            example: Example data dictionary.

        Returns:
            Weighted sum of all reward components.
        """
        # Compute individual rewards
        format_reward = output_format_check(prompt, completion, example, self.word_list_path)
        feedback_reward = uses_previous_feedback(prompt, completion, example)
        value_reward = guess_value(prompt, completion, example, self.word_list_path)

        # Compute weighted sum
        total_reward = (
            self.format_weight * format_reward +
            self.feedback_weight * feedback_reward +
            self.value_weight * value_reward
        )

        logger.debug(
            f"Combined reward: {total_reward:.3f} "
            f"(format={format_reward:.3f}, feedback={feedback_reward:.3f}, value={value_reward:.3f})"
        )

        return total_reward

    def get_individual_rewards(
        self,
        prompt: str,
        completion: str,
        example: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get individual reward scores without combining them.

        Useful for logging and debugging.

        Args:
            prompt: The input prompt.
            completion: The model's completion.
            example: Example data dictionary.

        Returns:
            Dictionary with individual reward scores.
        """
        return {
            "format": output_format_check(prompt, completion, example, self.word_list_path),
            "feedback": uses_previous_feedback(prompt, completion, example),
            "value": guess_value(prompt, completion, example, self.word_list_path),
        }
