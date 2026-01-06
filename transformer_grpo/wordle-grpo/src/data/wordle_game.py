"""
Wordle game utilities for validation and feedback.

This module contains core Wordle game logic including guess validation,
feedback generation, and helper functions for reward computation.
Logic adapted from the course's reward_functions.py.
"""

import logging
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
    """
    Validate a guess against the secret word and return feedback.

    This function implements the standard Wordle feedback mechanism:
    - Correct letter in correct position: marked with ✓
    - Correct letter in wrong position: marked with -
    - Letter not in word: marked with x

    Logic copied from reward_functions.py validate_guess function.

    Args:
        secret: The secret word to guess.
        guess: The guessed word.
        raw_feedback: If True, return list of feedback strings instead of joined string.

    Returns:
        Feedback string in format "A(✓) B(-) C(x) D(✓) E(x)" or list if raw_feedback=True.

    Example:
        >>> validate_guess("CRANE", "TRAIN")
        "T(x) R(✓) A(✓) I(x) N(✓)"
    """
    feedback = []
    secret_list = list(secret)

    # First pass: Check for correct positions
    for i, (g_char, s_char) in enumerate(zip(guess, secret)):
        if g_char == s_char:
            feedback.append(f"{g_char}(✓) ")
            secret_list[i] = None  # Mark as used
        else:
            feedback.append(None)  # Placeholder for second pass

    # Second pass: Check for misplaced letters
    for i, g_char in enumerate(guess):
        if feedback[i] is None:
            if g_char in secret_list:
                feedback[i] = f"{g_char}(-) "
                secret_list[secret_list.index(g_char)] = None  # Mark as used
            else:
                feedback[i] = f"{g_char}(x) "

    if raw_feedback:
        return feedback
    return "".join(feedback).strip()


def format_feedback(feedback: str) -> str:
    """
    Format feedback string with visual symbols for display.

    Args:
        feedback: Raw feedback string like "A(✓) B(-) C(x) D(✓) E(x)".

    Returns:
        Formatted string for display.

    Example:
        >>> format_feedback("C(✓) R(✓) A(x) N(-) E(x)")
        "C✓ R✓ A✗ N⚠ E✗"
    """
    if not feedback:
        return ""

    # Replace symbols for better readability
    formatted = feedback.replace("(✓)", "✓")
    formatted = formatted.replace("(-)", "⚠")
    formatted = formatted.replace("(x)", "✗")
    formatted = formatted.replace(" ", " ")

    return formatted


def is_valid_word(word: str, word_list: List[str]) -> bool:
    """
    Check if a word is valid (5 letters and in word list).

    Args:
        word: The word to validate.
        word_list: List of valid Wordle words.

    Returns:
        True if word is valid, False otherwise.
    """
    if not word or not isinstance(word, str):
        return False

    word = word.strip().upper()

    # Check length
    if len(word) != 5:
        return False

    # Check if alphabetic
    if not word.isalpha():
        return False

    # Check if in word list
    if word not in word_list:
        return False

    return True


def parse_feedback(feedback: str) -> List[Tuple[str, str]]:
    """
    Parse feedback string into list of (letter, status) tuples.

    Args:
        feedback: Feedback string like "A(✓) B(-) C(x) D(✓) E(x)".

    Returns:
        List of tuples: [('A', '✓'), ('B', '-'), ('C', 'x'), ('D', '✓'), ('E', 'x')]
    """
    parts = feedback.split()
    parsed = []

    for part in parts:
        if '(' in part and ')' in part:
            letter = part[0]
            status = part[part.index('(') + 1:part.index(')')]
            parsed.append((letter, status))

    return parsed


def extract_constraints_from_history(
    past_guess_history: List[Tuple[str, str]]
) -> Tuple[dict, dict, set]:
    """
    Extract letter position constraints from past guess history.

    This is used by reward functions to check if a new guess respects previous feedback.

    Args:
        past_guess_history: List of (guess, feedback) tuples from previous guesses.

    Returns:
        Tuple of:
        - correct_letter_to_position: Dict mapping letters to positions where they're correct
        - valid_letter_to_position: Dict mapping letters to positions where they appeared but wrong
        - wrong_letters: Set of letters that are not in the word
    """
    correct_letter_to_position = {}
    valid_letter_to_position = {}
    wrong_letters = set()

    for guess, past_feedback in past_guess_history:
        past_feedback_parts = past_feedback.split(" ")

        for i, fb in enumerate(past_feedback_parts):
            if not fb or '(' not in fb:
                continue

            letter = fb[0]

            if '✓' in fb:
                # Letter in correct position
                if letter not in correct_letter_to_position:
                    correct_letter_to_position[letter] = set()
                correct_letter_to_position[letter].add(i)
            elif '-' in fb:
                # Letter in word but wrong position
                if letter not in valid_letter_to_position:
                    valid_letter_to_position[letter] = set()
                valid_letter_to_position[letter].add(i)
            else:  # 'x' in fb
                # Letter not in word
                wrong_letters.add(letter)

    return correct_letter_to_position, valid_letter_to_position, wrong_letters


def filter_candidates(
    all_candidate_words: List[str],
    past_guesses: List[Tuple[str, str]]
) -> List[str]:
    """
    Filter candidate words based on past guesses and their feedback.

    This function is used for information gain calculations in reward functions.
    Copied from reward_functions.py.

    Args:
        all_candidate_words: List of all possible words.
        past_guesses: List of (guess, feedback) tuples.

    Returns:
        List of words that are still possible given the feedback.
    """
    filtered = []

    for word in all_candidate_words:
        valid = True
        for past_guess, past_feedback in past_guesses:
            # Compute what the feedback would be if 'word' were the secret
            candidate_feedback = validate_guess(word, past_guess)
            if candidate_feedback != past_feedback:
                valid = False
                break
        if valid:
            filtered.append(word)

    return filtered


def is_winning_guess(feedback: str) -> bool:
    """
    Check if the feedback indicates a winning guess (all correct).

    Args:
        feedback: Feedback string.

    Returns:
        True if all letters are marked with ✓, False otherwise.
    """
    parsed = parse_feedback(feedback)
    return len(parsed) == 5 and all(status == '✓' for _, status in parsed)


def get_letter_info(feedback: str, position: int) -> Optional[Tuple[str, str]]:
    """
    Get information about a specific position in the feedback.

    Args:
        feedback: Feedback string.
        position: Position to query (0-4).

    Returns:
        Tuple of (letter, status) or None if position invalid.
    """
    parsed = parse_feedback(feedback)
    if 0 <= position < len(parsed):
        return parsed[position]
    return None


def count_feedback_type(feedback: str, feedback_type: str) -> int:
    """
    Count how many letters have a specific feedback type.

    Args:
        feedback: Feedback string.
        feedback_type: Type to count ('✓', '-', or 'x').

    Returns:
        Count of letters with that feedback type.
    """
    parsed = parse_feedback(feedback)
    return sum(1 for _, status in parsed if status == feedback_type)


def format_guess_history(guess_history: List[Tuple[str, str]]) -> str:
    """
    Format guess history for display.

    Args:
        guess_history: List of (guess, feedback) tuples.

    Returns:
        Formatted string showing the guess history.
    """
    if not guess_history:
        return "No previous guesses"

    lines = []
    for i, (guess, feedback) in enumerate(guess_history, 1):
        formatted = format_feedback(feedback)
        lines.append(f"Guess {i}: {guess} -> {formatted}")

    return "\n".join(lines)
