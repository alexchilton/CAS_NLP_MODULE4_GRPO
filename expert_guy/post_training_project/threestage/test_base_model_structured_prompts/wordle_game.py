"""
Wordle Game Engine
==================

Simple Wordle game simulator compatible with structured prompts.
"""

import re
from typing import Optional, Tuple, List
from prompt_system import GuessFeedback, get_feedback


class WordleGame:
    """Manages a single Wordle game"""

    def __init__(self, secret_word: str, word_list: set):
        self.secret = secret_word.upper()
        self.word_list = set(w.upper() for w in word_list)
        self.past_feedback: List[GuessFeedback] = []
        self.max_guesses = 6

    def make_guess(self, guess: str) -> Tuple[Optional[GuessFeedback], str]:
        """
        Make a guess and return feedback.

        Args:
            guess: The guessed word

        Returns:
            (GuessFeedback | None, status_message)
            status: "win" | "loss" | "continue" | "invalid"
        """
        guess = guess.upper()

        # Validation
        if len(guess) != 5:
            return None, "Invalid: not 5 letters"

        if guess not in self.word_list:
            return None, "Invalid: not in word list"

        if any(fb.guess == guess for fb in self.past_feedback):
            return None, "Invalid: already guessed"

        # Generate feedback
        feedback = get_feedback(guess, self.secret)
        self.past_feedback.append(feedback)

        # Check win/loss
        if guess == self.secret:
            return feedback, "win"
        elif len(self.past_feedback) >= self.max_guesses:
            return feedback, "loss"
        else:
            return feedback, "continue"

    def get_history(self) -> List[GuessFeedback]:
        """Get the current game history"""
        return self.past_feedback.copy()

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        if not self.past_feedback:
            return False
        if self.past_feedback[-1].guess == self.secret:
            return True
        return len(self.past_feedback) >= self.max_guesses


def extract_guess_from_completion(completion: str) -> Optional[str]:
    """
    Extract the guess from model completion.

    Looks for <guess>WORD</guess> pattern.

    Args:
        completion: Model's generated text

    Returns:
        Extracted guess (uppercase) or None if not found
    """
    # Try to find <guess>WORD</guess>
    guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)

    if not guess_match:
        return None

    guess_text = guess_match.group(1).strip()

    # Clean up common patterns
    guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)

    # Extract only letters
    letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())

    if len(letters_only) != 5:
        return None

    return letters_only


def extract_thinking(completion: str) -> str:
    """
    Extract the reasoning from <think> tags.

    Args:
        completion: Model's generated text

    Returns:
        Extracted thinking or empty string
    """
    think_match = re.search(r"<think>(.*?)</think>", completion, re.IGNORECASE | re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return ""
