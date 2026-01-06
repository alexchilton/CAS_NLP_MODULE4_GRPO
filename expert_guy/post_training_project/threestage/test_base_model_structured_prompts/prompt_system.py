"""
Structured Prompt System for Wordle
====================================

Adapted from wordle-rl-gemma to work with PyTorch/Qwen models.
Uses their prompt format with explicit state summaries.
"""

from typing import List, Tuple
from collections import Counter

SYSTEM_PROMPT = """You are an expert Wordle-solving AI. Your primary directive is to deduce the secret 5-letter English word with flawless logic and strategy. Adherence to the rules and format is critical.

### Core Principles
1.  **Deductive Reasoning:** Analyze all available clues from the "Current Knowledge" summary to logically eliminate possibilities.
2.  **Strategic Guessing:** In early turns, your goal is to reveal the most information. In later turns, your goal is to pinpoint the exact word.
3.  **Self-Correction & Rule Adherence:** Before finalizing a guess, ALWAYS double-check that it does not violate any Green, Yellow, or Gray clues. Your guess must be a valid 5-letter English word that has not been used before.

### Rules of Engagement
1.  **Clue Analysis:** The clues are provided in a structured "Current Knowledge" block.
    *   **Correct Position (Green):** Shows letters in their exact, confirmed positions. Your guess MUST match this pattern.
    *   **Wrong Position (Yellow):** Lists letters that are in the word. Your guess MUST include these letters.
    *   **Not in Word (Gray):** Lists letters that are not in the word. Your guess must NOT use any of these letters.
    *   **Words Already Guessed:** A list of words you cannot use again.

2.  **Chain of Thought:** You MUST explain your reasoning inside `<think>` tags. Detail your deductions from the clues, your strategy, and why your chosen word is the optimal choice.

3.  **Final Guess:** You MUST provide your final 5-letter English word guess inside `<guess>` tags.

---
### EXAMPLES
---

**Example 1: Optimal First Guess**

You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `_ _ _ _ _`
*   **Wrong Position (Yellow):** None
*   **Not in Word (Gray):** None
*   **Words Already Guessed:** None

<think>
This is the first guess with no prior clues. The best strategy is to use a word with common, distinct letters to maximize information gain. 'SLATE' is an excellent choice as it tests three common consonants and two common vowels.
</think>
<guess>SLATE</guess>

**Example 2: Complex Mid-Game Deduction**

You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `A _ _ _ _`
*   **Wrong Position (Yellow):** 'O', 'R', 'T', 'U'
*   **Not in Word (Gray):** B, E, I, S
*   **Words Already Guessed:** ARISE, ABOUT

<think>
From the clues, I have a strong set of constraints.
- The word must match the pattern `A _ _ _ _`.
- It must contain the letters O, R, T, and U in the remaining four slots.
- It must not contain the gray letters B, E, I, or S.
- It cannot be ARISE or ABOUT.
The only possible anagram of the yellow letters that fits the green pattern is 'AUTOR'. This word satisfies all known clues and is the only logical solution.
</think>
<guess>AUTOR</guess>

--- END OF EXAMPLES ---

You are now ready. The new puzzle begins. Take a deep breath and play!
"""


class GuessFeedback:
    """Represents a guess and its feedback"""
    def __init__(self, guess: str, feedback: str):
        self.guess = guess.upper()
        # Feedback format: "G X Y X G" (space-separated)
        # G = Green (correct position)
        # Y = Yellow (wrong position)
        # X = Gray (not in word)
        self.feedback = feedback


def get_feedback(guess: str, secret: str) -> GuessFeedback:
    """
    Generate feedback for a guess.
    Returns GuessFeedback with format: "G X Y X G"
    """
    guess = guess.upper()
    secret = secret.upper()

    feedback = [''] * 5
    secret_counts = Counter(secret)

    # First pass: mark greens
    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = 'G'
            secret_counts[guess[i]] -= 1

    # Second pass: mark yellows and grays
    for i in range(5):
        if feedback[i] == 'G':
            continue
        if guess[i] in secret and secret_counts[guess[i]] > 0:
            feedback[i] = 'Y'
            secret_counts[guess[i]] -= 1
        else:
            feedback[i] = 'X'

    return GuessFeedback(guess, ' '.join(feedback))


def format_prompt_for_model(past_feedback: List[GuessFeedback]) -> str:
    """
    Build the user prompt with current knowledge state.
    This is called EVERY TURN to dynamically rebuild the prompt.

    Args:
        past_feedback: List of previous GuessFeedback objects

    Returns:
        Formatted user prompt string
    """
    if not past_feedback:
        return "This is the first turn. Please provide your best starting word."

    # Extract state from ALL past feedback
    known_green = {}  # {position: letter}
    known_yellow = Counter()  # {letter: min_count}
    known_gray = set()

    for fb in past_feedback:
        counts_in_secret_this_turn = Counter()

        # First pass: count greens and yellows
        for i, f_char in enumerate(fb.feedback.split()):
            if f_char in ('G', 'Y'):
                counts_in_secret_this_turn[fb.guess[i]] += 1

        # Update yellow counts (track minimum required)
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)

        # Second pass: mark positions
        for i, f_char in enumerate(fb.feedback.split()):
            letter = fb.guess[i]
            if f_char == 'G':
                known_green[i] = letter
            elif f_char == 'X':
                # Only mark as gray if it wasn't green/yellow elsewhere
                if counts_in_secret_this_turn[letter] == 0:
                    known_gray.add(letter)

    # Clean up: remove greens from yellow/gray
    green_letters = set(known_green.values())
    for letter in green_letters:
        if letter in known_yellow:
            del known_yellow[letter]
        if letter in known_gray:
            known_gray.remove(letter)

    # Build the prompt
    prompt_parts = [
        "You are playing a game of Wordle. Analyze the clues and provide your next guess.",
        "**Current Knowledge:**"
    ]

    # Green pattern: "_ _ A _ _"
    green_display = ['_'] * 5
    for idx, letter in known_green.items():
        green_display[idx] = letter
    prompt_parts.append(f"*   **Correct Position (Green):** `{' '.join(green_display)}`")

    # Yellow letters
    if known_yellow:
        yellow_display = [f"'{k}' (at least {v})" if v > 1 else f"'{k}'"
                         for k, v in sorted(known_yellow.items())]
        prompt_parts.append(f"*   **Wrong Position (Yellow):** {', '.join(yellow_display)}")
    else:
        prompt_parts.append(f"*   **Wrong Position (Yellow):** None")

    # Gray letters
    if known_gray:
        gray_display = sorted(list(known_gray))
        prompt_parts.append(f"*   **Not in Word (Gray):** {', '.join(gray_display)}")
    else:
        prompt_parts.append(f"*   **Not in Word (Gray):** None")

    # Already guessed words
    past_guesses = [fb.guess for fb in past_feedback]
    prompt_parts.append(f"*   **Words Already Guessed:** {', '.join(past_guesses)}")

    prompt_parts.append("\nYour task is to find a valid 5-letter English word that fits all the clues above.")
    prompt_parts.append("Provide your reasoning within <think> tags, and then your final guess within <guess> tags.")

    return "\n".join(prompt_parts)


def build_messages(past_feedback: List[GuessFeedback]) -> List[dict]:
    """
    Build the full message list for chat models.

    Returns:
        List of message dicts: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    user_content = format_prompt_for_model(past_feedback)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
