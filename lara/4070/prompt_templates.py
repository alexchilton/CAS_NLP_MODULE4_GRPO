"""
Prompt templates and wrappers for Wordle GRPO training.

This module provides utilities to enhance prompts with few-shot examples
and better instructions to guide model behavior.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# Few-shot examples demonstrating the correct format
FEW_SHOT_EXAMPLES = """
### Example 1: First Guess
No previous guesses yet.

<think>I'll start with a common word that has diverse vowels and consonants to maximize information gain. CRANE is a popular opening word.</think>
<guess>CRANE</guess>

### Example 2: Using Feedback
Previous guesses:
Guess 1: C(✓) R(x) A(-) N(x) E(x)

<think>C is correct in position 1, A is in the word but not position 3. R, N, E are not in the word. I need a word starting with C, containing A but not in position 3. Let me try MATCH.</think>
<guess>MATCH</guess>

### Example 3: Narrowing Down
Previous guesses:
Guess 1: S(x) T(-) A(✓) R(-) T(x)
Guess 2: B(✓) R(✓) A(✓) I(-) N(x)

<think>B and R are correct in positions 1-2, A is correct in position 3. T and I are in the word but wrong positions. T was wrong in positions 2 and 5, so it must be in position 4. I was wrong in position 4, so it's likely in position 5. The word is BRAIT.</think>
<guess>BRAIT</guess>

"""


def add_few_shot_examples(prompt: str, num_examples: int = 3) -> str:
    """
    Add few-shot examples to a prompt to teach the model the correct format.

    Args:
        prompt: The original prompt from the dataset.
        num_examples: Number of examples to include (1-3).

    Returns:
        Enhanced prompt with few-shot examples.
    """
    if num_examples <= 0:
        return prompt

    # Insert examples before the actual game starts
    # Find the end of the rules section
    if "### Your Turn:" in prompt or "<|im_start|>assistant" in prompt:
        # Insert examples before "Your Turn" or before assistant starts
        split_marker = "### Your Turn:" if "### Your Turn:" in prompt else "<|im_start|>assistant"
        parts = prompt.split(split_marker, 1)

        enhanced_prompt = (
            parts[0] +
            "\n\n### FORMAT EXAMPLES:\n" +
            "Here are examples of the correct format for your responses:\n" +
            FEW_SHOT_EXAMPLES +
            "\nNow it's your turn!\n\n" +
            split_marker +
            (parts[1] if len(parts) > 1 else "")
        )

        return enhanced_prompt
    else:
        # If no clear split point, append examples before the prompt content
        return prompt + "\n\n" + FEW_SHOT_EXAMPLES


def simplify_prompt(prompt: str) -> str:
    """
    Simplify and shorten the prompt while keeping essential information.

    Args:
        prompt: The original prompt.

    Returns:
        Simplified prompt.
    """
    # This is a placeholder - you can customize based on prompt structure
    return prompt


def create_wordle_prompt(
    past_guesses: List[tuple] = None,
    use_few_shot: bool = True,
    num_examples: int = 2
) -> str:
    """
    Create a Wordle prompt from scratch with optional few-shot examples.

    Args:
        past_guesses: List of (guess, feedback) tuples.
        use_few_shot: Whether to include few-shot examples.
        num_examples: Number of few-shot examples to include.

    Returns:
        Formatted prompt string.
    """
    prompt = """<|im_start|>system

You are playing Wordle. Guess a 5-letter word.

### Rules:
- Make strategic guesses to find the secret word
- Use feedback: ✓ = correct position, - = wrong position, x = not in word
"""

    if use_few_shot and num_examples > 0:
        prompt += "\n### IMPORTANT - Response Format:\n"
        prompt += "You MUST respond in this exact format:\n"
        prompt += "<think>your reasoning here</think>\n"
        prompt += "<guess>WORD</guess>\n\n"

        if num_examples >= 1:
            prompt += "### Example:\n"
            prompt += "<think>I'll try CRANE as a starting word with good coverage.</think>\n"
            prompt += "<guess>CRANE</guess>\n\n"

    prompt += "<|im_end|>\n<|im_start|>user\n\n"

    # Add game history
    if past_guesses and len(past_guesses) > 0:
        prompt += "Previous guesses:\n"
        for i, (guess, feedback) in enumerate(past_guesses, 1):
            prompt += f"Guess {i}: {feedback}\n"
    else:
        prompt += "No previous guesses yet.\n"

    prompt += "\nMake your next guess. Remember to use the format:\n"
    prompt += "<think>your reasoning</think>\n<guess>WORD</guess>\n"
    prompt += "<|im_end|>\n<|im_start|>assistant\n\n<think>"

    return prompt
