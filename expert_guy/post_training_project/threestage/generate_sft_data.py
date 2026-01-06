"""
Synthetic SFT Data Generator for Wordle Format Learning
========================================================

Stage 1 focuses on teaching the model the correct output format:
- <think>reasoning</think><guess>WORD</guess>

This generator creates diverse examples with:
- Different reasoning styles (letters, positions, constraints)
- Various secret words

- Different game states (first guess, mid-game, late-game)
- Valid 5-letter words only

Goal: 1000+ training examples for robust format learning
"""

import pandas as pd
import random
import json
from pathlib import Path
from typing import List, Tuple


def load_word_list(path: str = "../five_letter_words.csv") -> List[str]:
    """Load the Wordle word list"""
    df = pd.read_csv(path)
    return df["Word"].str.upper().tolist()


def generate_first_guess_examples(words: List[str], n: int = 300) -> List[dict]:
    """
    Generate examples for first guess (no history).
    Focus on common starting words and reasoning patterns.
    """
    examples = []
    common_starters = ["CRANE", "SLATE", "ARISE", "STARE", "AUDIO", "ROAST", "TEARS", "PILOT"]

    reasoning_templates = [
        "I'll start with a word containing common vowels and consonants to maximize information gain.",
        "Let me begin with a word that has high-frequency letters like E, A, R, S, and T.",
        "Starting with a word containing multiple vowels will help narrow down possibilities quickly.",
        "I'll choose a word with common letters in typical positions for English words.",
        "Beginning with a balanced mix of vowels and consonants in common positions.",
    ]

    for i in range(n):
        if i < len(common_starters):
            guess = common_starters[i % len(common_starters)]
        else:
            guess = random.choice([w for w in words if len(set(w)) >= 4])  # Prefer diverse letters

        reasoning = random.choice(reasoning_templates)

        prompt = """You are playing Wordle. Your goal is to guess a 5-letter word.

Output format:
<think>your reasoning</think><guess>WORD</guess>

Make your first guess:"""

        completion = f"<think>{reasoning}</think><guess>{guess}</guess>"

        examples.append({
            "prompt": prompt,
            "completion": completion,
            "guess": guess,
            "stage": "first_guess"
        })

    return examples


def generate_feedback_string(guess: str, secret: str) -> str:
    """Generate Wordle feedback for a guess"""
    feedback = []
    secret_counts = {}
    for c in secret:
        secret_counts[c] = secret_counts.get(c, 0) + 1

    # First pass: mark correct positions
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            feedback.append(f"{g}(✓)")
            secret_counts[g] -= 1
        else:
            feedback.append(None)

    # Second pass: mark wrong positions and misses
    for i, (g, s) in enumerate(zip(guess, secret)):
        if feedback[i] is None:
            if g in secret and secret_counts.get(g, 0) > 0:
                feedback[i] = f"{g}(-)"
                secret_counts[g] -= 1
            else:
                feedback[i] = f"{g}(x)"

    return " ".join(feedback)


def generate_mid_game_examples(words: List[str], n: int = 400) -> List[dict]:
    """
    Generate examples with 1-3 previous guesses.
    Focus on using feedback correctly.
    """
    examples = []

    for _ in range(n):
        secret = random.choice(words)
        num_guesses = random.randint(1, 3)

        # Generate random previous guesses (not the secret)
        past_guesses = []
        available_words = [w for w in words if w != secret]

        for _ in range(num_guesses):
            guess = random.choice(available_words)
            feedback = generate_feedback_string(guess, secret)
            past_guesses.append((guess, feedback))

        # Generate reasoning based on feedback
        reasoning = generate_reasoning_from_feedback(past_guesses, secret, words)

        # Create prompt with history
        prompt_text = "You are playing Wordle. Your goal is to guess a 5-letter word.\n\nPrevious guesses:\n"
        for guess, feedback in past_guesses:
            prompt_text += f"{guess}: {feedback}\n"

        prompt_text += "\nOutput format:\n<think>your reasoning</think><guess>WORD</guess>\n\nMake your next guess:"

        # Pick a valid next guess that uses the feedback
        next_guess = pick_smart_guess(past_guesses, secret, words)

        completion = f"<think>{reasoning}</think><guess>{next_guess}</guess>"

        examples.append({
            "prompt": prompt_text,
            "completion": completion,
            "guess": next_guess,
            "stage": "mid_game",
            "secret": secret
        })

    return examples


def generate_reasoning_from_feedback(past_guesses: List[Tuple[str, str]], secret: str, words: List[str]) -> str:
    """Generate plausible reasoning based on past feedback"""
    confirmed = []
    wrong_pos = []
    dead = []

    for guess, feedback in past_guesses:
        parts = feedback.split()
        for i, part in enumerate(parts):
            letter = guess[i]
            if "(✓)" in part:
                confirmed.append((letter, i))
            elif "(-)" in part:
                wrong_pos.append(letter)
            elif "(x)" in part:
                dead.append(letter)

    reasoning_parts = []

    if confirmed:
        reasoning_parts.append(f"I know {', '.join([f'{l} is at position {p+1}' for l, p in confirmed[:2]])}.")

    if wrong_pos:
        unique_wrong = list(set(wrong_pos))[:2]
        reasoning_parts.append(f"The word contains {', '.join(unique_wrong)} but in different positions.")

    if dead:
        unique_dead = list(set(dead))[:3]
        reasoning_parts.append(f"I can eliminate {', '.join(unique_dead)}.")

    reasoning_parts.append("Let me choose a word that satisfies these constraints.")

    return " ".join(reasoning_parts)


def pick_smart_guess(past_guesses: List[Tuple[str, str]], secret: str, words: List[str]) -> str:
    """Pick a guess that respects feedback constraints (and ideally makes progress)"""
    confirmed_positions = {}
    valid_letters = set()
    dead_letters = set()
    wrong_positions = {}  # letter -> positions where it's wrong

    for guess, feedback in past_guesses:
        parts = feedback.split()
        for i, part in enumerate(parts):
            letter = guess[i]
            if "(✓)" in part:
                confirmed_positions[i] = letter
                valid_letters.add(letter)
            elif "(-)" in part:
                valid_letters.add(letter)
                if letter not in wrong_positions:
                    wrong_positions[letter] = []
                wrong_positions[letter].append(i)
            elif "(x)" in part:
                dead_letters.add(letter)

    # Filter candidates
    candidates = []
    for word in words:
        # Check confirmed positions
        valid = True
        for pos, letter in confirmed_positions.items():
            if word[pos] != letter:
                valid = False
                break
        if not valid:
            continue

        # Check valid letters are present
        for letter in valid_letters:
            if letter not in word:
                valid = False
                break
        if not valid:
            continue

        # Check wrong positions
        for letter, positions in wrong_positions.items():
            for pos in positions:
                if word[pos] == letter:
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

    if not candidates:
        # If no valid candidates (shouldn't happen with correct secret), return secret
        return secret

    # Prefer the actual secret if it's in candidates (makes example more realistic)
    if secret in candidates and random.random() < 0.3:
        return secret

    return random.choice(candidates)


def generate_late_game_examples(words: List[str], n: int = 300) -> List[dict]:
    """
    Generate examples with 4-5 previous guesses (near the end).
    Focus on precision and using all constraints.
    """
    examples = []

    for _ in range(n):
        secret = random.choice(words)
        num_guesses = random.randint(4, 5)

        past_guesses = []
        available_words = [w for w in words if w != secret]

        for _ in range(num_guesses):
            if len(available_words) == 0:
                break
            guess = random.choice(available_words)
            feedback = generate_feedback_string(guess, secret)
            past_guesses.append((guess, feedback))
            available_words = [w for w in available_words if w != guess]

        reasoning = generate_late_game_reasoning(past_guesses, secret)

        prompt_text = "You are playing Wordle. Your goal is to guess a 5-letter word.\n\nPrevious guesses:\n"
        for guess, feedback in past_guesses:
            prompt_text += f"{guess}: {feedback}\n"

        prompt_text += "\nOutput format:\n<think>your reasoning</think><guess>WORD</guess>\n\nMake your next guess:"

        # For late game, often guess the secret
        next_guess = secret if random.random() < 0.5 else pick_smart_guess(past_guesses, secret, words)

        completion = f"<think>{reasoning}</think><guess>{next_guess}</guess>"

        examples.append({
            "prompt": prompt_text,
            "completion": completion,
            "guess": next_guess,
            "stage": "late_game",
            "secret": secret
        })

    return examples


def generate_late_game_reasoning(past_guesses: List[Tuple[str, str]], secret: str) -> str:
    """Generate reasoning for late-game scenarios"""
    reasoning_templates = [
        "Based on all the feedback, I can narrow down the possibilities to very few words. Let me choose the most likely candidate.",
        "With so many constraints identified, there are only a handful of valid words remaining. I'll pick the one that best fits.",
        "Combining all the position and letter information, I can deduce the most probable answer.",
        "After eliminating so many options, the answer should be clear from the remaining constraints.",
    ]
    return random.choice(reasoning_templates)


def main():
    """Generate synthetic SFT dataset"""
    print("Loading word list...")
    words = load_word_list()
    print(f"Loaded {len(words)} words")

    print("\nGenerating examples...")
    all_examples = []

    print("- First guess examples (1000)...")
    all_examples.extend(generate_first_guess_examples(words, n=1000))

    print("- Mid-game examples (1000)...")
    all_examples.extend(generate_mid_game_examples(words, n=1000))

    print("- Late-game examples (1000)...")
    all_examples.extend(generate_late_game_examples(words, n=1000))

    print(f"\nGenerated {len(all_examples)} total examples")

    # Shuffle
    random.shuffle(all_examples)

    # Save as JSONL
    output_path = Path("sft_synthetic_data.jsonl")
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {output_path}")

    # Also save as CSV for inspection
    df = pd.DataFrame(all_examples)
    df.to_csv("sft_synthetic_data.csv", index=False)
    print(f"Also saved as sft_synthetic_data.csv for inspection")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total examples: {len(all_examples)}")
    print(f"First guess: {sum(1 for ex in all_examples if ex['stage'] == 'first_guess')}")
    print(f"Mid-game: {sum(1 for ex in all_examples if ex['stage'] == 'mid_game')}")
    print(f"Late-game: {sum(1 for ex in all_examples if ex['stage'] == 'late_game')}")

    # Show sample
    print("\n=== Sample Examples ===")
    for i, ex in enumerate(all_examples[:3]):
        print(f"\n--- Example {i+1} ({ex['stage']}) ---")
        print(f"Prompt: {ex['prompt'][:100]}...")
        print(f"Completion: {ex['completion'][:100]}...")


if __name__ == "__main__":
    random.seed(42)
    main()