"""
Model Comparison Script for Wordle
Compares three models:
1. Baseline: Pre-trained Qwen2.5-3B-Instruct (no training)
2. SFT: After Supervised Fine-Tuning
3. GRPO: After Group Relative Policy Optimization

This script runs the same validation games on all three models to compare performance.
"""

import os
import re
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List
from logger_setup import logger
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
import torch

# ----------------------#
# 1. CONFIGURATION      #
# ----------------------#

# Model paths
BASELINE_MODEL = "google/gemma-3-4b-it"
SFT_MODEL = "output_sft/wordle-sft-peft/final_model"
GRPO_MODEL = "output5/wordle-grpo-optimized/checkpoint-120"

# Test configuration
NUM_TEST_GAMES = 10  # Number of games to test per model
RANDOM_SEED = 42

# ----------------------#
# 2. DATA STRUCTURES    #
# ----------------------#

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

# ----------------------#
# 3. SYSTEM PROMPT      #
# ----------------------#

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example (GENERIC - NOT from training data):
Secret Word: PLUMB

Guess 1: CLIMB → Feedback: C(x) L(✓) I(x) M(✓) B(✓)
  Analysis: L is correct at position 2, M at position 4, B at position 5
  C and I are not in the word at all

Guess 2: PLUMB → Feedback: P(✓) L(✓) U(✓) M(✓) B(✓)
  SUCCESS!

### FIX: EXPLICIT POSITION MASKS
When you see feedback like "R(-)", it means:
- The letter R IS in the secret word
- BUT R is NOT in the position where you guessed it
- You MUST try R in a DIFFERENT position
- NEVER reuse the same position that gave you "(-)"

When you see feedback like "T(x)", it means:
- The letter T is NOT in the secret word AT ALL
- NEVER use T again in any position
- This is a "dead letter" - eliminate it completely

When you see feedback like "B(✓)", it means:
- The letter B IS in the secret word
- AND B IS in the CORRECT position
- ALWAYS keep B in this exact position for future guesses

### Response Format:
Think through the problem and feedback step by step. Make sure to
first add your step by step thought process within <think> </think>
tags. Then, return your guessed word in the following format:
<guess> guessed-word </guess>.

### IMPORTANT CONSTRAINTS:
- DO NOT reference any specific words from previous games
- DO NOT say "In the first guess..." when you haven't made a first guess yet
- ONLY use information from the feedback you actually received in THIS game
- If you have no past feedback, start fresh with common letters (like ADIEU, RAISE, OCEAN)
"""

# ----------------------#
# 4. PROMPT FUNCTIONS   #
# ----------------------#

def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """Enhanced prompt with explicit position tracking"""
    prompt = "Make a new 5-letter word guess."

    if past_guesses:
        prompt += "\n\nHere is the feedback from your previous guesses in THIS CURRENT GAME:"

        # Track letters explicitly with position information
        confirmed_positions = {}  # position -> letter (✓)
        wrong_positions = {}      # letter -> set of wrong positions (-)
        dead_letters = set()      # letters not in word (x)

        for i, guess_obj in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess_obj}"

            # Track letter states
            for pos, (letter, fb) in enumerate(zip(guess_obj.guess, guess_obj.feedback)):
                if fb == LetterFeedback.CORRECT:
                    confirmed_positions[pos] = letter
                elif fb == LetterFeedback.WRONG_POS:
                    if letter not in wrong_positions:
                        wrong_positions[letter] = set()
                    wrong_positions[letter].add(pos)
                elif fb == LetterFeedback.WRONG_LETTER:
                    dead_letters.add(letter)

        # Add explicit summary
        if confirmed_positions or wrong_positions or dead_letters:
            prompt += "\n\n### FEEDBACK SUMMARY:"

            if confirmed_positions:
                prompt += "\n✓ Confirmed positions (KEEP these exact positions):"
                for pos, letter in sorted(confirmed_positions.items()):
                    prompt += f"\n  Position {pos+1}: {letter}"

            if wrong_positions:
                prompt += "\n- Letters in word but WRONG position (try different positions):"
                for letter, positions in sorted(wrong_positions.items()):
                    pos_list = ", ".join([str(p+1) for p in sorted(positions)])
                    prompt += f"\n  {letter}: DO NOT use at position(s) {pos_list}"

            if dead_letters:
                prompt += "\nx Dead letters (NEVER use these again):"
                prompt += f"\n  {', '.join(sorted(dead_letters))}"

    return prompt

def render_prompt(past_guesses: List[GuessWithFeedback]):
    return SYSTEM_PROMPT + "\n" + render_user_prompt(past_guesses) + "\nLet me solve this step by step.\n<think>"

# ----------------------#
# 5. GAME LOGIC         #
# ----------------------#

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """Calculate Wordle feedback for a guess"""
    guess = guess.upper()
    secret = secret_word.upper()
    feedback = [None] * 5
    secret_counts = {}

    # Count letters in secret
    for letter in secret:
        secret_counts[letter] = secret_counts.get(letter, 0) + 1

    # First pass: Mark correct positions
    for i, (g_letter, s_letter) in enumerate(zip(guess, secret)):
        if g_letter == s_letter:
            feedback[i] = LetterFeedback.CORRECT
            secret_counts[g_letter] -= 1

    # Second pass: Mark wrong positions
    for i, g_letter in enumerate(guess):
        if feedback[i] is None:  # Not already marked as correct
            if g_letter in secret_counts and secret_counts[g_letter] > 0:
                feedback[i] = LetterFeedback.WRONG_POS
                secret_counts[g_letter] -= 1
            else:
                feedback[i] = LetterFeedback.WRONG_LETTER

    return feedback

def generate_guess(model, tokenizer, prompt: str, verbose: bool = False) -> str:
    """Generate a guess from the model"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt from the output
    completion = completion[len(prompt):]

    if verbose:
        print(f"Model completion: {completion[:200]}...")

    return completion

def extract_guess_from_completion(completion: str) -> str:
    """Extract the guess word from model completion"""
    match = re.search(r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL)
    if not match:
        return None
    guess = match.group(1).strip().upper()
    return guess

def play_game(model, tokenizer, secret_word: str, model_name: str, verbose: bool = True) -> dict:
    """Play one game of Wordle with a model"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing game with {model_name}")
        print(f"Secret word: {secret_word}")
        print(f"{'='*60}")

    past_guesses = []
    game_log = []

    for turn in range(1, 7):
        if verbose:
            print(f"\n--- Turn {turn} ---")

        # Generate prompt
        prompt = render_prompt(past_guesses)

        # Get model's guess
        completion = generate_guess(model, tokenizer, prompt, verbose=verbose)
        guess = extract_guess_from_completion(completion)

        if not guess:
            if verbose:
                print(f"❌ Invalid output format (no <guess> tag found)")
            game_log.append({
                "turn": turn,
                "guess": None,
                "error": "Invalid format",
                "completion_preview": completion[:200]
            })
            continue

        if len(guess) != 5:
            if verbose:
                print(f"❌ Invalid guess length: {len(guess)} (expected 5): {guess}")
            game_log.append({
                "turn": turn,
                "guess": guess,
                "error": f"Invalid length: {len(guess)}"
            })
            continue

        # Get feedback
        feedback = get_feedback(guess, secret_word)
        guess_obj = GuessWithFeedback(guess, feedback)
        past_guesses.append(guess_obj)

        if verbose:
            print(f"Guess: {guess_obj}")

        game_log.append({
            "turn": turn,
            "guess": guess,
            "feedback": [fb.value for fb in feedback],
            "feedback_str": str(guess_obj)
        })

        # Check if won
        if guess == secret_word.upper():
            if verbose:
                print(f"🎉 SUCCESS in {turn} turns!")
            return {
                "secret": secret_word,
                "model": model_name,
                "result": "WIN",
                "turns": turn,
                "guesses": game_log
            }

    # Game over - no more turns
    if verbose:
        print(f"❌ LOSS - Failed to guess {secret_word}")

    return {
        "secret": secret_word,
        "model": model_name,
        "result": "LOSS",
        "turns": 6,
        "guesses": game_log
    }

# ----------------------#
# 6. MODEL LOADING      #
# ----------------------#

def load_model(model_path: str, model_name: str):
    """Load a model and tokenizer"""
    logger.info(f"Loading {model_name} from {model_path}...")
    print(f"\nLoading {model_name}...")

    try:
        # Check if this is a PEFT model (has adapter_config.json)
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_peft:
            # Load base model first, then adapters
            base_model_name = "google/gemma-3-4b-it"
            print(f"  Loading base model: {base_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="mps",
                torch_dtype=torch.float32
            )

            print(f"  Loading PEFT adapters from: {model_path}...")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model directly
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="mps")

        # Setup tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info(f"✓ {model_name} loaded successfully")
        print(f"✓ {model_name} loaded successfully")

        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None

# ----------------------#
# 7. MAIN COMPARISON    #
# ----------------------#

def main():
    print("="*80)
    print("WORDLE MODEL COMPARISON TEST")
    print("="*80)

    # Load dataset for test words
    print("\nLoading test dataset...")
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()
    valid_secrets = dataset['secret'].astype(str)
    valid_secrets = valid_secrets[valid_secrets.str.len() == 5]
    valid_secrets = valid_secrets[valid_secrets.str.isalpha()]

    # Get test set
    _, test_secrets = train_test_split(valid_secrets, test_size=0.1, random_state=RANDOM_SEED)
    test_words = test_secrets.head(NUM_TEST_GAMES).tolist()

    print(f"Testing on {NUM_TEST_GAMES} words: {test_words}")

    # Results storage
    all_results = {
        "baseline": [],
        "sft": [],
        "grpo": []
    }

    # Test each model
    models_to_test = [
        # ("baseline", BASELINE_MODEL, "Baseline (Pre-trained)"),
        # ("sft", SFT_MODEL, "SFT (Supervised Fine-Tuning)"),
        ("grpo", GRPO_MODEL, "GRPO Checkpoint-120")
    ]

    for model_key, model_path, model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print(f"{'='*80}")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}, skipping...")
            continue

        # Load model
        model, tokenizer = load_model(model_path, model_name)
        if model is None:
            continue

        # Test on all words
        for i, secret_word in enumerate(test_words, 1):
            print(f"\n{'─'*60}")
            print(f"Game {i}/{NUM_TEST_GAMES}: {secret_word}")
            print(f"{'─'*60}")

            result = play_game(model, tokenizer, secret_word, model_name, verbose=True)
            all_results[model_key].append(result)

        # Calculate stats for this model
        wins = sum(1 for r in all_results[model_key] if r["result"] == "WIN")
        avg_turns = sum(r["turns"] for r in all_results[model_key] if r["result"] == "WIN") / wins if wins > 0 else 0

        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {model_name}")
        print(f"{'='*60}")
        print(f"Games played: {len(all_results[model_key])}")
        print(f"Wins: {wins}/{len(all_results[model_key])} ({100*wins/len(all_results[model_key]):.1f}%)")
        print(f"Average turns (wins only): {avg_turns:.2f}")
        print(f"{'='*60}")

    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}\n")

    comparison_table = []
    for model_key, _, model_name in models_to_test:
        if all_results[model_key]:
            wins = sum(1 for r in all_results[model_key] if r["result"] == "WIN")
            total = len(all_results[model_key])
            win_rate = 100 * wins / total if total > 0 else 0
            avg_turns = sum(r["turns"] for r in all_results[model_key] if r["result"] == "WIN") / wins if wins > 0 else 0

            comparison_table.append({
                "model": model_name,
                "wins": wins,
                "total": total,
                "win_rate": win_rate,
                "avg_turns": avg_turns
            })

    # Print comparison table
    print(f"{'Model':<40} {'Wins':<10} {'Win Rate':<15} {'Avg Turns'}")
    print(f"{'-'*80}")
    for row in comparison_table:
        print(f"{row['model']:<40} {row['wins']}/{row['total']:<7} {row['win_rate']:>6.1f}%        {row['avg_turns']:>6.2f}")

    # Save detailed results to JSON
    output_file = "model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_words": test_words,
            "results": all_results,
            "summary": comparison_table
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
