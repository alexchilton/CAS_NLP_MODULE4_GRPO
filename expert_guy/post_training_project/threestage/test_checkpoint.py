"""
Test a checkpoint by playing Wordle games
==========================================

Loads a checkpoint and plays N games of Wordle, showing:
- The model's reasoning (<think> content)
- The model's guess
- Feedback from the game
- Whether the guess was valid
- Game outcome (win/loss)

Usage:
    python test_checkpoint.py --checkpoint stage1_output/checkpoint-650 --games 10
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

sys.path.append('..')
from logger_setup import logger


class WordleGame:
    """Simple Wordle game simulator"""

    def __init__(self, secret_word, word_list):
        self.secret = secret_word.upper()
        self.word_list = set(w.upper() for w in word_list)
        self.guesses = []
        self.max_guesses = 6

    def make_guess(self, guess):
        """Make a guess and return feedback"""
        guess = guess.upper()

        if len(guess) != 5:
            return None, "Invalid: not 5 letters"

        if guess not in self.word_list:
            return None, "Invalid: not in word list"

        # Generate feedback
        feedback = []
        secret_counts = {}
        for c in self.secret:
            secret_counts[c] = secret_counts.get(c, 0) + 1

        # First pass: mark correct positions
        for i, (g, s) in enumerate(zip(guess, self.secret)):
            if g == s:
                feedback.append(f"{g}(✓)")
                secret_counts[g] -= 1
            else:
                feedback.append(None)

        # Second pass: mark wrong positions and misses
        for i, (g, s) in enumerate(zip(guess, self.secret)):
            if feedback[i] is None:
                if g in self.secret and secret_counts.get(g, 0) > 0:
                    feedback[i] = f"{g}(-)"
                    secret_counts[g] -= 1
                else:
                    feedback[i] = f"{g}(x)"

        feedback_str = " ".join(feedback)
        self.guesses.append((guess, feedback_str))

        # Check win
        won = guess == self.secret
        game_over = won or len(self.guesses) >= self.max_guesses

        return feedback_str, "win" if won else ("loss" if game_over else "continue")


def extract_guess_from_completion(completion, word_list):
    """Extract the guess from model completion"""
    # Try to find <guess>WORD</guess>
    guess_match = re.search(r"<guess>\s*([^<>]*?)\s*</guess>", completion, re.IGNORECASE | re.DOTALL)

    if not guess_match:
        return None, "No <guess> tag found"

    guess_text = guess_match.group(1).strip()

    # Clean up common patterns
    guess_text = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess_text, flags=re.IGNORECASE)

    # Extract only letters
    letters_only = re.sub(r'[^A-Z]', '', guess_text.upper())

    if len(letters_only) != 5:
        return None, f"Invalid length: {len(letters_only)} letters ('{letters_only}')"

    if letters_only not in word_list:
        return None, f"Not in word list: '{letters_only}'"

    return letters_only, "valid"


def extract_thinking(completion):
    """Extract the reasoning from <think> tags"""
    think_match = re.search(r"<think>(.*?)</think>", completion, re.IGNORECASE | re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return "No reasoning found"


def play_game(model, tokenizer, secret_word, word_list, max_guesses=6, verbose=True):
    """Play one game of Wordle"""
    game = WordleGame(secret_word, word_list)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Playing Wordle | Secret: {secret_word}")
        print(f"{'='*80}")

    for turn in range(1, max_guesses + 1):
        # Build prompt
        if len(game.guesses) == 0:
            prompt = """You are playing Wordle. Your goal is to guess a 5-letter word.

Output format:
<think>your reasoning</think><guess>WORD</guess>

Make your first guess:"""
        else:
            prompt = "You are playing Wordle. Your goal is to guess a 5-letter word.\n\nPrevious guesses:\n"
            for guess, feedback in game.guesses:
                prompt += f"{guess}: {feedback}\n"
            prompt += "\nOutput format:\n<think>your reasoning</think><guess>WORD</guess>\n\nMake your next guess:"

        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract guess and thinking
        thinking = extract_thinking(completion)
        guess, validity = extract_guess_from_completion(completion, word_list)

        if verbose:
            print(f"\n--- Turn {turn} ---")
            print(f"RAW OUTPUT:")
            print(f"{completion}")
            print(f"\nExtracted thinking: {thinking[:150]}..." if len(thinking) > 150 else f"\nExtracted thinking: {thinking}")

        if guess is None:
            if verbose:
                print(f"❌ Invalid guess: {validity}")
            return False, turn, "invalid_guess"

        # Make guess
        feedback, status = game.make_guess(guess)

        if verbose:
            print(f"Guess: {guess}")
            print(f"Feedback: {feedback}")

        if status == "win":
            if verbose:
                print(f"\n🎉 WON in {turn} guesses!")
            return True, turn, "win"
        elif status == "loss":
            if verbose:
                print(f"\n😞 LOST after {turn} guesses. Secret was: {secret_word}")
            return False, turn, "loss"

    if verbose:
        print(f"\n😞 LOST - ran out of guesses. Secret was: {secret_word}")
    return False, max_guesses, "loss"


def main():
    parser = argparse.ArgumentParser(description="Test a checkpoint by playing Wordle games")
    parser.add_argument("--checkpoint", type=str, default="stage1_output/checkpoint-650",
                        help="Path to checkpoint directory")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for word selection")
    parser.add_argument("--word-list", type=str, default="../five_letter_words.csv",
                        help="Path to word list CSV")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load word list
    print(f"Loading word list from {args.word_list}...")
    word_df = pd.read_csv(args.word_list)
    word_list = set(word_df["Word"].str.upper().tolist())
    print(f"Loaded {len(word_list)} words")

    # Load model and tokenizer
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        torch_dtype=torch.float32
    )
    model.eval()
    print("Model loaded successfully!")

    # Select random secret words
    secret_words = random.sample(list(word_list), args.games)

    # Play games
    print(f"\n{'='*80}")
    print(f"Playing {args.games} games...")
    print(f"{'='*80}")

    results = []
    for i, secret in enumerate(secret_words, 1):
        print(f"\n\n{'#'*80}")
        print(f"GAME {i}/{args.games}")
        print(f"{'#'*80}")

        won, turns, status = play_game(model, tokenizer, secret, word_list, verbose=True)
        results.append({
            "game": i,
            "secret": secret,
            "won": won,
            "turns": turns,
            "status": status
        })

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    wins = sum(1 for r in results if r["won"])
    losses = sum(1 for r in results if r["status"] == "loss")
    invalid = sum(1 for r in results if r["status"] == "invalid_guess")

    print(f"Games played: {args.games}")
    print(f"Wins: {wins} ({100*wins/args.games:.1f}%)")
    print(f"Losses: {losses} ({100*losses/args.games:.1f}%)")
    print(f"Invalid guesses: {invalid} ({100*invalid/args.games:.1f}%)")

    if wins > 0:
        avg_turns = sum(r["turns"] for r in results if r["won"]) / wins
        print(f"Average turns to win: {avg_turns:.2f}")

    print(f"\nDetailed results:")
    for r in results:
        status_emoji = "🎉" if r["won"] else ("❌" if r["status"] == "invalid_guess" else "😞")
        print(f"  {status_emoji} Game {r['game']}: {r['secret']} - {r['status']} in {r['turns']} turns")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()