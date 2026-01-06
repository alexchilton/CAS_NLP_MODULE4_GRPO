"""
Test Gemma-3-4b-it Model with Structured Prompts
=================================================

Tests the google/gemma-3-4b-it model using the structured prompt
system from wordle-rl-gemma to see how well it performs WITHOUT any training.

Optimized for memory efficiency with bfloat16.

Usage:
    python test_gemma.py --games 20 --temperature 0.1
    python test_gemma.py --games 150 --temperature 0.7 --seed 123
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path

# Simple logging (no external dependencies)
class SimpleLogger:
    def info(self, msg):
        print(msg)

logger = SimpleLogger()

from prompt_system import build_messages, SYSTEM_PROMPT
from wordle_game import WordleGame, extract_guess_from_completion, extract_thinking


def play_game(
    model,
    tokenizer,
    secret_word: str,
    word_list: set,
    temperature: float = 0.7,
    max_guesses: int = 6,
    verbose: bool = True,
    detailed_log_file = None
) -> dict:
    """
    Play one game of Wordle with the structured prompt system.

    Args:
        model: The language model
        tokenizer: The tokenizer
        secret_word: The secret word to guess
        word_list: Set of valid words
        temperature: Sampling temperature
        max_guesses: Maximum number of guesses
        verbose: Whether to print detailed output

    Returns:
        Dictionary with game results
    """
    game = WordleGame(secret_word, word_list)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Playing Wordle | Secret: {secret_word.upper()}")
        print(f"{'='*80}")

    if detailed_log_file:
        detailed_log_file.write(f"\n{'='*80}\n")
        detailed_log_file.write(f"Playing Wordle | Secret: {secret_word.upper()}\n")
        detailed_log_file.write(f"{'='*80}\n")

    for turn in range(1, max_guesses + 1):
        if verbose:
            print(f"\n--- Turn {turn}/{max_guesses} ---")

        # Build messages with current game state
        messages = build_messages(game.get_history())

        # Format for model (using chat template)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if detailed_log_file:
            detailed_log_file.write(f"\n--- Turn {turn}/{max_guesses} ---\n")
            detailed_log_file.write(f"\nFULL PROMPT:\n")
            detailed_log_file.write(f"System: {messages[0]['content'][:200]}...\n")
            detailed_log_file.write(f"\nUser: {messages[1]['content']}\n")

        if verbose and turn == 1:
            # Show the prompt for first turn
            print(f"\nPROMPT (Turn 1):")
            print(messages[-1]['content'][:300] + "...")

        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Gemma should be more concise
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.95,
            )

        # Extract completion (remove prompt)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_output[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]

        # Extract guess and thinking
        thinking = extract_thinking(completion)
        guess = extract_guess_from_completion(completion)

        if detailed_log_file:
            detailed_log_file.write(f"\nFULL RAW OUTPUT:\n")
            detailed_log_file.write(f"{completion}\n")
            detailed_log_file.write(f"\nFULL THINKING:\n")
            detailed_log_file.write(f"{thinking}\n")
            detailed_log_file.write(f"\nEXTRACTED GUESS: {guess}\n")

        if verbose:
            print(f"\nRAW OUTPUT:")
            print(completion[:400] if len(completion) > 400 else completion)
            if thinking:
                thinking_preview = thinking[:150] + "..." if len(thinking) > 150 else thinking
                print(f"\nEXTRACTED THINKING: {thinking_preview}")

        # Validate and make guess
        if guess is None:
            if verbose:
                print(f"\n❌ INVALID: Could not extract valid guess from output")
            return {
                "secret": secret_word,
                "won": False,
                "turns": turn,
                "status": "invalid_format",
                "guesses": [fb.guess for fb in game.get_history()],
            }

        feedback, status = game.make_guess(guess)

        if feedback is None:
            if verbose:
                print(f"\n❌ INVALID GUESS: {status}")
            return {
                "secret": secret_word,
                "won": False,
                "turns": turn,
                "status": f"invalid_guess_{status}",
                "guesses": [fb.guess for fb in game.get_history()],
            }

        if detailed_log_file:
            readable_feedback = []
            for i, (letter, fb_char) in enumerate(zip(guess, feedback.feedback.split())):
                if fb_char == 'G':
                    readable_feedback.append(f"{letter}(✓)")
                elif fb_char == 'Y':
                    readable_feedback.append(f"{letter}(-)")
                else:
                    readable_feedback.append(f"{letter}(x)")
            detailed_log_file.write(f"\nGUESS: {guess}\n")
            detailed_log_file.write(f"FEEDBACK: {' '.join(readable_feedback)}\n")
            detailed_log_file.write(f"STATUS: {status}\n")

        if verbose:
            print(f"\nGUESS: {guess}")
            # Convert feedback to readable format
            readable_feedback = []
            for i, (letter, fb_char) in enumerate(zip(guess, feedback.feedback.split())):
                if fb_char == 'G':
                    readable_feedback.append(f"{letter}(✓)")
                elif fb_char == 'Y':
                    readable_feedback.append(f"{letter}(-)")
                else:
                    readable_feedback.append(f"{letter}(x)")
            print(f"FEEDBACK: {' '.join(readable_feedback)}")

        if status == "win":
            if verbose:
                print(f"\n🎉 WON in {turn} guesses!")
            return {
                "secret": secret_word,
                "won": True,
                "turns": turn,
                "status": "win",
                "guesses": [fb.guess for fb in game.get_history()],
            }
        elif status == "loss":
            if verbose:
                print(f"\n😞 LOST after {turn} guesses. Secret was: {secret_word}")
            return {
                "secret": secret_word,
                "won": False,
                "turns": turn,
                "status": "loss",
                "guesses": [fb.guess for fb in game.get_history()],
            }

    # Should not reach here, but just in case
    return {
        "secret": secret_word,
        "won": False,
        "turns": max_guesses,
        "status": "loss",
        "guesses": [fb.guess for fb in game.get_history()],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemma-3-4b-it model with structured prompts"
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                       help="Model name or path")
    parser.add_argument("--games", type=int, default=20,
                       help="Number of games to play")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (0.1 = deterministic, 0.7 = creative)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for word selection")
    parser.add_argument("--word-list", type=str, default="../../five_letter_words.csv",
                       help="Path to word list CSV")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed game output")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load word list
    logger.info(f"Loading word list from {args.word_list}...")
    word_df = pd.read_csv(args.word_list)
    word_list = set(word_df["Word"].str.upper().tolist())
    logger.info(f"Loaded {len(word_list)} words")

    # Load model with bfloat16 for memory efficiency
    logger.info(f"\nLoading model: {args.model}...")
    logger.info("Using bfloat16 for memory efficiency on MPS")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Memory efficient: ~4GB instead of ~8GB
    )
    model.eval()
    logger.info("Model loaded successfully!")

    # Select random secret words
    secret_words = random.sample(list(word_list), args.games)

    # Play games
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing BASE MODEL: {args.model}")
    logger.info(f"Prompt System: Structured (wordle-rl-gemma style)")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Precision: bfloat16")
    logger.info(f"Playing {args.games} games...")
    logger.info(f"{'='*80}")

    # Create detailed log file
    detailed_log_path = f"gemma_detailed_log_temp{args.temperature}_games{args.games}.txt"
    with open(detailed_log_path, 'w') as detailed_log:
        detailed_log.write(f"DETAILED LOG - Gemma-3-4b-it Testing\n")
        detailed_log.write(f"Model: {args.model}\n")
        detailed_log.write(f"Temperature: {args.temperature}\n")
        detailed_log.write(f"Precision: bfloat16\n")
        detailed_log.write(f"Games: {args.games}\n")
        detailed_log.write(f"{'='*80}\n")

        results = []
        for i, secret in enumerate(secret_words, 1):
            detailed_log.write(f"\n\n{'#'*80}\n")
            detailed_log.write(f"GAME {i}/{args.games}\n")
            detailed_log.write(f"{'#'*80}\n")

            if args.verbose:
                print(f"\n\n{'#'*80}")
                print(f"GAME {i}/{args.games}")
                print(f"{'#'*80}")

            result = play_game(
                model,
                tokenizer,
                secret,
                word_list,
                temperature=args.temperature,
                verbose=args.verbose,
                detailed_log_file=detailed_log
            )
            results.append(result)
            detailed_log.flush()  # Ensure it's written immediately

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    wins = sum(1 for r in results if r["won"])
    losses = sum(1 for r in results if r["status"] == "loss")
    invalid = sum(1 for r in results if r["status"].startswith("invalid"))

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Games played: {args.games}")
    print(f"Wins: {wins} ({100*wins/args.games:.1f}%)")
    print(f"Losses: {losses} ({100*losses/args.games:.1f}%)")
    print(f"Invalid: {invalid} ({100*invalid/args.games:.1f}%)")

    if wins > 0:
        avg_turns = sum(r["turns"] for r in results if r["won"]) / wins
        print(f"Average turns to win: {avg_turns:.2f}")

    print(f"\nDetailed results:")
    for i, r in enumerate(results, 1):
        status_emoji = "🎉" if r["won"] else ("❌" if r["status"].startswith("invalid") else "😞")
        print(f"  {status_emoji} Game {i}: {r['secret']} - {r['status']} in {r['turns']} turns")

    print(f"\n{'='*80}")

    # Save results
    output_file = f"gemma_results_temp{args.temperature}_games{args.games}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Precision: bfloat16\n")
        f.write(f"Games: {args.games}\n")
        f.write(f"Wins: {wins}/{args.games} ({100*wins/args.games:.1f}%)\n")
        f.write(f"Losses: {losses}/{args.games} ({100*losses/args.games:.1f}%)\n")
        f.write(f"Invalid: {invalid}/{args.games} ({100*invalid/args.games:.1f}%)\n")
        if wins > 0:
            f.write(f"Avg turns to win: {avg_turns:.2f}\n")
        f.write("\nDetailed results:\n")
        for i, r in enumerate(results, 1):
            f.write(f"Game {i}: {r['secret']} - {r['status']} in {r['turns']} turns - {r['guesses']}\n")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Detailed log saved to: {detailed_log_path}")


if __name__ == "__main__":
    main()
